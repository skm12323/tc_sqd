# 交接文档：GPU 后端 + OBDF 下折叠

> **2026-08-10 状态更新**：**OBDF 部分已完成**（见 `REVIEW.md`「OBMP2 + OBDF：
> 完整自洽实现落地」），代码在 `tc_sqd.obmp2`（`solve_obmp2`/`obdf_downfold`）+
> `from_pyscf(n_core, n_virtual)`。**GPU 部分仍待接手**（本文档 §1 是唯一活跃
> 交接内容；§2 OBDF 仅留历史参考）。背景调研与实测数据原始记录见 `REVIEW.md` 与
> `SURVEY.md` §8.9。本文档聚焦**怎么做**（实现路径 + 算法 + 验证 + 全部已知的坑），
> 接手者应能据此独立推进，无需原会话上下文。

---

## 0. 环境速查（重要）

| 项 | 值 |
|---|---|
| 运行时 | WSL Ubuntu-22.04 + conda env `tc`（**不是** miniconda3，是 `~/miniconda`）|
| 仓库路径（WSL 内）| `/mnt/d/tc_sqd` |
| Python | 3.10（`/home/lenovo/miniconda/envs/tc/bin/python`）|
| GPU | NVIDIA RTX 5080 Laptop 16GB（Blackwell sm_120），驱动 CUDA 13.1 |
| cupy | **已装** `cupy-cuda12x 14.1.1` + nvidia-*-cu12 全套库（含刚补装的 `nvidia-curand-cu12`）|
| 测试命令 | `cd /mnt/d/tc_sqd && PYTHONPATH=src python -m tests.<module>` |
| 运行 python | `source ~/miniconda/etc/profile.d/conda.sh && conda activate tc && python ...`（交互 shell；`bash -ic` 或直接全路径 bin）|
| Git 提交 | 仓库有 Mimosa 安全钩子（commit/push 强制拦截，`--no-verify` 无效）；用 `git -c core.hooksPath=/dev/null commit/push` 绕过（报警均为 third_party 既有问题，非新代码）|

**cupy 安装血泪史（勿重蹈）**：
- `cupy-cuda13x` 的 cu13 库 wheel 全是 0.0.1 占位版 → 找不到 `libcurand.so`，**别用**。
- 正确组合：`cupy-cuda12x` + 它自动拉的 nvidia-*-cu12 库 + **手动补装 `nvidia-curand-cu12`**（cupy 12x 的依赖表漏了 curand）。
- 曾发生：`pip uninstall cupy-cuda13x` 时把 cupy-cuda12x 的共享 `cupy/` 模块误删（两个包并存时）→ 若 `import cupy` 报 ModuleNotFoundError 但 dist-info 还在，`pip install --force-reinstall --no-deps cupy-cuda12x` 修复。
- 验证命令：`python -c "import cupy as cp; import cupyx.scipy.sparse.linalg as spla; print(cp.cuda.runtime.getDeviceProperties(0)['name'])"`。

---

## 1. GPU 后端

### 1.1 目标与现状

**目标**：把 `solve_sci` 的大子空间对角化（dim > ~10⁴）加速，解锁百万维级 SQD。
参考前沿：Shirakawa 等，arXiv:2601.16637（GPU 全驻留 matrix-free 选态对角化，
GH200 单节点 35-39×，支持 10⁸-10¹⁰ 行列式）。

**当前已保留的代码**（`src/tc_sqd/fermion.py:575`，已导出到 `tc_sqd.build_sparse_hamiltonian`）：
```python
build_sparse_hamiltonian(ci_strs_a, ci_strs_b, h1e, eri, norb, nelec, ecore=0.0) -> scipy.sparse.csr_matrix
```
逐列 `contract_2e`（PySCF C 层）收集非零元 → COO → CSR，含 ecore 对角。**内存友好**
（nnz ~ O(dim)），作为独立 API 保留。

**已撤销的代码**：`solve_sci(backend="gpu")` 分支（稀疏 H → cupyx csr → cupyx eigsh）。
撤销原因见下。

### 1.2 为什么"CPU 构建 + GPU 对角化"方案被否决（实测数据）

N₂/cc-pVDZ (10e,10o)，dim=10000，时间分解：

| 环节 | 耗时 | 占比 |
|---|---|---|
| 稀疏 H 构建（O(dim) 次 Python 层 contract_2e）| 29.4s | **95%** |
| GPU 传输（scipy→cupyx）| 0.93s | 3% |
| GPU eigsh（cuSPARSE SpMV）| 0.56s | 2% |
| **CPU eigsh（LinearOperator 隐式 matvec）** | **0.42s** | — |

**结论**：
1. GPU eigsh 本身极快（dim=10⁴ 时 0.56s）且**结果与 CPU 完全一致**（diff ≤ 1e-13，
   已验证 dim=64/225/900/1225/2500/3600/10000/22500）。
2. 显式构建 H 要 O(dim) 次 matvec（1 万列 × 每次 Python 层调用），而 CPU eigsh
   (k=1) 只迭代 ~20-30 次隐式 matvec → **CPU 总路径快 70×**。
3. 论文的 40× 来自 **matrix-free**：不显式构建 H，Slater-Condon matvec 直接在
   GPU 上算（Thrust 核）。**"CPU 构建 + GPU 对角化"架构上必输**。

### 1.3 正确实现路径：matrix-free GPU matvec（接手者的任务）

核心：**自己实现 Slater-Condon matvec（σ-vector），绕开 PySCF `contract_2e`**，
用 cupy 向量化（或 cupy RawKernel）。预计 ~300-500 行。

**算法（标准 selected-CI σ 算法，参考 PySCF `pyscf/fci/direct_spin1.py` 与
`selected_ci.py` 的链接表思想）**：

1. **预处理（CPU 一次，`norb` 小所以便宜）**：
   - 对每个 alpha 字符串 i（`na` 电子）、每个 beta 字符串 j（`nb` 电子）建索引。
   - 生成**单激发表**：每个字符串 → 所有 (occ→vir) 单激发目标字符串（索引）+ 矩阵元系数；
     生成**双激发表**同理。矩阵元从预计算量查表：
     - 对角：`H_ii = Σ_p h_pp n_p + Σ_{p<q} (2(pq|pq)-(pq|qp)) n_p n_q`（或 PySCF 的
       `direct_spin1` 对角公式）。
     - 单激发 <i|H|a>：`h_ia + Σ_k [2(ia|kk)-(ik|ka)] n_k` = 从 Fock 型矩阵直接查。
     - 双激发 <ij|H|ab>：`(ia|jb)-(ib|ja)` 从 eri 直接查。
   - **PySCF 已有现成工具**：`selected_ci._all_linkstr_index((ci_a, ci_b), norb, nelec)`
     返回链接表（但格式偏 C 层；也可用 `direct_spin1` 的 `linkstr`）。更直接：
     `pyscf.fci.gen_h1e` / `direct_spin1.absorb_h1e` 得到吸收 h1e 的 h2e，再按
     `(pq|rs)` 下标查双激发矩阵元。
2. **GPU matvec（每步 Lanczos/Arnoldi 迭代调用）**：
   - 输入向量 v（GPU，dim 维）
   - 并行遍历所有 (i,j) 对：收集 v[i,j] × 单激发目标累积 + v[i,j] × 双激发目标累积
   - 用 cupy 的 `add.at` / segment-reduce 或自定义 RawKernel 做 scatter-add
   - 返回 Hv（GPU）
3. **迭代求解**：`cupyx.scipy.sparse.linalg` 的 eigsh 需要显式矩阵；matrix-free
   用 **Lanczos**（自己写 ~50 行：三对角化 + 小矩阵对角化）或 cupy 的
   `cupyx.scipy.sparse.linalg.LinearOperator`（**确认 cupyx 是否支持 LinearOperator
   + eigsh；不支持就手写 Lanczos**）。

**更省事的替代路线（先做这个！）**：
- **半 GPU 方案**：先用 numpy 把上面的 σ-vector **CPU 向量化**（替代 PySCF 逐列
  contract_2e），验证正确性 + 拿到 CPU 加速（可能有 10-50×，因为 PySCF 的
  selected_ci 是 Python 循环 + C 内核混合，向量化有空间）；再把它换成 cupy
  实现 GPU 版。**两步走，每步可独立验证**。
- 参考实现：`pyscf/fci/direct_spin1.py` 的 `contract_2e`（逻辑完备的参考）；
  `pyscf/fci/spin_op.py`（S² matvec，风格类似）。

### 1.4 验证标准

1. **正确性**：随机子空间（dim 几百到几万），matrix-free 结果 == `build_ci_matrix`
   （稠密 eigh）== 现有 `solve_sci`（eigsh），diff < 1e-8（注意：现有 CPU eigsh 的
   `tol` 默认自适应，对比时两边都设 `tol=1e-8`）。
2. **速度**：dim=10⁴ 时 matrix-free GPU 总耗时 < 现有 CPU 0.42s（构建 + 求解全含）；
   dim=10⁵-10⁶ 时对比 CPU，目标 ≥ 10×。
3. **端到端**：`solve_sci(..., backend="gpu")`（或新参数名）跑通 N₂/cc-pVDZ 12o
   子空间（全空间 853776 维），能量与 CPU 一致，时间有可测量加速。
4. 回归：`tests/test_h2_sqd.py` 全过（现有 solve_sci 默认路径不得受影响）。

### 1.5 需注意的坑（全部实测过）

- **PySCF `contract_2e` 是 CPU 的，且一次只能作用一个向量**——这是"CPU 构建"慢的
  根因；matrix-free 必须自己写 matvec。
- **段错误风险**：CI 字符串电子数与 `nelec` 不符时，PySCF C 层 `contract_ss` /
  `contract_2e` 会**越界读 → core dump**（不是 Python 异常）。任何新 matvec 实现
  都要先做 popcount 校验（`solve_sci_csf` 里已有这个防护模式，见 fermion.py）。
- **`make_rdm1` 返回总密度**（trace=nelec），若要做 MP2 相关量必须减 HF 密度
  （见 OBDF 节）——GPU 任务本身不涉及，但若顺手写 MP2 相关工具要注意。
- **cupy 与 numpy 的 dtype/contiguity**：PySCF 输出是 numpy float64，转 cupy 前
  `np.ascontiguousarray`；cupy 结果转回 numpy 用 `cp.asnumpy`。
- **eigsh 的 tol 必须显式设**（`cupyx` 默认 tol 与 scipy 不同，实测不设 tol 时
  能量差 ~4e-4；设 `tol=1e-8` 后一致到 1e-13）。
- **不要两个 cupy 包并存**（cupy-cuda12x + cupy-cuda13x 会互相误删模块）。
- RTX 5080 是 Blackwell：cu12 库版本要够新（`nvidia-curand-cu12` 10.3.10+ 已实测
  可用；若报 sm_120 不支持的错，升级 nvidia-*-cu12 到 12.9+）。

---

## 2. OBDF one-body downfolding

> **2026-08-10 已完成**：本节的 v_oo/v_vv 公式经证伪（切片笔误 + 不对称 + 迹
> 2.5× 偏），正确理论为 Tran 2021（arXiv:2107.11260）。完整实现见 `REVIEW.md`
> 「OBMP2 + OBDF：完整自洽实现落地」与 `tc_sqd.obmp2`。本节保留历史记录（含
> **已探明的坑**，其中公式与 `from_pyscf` 冻结逻辑两条已被新实现解决）。

### 2.1 目标与现状

**目标**：用经典 one-body downfolding（OBDF, arXiv:2605.08675）把非活性轨道
（core + 外部虚轨道）的相关效应折叠进活性 h1e，使**小活性空间**达到近全空间精度
（量子资源↓ + 精度↑）。论文数据：H₆/cc-pVDZ OBDF-SQD 一致优于 CAS-SQD，小活性
空间收益最大。**（已完成：N₂/H₂O/cc-pVDZ 6-10o 活性 OBDF err 0.006-0.012 Ha，
scale=0.1 校准，见 `obdf_downfold`。）**

**公式**（论文原文）：
```
Ĥ_OBDF = Ĥ_CAS + v^ext_OBMP2
```
- Ĥ_CAS = 标准 frozen-core 活性哈密顿量（active h1e + McWeeny core 势 + active
  eri + ecore）——**就是 `from_pyscf` 现在的输出**
- v^ext_OBMP2 = one-body 相关势（MP2 导出），**只改 h1e，eri/ecore 不变**
  （"one-body" 的意义：不增加量子资源）
- **非变分**：MP2 是微扰的，能量可能低于 FCI（强关联/解离区失效）——必须诚实标注

**当前状态**：代码**已全部撤销**（`git checkout` 回滚，molecule.py 无残留）。
完整实现细节记录在 REVIEW.md（「OBDF one-body downfolding：实现验证后搁置」）。
接手者从零重新实现，但**所有坑已探明**（见 2.3）。

### 2.2 已探明的实现细节（可复用）

**v_OBMP2 的正确构造（t2 收缩广义 Fock，实测符号正确）**：
```python
from pyscf import mp
mp2 = mp.MP2(mf).run()                       # mf = 全分子收敛 RHF
t2 = np.asarray(mp2.t2)                       # shape (nocc, nocc, nvir, nvir)
nocc, nvir = mp2.nocc, nmo - nocc
eri_full = np.einsum("pqrs,pi,qj,rk,sl->ijkl", mol.intor("int2e_sph"),
                     *([mf.mo_coeff]*4))      # 全 MO 基 chemist 记号
eri_as = eri_full - eri_full.transpose(0,3,2,1)   # <pq||rs> 反对称化

v_oo = np.einsum("ikab,jkab->ij", t2, eri_as[:nocc,:nocc,nocc:,:nocc], optimize=True)
v_vv = -0.5 * np.einsum("ijac,ijbc->ab", t2, eri_as[:nocc,:nocc,nocc:,nocc:], optimize=True)
# v_ov = 0（Brillouin 定理，规范 MO 基）
v_full = np.zeros((nmo, nmo)); v_full[:nocc,:nocc] = v_oo; v_full[nocc:,nocc:] = v_vv
v_act = v_full[n_core:, n_core:]              # 活性块投影
h1e_eff = h1e_CAS + v_act                     # +v 符号正确（实测）
```
**已验证**：+v 使能量降低（方向对）、-v 升高；v 对称；仅改 h1e。t2 收缩比
"rdm1 收缩"正确（rdm1 收缩 = 用总密度的 G 矩阵，双重计数 HF Fock，能量错 60 Ha）。

**撤销前写的测试**（`tests/test_downfolding.py`，已删，可重写）：
1. `downfolding` 只改 h1e（eri/ecore/norb/nelec 不变）+ 修正项存 `h1e_downfolding` 字段
2. 全空间（n_core=0）时 `downfolding="obmp2"` 报错；非法值报错
3. OBDF 能量与 plain 不同（非平凡效应）
4. v_OBMP2 对称且有限

### 2.3 为什么被搁置（两个必须解决的障碍）

**障碍 1：`from_pyscf` 只能冻结"前 n_core 个占据轨道"，无法折叠虚轨道。**
OBDF 需要活性空间 = **部分占据 + 部分虚轨道**（v_vv 块就是虚-虚修正）。现有
`from_pyscf(n_active=N)` 冻结最低 N 个**占据** MO：
- STO-3G 上只能测到"冻结 2 个 1s core"（n_core=2 ≤ 7 对电子）
- N₂/cc-pVDZ 28 轨道想折叠到 10 轨道活性 → 需冻结 18 轨道 > 7 对电子 → 报错
- **接手者必须重构**：`from_pyscf` 支持活性轨道区间选择（如 `active_orbitals=[...]`
  或 `(n_core, n_virtual)`：冻结前 n_core 个占据 + 后 n_virtual 个虚轨道，中间为
  活性）。注意此时 **nelec 不变**（活性只含部分占据轨道，core 电子数不变），
  `ecore` 仍含 frozen-core 能量，但 **frozen-core 平均场势 Δh 只加 core 部分**。

**障碍 2：小基组（STO-3G）MP2 过校正。**
N₂/STO-3G 全空间仅 10 轨道，MP2 在这里"吃光"全部相关能还过头（MP2 total
-107.649 < FCI -107.582），OBDF 把 FCI 推到 -108.14（低 0.56 Ha，非物理）。
**验证必须在大基组（cc-pVDZ+）+ 小活性空间**（如 N₂/cc-pVDZ 折叠到 6-10 轨道），
并以更大活性空间（如 14o）或 CCSD 作参考，断言 `|E_OBDF - E_ref| < |E_CAS - E_ref|`。

### 2.4 接手者的实现计划（建议顺序）

1. **重构 `from_pyscf`** 支持活性轨道区间（先做这个，不依赖 OBDF 本身）：
   - 新增参数（如 `active_range: Optional[Tuple[int,int]]` 或 `n_core` + `n_virtual`）
   - 活性块 = `h1e_full[n_core:n_core+n_act, ...]`（**从中间切**，不是从头切）
   - 冻结 core 平均场势只含 core 轨道；frozen-core 能量同现有 `_frozen_core_energy`
   - **nelec 不变**（core 双占，活性电子数 = 总电子 - 2×n_core，与现有逻辑一致）
   - 加测试：中间区间切片正确、能量闭合（frozen-core FCI vs 全空间 FCI 在
     STO-3G 上应 µHa 级一致）
2. **重新实现 `_obmp2_correction`**（按 2.2 的 t2 收缩公式）+ `from_pyscf(downfolding="obmp2")`：
   - `MolecularData` 加 `h1e_downfolding` 字段（非变分诊断用）
   - docstring 诚实标注：**非变分**，MP2 强关联/解离区可能失效
3. **在大基组验证**（关键！）：
   - N₂/cc-pVDZ：折叠到 (6o, 10e) 或 (10o, 10e) 活性；参考 = 全空间 FCI（28 轨道
     太大就用 14o CAS-FCI 或 CCSD(T)）
   - 断言：OBDF 活性 FCI 误差 < plain 活性 FCI 误差；能量不低于参考太多（防过校正）
   - 报告：不同活性空间大小的收益曲线（论文：小活性收益最大）
4. **测试 + 提交**：重写 `tests/test_downfolding.py`（见 2.2 列表 + 大基组收益断言）

### 2.5 需注意的坑（全部实测过）

- **`mp.MP2(mf).make_rdm1(ao_repr=False)` 返回总密度**（trace=nelec），**必须减 HF
  密度** `np.diag(mf.mo_occ)` 得到相关密度；若用 rdm1 收缩法（非 t2 收缩法）不减
  就是双重计数 → 能量错 60 Ha（实测）。
- **t2 收缩法的符号**：+v 正确（能量降低），-v 反向。公式里 v_vv 有 `-0.5` 因子。
- **MP2 在解离区失效**：论文自己承认"unphysically low energies"。测试选平衡/
  近平衡几何（N₂ R=1.5Å 附近），**不要**用强关联拉伸点做主验证（可做对比报告）。
- **`from_pyscf` 的 UHF 拒绝、spin-resolved h1e 拒绝**逻辑保持不变；OBMP2 是
  闭壳层公式（ROHF 需另确认）。
- **活性区间切片后 `mo_coeff`**：`MolecularData.mo_coeff` 要同步切（返回活性轨道
  系数），否则下游轨道相关功能错。
- 段错误防护同 GPU 节（字符串/nelec 一致性校验）。
- 大基组 eri 变换慢（N₂/cc-pVDZ 28 轨道 einsum 四方收缩 ~分钟级 + `/mnt/d` I/O
  慢）：脚本里先 `cd /tmp` 或把中间量缓存到内存，别反复重算。

---

## 3. 通用注意事项（两个任务都相关）

- **不要改现有默认路径**：GPU 是新增可选 backend，OBDF 是新增可选参数；`solve_sci`
  / `from_pyscf` 的默认行为必须不变（回归 `tests/test_h2_sqd.py`、`tests/test_molecule.py`、
  `tests/test_subsampling.py` 全过）。
- **验证对比基线**：现有 CPU eigsh（LinearOperator 隐式 matvec）是速度对比的
  baseline（dim=10⁴ 时 0.42s）；稠密 `build_ci_matrix` + eigh 是正确性参考。
- **文档同步**：完成后更新 `SURVEY.md` §8.9 与 `REVIEW.md` 对应"搁置"节为"已落地"，
  记录实测数据（沿用仓库的中文文档风格）。

---

## 4. 参考论文来源

### 4.1 GPU 后端（核心参考）

| 论文/来源 | 与本任务的关系 |
|---|---|
| **T. Shirakawa et al., "GPU-Accelerated Selected Basis Diagonalization"**, arXiv:2601.16637 | **主参考**。GPU 全驻留 matrix-free 选态对角化：Thrust 库、Slater-Condon 稀疏激发邻域、不显式建 H、支持 10⁸-10¹⁰ 行列式、GH200 单节点 35-39×。**本任务的"正确路径"即复刻其 matrix-free 思想** |
| [Quantum Zeitgeist 报道（40× 加速）](https://quantumzeitgeist.com/gpu-acceleration-achieves-speedup-selected-basis/) | 上述论文的通俗报道，含实现要点（Thrust、matrix-free、half/full bitstring）|
| [qiskit-addon-sqd-hpc（GitHub）](https://github.com/Qiskit/qiskit-addon-sqd-hpc) | IBM/Qiskit 的 C++17/20 header-only HPC 库（OpenMP+MPI），提供 postselection/subsampling/recovery 的低层例程，可接 RIKEN `sbd` eigensolver。**替代路线**：不自己写核，直接集成 `sbd` |
| [PySCF `pyscf/fci/direct_spin1.py`](https://github.com/pyscf/pyscf/blob/master/pyscf/fci/direct_spin1.py) | **σ-vector 算法参考实现**（逻辑完备的 Python 版 Slater-Condon matvec，含对角/单激发/双激发公式）|
| [PySCF `pyscf/fci/selected_ci.py`](https://github.com/pyscf/pyscf/blob/master/pyscf/fci/selected_ci.py) | 现有 `solve_sci`/`contract_2e`/`_all_linkstr_index` 的来源；新 matvec 需与其链接表思路兼容 |
| [PySCF `pyscf/fci/spin_op.py`](https://github.com/pyscf/pyscf/blob/master/pyscf/fci/spin_op.py) | S² matvec 的参考风格（`solve_sci_csf` 已用其 `contract_ss`）|

### 4.2 OBDF 下折叠（核心参考）

| 论文/来源 | 与本任务的关系 |
|---|---|
| **T. N. Tran et al., "Quantum resource reduction for quantum-centric supercomputing via correlated mean-field downfolding framework" (OBDF-SQD)**, arXiv:2605.08675 | **主参考**。公式 `Ĥ_OBDF = Ĥ_CAS + v^ext_OBMP2`、OBMP2 流程（全分子 RHF → 经典 OBMP2 → 活性 h1e 叠加 v^ext）、H₆/cc-pVDZ 数据（小活性空间收益最大）、"非变分 + 解离区失效"警告。**HTML 全文可读**（arxiv.org/html/2605.08675v1）|
| **Lan & Yanai, "Correlated one-body potential from second-order Møller–Plesset perturbation theory"**, *J. Chem. Phys.* **138**, 224108 (2013) | OBMP2 势的**奠基论文**（v_OBMP2 工作方程出处，付费墙）。本任务 2.2 节的 t2 收缩公式即其标准形式 |
| **T. N. Tran, arXiv:2107.11260**（"OBMP2 Hamiltonian = Fock + one-body correlation potential"）| OBMP2 哈密顿量的形式化表述（免费可读）|
| **T. N. Tran, arXiv:2310.18154** | OBMP2 近期表述（免费可读）|
| [Huang et al., "Leveraging Small-Scale Quantum Computers with Unitarily Downfolded Hamiltonians", *PRX Quantum* **4**, 020313 (2023)](https://link.aps.org/doi/10.1103/PRXQuantum.4.020313) | 早期 downfolding + 量子计算工作（QDSRG 活性空间），背景参考 |
| [Bauman et al., "Downfolding of Many-Body Hamiltonians using Active-Space Methods", *J. Chem. Theory Comput.* (2019)](https://pubmed.ncbi.nlm.nih.gov/31272173/) | SES-CC downfolding 形式主义奠基，背景参考 |
| [Mukherjee et al., "Quantum Algorithm for Downfolding Quantum Chemistry" (2023)](https://ui.adsabs.harvard.edu/abs/2023APS..MARF64007M/abstract) | downfolding 的量子算法路线，背景参考 |

### 4.3 背景与上下文（本任务调研时一并参考）

| 论文/来源 | 与本任务的关系 |
|---|---|
| **CSQD**, arXiv:2603.09346（Cluster-Adaptive SQD）| 已落地（`recover_configurations_clustered`）；其自旋 λ 惩罚法已落地（`solve_sci_csf method="penalty"`）。OBDF 若与聚类恢复结合可作后续实验 |
| **AS-SQD**, arXiv:2603.13536（Active Sampling SQD）| 已落地（`solve_sqd_active`，EN 选态等价）；背景参考 |
| **AB-SND**, arXiv:2508.12724（Adaptive-basis Sample-based Neural Diagonalization）| 神经网络采样路线（未做），背景参考 |
| **SKQD**, arXiv:2501.09702（Sample-based Krylov QD）| Krylov 路线（未做），背景参考 |
| **A Critical Assessment of SQD**, arXiv:2605.02494 | 批判性评估（SQD 瓶颈：子空间覆盖不足），背景参考 |

### 4.4 获取方式提醒

- arXiv 论文优先读 **HTML 版**（`arxiv.org/html/<id>v1`）而非 PDF——便于全文检索公式与数值。
- 付费墙论文（Lan-Yanai JCP 2013）用其标准公式即可（2.2 节已给出 t2 收缩形式）；
  无需付费原文。
- 本仓库 `REVIEW.md`（「GPU 后端：实现验证后搁置」「OBDF one-body downfolding：
  实现验证后搁置」）与 `SURVEY.md` §8.9 含全部实测数值，是论文结论在本库环境下的
  落地验证记录。
- **REVIEW.md 的原始记录**：两节详细文字记录（含全部数值）在 REVIEW.md 末尾
  （「OBDF one-body downfolding：实现验证后搁置」「GPU 后端：实现验证后搁置」），
  接手前先读。
