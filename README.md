# tc_sqd

**Sample-based Quantum Diagonalization (SQD) for TensorCircuit**

一个适配 TensorCircuit + numpy 1.x/2.x + PySCF 的轻量级 SQD 包，参考
[`qiskit-addon-sqd`](https://github.com/qiskit/qiskit-addon-sqd) 设计，但**不依赖
numpy≥2 / jax 的硬性要求**。

## 特性

- 统一入口 `compute_ground_state_energy`（积分→能量，`fci`/`sqd`/`direct`，返回 float）+
  `solve_sqd`（端到端，含采样/迭代，返回 SCIResult；分工见 `docs/solve_sqd_api.md` §9）
- 比特串矩阵 ↔ 整数互转，TensorCircuit 采样适配
- 基于平均占据数的配置恢复（纠正噪声导致的粒子数违例）+
  **T1 感知恢复**（`estimate_true_occupancies`：从观测位串反卷积真实平均占据，
  喂回 recover / `initial_occupancies`；per-qubit γ 不均匀时 RMSE 降 ~30%）
- 批量子采样、汉明权重后选择、**`max_dim` 子空间维度限制**（int / (na, nb)）
- CI 矩阵构造（Slater–Condon）、子空间对角化、迭代 SQD、轨道优化
- CCSD 振幅驱动的 LUCJ ansatz 电路构造（量子态制备侧）+
  **真机深度预算报告**（`circuit_stats` / `lucj_report`：1Q/2Q 门统计，2Q 门数作保守深度代理）
- **SQD+VQE 混合优化**（`optimize_ansatz_parameters`）：以**采样后的 SQD 总能量**为损失，
  Nelder-Mead 变分优化 LUCJ 角度（`theta_list` 变分入口，`n_seeds` 多 seed 平均消除过拟合）。
  LiH 验证：误差 +5.9e-3 → +1.1e-3（改善 ~4.8 mHa）
- **误差优化关键发现**：SQD 误差根源是**采样子空间覆盖不足**。用 `excited_configurations`
  强制纳入单双激发配置（经典生成确定性覆盖相关空间），采样仅提供权重——LiH 上
  **误差 +4e-3 → 1e-16（= FCI 精确），1000 shots 即达，零统计波动**（见 `ansatz_opt_demo`）。
  ⚠ 注：LiH/STO-3G **每自旋仅 2 电子**，单+双激发已穷尽该自旋全部行列式，故 include(S+D)
  等价于全 FCI 空间。对 **>2 占据/自旋**的体系（如 N₂ 7e/spin）单双激发覆盖不足，
  include(S+D) 误差 ~2e-2（≠FCI，见 REVIEW N₂ 反例）。
- Pauli 哈密顿量在比特串子空间的投影与对角化（非费米子问题，如 QAOA-MaxCut）
- **激发态**：`solve_sci(..., n_roots=k)` 取前 k 个本征值（基态 + 低激发态）+
  **激发态采样策略**（`excited_configurations` 生成 HF+单/双激发配置强制纳入子空间，保障 n_roots 变分下界；
  `truncate_excited_configurations` 按 Slater-Condon 对角能量截断，控制大体系子空间维度）
- **密度矩阵噪声模拟**（`noise`）：退相干/振幅阻尼/去极化 Kraus 通道，cupy GPU 可选
- **噪声容限预测器**（`predict`）：输入 T₁/电路/shots → 预测 SQD 基态/激发态精度；
  `depth_budget` 结构化深度预算；`plan_sampling` 自动找最优 (shots, depth) 采样方案
- **一键分子接口**（`molecule`）：`from_pyscf(mol_or_mf)` 自动算 MO 基
  h1e/eri/ecore/norb/nelec，支持活性空间（冻结 core）与**开壳层**（ROHF，`n_α≠n_β`）
  以及 **UHF**（五积分 `h1e (2,norb,norb)` + `eri (aa,ab,bb)`，round_011）
- **UCJ 精确化**（`lucj`）：`ucj_decomposition` t2→SVD→多层 (κ, J)（简化 UCJ，
  诚实标注非 ffsim 精确）+ `build_ucj_circuit` Û Givens 电路 + `ucj_subspace_energy`
  确定性 SQD 验证——LiH 误差 简化 LUCJ 7.5e-4 → **~2e-4**
- **采样诊断**（`diagnostics`）：采样熵 / 子空间维度 / 配置分布 / 能量随 shots 收敛曲线
- **改进 SQD 工具链**（`cipsi`/`diagnostics`）：`solve_sqd_active`（采样↔PT2 选态双闭环，
  AS-SQD 思想）、`solve_hci`（库内 SHCI = heat-bath 选态 + PT2）、`solve_sqd_ev`
  （active + PT2 / evpt2 能量修正）、`solve_sqd_distill`（自蒸馏重采样）、
  `solve_sqd_adaptive`（自洽 NO 换基 + active）、`extrapolate_ev_pt2`（E_V vs E_PT2 外推）。
- **整合 SQD 入口**（`integrated`）：`solve_sqd_improved`（= active+PT2 显式入口，improved SQD）、
  `solve_sqd_best`（+ evpt2 多 shots 外推，**当前最优**，近收敛体系实测改进 30×）、`solve_sqd_auto`
  （一键流水线，`correction`=pt2/evpt2/none）。**修正层级**：变分（active）→ +PT2（improved，
  普适）→ +evpt2 外推（best，近收敛精修）。
  **适应边界（2026-08-10 实测）**：evpt2 = 近收敛精修（N₂/cc-pVDZ 10o 改进 30×、12o 远未
  收敛仅 1.7×）；distill 依赖首轮 |Ψ⟩ 质量（近收敛边际、远未收敛有害）；PT2 修正普适。
  adaptive NO 换基在大体系默认参数下差于 active（子空间缩）；UCJ 采样需 CCSD 收敛
  （强关联 R=3.0 不收敛则失效）。详见 REVIEW「L1/L2 改进实验」
  **BFS 覆盖闭包**（`coverage_closure=True`，2026-08-20 round_012）：采样得高权重字符串后，
  单激发 BFS 确定性补全到 `max_strings` 上限（默认全空间）。N₂/cc-pVDZ (12,12) @500 shots
  采得 908/924 串（97.8%），BFS 补全缺 16 串 → 全空间 FCI（err 3.6e-7→2.25e-10，1600×，
  sigma²→0，wall 1.1×，**3 seed 无关**）。修复 round_008 triple 注入的 `pt2_floor` 断链
  （默认门控 1e-7 过滤低分中间父串→BFS 断）。大体系全空间不可对角化时给较小 `max_strings`。
  **对角化提速配方**（`eigsh_tol`，2026-08-21 round_013）：闭包路径 GPU 可加
  `eigsh_tol=1e-6` + `n_active_per_round=90`，联合实测 n_mv 3.9× 少、wall 2.7× 快，
  err 不降（ARPACK 停机界远悲观于实际收敛）；CPU 路径 `eigsh_tol=1e-8`（相对
  tol=0 基线 n_mv 减半，wall 视机器负载 2-3.6×）。
  CPU/except 分支默认 tol 已从 0 改为 1e-10（E diff ≤1e-13，n_mv 0.62×）。
  非闭包路径（残余误差由缺串主导）建议保持默认 tol。
  **shots 无关性**（2026-08-28 round_015）：12,12 上 @20 与 @500 shots 给出同样的
  全空间 FCI（err ~1.8e-10，wall 47-55s 平坦）——采样坍缩为种子，确定性 PT2 循环
  + BFS 闭包完成全部工作。前提：全字符串空间可对角化（924 串/853k dim）；更大
  活性空间须 `max_strings` 截断，此结论不外推。
- **OBMP2 自洽方法 + OBDF 下折叠**（`obmp2`，2026-08-10）：自旋轨道显式实现 Tran 2021
  一体相关势（1st+2nd BCH、Ω̂ 对称化），`solve_obmp2` 自洽收敛 E≈CCSD（N₂/STO-3G 平衡差
  0.3 mHa）。`obdf_downfold` 把外部相关折叠进活性 h1e（`H_OBDF=H_CAS+scale·v^ext`，仅改
  h1e）——弱关联 N₂/H₂O/cc-pVDZ 6-10o 活性 OBDF err **0.006-0.012 Ha**（CAS 0.21-0.30，
  改善 18-38×，近 CCSD(T)）。配套 `from_pyscf(n_core, n_virtual)` 中间区间活性空间。
  ⚠ 强关联（R=3.0）过校正、scale 几何依赖（校准 ~0.1 弱关联 / ~0.01 强关联），详见 REVIEW。
- **matrix-free GPU 对角化**（`matrixfree`）：向量化 Slater-Condon σ-vector
  （`sigma_vector`/`sigma_vector_ops`，后端无关 numpy/cupy）+ `solve_sci(backend="gpu")`
  （cupyx eigsh，结果与 CPU 一致 ≤1e-13）。绕开"显式构建稀疏 H"瓶颈（O(dim) 次
  contract_2e），是 arXiv:2601.16637 matrix-free 路线；dim 大时 GPU 优势显现。
- **真机一站式**（`hardware`）：腾讯 qcloud 校准加载 / 选最优 qubit 子图 / 真机采样 / SQD 后处理
- **统一采样后端**（`sampler`）：`sample(circuit, n_samples, backend="tc"/"qcloud")` 一行切换
  模拟器 / 真机，下游 SQD 流水线不变（开发用 tc，交付用真机）

## 安装

支持 Python **3.10–3.12**（实测 3.10.20；3.11/3.12 下 tensorcircuit/pyscf/numpy/scipy 均有 wheel）。

```bash
conda create -n tc python=3.10   # 3.10 / 3.11 / 3.12 均可
conda activate tc
pip install -e .          # editable 安装, 之后任何目录都可 import tc_sqd
```

`pip install -e .` 以 editable 模式安装 tc_sqd（源码改动即时生效，无需 PYTHONPATH）。
也可仅装依赖：`pip install -r requirements.txt`，但运行范例时需手动设置
`PYTHONPATH=/path/to/tc_sqd/src`。

依赖：`tensorcircuit==0.12.0`、`numpy>=1.17`、`scipy>=1.10`、`pyscf>=2.0`。

> **numpy 版本（tensorcircuit 0.12 兼容）**：tensorcircuit 0.12.0 用了 numpy 在 2.x
> 中搬走/移除的 `np.ComplexWarning` 与 `np.reshape(newshape=)`。两条路径：
>
> - **路径 A（省心）**：固定 `numpy<2.0`（如 1.26.4）+ `scipy<1.14`，无需任何 patch。
> - **路径 B（用 numpy 2.x，例如要与 Vayesta 等倾向 numpy 2 的库共存）**：装标准
>   numpy 2.x，再启用 tc_sqd 的兼容补丁。最省事的是写入 sitecustomize（一劳永逸）：
>   ```bash
>   python -m tc_sqd._compat install
>   ```
>   之后该环境任何脚本 `import tensorcircuit` 前都自动 patch。也可在脚本里
>   `import tc_sqd`（导入即 patch）后再 `import tensorcircuit`。

## 快速开始（H₂ 三步）

```python
import numpy as np
import tensorcircuit as tc
from pyscf import gto, scf
import tc_sqd

# 1. PySCF 构建分子哈密顿量（MO 基积分）
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
mf  = scf.RHF(mol).run()
mo  = mf.mo_coeff
h1e  = mo.T @ mf.get_hcore() @ mo
eri  = np.einsum("pqrs,pi,qj,rk,sl->ijkl", mol.intor("int2e_sph"), mo, mo, mo, mo)
ecore = mf.energy_nuc()
norb, nelec = mol.nao_nr(), (mol.nelectron // 2, mol.nelectron // 2)

# 2. TensorCircuit 采样比特串
c = tc.Circuit(2 * norb)
c.x(0); c.x(norb)                                  # HF 初态
c.ry(0, theta=0.8); c.cnot(0, 1); c.ry(0, theta=-0.8)
c.ry(norb, theta=0.8); c.cnot(norb, norb+1); c.ry(norb, theta=-0.8)
bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=2000)

# 3. SQD 求解基态能量
e = tc_sqd.compute_ground_state_energy(
    h1e, eri, norb, nelec, ecore=ecore, method="sqd",
    bitstring_matrix=bsm, probabilities=probs,
)
print(f"E(SQD) = {e:.8f}")   # -1.13728383
```

## Web 计算面板（WebUI）

克隆仓库后**不写一行代码**，在浏览器里调参并在本机跑计算：

```bash
pip install flask          # 唯一额外依赖（或 pip install -e .[webui]）
python -m tc_sqd.webui     # 自动打开 http://127.0.0.1:8765
```

面板能力：

- **体系**：预设下拉（H₂/CH/N₂ 拉伸/C₂ 等，均为项目验证过的体系）或自定义
  （PySCF 几何串 + 基组 + 电荷/自旋 + 冻结核/虚轨道 + RHF/ROHF/UHF）；
  提交前实时预览活性空间 (nα,nb)@norb 与全空间维度。
- **方法**：SQD active（采样↔PT2 双闭环，全配方参数：max_strings、
  coverage_closure、warm_start、tail_suppression、eigsh_tol 等）、CIPSI、
  SHCI、全空间 SCI；backend 可选 cpu/gpu（本机装有 cupy 时）。
- **采样**：shots、随机种子、均匀/HF 偏置两种采样模式；**多 seed 平均**
  （同一配方跑 n 个种子，汇总 mean±std，逐 seed 轨迹叠加对比）。
- **参考能量**：auto（库内 `solve_sci` 全空间真基态，带维度上限保护）/
  手动输入 / 关闭；err 口径与库内一致。
- **实时进度与可视化**：运行中逐轮刷新 E/dim/σ²；结束后出能量收敛、
  err-per-seed、维度增长、σ² 四张图 + 汇总卡片 + 原始 JSON。

注意：单时间只跑一个任务（避免 CPU/GPU 抢占污染计时）；取消在 seed 间生效；
服务重启后任务历史清空；积分按体系指纹缓存于仓库根 `_webui_*_ints.npz`
（已 gitignore），换 shots/seed/方法重跑不重复 SCF。测试见
`tests/test_webui.py`（flask 未安装时自动跳过）。

## 三种求解方法

`compute_ground_state_energy(h1e, eri, norb, nelec, *, ecore, method, ...)` 一键切换：

| method | 用途 | 特点 |
|---|---|---|
| `"fci"` | 精确基准 | 枚举全部行列式，PySCF Davidson 对角化 |
| `"direct"` | 小体系 / 教学 | 显式构造 CI 矩阵 + `numpy.linalg.eigvalsh` |
| `"sqd"` | 量子采样 | 迭代 SQD：恢复 → 采样 → 对角化 → 更新占据数 |

H₂/STO-3G 三种方法一致：`E = -1.13728383`（= PySCF FCI）。

## 分步 API（进阶）

```python
# 配置恢复：修正违反粒子数的比特串
recovered, p = tc_sqd.recover_configurations(
    bsm, probs, (occ_a, occ_b), nelec[0], nelec[1], rand_seed=42)

# 比特串 → CI 字符串
ci_a, ci_b = tc_sqd.bitstring_matrix_to_ci_strs(recovered)

# 子空间对角化（返回 SCIResult：energy / sci_state / occupancies / spin_square）
res = tc_sqd.solve_sci((ci_a, ci_b), h1e, eri, norb, nelec)

# 显式构造 CI 矩阵（可导出 / 检查）
H = tc_sqd.build_ci_matrix(ci_a, ci_b, h1e, eri, norb, nelec, ecore=ecore)

# 迭代 SQD 循环
res = tc_sqd.diagonalize_fermionic_hamiltonian(
    h1e, eri, (bsm, probs), samples_per_batch=200,
    norb=norb, nelec=nelec, max_iterations=5, seed=42)
```

## 非费米子问题（Qubit 哈密顿量）

适用于 QAOA-MaxCut 等 Pauli 哈密顿量问题：

```python
hamiltonian = [("ZZI", -1.0), ("IZZ", -1.0), ("XII", -0.5)]
vals, vecs = tc_sqd.solve_qubit(bsm, hamiltonian)   # 支持稠密 k 与稀疏 eigsh 分支
```

## LUCJ ansatz（CCSD 振幅驱动）

从 PySCF CCSD 双激发振幅 t2 构造 LUCJ 电路，采样后交给 SQD（替代上文的手写纠缠电路）：

```python
c = tc_sqd.build_lucj_circuit(mf, norb, nelec, ccsd_scale=1.0)
bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=3000)
e = tc_sqd.compute_ground_state_energy(
    h1e, eri, norb, nelec, ecore=ecore, method="sqd",
    bitstring_matrix=bsm, probabilities=probs)
# H2: e = -1.13728383 (= FCI)；LiH: 误差 ~7.5e-4 vs FCI
```

关键：必须由 t2（而非 t1）驱动 —— H2/STO-3G 的 t1≈0（Brillouin 定理），相关能几乎全来自 t2 双激发。

## API 速查

| 模块 | 函数 | 作用 |
|---|---|---|
| counts | `bitarray_to_int(bsm)` | 比特串矩阵 → 整数数组 |
| counts | `int_to_bitarray(vals, nbits)` | 整数 → 比特串矩阵 |
| counts | `counts_dict_to_bitstring_matrix(counts, nbits)` | 计数字典 → (比特串矩阵, 概率)，等价键自动合并 |
| counts | `sample_from_circuit(circuit, n_samples)` | 从 TC 电路采样 → (比特串矩阵, 概率) |
| configuration_recovery | `recover_configurations(bsm, probs, avg_occ, na, nb)` | 基于平均占据数的配置恢复 |
| configuration_recovery | `estimate_true_occupancies(bsm, na, nb, t1_gamma)` | T1 反卷积估计真实平均占据（per-qubit γ；喂 recover / initial_occupancies）|
| configuration_recovery | `postselect_by_hamming_weight(bsm, *, hamming_right, hamming_left)` | 按汉明权重筛选 |
| subsampling | `subsample(bsm, probs, samples_per_batch, num_batches)` | 按概率无放回批量子采样 |
| subsampling | `postselect_by_hamming_right_and_left(bsm, probs, ...)` | 汉明权重后选择 + 重归一化 |
| subsampling | `limit_subspace(bsm, max_dim, norb, *, probabilities)` | 按概率裁剪子空间（int=总行列式数 / tuple=(na, nb)）|
| fermion | `bitstring_matrix_to_ci_strs(bsm)` | 比特串 → PySCF CI 字符串 |
| fermion | `build_ci_matrix(ci_a, ci_b, h1e, eri, norb, nelec, ecore)` | Slater–Condon 构造 CI 矩阵 |
| fermion | `solve_sci(ci_strs, h1e, eri, norb, nelec, *, spin_sq)` | 子空间对角化（可选目标自旋） |
| fermion | `solve_fermion(bsm, hcore, eri, ...)` | 从比特串出发的 SQD 求解 |
| fermion | `diagonalize_fermionic_hamiltonian(h1e, eri, bit_array, ...)` | 迭代 SQD 循环 |
| fermion | `optimize_orbitals(bsm, hcore, eri, k_flat, ...)` | 轨道优化（best-so-far） |
| fermion | `rotate_integrals(hcore, eri, k_flat)` | 应用轨道旋转 U=exp(K) |
| fermion | `compute_ground_state_energy(...)` | **统一入口**，method = fci / sqd / direct |
| fermion | `SCIState` / `SCIResult` | SQD 波函数 / 结果数据类（支持 save/load、rdm、spin_square） |
| qubit | `sort_and_remove_duplicates(bsm)` | 排序 + 去重 |
| qubit | `matrix_elements_from_pauli(bsm, pauli)` | 单个 Pauli 算符的子空间矩阵元 |
| qubit | `project_operator_to_subspace(bsm, hamiltonian)` | Pauli 哈密顿量投影为稀疏矩阵 |
| qubit | `solve_qubit(bsm, hamiltonian)` | Pauli 哈密顿量子空间求解 |
| lucj | `get_ccsd_amplitudes(mf)` | 跑 RHF-CCSD，返回 (t1, t2, mycc) |
| lucj | `build_lucj_circuit(mf, norb, nelec, *, ccsd_scale)` | 从 CCSD t2 构造简化 LUCJ 电路（HF + 占据-空 Givens；`theta_list` 变分入口）|
| lucj | `optimize_ansatz_parameters(mf, h1e, eri, norb, nelec, ...)` | SQD+VQE：Nelder-Mead 优化 LUCJ 角度，SQD 能量作损失（固定 seed 可复现）|
| lucj | `circuit_stats(circuit)` | 门统计：n_1q / n_2q / n_multi / n_gates |
| lucj | `lucj_report(mf, norb, nelec, *, max_excitations, max_depth)` | 真机深度预算：2Q 门数代理 / within_budget / max_entries |
| lucj | `ucj_decomposition(t2, norb, nocc, *, nlayers, scale)` | t2→SVD→多层 (κ, J) UCJ 参数（简化，非 ffsim 精确）|
| lucj | `ucj_subspace_energy(layers, h1e, eri, norb, nelec)` | 确定性 SQD：UCJ 态支持的 det 子空间对角化（H₂=FCI，LiH 趋近）|
| lucj | `build_ucj_circuit(mf, norb, nelec, *, nlayers, scale, include_jastrow)` | UCJ 电路：Û Givens（默认省略 e^{iJ}，SQD 采样相位无关）|
| fermion | `solve_sci(..., n_roots=k)` | 激发态：n_roots>1 返回前 k 个本征态 list[SCIResult] |
| fermion | `excited_configurations(norb, nelec, *, max_excitations)` | 从 HF 生成单/双激发位串（喂 include_configurations，激发态采样策略）|
| fermion | `truncate_excited_configurations(norb, nelec, h1e, eri, ...)` | 按 Slater-Condon 对角能量截断单/双激发（max_configs / energy_threshold，强制含 HF，大体系用）|
| noise | `statevector_to_density(psi)` | 纯态 → 密度矩阵 ρ=\|ψ⟩⟨ψ\| |
| noise | `apply_dephasing(rho, p, nq)` / `apply_amp_damping(rho, γ, nq)` / `apply_depolarizing(rho, p, nq)` | 退相干(T₂)/振幅阻尼(T₁)/去极化 Kraus 通道（gpu=True 走 cupy）|
| noise | `density_to_bitstring_matrix(diag, norb, n_samples)` | 密度矩阵 diag → 采样 bsm（接 recover_configurations）|
| predict | `gamma_T1(depth, t_gate_ns, T1_us)` | 真机振幅阻尼率 γ = 1−exp(−depth·t_gate/T₁) |
| predict | `predict_sqd_error(T1, depth, t_gate, shots, n_excited)` | 预测 SQD 基态/激发态误差（退相干免疫，T₁ 主导）|
| predict | `depth_budget(T1, t_gate, shots, target, excited)` | 结构化深度预算（`DepthBudget`：max_depth/status/reason）|
| predict | `max_depth_for_accuracy(T1, t_gate, shots, target, excited)` | 反向预测达目标精度的 depth 上限（int 薄封装）|
| predict | `plan_sampling(T1, t_gate, *, target, excited, ...)` | 采样预算分配：枚举 (shots, depth) 网格，按成本排序可行方案 |
| predict | `calibrate(h1e, eri, norb, nelec, *, circuit, ...)` | 跨体系校准 KS/KT1（二元 LSQ；circuit= 实际采样 / None=FCI 密度 benchmark）|
| molecule | `from_pyscf(mf_or_mol, *, n_active, n_core, n_virtual)` | 一键构建 SQD 输入（MO 积分 + 核能 + 电子数，活性空间冻结 core，`n_core`+`n_virtual` 中间区间可折叠虚轨道）|
| molecule | `MolecularData.solve(method, ...)` | 一键求基态能量（fci/sqd/direct）|
| obmp2 | `solve_obmp2(mf, ...)` | OBMP2 自洽求解（一体相关势，E≈CCSD）|
| obmp2 | `obdf_downfold(mf, n_core, n_virtual, *, scale)` | OBDF 下折叠：外部相关折叠进活性 h1e（`H_OBDF=H_CAS+scale·v^ext`）|
| matrixfree | `sigma_vector(v, ci_a, ci_b, norb, nelec, h1e, eri)` | 向量化 Slater-Condon σ-vector（matrix-free matvec）|
| matrixfree | `sigma_vector_ops(v, ops, xp)` | 后端无关 σ-vector（预计算算子，numpy/cupy 复用）|
| matrixfree | `eigsh_gpu(ops, dim, ...)` | matrix-free GPU 本征求解（cupyx eigsh）|
| fermion | `solve_sci(..., backend="gpu")` | 子空间对角化 GPU 后端（matrix-free cupy，结果与 CPU 一致）|
| diagnostics | `sampling_report(h1e, eri, norb, nelec, bsm, ...)` | 采样质量综合报告（熵/维度/配置/收敛曲线）|
| diagnostics | `energy_convergence(...)` | 能量随 shots 收敛曲线 |
| diagnostics | `shannon_entropy(probs)` / `subspace_dimension(bsm)` | 采样熵 / 子空间维度 |
| hardware | `load_calibration(device_name)` | 从 tc qcloud 读校准快照（T₁/T₂/读出/CZ/拓扑）|
| hardware | `select_qubits(calibration, nq)` | 多起点贪心选最优 nq 物理 qubit 子图（min T₂ 最大化）|
| hardware | `bitstring_matrix_to_energy(bsm, h1e, eri, norb, nelec, ecore)` | 采样 bsm → recover → 子空间对角化 → 能量 |
| hardware | `sample_on_hw(device, circuit, physical_qubits, ...)` | 真机采样（编译+submit_task+REM+字节序自校准）|
| sampler | `sample(circuit, n_samples, *, backend, backend_kwargs)` | 统一采样后端：`"tc"` 模拟 / `"qcloud"` 真机，返回 (bsm, probs) |

## 比特串约定

```
[ beta_{norb-1} ... beta_0 | alpha_{norb-1} ... alpha_0 ]
  ^------- 左半 (beta) -----^  ^----- 右半 (alpha) -----^
```

与 `qiskit-addon-sqd` 一致：右半编码 alpha（自旋向上），左半编码 beta（自旋向下）。

## 与 qiskit-addon-sqd 的区别

| 特性 | qiskit-addon-sqd | tc_sqd |
|---|---|---|
| numpy 要求 | >= 2.0 | >= 1.17（实测兼容 2.x） |
| 量子电路 | qiskit QuantumCircuit | tensorcircuit.Circuit |
| 采样接口 | BitArray (qiskit) | 直接 numpy 数组 |
| CI 矩阵后端 | jax 加速 | PySCF selected_ci |
| jax 依赖 | 必须 | 不需要 |
| 基态能量入口 | 无统一入口 | `compute_ground_state_energy` |

## 运行测试

28 个测试模块，**255 个测试函数**（2026-09-03 统计；GPU 测试须拆分单跑，避免与
CPU 全库并行造成计时污染）：

```bash
PYTHONPATH=src python -m pytest tests/ -q --ignore=tests/test_subspace_gpu.py
PYTHONPATH=src python -m pytest tests/test_subspace_gpu.py -q
```

| 模块 | 数量 | 覆盖 |
|---|---|---|
| test_h2_sqd | 18 | H₂ SQD 主路径 / FCI 一致性 |
| test_tail_sampling | 22 | C1 尾部发现采样（tail_suppression / 预算缩放） |
| test_sqd_active | 14 | active 循环 solve_sqd_active |
| test_subspace_gpu | 12 | GPU _Subspace（hybrid/cupyx/fallback + eigsh_tol 消融） |
| test_noise | 11 | 噪声模型 / T1 反卷积 |
| test_molecule | 11 | 分子体系集成 |
| test_lucj | 11 | LUCJ/UCJ 分解 / 电路 |
| test_triple_injection | 10 | 三激发定向注入 |
| test_spin_resolved | 16 | 自旋分辨积分 / UHF（含 round_017 active 闭环） |
| test_predict | 10 | 误差预测 / 校准 |
| test_pruning | 9 | PT2 排序剪枝 prune_keep |
| test_obmp2 | 8 | OBMP2/OBDF |
| test_excited | 8 | 激发态采样 |
| test_coverage_closure | 8 | BFS 覆盖闭包 coverage_closure |
| test_basis | 8 | 基组 / 自然轨道 |
| test_warm_start | 7 | warm-start eigsh 初猜 |
| test_recovery_clustered | 7 | CSQD 聚类恢复 |
| test_diagnostics | 7 | 诊断 |
| test_ansatz | 7 | SQD+VQE |
| test_eigsh_tol | 6 | eigsh_tol 透传 / 等价性 |
| test_subsampling | 5 | subsampling |
| test_sampler | 5 | 统一采样后端 |
| test_open_shell | 5 | 开壳层 CH (3,2) |
| test_cipsi | 5 | CIPSI/HCI 基础路径 |
| test_t1_recovery | 4 | T1 感知恢复 |
| test_matrixfree | 4 | matrix-free σ-vector |
| test_hardware | 3 | 真机 mock |

```bash
PYTHONPATH=src python examples/h2_sqd_demo.py    # H2 完整演示
PYTHONPATH=src python examples/excited_sqd_demo.py  # 激发态 SQD 全链路 (LiH)
PYTHONPATH=src python examples/noise_aware_demo.py  # 噪声感知全链路 (T1反卷积+规划+诊断)
PYTHONPATH=src python examples/ucj_demo.py          # UCJ 全链 (分解 → 确定性 SQD → 电路采样)
```

## 限制与已知边界

- **开壳层 / UHF**：SQD 核心（`solve_sci`/三路径）与 `recover_configurations`/`estimate_true_occupancies` 原生支持 `n_α≠n_β`；`from_pyscf` 支持 ROHF（`mol.spin!=0` 自动）；**UHF 支持**（round_011：`from_pyscf`/`solve_sci`/`build_ci_matrix`/`compute_ground_state_energy(method="fci")` 走自旋分辨 matrixfree / `pyscf.fci.direct_uhf` 路径，全空间与任意子空间，CPU；round_017 起 **active/PT2/ev/best/auto 闭环与 coverage_closure 也支持三元组 eri**（matrixfree ops 路径，CPU）；round_019 起上述 _Subspace 系路径 **GPU 可用**（hybrid：scipy eigsh 引擎 + GPU ops matvec，per-mv 实测 8-10× CPU；无 GPU 静默回退 CPU）；仍首期 raise：HCI/CIPSI/adaptive/distill、CSF、`solve_sci` GPU、linkstr 三元组、UHF+frozen-core）；UHF 轨道基下的 S² 为共享轨道近似（与 `direct_uhf` 行为一致）；UHF 基下的采样/电路恢复行为未验证（实验性）。`build_lucj_circuit`/`build_ucj_circuit` 仍闭壳层（开壳层用 HF 电路采样）。
- **一电子积分**：单个 `(norb, norb)`（或两块相同的 `(2, norb, norb)`，collapse）；
  自旋分辨 `h_alpha ≠ h_beta` 须配 `eri (aa,ab,bb)` 三元组（round_011 新路径），非法组合显式 raise。
- **`max_dim`**：已实现（`limit_subspace` 按概率贪心裁剪；int=总行列式数、tuple=(na, nb)）。`include_configurations` / carryover 强制配置不受裁剪。
- **`spin_sq`**：在 `solve_sci` / `fci` 路径通过多根 S² 匹配实现真正的目标自旋选态（不可达时 raise）；`sqd` / `direct` 路径显式拒绝。
- **状态持久化**：`SCIState.save/load` 后通过 `_as_scivector` 重建 PySCF `SCIvector` 元数据，`rdm` / `spin_square` 在加载后仍可用。
- **LUCJ / UCJ**：`build_lucj_circuit` 为简化实现（t2 范数 Givens，LiH 误差 ~7.5e-4）；`build_ucj_circuit` + `ucj_decomposition` 为 **UCJ-inspired 简化 SVD**（诚实标注非 ffsim `UCJOpSpinBalanced` 精确；J 对角启发式，LiH 误差 ~2e-4）。均仅闭壳层。
- **`optimize_orbitals`**：基于 scipy Nelder-Mead 无导数优化（旧版数值梯度每梯度分量一次 SQD 对角化，实际不可用）。`learning_rate` 仅保留兼容，不再使用。
- **`predict` 校准常数**：KS/KT1 来自 H₄/STO-3G 拟合，跨体系只作数量级参考；`plan_sampling` / `depth_budget` 的误差界在同一近似下成立。
- **`from_pyscf` 冻结 core**：frozen-core 近似冻结 core-valence 关联（~2e-4 Ha 量级，对 LiH），活性 FCI 与"core 严格双占据受限对角化"精确一致。

## 目录结构

```
tc_sqd/
├── README.md                 # 本文件
├── REVIEW.md                 # 代码审查与验证历史
├── requirements.txt
├── src/tc_sqd/               # _compat, basis, cipsi, configuration_recovery, counts, diagnostics, fermion, hardware, integrated, lucj, matrixfree, molecule, noise, obmp2, predict, qubit, sampler, selected_ci_gpu, subsampling, tail_sampling, webui (可选 flask)
├── tests/                    # 28 个测试模块 / 255 个测试函数（清单见"运行测试"节）
└── examples/
    ├── h2_sqd_demo.py        # H2 完整演示
    ├── excited_sqd_demo.py   # 激发态 SQD 全链路 (LiH: n_roots + 激发配置强制纳入)
    ├── noise_aware_demo.py   # 噪声感知全链路 (T1 反卷积 + 误差预测 + 采样规划 + 诊断)
    ├── ansatz_opt_demo.py    # SQD+VQE 混合优化 (LiH: 固定 3.9e-3 → VQE 1.1e-3 → include 单双激发 = FCI)
    └── ucj_demo.py           # UCJ 全链 (分解 → 确定性 SQD → 电路采样，LiH 7.5e-4 → ~2e-4)
```

> 审查与验证历史（逐轮追加）见 [`REVIEW.md`](REVIEW.md)。

## License

Apache License 2.0（见 [LICENSE](LICENSE)）。

**Attribution**：`tc_sqd.selected_ci_gpu` 的 GPU RawKernel 派生自
[PySCF](https://github.com/pyscf/pyscf)（Apache 2.0）`pyscf/fci/select_ci.c` 的
selected-CI contraction 算法逻辑；`tc_sqd.matrixfree` 的 linkstr 算法参考其
`direct_spin1.c` 实现。
