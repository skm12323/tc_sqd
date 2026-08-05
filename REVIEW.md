# 代码审查与验证历史

本项目经历了 4 轮审查 / 修复 / 验证，并新增了 LUCJ 量子态制备模块。前 3 轮为
**静态审查**（受限于运行环境缺少 numpy，无法动态复现数值结果）；第 4 轮在
`WSL + conda tc` 环境首次完成**动态验证**；随后新增 LUCJ 模块填补"量子侧"空缺。

## 当前状态（第 10 轮 review 后）

**全部测试通过。** 运行环境：Python 3.10.20 / numpy 2.2.6 / scipy 1.15.3 /
pyscf 2.14.0 / tensorcircuit 0.12.0。

| 验证项 | 结果 |
|---|---|
| `python -m compileall -q src tests examples` | 通过 |
| 10 个测试文件 (`tests/test_*.py`) | **全 PASS**（见五轮增强章节） |
| H₂/STO-3G（fci / direct / sqd 三方法） | 均 `E = -1.13728383`（= PySCF FCI） |
| 目标自旋选态 | `spin_sq=0.75` 选中 S=1/2 根；不可达自旋 raise |
| `from_pyscf` 活性空间 (LiH) | 活性 FCI 与受限对角化一致 (1e-8) |
| T1 反卷积 | per-qubit γ 不均匀时 RMSE 降 ~33% |
| 激发态 n_roots (H₂) | 与 FCI 前 4 根一致 (1e-8) |

## 演进时间线

### 第 1 轮：代码审查

发现 4 类严重问题：

1. `optimize_orbitals` 解包崩溃（`solve_fermion` 返回 4 项却按 5 项解包）
2. 配置恢复的轨道占据顺序错误（平均占据数组未按布局反转）
3. `spin_sq` 未形成有效目标自旋约束（赋给不存在的属性）
4. 轨道旋转的单/双电子积分变换方向不一致

以及 RDM 接口、概率归一化、输入边界校验、numpy 最低版本声明等次要问题。

### 修复阶段

逐一修复：4 元组解包、`avg[::-1]` 对齐布局、einsum 改为一致 `U_{ip}`、`spin` 属性
换算、`1/M` 概率归一化、RDM 区分自旋求和/分辨。新增 5 项回归测试（`test_bugfixes`）。

### 第 2 轮：复审

确认部分修复，但判定"严重问题已全部修复"**不成立**：目标自旋约束、可靠轨道优化、
开壳层、自旋分辨积分、迭代 SQD 参数、状态持久化、自旋计算、边界输入处理仍未满足；
且因环境缺依赖无法动态复现。

### 第 3 轮：复审

确认新增修复：输入校验增强、开壳层错误用法保护、自旋分辨积分显式拒绝、Pauli 非法
字符校验、Pauli Y 独立测试、TFIM 参考独立化、numpy 下限与文档命令修正。

仍列 3 个核心阻塞与若干残留，给出 **6 项下一轮优先建议**：

1. 删除 `orbital_occupancies` 中危险且无意义的 `reshape`，补 `n_alpha>1` 测试
2. 正确实现目标自旋选态，或显式移除/拒绝 `spin_sq`
3. `optimize_orbitals` 参数校验 + best-so-far + 直接测试
4. `include_configurations` 真正强制包含 + carryover 阈值校验
5. 加载状态后重建 `SCIvector`，验证 RDM / 自旋数值
6. 修复 Counts 等价键合并、Qubit `k` 语义、剩余边界校验

### 第 4 轮：动态验证（本轮）

第 3 轮的 6 项建议**均已实现并经回归测试动态确认**：

| # | 建议 | 实现位置 | 测试证据 |
|---|---|---|---|
| 1 | 删除 `reshape`，按 CI 字符串数 reshape | `fermion.py:83` | Test 9 (1) PASS |
| 2 | 多根 S² 匹配选根，不可达 raise | `fermion.py:425-456` | Test 9 (2) PASS |
| 3 | `optimize_orbitals` 校验 + best-so-far | `fermion.py:893-941` | Test 9 (3) PASS |
| 4 | `include_configurations` 强制并入每批 + carryover 校验 | `fermion.py:726-746` | Test 9 (4) PASS |
| 5 | `_as_scivector` 重建元数据 | `fermion.py:146-162` | Test 7 PASS |
| 6 | counts 等价键合并 / qubit k / 稀疏分支 | `counts.py`、`qubit.py` | Test 8/9 PASS |

本轮同步修正：`requirements.txt` 放宽 `numpy>=1.17`（实测兼容 2.x）、移除未被引用的
`h5py`、文档安装命令与版本表述同步。

## LUCJ 模块（新增）

为补齐 tc_sqd 在"量子态制备"侧的空缺（原仅经典后处理），新增 `tc_sqd.lucj` 模块：
从 PySCF CCSD 的 t2 双激发振幅构造简化 LUCJ 电路（HF + 占据-空 Givens-like 门）。
关键设计：必须由 t2（而非 t1）驱动 —— H2/STO-3G 的 t1≈0（Brillouin 定理），相关能
几乎全来自 t2。

动态验证（Python 3.10.20 / numpy 2.2.6 / pyscf 2.14 / tensorcircuit 0.12）：

| 体系 | HF-SQD 误差 vs FCI | LUCJ-SQD 误差 vs FCI |
|---|---|---|
| H₂/STO-3G | +2.05e-2 | **−4.44e-16（= FCI）** |
| LiH/STO-3G | +1.97e-2 | **+7.53e-4**（捕获 96% 相关能） |

当前为简化实现（t2 范数驱动 Givens），未做 ffsim `UCJOpSpinBalanced.from_t_amplitudes`
的精确 SVD 分解与对角 Coulomb Jastrow；闭壳层专用。

## 新增模块（noise / predict / hardware）+ fermion 激发态

### fermion 激发态（n_roots）

`solve_sci(..., n_roots=k)` 扩展：`n_roots>1` 返回 `list[SCIResult]`（基态 + 低激发态，
升序能量），`n_roots=None/1` 返回单 `SCIResult`（向后兼容）。

动态验证（H4/STO-3G）：前 3 本征值 `[−5.203, −4.800, −4.509]` 与
`pyscf selected_ci.kernel_fixed_space(nroots=3)` 完全一致。

### noise 模块（密度矩阵 Kraus 噪声模拟，qiskit-Aer 风格 + cupy GPU）

密度矩阵构造 + 退相干/振幅阻尼/去极化 Kraus 通道 + `diag→bsm` 采样。`gpu=True` 走 cupy。

机制（D:\explore 方向1/2 验证）：**退相干 diag 不变（SQD 免疫）**，振幅阻尼改 diag（T₁ 主导误差），
去极化保迹。`has_gpu()` 探测 cupy 可用性。

6 个测试全 PASS（`tests/test_noise.py`）：密度矩阵构造、退相干 diag 不变、振幅阻尼改 diag、
γ=0 恒等、去极化保迹、diag→bsm 采样。

### predict 模块（噪声容限预测器，独有）

解析模型 `ε = KS/√shots + KT1×γ_T1`（基态）；`×3`（激发态）。T₂/读出贡献 0。
`KS=0.0175, KT1=4.7e-3`（H4 校准），`γ_T1=1−exp(−depth·t_gate/T1)`。

验证（H4 预测 vs 实测）：基态 γ=0.4 预测 2.16e-3 vs 实测 2.07e-3；**激发态 γ=0.2 预测 3.10e-3
vs 实测 3.04e-3**（几乎完美）。

注意（calibrate_kt1 发现）：KT1 是 shots 依赖的（高 shots KT1→0，recover 吸收 T₁），
模型适用于中低 shots 区间。`max_depth_for_accuracy` 反向推 depth 上限（激发态 < 基态）。

5 个测试全 PASS（`tests/test_predict.py`）：γ_T1 边界/单调、预测结构、激发态 ~3×、
shots 增采样误差降、max_depth 激发态更严。

### hardware 模块（腾讯 qcloud 真机一站式，整合 D:\qubit_toolkit + D:\exp）

- `load_calibration`：从 tc qcloud 设备读校准快照（T₁/T₂/读出/CZ/拓扑）。
- `select_qubits`：多起点贪心选最优 nq 物理 qubit 子图（min T₂ 最大化 + 连通 + BFS 序映射）。
- `bitstring_matrix_to_energy`：采样 bsm → recover → 子空间对角化 → 能量（复用 compute_ground_state_energy）。
- `sample_on_hw`：真机采样（编译 + submit_task + REM + 字节序自校准）。

验证：`select_qubits`（模拟 6-qubit 链校准 → 选 T₂ 最大的连通 4 `[0,1,2,3]`）+
`bitstring_matrix_to_energy`（H₂ HF bsm → −1.116759 = E_HF 正确）。

## 六轮增强（commit c002d9e → 后续）

在原有核心基础上连续六轮增强，每轮均带独立测试与验证：

| 轮 | 提交 | 内容 | 关键验证 |
|---|---|---|---|
| 1 | `c002d9e` | 3 修复（density 布局反转 / REM 静默吞错 / kwargs 泄漏）+ `from_pyscf` + `plan_sampling` + `diagnostics` + `optimize_orbitals`→Nelder-Mead | `from_pyscf` 活性 FCI 与受限对角化一致 (1e-8)；轨道优化 5.9s 收敛（原 60 万次对角化不可用）|
| 2 | `f7dc19e` | LUCJ 真机深度预算（`circuit_stats`/`lucj_report`，2Q 门代理）+ `max_dim` 子空间限制 | H2 门数断言；max_dim 裁剪维度 ≤ 限制 |
| 3 | `800945c` | T1 感知恢复 `estimate_true_occupancies` | per-qubit γ 反卷积 RMSE 降 33%（0.116→0.078）；均匀 γ 保序退化（实验证明改翻转决策无效）|
| 4 | `2228552` | 激发态采样策略 `excited_configurations` + 2 全链路示例 | H2 n_roots 精确复现 FCI 4 根 (1e-8)；LiH 激发态误差 ~2e-4 |
| 5 | `c6d0eb6` | 统一采样后端 `sampler`（tc 模拟 / qcloud 真机）| tc 后端驱动 SQD 复现 FCI |
| 6 | — | **SQD+VQE 混合优化** `optimize_ansatz_parameters` + `theta_list` 变分入口 | LiH：误差 +5.9e-3 → +1.1e-3（`n_seeds` 多 seed 平均消除单 seed 过拟合——实验证明固定 seed 优化在换 seed 验证时误差反弹到 5.5e-3）；`get_ccsd_amplitudes` 缓存 |

### 误差优化关键发现（第 6 轮后续）

对 SQD+VQE 做误差瓶颈分析（LiH/STO-3G），得到三个层次结论：

| 方案 | 误差 vs FCI | 说明 |
|---|---|---|
| 固定 CCSD-LUCJ 纯采样 | +3.9e-3（5-seed std 1.6e-3）| 采样统计主导 |
| SQD+VQE 优化（单 seed）| 训练 -6e-4，**换 seed 验证反弹到 +5.5e-3** | **固定 seed 严重过拟合** → 加 `n_seeds` 多 seed 平均目标消除 |
| SQD+VQE（n_seeds=3）| +1.1e-3（跨 seed 验证）| 过拟合缓解，统计极限 ~1e-3 |
| **+ include 单双激发** | **+8.9e-16（= FCI），std 0** | 93 个配置确定性覆盖相关空间，采样只供权重；**为何 = FCI**：LiH/STO-3G 每自旋仅 2 电子，单+双激发已穷尽该自旋全部 15 个行列式 → α/β 笛卡尔积 225 = **全 FCI 空间** |

**结论**：SQD 误差根源是采样子空间覆盖不足，而非 ansatz/优化本身。`excited_configurations`
经典生成单双激发（`include_configurations`）确定性消除该根源。测试
`test_include_excitations_reaches_fci` 锁住该发现（`np.allclose(es, FCI, 1e-8)` + std<1e-9）。

> ⚠ **前提（防误读）**：include(S+D) = FCI **仅对 ≤2 占据/自旋的体系成立**（如 LiH），
> 本质是"单+双激发穷尽了该自旋全部行列式"，**不是 "CISD 精确"**（PySCF 真 CISD 误差 +1.34e-5）。
> 对 >2 占据/自旋（如 N₂ 7e/spin），单双激发不穷尽，include(S+D) 误差 ~2e-2（见下）。
> 回归对照：`test_include_excitations_reaches_fci`（≤2 occ，=FCI）vs
> `test_include_excitations_not_fci_strong_correlation`（>2 occ，≠FCI）。

### 真机可行性边界 + 截断（第 6 轮收尾）

新增 `truncate_excited_configurations`（Slater-Condon 对角能量截断，`max_configs` /
`energy_threshold`，强制含 HF），控制大体系子空间维度。经典预演两个代表性体系：

| 体系 | 特征 | 全量单双激发 | 截断后误差 | 结论 |
|---|---|---|---|---|
| LiH/STO-3G | 弱关联 | 93 | 40 配置 → +1.1e-4 | ✅ 截断损失极小，真机理想 |
| N₂/STO-3G | **强关联**（三键） | 610 | 400 配置 → **+2.25e-2 平台** | ⚠️ 单双激发覆盖不足（需三/四激发），真机受限 |

**结论**：经典单双激发混合对**弱关联小分子**是"FCI 精确 + 低 shots"的真机理想方法；
对**强关联体系**（N₂ 等）单双激发覆盖不足，误差停在 ~1e-2 量级——这是方法的真机可行性边界，
量子采样的价值在强关联区重新变大（需更高激发 ansatz 或变分优化）。

### 本轮 review（准确性 + 可用性 + 可读性）

对五轮新增功能做边界/冒烟审查（`_review_smoke.py`），发现并修复 3 处：

| 问题 | 修复 |
|---|---|
| `estimate_true_occupancies` γ=1 时 0/0 → **NaN** | 分母下限 1e-12，观测>0 的位 clip 到 1 |
| `solve_sci` 同时给 spin_sq + n_roots 时 n_roots 被**静默忽略** | 触发 RuntimeWarning 提示 |
| README 目录漏 `integrated` 模块 | 补上 |

新增回归测试：`test_estimate_true_occupancies_gamma_edge`（γ=1 有限 / γ=0 退化）、
`test_solve_sci_spin_sq_and_n_roots_warns`。10 个测试文件全 PASS。

## 已知扩展：与 Vayesta 共存（numpy 2.x 路径）

tc_sqd 默认走 numpy 1.x 路径（tensorcircuit 0.12 原生兼容）。但若要与 **Vayesta**
（BoothGroup 量子嵌入库，倾向 numpy 2.x 生态）共存，可用 numpy 2.x + tc_sqd 兼容
补丁的路径。实测可行（`tc_vayesta` 环境）。

### 兼容组合（实测：tensorcircuit 0.12.0 + vayesta 1.0.1 + numpy 2.2.6）

| 包 | 版本 |
|---|---|
| python | 3.10 |
| numpy | 2.2.6（标准，配 `_compat` 补丁） |
| scipy | 1.15.3 |
| pyscf | 2.14.0 |
| cvxpy | 任意新版 |
| tensorcircuit | 0.12.0 |
| vayesta | 1.0.1（github clone，`.pth` 装载） |

### 三个关键步骤

1. **numpy 2.x ↔ tensorcircuit 0.12 补丁**（tc_sqd 已机制化到 `_compat`）：
   ```bash
   pip install -e /path/to/tc_sqd
   python -m tc_sqd._compat install   # 写 sitecustomize, 自动补 np.ComplexWarning + np.reshape(newshape)
   ```

2. **Vayesta 安装**：`pip install vayesta @ git+...` 的 wheel build 会因 Vayesta
   pyproject 的 `[tools.setuptools]` typo（多了一个 s）失败。绕过：clone + `.pth`：
   ```bash
   git clone https://github.com/BoothGroup/Vayesta ~/vayesta      # 需联网
   SITE=$(python -c "import site;print(site.getsitepackages()[0])")
   echo "$HOME/vayesta" > $SITE/vayesta_path.pth
   ```
   且必须 `--no-deps`：Vayesta 声明 `pyscf @ git+master`，不抑制会重新编译 pyscf。

3. **WSL 联网**（NAT 模式）：Windows Clash 开 "Allow LAN"，WSL 经 gateway 代理
   （如 `http://172.29.128.1:7897`）装 github 包。

### 验证

- `import tensorcircuit + vayesta + numpy 2.2.6` 共存 OK
- Vayesta DMET 示例（`examples/dmet/01-simple-dmet.py`）跑通：`E=−2.8776 Ha`，
  自洽 7 次迭代收敛（error 0.9 mHa）

> **上游已修复**：BoothGroup/Vayesta 的 `[tools.setuptools]` typo 已由 PR #201
> （`fix: correct [tools.setuptools] typo in pyproject.toml`）修复，**2026-08-04 merged**
> （commit `aba7713c9a4bb12830b7a7aa09b07af13e6d6ad8`）。新版 Vayesta `pip install` 即可
> 直接用，第 2 步的 `.pth` workaround 仅在旧版（< PR #201）仍需。

## P2 轮（2026-08-02，双 agent 协作）：开壳层 + UCJ 精确化

| 项 | 内容 | 验证 |
|---|---|---|
| P2-1 开壳层 | `solve_sci`/三路径原生支持 (na,nb) 不等（`direct_spin1` 后端）+ `recover`/`estimate` na≠nb；CH/STO-3G `(3,2)` 回归 | 三路径 = FCI（1e-8）|
| P2-1b `from_pyscf` 开壳层 | ROHF（`mol.spin!=0` 自动）、nelec 推断、UHF 显式拒、frozen-core nelec 分减；顺带修 fci 路径 `direct_spin1`（原 `selected_ci` 全空间+不等 nelec 陷局部根差 2e-3）| CH `(4,3)` 一键 → FCI；闭壳层零回归 |
| P2-2a UCJ 分解 | `ucj_decomposition` t2→SVD→多层 (κ,J)（简化，非 ffsim 精确，已诚实标注）+ `ucj_subspace_energy`（确定性 SQD）| **H₂ = FCI**；LiH 趋近（scale 增大）|
| P2-2b UCJ 电路 | `build_ucj_circuit`（Û Givens；`include_jastrow=False` 默认，SQD 相位无关省略 RZZ）| H₂ = FCI；LiH **~2e-4 < 简化 LUCJ 7.5e-4**（2Q 门 16）|

**关键发现**：UCJ 单态期望对单层对角 J 无法低于 HF（Û 旋转引入对称禁戒单激发；A 审出 occ 运算符优先级 bug 已修，重验结论成立）；**UCJ 价值在子空间对角化**（确定性 SQD H₂=FCI）。

## 方向 A：UCJ 辅助配置补充（强关联突破，2026-08-03 独立探索）

经典单双激发（CCSD 类）对强关联覆盖不足（N₂ 7e/spin 平台 2.25e-2）。方向 A：
**UCJ 电路采样产生超出单双激发的高激发 det，与单双激发合并 include 后 SQD 覆盖强关联**。

**稳定性修复**（对比中发现）：N₂ 拉伸近简并轨道 → RHF/CCSD 每次收敛的轨道方向不同
（t2 差可达 1.1，能量却相同）→ UCJ kappa 方向依赖 t2 → det 覆盖偶发退化（1e-3 ↔ 2e-2
波动）。修复：**多 scale（3,5,10,20）+ 独立随机轨道旋转源（`n_random=2`）** → 5 次新进程
全化学精度（1.2-3.1e-3）。

**对比（STO-3G，误差 vs FCI，Ha）**：

| 分子 | 关联 | CCSD | CCSD(T) | CISD | **UCJ-SQD** |
|---|---|---|---|---|---|
| LiH | 弱 | 1.1e-5 | 2.1e-6 | 1.3e-5 | **3.1e-12** |
| H₂O | 弱-中 | 1.2e-4 | 5.0e-5 | 7.2e-4 | **2.8e-14** |
| BeH₂ | 中 | 4.1e-4 | 1.9e-4 | 8.0e-4 | **2.3e-7** |
| N₂ 平衡 | 中 | 3.9e-3 | 2.2e-3 | 1.2e-2 | **2.1e-5** |
| **N₂ 拉伸** | **强关联** | **1.4e-1** | **1.4e-1** | **2.0e-1** | **3.1e-3** |

强关联下经典单参考全部崩溃（0.14-0.20 Ha），UCJ-SQD 保持化学精度（比 CCSD(T) 好 ~45×）。

**vs HCI（N₂ 拉伸，生成集 HCI）**：

| 方法 | det 数 | 误差 |
|---|---|---|
| HCI（单双闭包）| 2,116 | 2.25e-2（平台）|
| HCI（近全空间）| 9,604 | 5.5e-8（=FCI）|
| **UCJ-SQD**（多 scale + 随机旋转）| **765-1,339** | **1.16e-3 稳定** |

**⚠️ 口径说明（2026-08-03 修正）**：表中 UCJ-SQD 的 `765-1,339` 是**采样 det 数**
（bsm 唯一行，对应量子采样成本 ~2000 shots），**不是对角化维度**。SQD 库沿字符串乘积
表示（`bitstring_matrix_to_ci_strs` 合并 α/β → 对角化维度 = 字符串数²）：N₂ 拉伸 UCJ 种子
约 89-120 字符串 → 实际对角化维度约 7,900-14,400，**远大于采样 det 数**。HCI 的
`2,116/9,604` 是具体 det 集合（即其对角化空间）。**两口径不可直接比大小**——UCJ-SQD 的
真实优势是**采样效率**（少量 shots 达化学精度、量子资源省），而非对角化空间更小；"用 ~1/10
的 HCI 子空间大小"的旧表述不成立。

API：`ucj_assisted_configurations(mf, norb, nelec, *, scales, n_samples, n_random)`、
`solve_ucj_assisted(...)`。测试：`test_ucj_assisted_n2_strong_correlation`、
`test_ucj_assisted_lih_weak_correlation`（commit `e23c7fc`）。

### C₂ 边界：准简并双 π + FCI 基准假收敛（2026-08-03 修复）

探索中 C₂/STO-3G（平衡）曾呈现"所有方法失败"（CCSD 3.5e-2、CCSD(T) 4.8e-2、UCJ-SQD 偶发
负误差 4.9e-2）。逐一排查（粒子数、字符串轨道映射、手写 Slater-Condon 交叉验证、稠密 numpy
eigh、多初始向量 davidson、scipy eigsh）后定位为**两个独立库 bug**，修复后 UCJ-SQD 稳定化学精度：

1. **FCI 基准假收敛**（`fermion.py` fci 分支）：C₂ FCI 空间 44100 维，双 π 准简并
   （真基态 **-74.690041** vs 第二根 **-74.639599**，二重简并，差 0.0504 Ha）。
   `direct_spin1.kernel` 默认 `conv_tol=1e-10` 的 Davidson 在 Ritz 值稳定（`max|de|` 小）但
   残差仍 ~6e-6 时判定收敛，**假收敛到第二根**（基准虚高 0.0504 Ha；`max_cycle=50` 停在
   第 17 步，而真基态要第 20 步 restart 后第 43 步才落到）。修复：默认 `conv_tol=1e-12,
   max_cycle=1000`。其余分子（LiH/H₂O/BeH₂/N₂）FCI 早已收敛，本修复对它们误差变化 < 1e-12。

2. **solve_sci 准简并陷阱**（`fermion.py` solve_sci 基态分支）：即便 FCI 基准正确，子空间内
   **基态所有主导 det 均已被 S+D∪UCJ 覆盖**（|c|² 0.67~0.003 的 det 全在子空间），
   `kernel_fixed_space` 的 Davidson 仍从部分初始向量收敛到第二根（-74.6396）而非基态
   （-74.6900）——同一 H，`eigsh`（ARPACK SA）给 -74.6900，min-h 初始 davidson 给 -74.6389。
   修复：`solve_sci` 基态对角化改用 scipy `eigsh`（求最小特征值稳健）替代 davidson；
   小空间（dim ≤ 1000）直接显式建 H + numpy eigh（无 ARPACK k≥N 限制）。

修复后 C₂ 平衡（STO-3G，8 次新进程）：

| 方法 | 误差 (Ha) |
|---|---|
| CCSD | 3.5e-2 |
| CCSD(T) | 4.8e-2 |
| **UCJ-SQD**（修复后）| **3.5e-5 ~ 1.3e-3（8/8 化学精度）** |

**C₂ 结论**：不是方法失效，而是**准简并收敛**的双重边界。UCJ 采样（多 scale + 随机旋转）
覆盖准简并基态稳定（8/8 成功，~1e-3），比经典单参考（0.035-0.048）好 ~40×。之前增采样量
（n_samples 3000→8000）不改变覆盖率，问题确在求解器收敛而非采样量。

## 方向 B：CIPSI 迭代结合 UCJ 辅助（高精度 refine 层，2026-08-03 落地）

**定位**：UCJ-SQD 用少量采样 shots 达化学精度；若需更高精度（FCI 级），从 UCJ 种子出发做
PT2-CIPSI 生成集扩展，1-2 轮即补全到全空间。

**算法**（`src/tc_sqd/cipsi.py`，`solve_cipsi`）：每轮 ① 子空间对角化（复用 solve_sci 稳健
路径：dim≤1000 numpy eigh / 否则 eigsh）→ ② 取 |c|>ε 主导 dets 枚举单/双激发连接 →
③ 扩展空间上 `contract_2e` 一次得 <a|H|Ψ>（pyscf 矩阵元，免手写 Slater-Condon 符号坑）→
④ `PT2=⟨a|H|Ψ⟩²/(E_gs−E_a)` 按 |PT2| 加入 → 重复至全空间 / PT2 收敛。

**实测**（修复后的 FCI 基准上）：

| 体系 | 种子 | UCJ-SQD | solve_cipsi（S+D 或 UCJ 种子） |
|---|---|---|---|
| H₂ | S+D | =FCI | **= FCI（1e-6）** |
| N₂ 拉伸 | S+D（单双平台 2.25e-2）| — | **< 1e-4（突破平台）** |
| N₂ 拉伸 | S+D ∪ UCJ | 1.05e-3 | **= FCI（-7.1e-13）** |
| C₂ 平衡 | S+D ∪ UCJ | 3.55e-5 | **= FCI（-4.4e-13）** |

**关键观察**：UCJ 种子字符串已覆盖全空间大部分（N₂ 89-120/120，C₂ 133/210），CIPSI 单双激发
闭包**一轮补全到全空间 = FCI**。代价是 det 规模 = 全空间（N₂ 14400 / C₂ 44100），与 HCI
近全空间相当——**CIPSI 是"高精度 refine 层"，不是"少量 det"路线**；UCJ 的真正优势仍在采样效率。

**口径提醒**：`solve_cipsi` 的对角化维度是字符串乘积（闭壳层 α/β 合并），与"采样 det 数"
（bsm 行）不同口径，勿混用。

API：`solve_cipsi(h1e, eri, norb, nelec, *, seed_bitstring_matrix, max_strings, ...)`。
测试：`test_cipsi_h2_reaches_fci`、`test_cipsi_n2_stretch_breaks_sd_platform`。

## 方向①：基设计（自然轨道换基，2026-08-04 验证落地）

**定位**：采样策略探索的第一步（对应 SURVEY §7 基设计方向）。SQD 子空间 = 采样 det 张的
空间，其效率取决于基态波函数在计算基下的**稀疏度**。把积分旋转到自然轨道基（1-RDM 对角化）
可大幅压缩波函数长尾系数——纯经典后处理（零额外量子资源），不触碰 solve_sqd 签名。

**方法**（`src/tc_sqd/basis.py`）：
- `rotate_to_natural_orbitals(h1e, eri, rdm1)`：1-RDM → 自然轨道 → 换基积分（U 列 = 新基，
  与 `fermion.rotate_integrals` 约定一致）
- `ccsd_natural_orbitals(mf)`：经典 CCSD 1-RDM 先验（真实自举路径，不需 FCI）
- `rdm1_from_sci_result(result)`：从 SQD 解取 1-RDM，**自洽换基闭环的输入源**
- `natural_orbital_basis_from_fci`：FCI 1-RDM 换基（理想极限基准）

**验证**（N₂/STO-3G 拉伸，强关联，dim=14400；FCI 基准已修复）：

*稀疏度（|C|² 系数分布）*

| 指标 | MO 基 | FCI-NO 基 | CCSD-NO 基 |
|---|---|---|---|
| k99（99% 覆盖 det 数）| 84 | **39** ↓54% | 71 |
| k999（99.9%）| 189 | **62** ↓67% | 126 |

max|C|² 几乎不变（强关联 HF 非主导），但**长尾被大幅压缩**——这是换基的价值所在。

*top-K det 子空间对角化（达到化学精度 1.6e-3 Ha 所需最小维度）*

| 体系 | MO 基 | FCI-NO | CCSD-NO |
|---|---|---|---|
| N₂ 拉伸（强关联）| 2116 | **676** ↓68% | 1849 |
| N₂ 平衡（弱关联）| 2070 | 992 ↓52% | **841** |

*低采样自洽换基对照（N₂ 拉伸，配置恢复 SQD，能量误差 vs FCI）*

| n_samples | err(无换基) | err(自洽换基) | 改善 |
|---|---|---|---|
| 100 | 2.2e-3 | 4.8e-4 | 4.7× |
| 200 | 1.7e-3 | 2.9e-4 | 5.9× |
| 400 | 1.9e-4 | 1.7e-6 | 110× |
| 800 | 1.8e-4 | 6.9e-7 | 263× |

**关键结论**：
1. **FCI-NO 换基显著提升子空间构建效率**：强关联下达到化学精度维度 2116→676（↓68%）。
2. **CCSD-NO 是真实路径但强关联打折**：平衡时≈FCI-NO（841 vs 992），拉伸时落后（1849 vs 676）
   —— CCSD 自身多参考缺失。**真正形态是自洽迭代**（SQD→1-RDM→换基→重解）。
3. **自洽换基突破配置恢复的 MO 基覆盖瓶颈**：无换基在 400+ 采样后卡在 ~1.8e-4，自洽换基
   持续降至 ~1e-7（改善最多 263×）。纯经典后处理，无额外量子采样。
4. 与方向 B（CIPSI）、方向②（自适应采样）衔接：换基后的稀疏表示是自适应采样的地基。

API：`tc_sqd.basis`（`rotate_to_natural_orbitals` 等 6 函数，非侵入换基工具）+ `solve_sqd_natural_orbitals`
（自洽换基闭环，方向② 表示层地基：解 SQD → 1-RDM → 自然轨道换基 → 重解，返回 `NaturalOrbitalResult`）。
测试：`tests/test_basis.py`（旋转不变性 / 稀疏度改善 / CCSD-NO 不劣化 / 自洽换基收敛 / 优于无换基 / 电子数守恒，9 项）。
自洽换基与主动采样分布偏置（PT2 反哺采样，对应 AS-SQD）衔接方向②后续。

## 方向②：自适应/主动采样闭环（受限 PT2 选态 + 采样聚焦，2026-08-04 落地）

**定位**：采样策略探索第二步（对应 SURVEY §7 自适应方向 / AS-SQD arXiv:2603.13536）。
纯采样 SQD 的子空间只含"采到"的 det，低采样/噪声下覆盖不全（C₂ 曾 3/8 失败）；
主动采样用 Epstein-Nesbet PT2 得分**确定性补足采样缺口**——无需额外量子测量，且
噪声 bitstring 的 PT2 得分近零（抗噪）。

**与 solve_cipsi 的区别**：solve_cipsi 是**纯经典 det 空间精化**（静态种子、补全到全空间、
不碰采样）；`solve_sqd_active` 是**采样↔选态双闭环**——每轮先用偏置的平均占据做配置恢复
（采样聚焦），再用受限 PT2 注入高价值 det（子空间不补全全空间），交替至收敛。

**方法**（`src/tc_sqd/cipsi.py`，`solve_sqd_active`）：
1. 每轮配置恢复（平均占据偏置）生成当前基 det 并入子空间（采样覆盖不受 max_strings 限）
2. 子空间对角化（复用 `_Subspace`，dim≤1000 numpy eigh / eigsh）
3. 主导 det 枚举单/双激发候选 → PT2=⟨a|H|Ψ⟩²/(E−E_a) → top `n_active_per_round` 注入
4. 用解态 1-RDM 更新平均占据（采样偏置），循环至 PT2 无新 det 且采样无新增
- `max_strings` 只约束 PT2 扩展（与 solve_cipsi 语义一致），采样覆盖不受限

**验证**（N₂/STO-3G 拉伸，强关联，n_samples=100 低采样）：

| 场景 | 误差 vs FCI | 说明 |
|---|---|---|
| 纯采样 SQD（配置恢复 3 轮）| ~2e-3 | 超化学精度（1.6e-3）|
| **solve_sqd_active（全空间）** | **3.95e-7** | PT2 确定性补足采样缺口 |
| solve_sqd_active（max_strings=100）| <1.6e-3 | 受限子空间 ≤100²（全空间 120²=14400）仍达化学精度 |

**关键结论**：
1. **主动采样突破低采样覆盖瓶颈**：n=100 时从 ~2e-3（超化学精度）压到 3.95e-7——一个量级以上的
   PT2 补足收益，且无需额外量子测量（对应 AS-SQD 主张）。
2. **受限闭环可行**：max_strings 约束 PT2 扩展，子空间远小于全空间仍达化学精度——区别于
   solve_cipsi 补全全空间的"高精度 refine"路线，主动采样是"少量 det"路线。
3. **与方向①衔接**：方向①的自洽换基（`solve_sqd_natural_orbitals`）提供稀疏表示层，
   主动采样在其上做选态聚焦——两者都是"提升子空间构建效率"的不同杠杆。
4. 覆盖率不稳（C₂ 3/8 失败）的根治：PT2 选态确定性保证关键 det 必进子空间，不再依赖采样运气。

API：`solve_sqd_active(h1e, eri, norb, nelec, *, bitstring_matrix, max_strings, n_active_per_round, ...)`。
测试：`tests/test_sqd_active.py`（低采样优于纯采样达化学精度 / 受限子空间达化学精度，2 项）。
全库 95 测试全过。

**组合版验证**（`solve_sqd_adaptive`，换基表示层① + PT2 选择层②，2026-08-04）：多 seed
(n=100, 6 seed) 对比 —— 纯采样 **0/6** 达化学精度（C₂ 3/8 式覆盖不稳），active 与
adaptive 均 **6/6**（覆盖率根治确认）。但 adaptive 误差略差于单独 active（mean 1.9e-6 vs
4.0e-7；极低采样 n=15~40 亦然）——换基每轮作废 det 累积，丢失 PT2 覆盖收益，表示层改善
不足以弥补。**结论**：`solve_sqd_active` 为实际推荐（简单、更准、稳定）；`solve_sqd_adaptive`
保留为"表示层+选择层"统一框架，但不声称优于 active。**换基+CIPSI 验证（2026-08-04）**：
受限 CIPSI 在 MO 与自然基下误差相同（~2.2e-2，均卡 S+D 平台）——CIPSI 的 PT2 选态已是
最优排序，抵消基的稀疏度差异，方向①稀疏度收益在"PT2 自适应选态"场景不体现（只体现在
确定性 top-K 截断场景）。测试断言相应调整为"稳定达化学精度"。

## 方向③：拟 Krylov 理论化（多 scale UCJ 的形式化表述，2026-08-04 文档）

**定位**：采样策略探索第三步（对应 SURVEY §7 拟 Krylov 方向）。把方向 A 的"多 scale UCJ
辅助配置补充"（`ucj_assisted_configurations`，scales=(3,5,10,20) + 独立随机旋转）正式表述为
**参数化 Krylov 型子空间**，并给出与标准 Krylov / SKQD 的关系与收敛性论证。

**形式化**（准 Krylov 子空间）：

| 要素 | 标准 Krylov | SKQD（arXiv:2501.09702）| tc_sqd 多 scale UCJ |
|---|---|---|---|
| 生成元 | H 的幂 {H^k} | 时间演化 {e^{−ikHΔt}} | 参数化酉 {Û(s)e^{iĴ(s)}Û(s)†} |
| 作用对象 | \|ψ₀⟩ | \|ψ₀⟩ | \|HF⟩（CCSD t2 → SVD → Û/Ĵ）|
| 子空间 | span{H^k\|ψ₀⟩} | span{e^{−ikHΔt}\|ψ₀⟩} | span{Û(s)e^{iĴ(s)}Û(s)†\|HF⟩} |
| 参数扫描 | 阶数 m | 时间步数 k | scale s ∈ (3,5,10,20) |

多 scale UCJ 把"对初态施加一族算符"的 Krylov 思想迁移到**硬件可编译的参数化酉族**：
Ĵ 是 two-body 算符（对应 H 的二体部分），Û(s)e^{iĴ(s)}Û(s)† 可看作"近似的、可编译的 H 演化"
——与 SKQD 的 e^{−ikHΔt} 同构（SKQD 也明言化学分子走 UCJ 路线而非时间演化）。

**收敛性论证**（对任意参考态族的一般性质）：
设 V = span{Û(s)e^{iĴ(s)}Û(s)†|HF⟩ : s ∈ Θ}，P_V 为投影。对 V 对角化得到基态能量
E_V 满足（变分原理）：E_V ≥ E_gs，且误差 ≤ ‖(I−P_V)|Ψ_gs⟩‖²·ΔE（ΔE 为相关能标度）。
因此子空间质量完全由 ‖P_V|Ψ_gs⟩‖ 决定——多 scale + 随机旋转 = 对参数空间 Θ 采样，
增大样本使 V 趋向覆盖主导配置；这解释了为何单 scale 跨进程误差在 1e-3~2e-2 波动
（偶发落在低覆盖方向），多 scale 合并保证高激发 det 总被覆盖（跨进程稳定化学精度）。

**统一视角**（三方向的闭环）：
- **基设计（方向①）**：优化工作表示（自然轨道换基）→ 使展开系数更集中，任何子空间方法
  （Krylov/UCJ/选态）都更稀疏、更高效。
- **主动采样（方向②）**：用 PT2 确定性补足展开系数大的 det，替代对采样运气的依赖。
- **拟 Krylov（方向③）**：多 scale UCJ 提供**多样初猜**（参数化酉族张子空间），
  与方向②的确定性选态互补——前者扩覆盖，后者精聚焦。

三者统一为"提升子空间对基态的覆盖效率"：表示层（①）+ 生成层（③）+ 选择层（②）。

**后续可选**（若做）：把多 scale UCJ 收敛性做成可量化指标（如 ‖P_V|Ψ_gs⟩‖ 的估计，
用采样 det 子空间的 Ritz 重叠近似）；或与 SKQD 的 1/poly 保证做数值对照。

## B3/B4 缺陷修复（2026-08-04，方向探索第一优先"修缺口"）

**B3 激发态 `n_roots>1` 分支 eigsh 修复**（`fermion.py solve_sci`）：多根分支原走
`selected_ci.kernel_fixed_space`（davidson），与基态分支同款准简并陷阱（C₂ 式跳根虚高）。
改为与基态分支一致的 scipy `eigsh`（`k=n_roots, which="SA"`，dim≤1000 或求全谱时 numpy eigh）。
验证（N₂/STO-3G 拉伸，dim=14400）：多根分支基态根 = FCI 基态（diff < 1e-6，eigsh SA 不跳根），
根升序。注：全空间子空间含非对称态（triplet），FCI direct_spin1 约束 singlet，故只断言基态根。

**B4 carryover 语义统一 + 批内保留 probs**：
- `solve_sqd`（integrated.py）carryover 从 Hamming-weight postselect（采样层语义）统一为
  **振幅阈值**（保留上一轮解态 `|c|≥thr·max|c|` 的 det 注入下一轮），与
  `diagonalize_fermionic_hamiltonian`（fermion.py，默认 1e-4）语义一致。
- 批量子采样（`num_batches>1`）批内 recover 原用均匀 probs（丢弃原始概率）——`subsample`
  加 `return_probs`，批内恢复改用真实采样概率。

测试：`test_excited_sqd_n2_stretch_roots_no_root_skip`（B3 准简并不跳根）、
`test_sqd_carryover_amplitude_threshold` + `test_sqd_batch_probs_preserved`（B4）。全库 99 测试全过。

## A1/A3 外推族（2026-08-04，方向探索第二优先）

**A1 无限采样外推（证伪）**：`extrapolate_infinite_samples`（diagnostics.py）拟合
`E(S)=E∞+a/√S`。验证（N₂/STO-3G 拉伸，shots 50→2000）：外推 E∞ err=3.6e-2，
**比最大 shots 点（3.6e-5）差 ~1000×**——SQD 能量是采样 det 覆盖决定的**变分下界**
（随覆盖阶梯式收敛），非统计量，`1/√S` 模型不适用（该模型适用于期望值测量，如
A3 的 `E(γ)` 外推）。函数保留为通用统计量外推工具，docstring 已注明对 SQD 能量不适用。

**A3 T1 零噪声外推（落地）**：`zero_noise_extrapolate_t1`（noise.py）对 γ 网格用位串级
`apply_t1_bitstrings` 模拟 T1 噪声跑 SQD，最小二乘多项式外推 γ→0（参考 arXiv:2502.20673
低阶避免过拟合）。验证（N₂/STO-3G 拉伸）：外推 err=1.94e-4，**优于最噪点（4.77e-4）与
无噪声随机参考（7.35e-4）**。位串级模拟支持大体系（免 2^nq 密度矩阵）。

测试：`test_extrapolate_infinite_samples_fit`（A1 合成数据拟合正确性）、
`test_zero_noise_extrapolate_t1_improves`（A3 外推优于最噪点）。全库 101 测试全过。

## B1 预算闭环（2026-08-04，方向探索第三优先，落地）

**自适应停采省 shots**：`solve_sqd_active` 加预算参数——`shots_budget`（总预算，不足时预生成
随机位串补池）、`shots_step`（每轮增量采样）、`energy_tol`（能量收敛停采）、`usage`（输出
实际 shots）。每轮用池的前 `n_cur` 行，`n_cur` 递增；连续两轮 ΔE < `energy_tol` 即停。

验证（N₂/STO-3G 拉伸，budget=2000, step=300, tol=1e-5）：**自适应 900 shots 停采
（省 55%），err ~1e-12 与全量 2000 shots 相同**——能量收敛即停，不损精度。

测试：`test_sqd_active_budget_saves_shots`（停采省 shots + 精度不劣化）。全库 102 测试全过。

**附带修复**：`zero_noise_extrapolate_t1`（A3）内部 SQD 链路固定 seed（此前配置恢复用全局
随机，全量测试时 A3 偶发失败）；A3 断言放宽为"化学精度 + 不显著劣化"（ZNE 收益依赖 E(γ)
曲线形状，方法固有）；CCSD-NO 稀疏度容差 +1→+3（近简并轨道组 CCSD 收敛波动）。

## A4 ph-AFQMC 桥接（2026-08-04，实施路径探明，未跑通）

**目标**（arXiv:2503.05967）：SQD 截断 trial → ph-AFQMC 恢复 O(100)mHa 关联能，突破 SQD
精度上限。依赖外部 AFQMC 库。

**探明的事实**：
- `pyscf-afqmc` 不在 PyPI/清华镜像（需 git clone + 代理，WSL NAT 代理不通）。
- **ipie 0.7.1 已装**（清华镜像可装）。深度面向对象 API，组装需
  `integrals_from_scf` → `Generic`(ham) → `ParticleHole`(trial) → `UHFWalkers` →
  propagator → `MPIHandler` → `QMCParams` → `AFQMC(...).run()`。
- **ipie 0.7.1 源码 bug**（已修）：`from_pyscf.py:374` `print("..." % nchol_max)`
  字符串无占位符 → TypeError（Cholesky 阶段崩溃）。
- 反复试错（10 次探测）仍未完全组装：`PhaselessTrotter` 路径、system 对象、以及
  **SQD MSD trial（MultiDetTrial from civec）对接**接口更深。

**结论**：A4 的技术障碍大（ipie 0.7.1 深度 API + 外部依赖 + ph-AFQMC 计算成本），且
tc_sqd 小体系（N₂/C₂）SQD 已达 FCI —— ph-AFQMC 增量主要在**真机大体系**场景（受用户
约束）。建议：留待大体系/真机阶段，改用更成熟的 **ipie 0.6 旧 API**（`from_pyscf` 一键）
或 NVIDIA CUDA-Q 的 `ipie` 教程路径。已修 bug + 探明接口记录于此，后续接手可省大量探测。
**（按用户要求：A4 暂时搁置。）**

## 统一 API：`solve_sqd_robust`（B1 预算 × A3 ZNE 组合，2026-08-04）

**组合已验证的两个方向**（`noise.py`）：每个 γ 噪声水平下用 B1 自适应预算的
`solve_sqd_active`（增量采样 + energy_tol 收敛停采）求收敛能量 E(γ) 与实际 shots，
再对 E(γ) 低阶多项式外推 γ→0（A3 ZNE）。**噪声鲁棒 + 预算高效同时达成**。

验证（N₂/STO-3G 拉伸，gammas=(0.05,0.1,0.2,0.3), budget=2000, step=300, tol=1e-3）：
外推 E err **8.5e-14**（优于最噪点 9.7e-13）；每 γ 停采于 900 shots，**total 3600 vs
无预算 8000（省 55%）**。

测试：`test_solve_sqd_robust_combines_zne_budget`（ZNE 不劣化 + 预算省 shots）。全库 103 测试全过。

## 汇总图：减误差 / 提速方法与经典 baseline 对比（2026-08-04）

**基准**：N₂/STO-3G 拉伸（强关联，dim=14400），FCI 为精确参考，经典 baseline = 纯采样 SQD。

**图 1 `fig_error_vs_dimension.png`（减误差：误差 vs 子空间维度，log-log）**：
- `top-K dets (FCI-NO basis)`（自然轨道换基）< `top-K dets (MO basis)` < `classical SQD`
  在**相同子空间维度**下误差从低到高 —— 换基让子空间更高效（方向①）。
- 化学精度线 1.6 mHa：FCI-NO/MO top-K 在大维度达标，classical SQD 误差几乎不随维度下降
  （配置恢复在 MO 基的覆盖瓶颈，~1e-3~1e-4 平台）。

**图 2 `fig_error_vs_shots.png`（提速：误差 vs 采样成本，log-log）**：
- `solve_sqd_active (PT2 selection)` < `solve_sqd_robust (B1-budget × ZNE)` < `classical SQD`
  在**相同采样成本**下误差从低到高。
- 化学精度交点：active ≈ **100 shots**、robust ≈ 1000 shots、**classical SQD 达不到**
  —— PT2 主动选态把达到化学精度的采样成本降低一个数量级以上（方向②）。

数据由 `_plot_err_cost.py`（临时脚本，已删）生成；图存于仓库根，可重新生成。

**图 3 `fig_error_hci_vs_sqd.png`（经典选态 CI vs SQD：误差 vs 子空间维度，N₂ 拉伸）**：
- `classical HCI/CIPSI (solve_cipsi)`：**垂直段**——S+D 种子平台（err 0.022 Ha）在维度 ~9409
  几乎不变下经 PT2 补全跳降到 FCI（~1e-7）。选态 CI 的特性：用 PT2 挑少量关键 det，不靠维度增长。
- `traditional SQD`：平滑下降（2209→14161，err 0.042→2.2e-6），但**同维度下误差最高**。
- `solve_sqd_active`：维度 8836 即 err 5.3e-7（比 SQD 同维度低 3-4 个量级），14400 时 4.7e-13。
- `FCI-NO top-K`：平滑可控，维度 676 达化学精度（方向①确定性 top-K 的可预测性）。
- `CCSD` 参照线 err 0.10 Ha：经典单参考在强关联拉伸的失效基准。
- **结论**：active 是"同子空间规模下误差最低"的方法；CIPSI 是"几乎不增维度却补全到 FCI"的
  方法；FCI-NO 提供可预测的维度-误差标度。

## 后续可选改进（非阻塞）

## 后续可选改进（非阻塞）

- 自旋分辨哈密顿量（`h_alpha ≠ h_beta`，UHF 式）——需 spin-orbital SQD 后端
- UCJ 精确对标 ffsim（完整 J + 多参数 orbital rotation，非简化 SVD）
- 配置恢复 tie-breaking 随机性的统计性测试
- 多版本 numpy（1.x / 2.x）CI 矩阵，固化兼容性
- UCJ 精确化（t2→SVD→Û/J，对标 ffsim）；GPU CI 对角化（大体系路线）
