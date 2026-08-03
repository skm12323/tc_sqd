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

> 待 Vayesta 上游修了 `[tools.setuptools]` typo，`pip install` 即可直接用，第 2 步
> 的 `.pth` workaround 可省。

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

## 后续可选改进（非阻塞）

- 自旋分辨哈密顿量（`h_alpha ≠ h_beta`，UHF 式）——需 spin-orbital SQD 后端
- UCJ 精确对标 ffsim（完整 J + 多参数 orbital rotation，非简化 SVD）
- 配置恢复 tie-breaking 随机性的统计性测试
- 多版本 numpy（1.x / 2.x）CI 矩阵，固化兼容性
- UCJ 精确化（t2→SVD→Û/J，对标 ffsim）；GPU CI 对角化（大体系路线）
