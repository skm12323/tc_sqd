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

机制（早期方向 1/2 验证）：**退相干 diag 不变（SQD 免疫）**，振幅阻尼改 diag（T₁ 主导误差），
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

### hardware 模块（腾讯 qcloud 真机一站式，整合早期 qubit_toolkit 与噪声实验代码）

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

## 真正的 HCI 实现（`solve_hci`，2026-08-04）

**SHCI（Holmes 2016 JCTC 12, 3674 + Sharma 2017）**——与 `solve_cipsi` 的本质区别在选态标准：
CIPSI 用完整 Epstein-Nesbet **PT2 排序**（⟨a|H|Ψ⟩²/(E−E_a)），HCI 用**单参考 det 对矩阵元
heat-bath 筛选**（|⟨j|H|i⟩| ≥ `eps_hb`）——更便宜（不求完整 ⟨a|H|Ψ⟩，只按与某主导 det 的
耦合强度选态）。

**两阶段 SHCI**（补 PT2 修正后）：
- **阶段 1（ε₁ = `eps_hb`）**：heat-bath 选态构建变分空间 V（到无新增）。
- **阶段 2（ε₂ = `pt2_floor`）**：对角化 V 得 `E_V`，对 V 外候选算 **PT2 能量修正**
  `E_PT2 = Σ_a |⟨a|H|Ψ⟩|²/(E−E_a)`。返回标准 SHCI 报告 `E_total = E_V + E_PT2`。
- `return_details=True` 返回 `(E_total, E_PT2, dim)`（诊断/绘图）。

**依赖确认**：外部 HCI 包在当前网络环境**均不可装**——`pyhci`（清华镜像无）、
`pyscf-forge`（wheel 编译失败）、`pyscf/naive-hci`（git+ GitHub 克隆失败）、`pyscf.hci`
（pyscf 2.14 无内置）。因此 **HCI 在库内实现（无额外依赖）**，复用 pyscf `contract_2e` +
`_Subspace`（传单位向量 `e_i` 得单对矩阵元 ⟨j|H|i⟩），与 pyscf/naive-hci 同思路的朴素实现。

**验证**（`tests/test_cipsi.py`，SHCI 相关 3 项）：H₂/N₂ 拉伸**从 HF 单 det 出发**
（seed=None）经 heat-bath 选态补全到 FCI（err < 1e-4）；`eps_hb` 控制变分空间规模 +
**PT2 修正补足**（N₂ 拉伸 eps_hb=5e-2 时 dim=3481, E_PT2=-3.9e-5 → E_total 接近 FCI；
eps_hb=1e-4 时 E_PT2≈-7e-11）。全库 106 测试全过（本小节的 HCI 三项）。

## 方向 D：能量-方差修正（PT2）+ 本征矢重要性采样（2026-08-07，实证修正）

**核心理论**：对截断 CI 子空间 V 对角化得的本征矢 |Ψ⟩，**精确能量方差**
`σ² = ⟨Ψ|H²|Ψ⟩ − E² = Σ_{a∉V} |⟨a|H|Ψ⟩|²`（只含子空间外矩阵元平方和）——
即 PT2 计算中分子项的平方和，**直接可算**。方差（或等价地 PT2 分子）给出
截断误差的估计，可构造**不增大最终子空间维度**的纯经典修正。

**实施**：
- `solve_sqd_active` 新增 `trajectory` 输出参数：每轮记录 `{round, E, sigma2,
  e_pt2, dim, shots}`（`sigma2` 为子空间外 PT2 分子平方和；`e_pt2` 为
  Epstein-Nesbet `Σ|⟨a|H|Ψ⟩|²/(E−E_a)`；最终对角化点也记录）。
- `diagnostics.extrapolate_energy_variance(E, σ², degree=1)`：多项式最小二乘
  外推到 σ²=0，返回 `(E∞, 斜率, r², 拟合误差带)`（通用工具；对 SQD/CIPSI/
  HCI 轨迹均适用）。
- `cipsi.solve_sqd_ev`：薄封装（active + trajectory + 修正，不重跑），
  `correction="pt2"`（**默认，推荐**）= `E + E_PT2`（SHCI 式）；`correction=
  "ev"` = σ² 线性外推（诊断）。

**实证修正（关键，2026-08-07）**：用 N₂ 与 C₂ 对比三种"改进"——σ² 线性外推、
PT2 修正：
| 体系 | 直接 err | σ² 外推 err | **PT2 修正 err** |
|---|---|---|---|
| N₂/STO-3G 受限 | +4.3e-4 | **−5.8e-4（过冲）** | **+6.2e-5** |
| N₂/STO-3G 收敛 | +4.0e-7 | −1.4e-4（过冲） | **−3.2e-9** |
| C₂/STO-3G | +7.9e-3 | **−1.7e-2（过冲）** | **+5.0e-4（达化学精度）** |

**σ² 线性外推全部过冲到 FCI 之下**（即使 r²=1.0）——对已近收敛的 active 轨迹
不可靠，**不作为默认**（保留为方差标度诊断）。**PT2 修正 `E+E_PT2` 行为良好且
真正改进**：N₂ 受限 4.3e-4→6.2e-5，C₂ 直接超化学精度（7.9e-3）而 PT2 修正达
化学精度（5.0e-4）——"active 变分空间 + PT2 修正"（与 SHCI 同构）是方向 D 的
实际改进。`solve_sqd_auto` 默认修正同步改为 PT2。

**验证**（`tests/test_diagnostics.py` + `tests/test_sqd_active.py`）：
- 合成 `E(σ²) = E∞ + a·σ²` 数据线性/二次拟合并行精确恢复 `E∞`（r²>0.9999）
  ——外推工具本身的数学正确性。
- N₂/STO-3G 轨迹单调性（E/σ² 不增、dim 增）；`solve_sqd_ev` **默认 PT2 修正
  优于直接能量**（断言 `|E_corr−FCI| ≤ |E_direct−FCI|`）；σ² 外推诊断模式可
  运行且落化学精度带。

**本征矢重要性采样（学习型采样先验，AI 结合点）**：`eigenvector_importance_sample`
从子空间对角化本征矢振幅平方分布 `p_i ∝ |c_i|^(2/temperature)` 采样 det 位串。
这是"从解态学分布"的最简实现（数据驱动先验），与 NQS/神经网络参数化采样分布
的衔接点；`temperature<1` 锐化聚焦主导 det（验证：H₂ 低温采样 ratio 明显高于
高温）。工程上可作下一轮采样的**改进先验**（替代均匀/随机），确定性、可验证。

## 方向 E：工程自动化（基准 / 噪声评估 / 超参推荐 / 自适应流程，2026-08-07）

**① 性能基准**（`benchmarks/benchmark_sqd.py`）：测量传统/active/ev/cipsi/hci
× h₂/n₂/lih × shots 网格的**墙钟耗时**（time.perf_counter）与**峰值内存**
（ru_maxrss, WSL），输出 CSV + Markdown 表。quick 模式（h2/n2 × [500] ×
traditional/active/ev）实测：
| 方法 | N₂/STO-3G 500 shots | wall | peak | err vs FCI |
|---|---|---|---|---|
| traditional SQD | 79 s | 375 MB | 1.8e-4 |
| active SQD | 296 s | 377 MB | 7.2e-13 |
| EV | 537 s | 377 MB | 3.4e-6 |
H₂ 全空间 = FCI（err 0，~0.3 s）；CIPSI 补全到全空间最慢（N₂ 76-227 s）。
**洞察**：active/ev 的"效率"体现在**量子 shots 大幅节省**（图 2，~100 shots 达
化学精度），**经典 wall-time 更高**（多轮对角化 + PT2 选态）——两种口径要分开
讲。内存峰值三种方法相近（~377 MB，Python/pyscf 常驻）。

**② 噪声影响评估**（`noise.noise_impact`）：对同一无噪声位串池施加逐级 T1
（γ 0→0.4），跑 SQD 得 `E(γ)`，量化"噪声把结果拖多远"。输出：各 γ 能量/误差、
**化学精度安全区 `safe_gamma`**（误差 < target 的最大 γ，网格内线性插值）、
可选真机参数换算 `safe_depth`（最大电路深度）、主导因子与建议文案。误差口径：
默认对 `E(0)`（纯噪声退化，与截断误差解耦）；传 `e_reference`（如 FCI）则对
绝对误差。**实测**：N₂/STO-3G active 采样对 T1 近免疫（errors 全 ~1.7e-7，
safe_gamma=0.4）——recover 纠正 + PT2 抗噪的实证。

**③ 超参自动推荐**（`predict.recommend_sqd_params`）：给定分子（norb, nelec）
+ 硬件（T1_us, t_gate_ns）+ 目标精度，返回 `SqdParams` 结构化推荐：
- `shots`/`depth`：`plan_sampling` 枚举 (shots, depth) 网格取**最便宜可行**
  方案（预测误差 < target）；
- `max_strings`：子空间维度上限启发式 `min(full, max(50, min(250, 25·norb)))`
  （对角化维度 ≈ n_str² 保持可解）；
- `n_active_per_round`：`max(10, min(50, max_strings//3))`（选态注入随规模缩放）；
- `dom_thresh`/`pt2_floor`：库默认；`feasible`/`reason`：无可行组合时明确警告
  并给改进建议（ZNE / 换基 / 加大 shots）。
注意：精度模型用 H₄ 拟合的 KS/KT1，跨体系只作数量级起点（建议先 `calibrate`）。

**④ 自适应流程**（`integrated.solve_sqd_auto`）：一键流水线——超参推荐（有
T1 时）→ 采样（电路/随机位串）→ `solve_sqd_active` B1 自适应停采（energy_tol
自动判断收敛）→ 轨迹能量-方差外推（饱和保护）。返回 `{energy, E_direct, E_ev,
shots_used, recommendation, trajectory, converged, n_rounds}`。实测 N₂/STO-3G：
500 shots 内收敛（shots_used=500），err ~2.4e-6（EV 版）。

**测试**：新增 9 项（EV 合成恢复、N₂ EV 化学精度、轨迹单调、重要性采样聚焦、
noise_impact 安全区、recommend 结构/上限/过紧、auto 端到端）。全库测试从
106 → 115（方向 D/E 后）。

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

**图 3 `fig_error_hci_vs_sqd.png`（SHCI vs SQD：误差 vs 子空间维度，N₂ 拉伸）**：
- `HCI variational E_V`（虚线）与 `SHCI E_V+E_PT2`（实线）：**低维度（dim~3481）处虚线误差
  高于实线（E_PT2=-3.9e-5 补足）**——直观展示 PT2 修正价值; 维度增大（eps_hb 减小）两者
  趋同（E_PT2→0）。
- `traditional SQD`：覆盖不全时误差最高（4489→13924，err 0.126→1.8e-4）；**右端垂直陡降到 1.3e-9 是"全有或全无"伪影**——N₂/STO-3G 全空间仅 14400（120²），shots≥800 时配置恢复恰好补全全部 120 det → 子空间=全空间 → 直接解 FCI（图上已标注 "full space reached = FCI"）。**任何未达全空间的维度下传统 SQD 都远差于 active/improved SQD/SHCI**。
- `solve_sqd_active`：维度 8836 即 err 5.3e-7（比 SQD 同维度低 3-4 个量级），14400 时 4.7e-13。
- `FCI-NO top-K`：**同维度下误差最低**（维度 3969 时 3.2e-6 vs HCI 3481 时 E_V 4.0e-5）——
  确定性 top-K 是选态上限，HCI 的 heat-bath 是近似选择。
- `CCSD` 参照线 err 0.10 Ha：经典单参考在强关联拉伸的失效基准。
- **结论**：SHCI 的 PT2 修正让"小变分空间 + 外推修正"达到高精度; active 是"同子空间规模下
  误差最低"的采样方法; FCI-NO top-K 提供选态方法的理论上限标度。

## 图 `fig_improved_sqd_vs_shci.png`：improved SQD vs SHCI 专对照（2026-08-07）

**体系**：C₂/STO-3G（强关联，全空间 44100，FCI 参考可得）——比 N₂ 全空间 14400
大三倍，维度覆盖范围更宽。**对标澄清**：qiskit-addon-sqd **本身不含 HCI**（用
配置恢复 + `kernel_fixed_space` 简单 CI）；其集成的经典强关联求解器是**外部
Dice/pyhci**，即 **SHCI**。故本图 `SHCI E_V+E_PT2`（`solve_hci`）与
qiskit-addon-sqd + Dice 的 HCI 报告值口径一致。

**取点均匀**：预定义 log 均匀目标维度（600→43000，12 档），对每方法取最近点，
去重后保代表性点；两条曲线维度覆盖——SHCI **7 点** [144, 41616]（eps_hb
1e-1→1e-3，含 144/5041/7056/9025/17424/31684/41616），improved SQD **7 点**
[2500, 43264]（shots×max_strings 组合，含 2500/6084/10609/14884/18769/30625/
43264）。维度 log 均匀、无堆叠重复点。

**结果**（误差 vs 维度，log-log）：
- **交叉点 ≈ dim 15000**：低维度 SHCI 误差更低（如 dim~5000：SHCI 2.3e-6 vs
  improved SQD 1.3e-5）；**高维度 improved SQD 反超**（dim~30000：improved SQD
  1.9e-9 vs SHCI 7.8e-9；dim~43000：improved SQD 6.6e-10）——improved SQD 收敛
  更平滑单调，SHCI 在 5000~17000 有一截 PT2 补不足的平台（~2.3e-6）。
- 虚线对照：两者变分层（`solve_sqd_active` 直接 / `HCI E_V`）都在各自 PT2 修正
  实线上方——**PT2 修正对两条方法都带来增益**（SHCI 修正幅度在低维度最大）。
- improved SQD 在 C₂ 上优势明显：采样恢复 + PT2 注入在 shots=4 即达 dim 2500，
  且修正后误差随维度单调下降到 6.6e-10。

生成脚本：`examples/plot_improved_sqd_vs_shci.py`（确定性 seed 0，断点缓存）。

## 图 `fig_improved_sqd_vs_shci_n2.png`：improved SQD vs SHCI（N₂/STO-3G，2026-08-07）

**与 C₂ 图同构**（同脚本逻辑换体系 N₂/STO-3G 拉伸，全空间 14400）。SHCI 用库内
`solve_hci`——与 pyhci/Dice 原生 SHCI 算法一致（Holmes 2016/Sharma 2017）。
**pyhci 获取失败结论**：换节点后 Windows 侧 GitHub 已通，但 pyhci 仓库不存在
（`gkclpt`/`sharma-lab`/`GKCLAB`/`pyhci` 全 `Repository not found`）；实际可用
接口是 `pyscf/shciscf`（对接 Dice），但需编译 C++（gcc/cmake/boost），WSL
工具链全缺。故用库内 `solve_hci` 作为 pyhci/Dice 等价实现。

**结果**（取点均匀，目标 log 维度 1200→14000 共 11 档；SHCI 5 点
[64, 12544]，improved SQD 6 点 [2500, 14161]）：
- **N₂ 上两条曲线非常接近，SHCI 略优或相当，无 C₂ 那种交叉**：
  dim~2500-3500 时 SHCI 8.4e-7 vs improved SQD 1.8e-5（SHCI 低 20 倍）；
  dim≥6000 两者都 ~5-9e-9（SHCI 略低）。
- 与 C₂ 图互补：**小全空间（N₂ 14400）下两者快速趋同、SHCI 确定性选态略优；
  大全空间强关联（C₂ 44100）下 SHCI 有 PT2 补不足的平台、improved SQD 高维度
  反超**——improved SQD 的价值在"强关联 + 受限子空间"场景更显著。
- 虚线对照：PT2 修正对两者都带来增益（变分层在实线上方）。

生成脚本：`examples/plot_improved_sqd_vs_shci_n2.py`（确定性 seed 0，断点缓存）。

## 图 `fig_improved_sqd_vs_shci_n2_ccpvdz.png`：N₂/cc-pVDZ (10e,10o) @ R=3.0（2026-08-07）

**动机**：N₂/STO-3G 全空间太小（14400），HCI 快速收敛（亚 mHa），参考价值有限；
换更大基组 cc-pVDZ (10e,10o)（全空间 63504，与用户 Desktop `bppp_opt_N2_lowmem.py`
同活性空间）。**R=2.0 与 3.0 探测**：R=2.0 时 SHCI 在 eps≤7e-2 直接 =FCI、
SQD shots≥60 直接 =FCI（弱关联无内容）；**R=3.0（近解离强关联）才复现受限误差**。

**关键探测发现——SHCI 维度对 eps_hb 极敏感（阈值悬崖）**：eps 7.0e-2→dim 3481、
6.9e-2→40401、6.8e-2→57600（0.1e-2 内维度涨 16 倍）；eps 网格须在 7.2e-2..6.5e-2
**加密**才能抓到中间维度。对比 C₂/STO-3G 的 SHCI 维度连续（144→5041→…→41616），
N₂/cc-pVDZ 的 heat-bath 选态有尖锐阈值。

**结果**（目标 log 均匀维度匹配；SHCI 5 点 [324, 63504]，improved SQD **9 点**
[1600, 62500]——eps 加密后 SHCI 维度重复率仍高（heat-bath 阈值悬崖），
有效维度 {144, 324, 900, 7396, 42025, 63504}；SQD 扫 shots×ms 维度连续）：
- **低维度相当**（dim ~3500：improved SQD 1.37e-3 vs SHCI 1.51e-3）；
- **中维度 improved SQD 明显反超 SHCI**（dim 40000：improved SQD **1.49e-6**
  vs SHCI 42025@1.34e-4，**~90 倍**）——**在第二个体系复现并放大 C₂ 的结论**；
- 高维度趋同（~4e-6，都逼近 FCI）。
- 虚线对照：PT2 修正对两方法都有增益（变分层在实线上方）。
- **图修复**：y 轴下限从 1e-5 放宽到 2e-7（原裁剪了 ~2e-6 的高维度趋同点）；
  脚本加 `--plot` 缓存模式（改样式秒出，不重跑收集）。

**参考口径偏移诊断（高维度非零收敛值的来源）**：全空间（63504）时三条线都
收敛到同一 **~1.4 µHa 非零值**，逐层排除定位为**系统性常数偏移**，非方法误差：
1. 全空间对角化（我们库）E=-108.7593747187 vs CASCI e_fci=-108.7593733633，
   差 **-1.355 µHa**（全部落在电子能部分，ecore 相同）；
2. **提高 eigsh 精度（maxiter 2000→20000 + tol=1e-12）能量完全不变**（排除
   ARPACK 未收敛）；
3. 结论：偏移来自**哈密顿量构造路径**（`h1e_for_cas` + `ao2mo aosym="1"`
   + `absorb_h1e(0.5)`）与 CASCI 参考之间的 µHa 级数值/约定累积差异。
**含义**：三条线在全空间精确解的是**同一个我们库哈密顿量**（互相一致 <µHa），
只是与外部 CASCI 参考恒差 ~1.4 µHa——不影响方法间的相对比较（同受一偏移），
但设定了"达 FCI"表述的 ~µHa 下限。图上已加灰色注释标注。

生成脚本：`examples/plot_improved_sqd_vs_shci_n2_ccpvdz.py`（确定性 seed 0，
断点缓存）。

## 图 `fig_improved_sqd_vs_shci_c2_ccpvdz.png`：C₂/cc-pVDZ (10e,10o)（2026-08-07）

**CASCI 参考跳根发现（重要）**：C₂/cc-pVDZ 近简并，CASCI 与 direct_spin1.kernel
（davidson）收敛到虚高 **9.3 mHa** 的第二根（-75.550874）；库内 eigsh(SA) 找到
**真基态 -75.560163**（eigsh(k=2) 明确两根差 9.289 mHa）。因此本图参考改用**库内
全空间对角化（真基态）**，规避参考陷阱。

**三方对账（Dice 交叉验证）**：真基态 -75.5601627525 下，Dice SHCI (eps=1e-4)
偏差 +0.003 mHa、solve_hci 偏差 0.000 mHa、全空间对角化一致——**三方一致到 µHa
级**，同时确认 CASCI 确实跳根（Dice 和库都到真基态而 CASCI 没有）。这也是
Dice/pyscf 官方 SHCI 环境（本次配置完成）与库内实现一致性的第二个实证。

**多 seed 取平均（按用户建议）**：improved SQD 每点 3 seed 取 mean±std 画误差带
（消除单 seed 涨落）；SHCI 确定性单次。

**结果**（SHCI 6 点 [4, 63504]，improved SQD 8 点 [1600, 51076] mean±std）：
- **存在双交叉**：约 **2.0–3.0×10⁴ 区间 improved SQD 优于 SHCI**（dim 20449：
  SQD 1.11e-7 vs SHCI 21316: 1.32e-7；dim 24336：SQD 7.25e-8），~3×10⁴ 后 SHCI
  反超（dim 29584：SHCI 5.1e-8 vs SQD 28900: 5.3e-8）；其余维度 SHCI 占优
  （高维度 dim 37249：2.0e-8 vs 3.8e-8）。与 C₂/STO-3G 相比反超区间更窄、更靠
  中维度——**方法优劣体系依赖**，多 seed 平均让结论更可靠。
- SHCI 全空间误差 0.0（真基态参考下收敛到参考）。
- improved SQD 低采样（shots=3, dim 1600）误差 0.19±0.27 Ha 异常不稳定（采样
  太少恢复失败），高采样收敛（dim 51076 时 2.0e-8±9.6e-9）。

生成脚本：`examples/plot_improved_sqd_vs_shci_c2_ccpvdz.py`（3 seed mean±std，
断点缓存）。

## 图 `fig_improved_sqd_vs_shci_n2_1212.png`：N₂/cc-pVDZ (12e,12o)（2026-08-08）

**能力边界验证（重要）**：全空间 **853,776 维**，参考 = 库内全空间对角化（**仅 3.1 分钟**，
真基态 -108.7686857）。**推翻了此前"12 轨道可能爆内存"的担忧**——受限
`max_strings` 下 PT2 枚举完全可控（单点 23-56s、内存 0.6GB）。库内 SQD 在
12 轨道活性空间可达受限 9000 维到无限制 51 万维。

**数据**（SHCI 单次 + SQD 3 seed mean±std）：
- SHCI：dim 144→831,744，err 3.65e-3→**1.82e-10**（eps 5e-1→1e-3，维度跳跃大）
- SQD：dim 9,025→509,796，err 1.15e-3→**6.02e-7**（单调下降）
- 整体趋势：SHCI 在类似维度下误差更低（需更小维度达同精度）。

**耗时 ~5.7 小时**（后台），主要瓶颈 = 无限制大维度 SQD 点（shots=200/500 各
~1 小时，12 轨道无限制 active 扩张到大维度对角化 + PT2 枚举）。

生成脚本：`examples/plot_improved_sqd_vs_shci_n2_1212.py`（3 seed，断点缓存；
积分/参考存 `_n2_1212_ints.npz`）。

**Dice 交叉验证（N₂ 12,12，2026-08-08）**：
- Dice SHCI 收敛到 **E=-108.768185**（eps→1e-6 后 0.502 mHa 平台不再下降），与库参考
  -108.768686 差 ~0.5 mHa。
- **三方诊断**：库 eigsh 全空间 -108.768686（183s）、pyscf direct_spin1 -108.768584
  （差 0.102 mHa）、Dice -108.768185（差 ~0.5 mHa）——**85 万维 FCI 的收敛/参考口径
  差异 ~0.1-0.5 mHa**，非单一真值可比。
- **无法画 Dice 完整曲线**：shciscf 无变分空间维度 API（mc 仅 nroots/printbestdeterminants，
  输出文件维度格式不可靠）→ 图上以蓝色文字标注 Dice 收敛平台。
- 对比小规模（C₂/cc-pVDZ 10o）：Dice 一致到 +0.003 mHa——**大规模 FCI 参考口径
  差异随体系规模放大**（此前 N₂ 10o 仅 1.4 µHa）。

## µHa 哈密顿量构造偏移诊断（2026-08-08，⑥ 定位完成）

**结论先行**: 基准图里的 ~µHa "common floor" **不是库 bug**, 是 benchmark 脚本
把 `cas.kernel()` (CASCI 内部 eri 路径) 当参考、而各方法跑在脚本自构的 eri
(`ao2mo.full(aosym="1")`) 上——**参考与方法用不同哈密顿量**。库自身两条 eri
构造路径 (`from_pyscf` einsum 与脚本 ao2mo) 完全等价 (机器精度)。

**三条独立 eri 路径在同一 N₂/cc-pVDZ (10e,10o) @ R=3.0 上跑全空间 FCI** (63504 维):

| 量 | 值 (Ha, total) |
|---|---|
| `ecore` (`h1e_for_cas`) | −85.0562455554 |
| CASCI `cas.kernel()` [内部 eri] | −108.7593738839 |
| `direct_spin1(eri=ao2mo.full aosym="1")` [脚本/方法用的 eri] | −108.7593747186 |
| `direct_spin1(eri=einsum(AO,mo_act))` [from_pyscf 风格] | −108.7593747186 |

关键差值:
- **\|CASCI − library\| = 8.35e-07** (≈0.8 µHa, 复现 REVIEW 报的 ~1.4 µHa 量级;
  精确值受 CASCI Davidson 默认 conv_tol 影响, 同量级)。
- **\|ao2mo eri − einsum eri\|_max = 2.22e-16**, 对应能量差 **0.00** —— 库的两条
  eri 路径**逐元素机器精度一致**, 库完全自洽。
- 故 µHa 偏移**全部**来自 CASCI 内部 eri 这第三条收缩路径与库 eri 在 ~1e-15/元素
  上的差异, 经 63504 维 CI 展开累积到 µHa。提高 eigsh 精度不变 (REVIEW 已验),
  因为本就不是对角化器收敛问题。

**库核心 (`molecule.from_pyscf`) 干净**: 直接 einsum, 无 ao2mo/aosym, 与脚本
`ao2mo.full("1")` 逐元素一致。`compute_ground_state_energy(method="fci")` 走
`direct_spin1.kernel(h1e, eri)`, 故"库 FCI on 库积分"自洽零偏移。

**修法 (benchmark 方法论)**: 参考应改为"库自身全空间 FCI 在方法所用 (h1e, eri) 上"
(`direct_spin1(h1e, eri_script)`), 而非 `cas.kernel()`。这样参考与方法同哈密顿量,
µHa floor 归零, 图中"误差 vs FCI"是干净的变分量。CASCI 降为**独立交叉校验**
(与库一致到 µHa, 印证 eri 路径差异的量级)。已落实于
`examples/plot_improved_sqd_vs_shci_n2_ccpvdz.py` (e_fci 改库自洽 FCI; CASCI 作
cross-check 打印)。注意: 已缓存的 `_plot_data_*.npy` 仍用旧 CASCI 参考, 再生成图
需删缓存重跑 (耗时); 现有图在 µHa floor 内仍有效。

诊断脚本: `_diag_muha.py` (临时, 已删)。

## 方向③-A：(E_V, E_PT2) 两点外推 + 轨迹外推脆弱性发现（2026-08-08）

**交付**: `diagnostics.extrapolate_ev_pt2(energies, e_pt2, degree=1)` —— SHCI 标准
(Holmes 2016 / Sharma 2017) 的 ``E_V`` vs ``E_PT2`` 线性外推工具; ``solve_sqd_ev``
新增 ``correction="evpt2"`` 模式。与 ``correction="ev"`` (σ² 线性) 的区别: x 轴用
带能量分母加权的 ``E_PT2`` (物理上更接近漏掉的关联能), 可正可负 (基态通常 <0)。

**关键实证发现 (轨迹外推脆弱性)**: 直接用 ``solve_sqd_active`` 的 within-run
``trajectory`` 做外推**不可靠** —— 轨迹常**退化**:
- **受限时** (小 ``max_strings``): round 间子空间不扩展, ``E_PT2`` 逐轮重复
  (LiH max_strings=8: 三轮 E_PT2 全 = −5.53e-4), 线性拟合病态, ``alpha`` 爆炸
  (实测 7130), 外推到 ``E_PT2→0`` 给出垃圾值 (err +3.9 Ha)。
- **饱和时** (大 ``max_strings``): ``E_PT2`` 一两轮即归零, 外推退化为饱和 ``E_V`` (= FCI)。
- ``evpt2`` 与 ``ev`` 在退化轨迹上**给出相同垃圾值** (σ² 与 E_PT2 仿射相关 → 同拟合)。

**护栏 (落地)**: ``evpt2`` 模式检测轨迹 ``E_PT2`` 互异点数, **<2 则自动退化为 ``pt2``
单点修正** (``details["fallback"]=="pt2"``), 故 evpt2 **永不劣于 pt2**。退化为 pt2 时
``e_evpt2 == e_pt2`` (测试锁定)。

**结论**:
- ``pt2`` (``E + E_PT2`` 单点 Epstein-Nesbet) 仍是**稳健推荐**默认 (N₂ 4.3e-4→6.2e-5,
  C₂→5.0e-4, 行为良好)。
- ``evpt2`` 仅在**非退化轨迹** (多个互异 ``E_PT2`` 点) 时给出真正的两点外推; 否则安全
  退化为 pt2。
- **稳健两点外推的正道**: 用**两次不同 ``max_strings``** 跑 ``solve_sqd_active`` 得两个
  well-separated ``(E_V, E_PT2)``, 喂 ``extrapolate_ev_pt2`` —— 而非 within-run 轨迹
  (其 round 间不独立)。这是 SHCI 社区的标准做法, 也是本工具的设计用途。

**未做 (方向③-B, 效率件)**: 半随机 PT2。当前 ``E_PT2`` 确定性枚举主导 det 的 S+D 连接,
大体系 (12,12 = 85 万维, 基准 5.7 h) 下枚举成本高。半随机 (近场确定性 + 远场抽样估
``E_PT2``) 是 SHCI 的效率杠杆, 降估计方差与枚举成本。属效率优化 (非精度), 留作大体系
路线 (Part 2 C1/C2) 的配套。

## 方向②：solve_sqd_distill 自蒸馏重采样闭环（2026-08-08）

**交付**: `cipsi.solve_sqd_distill(...)` + `solve_sqd_active` 新增 `state_out` 出参 (取出
最终本征矢供重采样)。库 TODO "把本征矢重要性采样做成 solve_sqd_distill 蒸馏闭环" 落地。

**算法**: EM 式量子-经典反馈。每轮 ① `solve_sqd_active` 对角化 → ② 取本征矢
``|Ψ⟩=Σ c_i|i⟩`` → ③ 按 ``p_i ∝ |c_i|^(2/T)`` 重要性重采 n_samples det
(:func:`eigenvector_importance_sample`, F1 已修 α/β 布局) → ④ 喂回 active。温度退火
``temperature_schedule`` (高→低): 高温探索、低温锐化。``keep_pool=True`` (默认) 保留
原始电路采样 + 蒸馏聚焦; 返回各轮 min 能量。

**实证 (LiH/STO-3G, max_strings=15, 4 轮蒸馏 vs 单次 active)**:

| n_shots | seed | single err | distill err | 改善 |
|---|---|---|---|---|
| 10 | 0 | +3.6e-15 (=FCI) | +3.6e-15 | 0 (已 FCI) |
| 10 | 1 | +3.6e-15 (=FCI) | +3.6e-15 | 0 (已 FCI) |
| 10 | 2 | **+4.5e-08** | **+3.6e-15 (=FCI)** | **+4.5e-08** |
| 20 | 2 | **+4.5e-08** | **+3.6e-15 (=FCI)** | **+4.5e-08** |

**关键结论**: 多数 seed 下 active 已饱和到 FCI (无事可补); 但**坏初始采样** (seed=2,
single 卡在 +4.5e-8) 经蒸馏**恢复到 FCI**。这正是 C₂ 式"3/8 覆盖失败"的解药 —— 蒸馏
从解态重导采样分布, 把单次 active 错过的高权重 det 找回来。distill **不劣于** single
(round 0 即单次, best_E 取 min), 且在覆盖失败时**显著修复**。

**边界 (2026-08-10 N₂(12,12) 实测修正 — 推翻原预期)**: 原“收益在更大/强相关体系覆盖失败”
的预期**被证伪**。N₂/cc-pVDZ(12o) R=3.0 (远未收敛, baseline err 2.28e-4, 落后 SHCI ~20×)
上 distill 4 轮**有害**: best=baseline(round 0), r2-4 单调恶化 (2.79e-4→3.01e-4→3.10e-4),
pool 翻倍 (80→160) 后 dim 反缩 (112896→75076)。根因: 重采按 ``|c|^(2/T)`` **聚焦主导 det**,
但 12,12 baseline ``|Ψ⟩`` 本身差 (远未收敛) → 从差 ``|Ψ⟩`` 学错误分布 → 丢覆盖广度。
**修正适用条件: distill 需首轮 ``|Ψ⟩`` 足够可靠** (近收敛; LiH seed=2 卡 4.5e-8 是 ``|Ψ⟩``
近对但漏采的甜点); **远未收敛 (``|Ψ⟩`` 差) 时 distill 学错、有害**。N₂/cc-pVDZ(10o) 近收敛
(baseline 9.76e-7) distill 仅边际 18% 且不稳定 (r2 变差), 印证“近收敛覆盖已足、边际小”。
即 distill 是“可靠 ``|Ψ⟩`` + 覆盖缺口”的精修, **非**“覆盖失败解药”。NQS 衔接 (Part 2 B1)
同理依赖好 ``|Ψ⟩`` 先验。

## L1 改进方法跨体系适应度（2026-08-10 实测）

L1 阶段在两个 benchmark 体系实测 distill / evpt2 改进 (单次 active+PT2 为 baseline):

| 体系 | 全空间 | baseline errPT2 | distill | evpt2 | 收敛状态 |
|---|---|---|---|---|---|
| N₂/cc-pVDZ(10o) R=3.0 | 63504 | 9.76e-7 (dim~47k) | 8.00e-7 (18%, 不稳) | **3.18e-8 (30×)** | 近收敛 |
| N₂/cc-pVDZ(12o) R=3.0 | 853776 | 2.28e-4 (dim~1e5) | **2.28e-4 (有害)** | 1.36e-4 (1.7×, 3 点) | 远未收敛 |

(STO-3G 体系 plot 历史: N₂ 拉伸 / C₂ 在 dim≥6k / 30k 已 ~5-9e-9 / 1.9e-9 近 FCI, evpt2 /
distill 边际; 未单独跑 L1。)

**三条结论**:

1. **evpt2 = 近收敛精修**: baseline 越近收敛 (PT2→0) 外推越准 — 10o 近收敛 30×, 12o 远
   未收敛仅 1.7× (3 点 r²=0.9995; 2 点 2.2× 过乐观)。适用近收敛体系, 远未收敛边际。
2. **distill = 依赖 ``|Ψ⟩`` 质量** (见上修正): 近收敛边际+不稳, 远未收敛有害。
3. **PT2 修正 = 普适基线** (所有体系行为良好), improved SQD 标配层。

**12,12 根本问题是采样覆盖** (baseline 2.28e-4 远未收敛, 落后 SHCI 20×), evpt2 / distill
精修无法弥补 (最好 1.36e-4 仍落后 SHCI 12×) → 需 L2 从**采样端** (UCJ/Krylov 电路采样
覆盖离域波函数) 和**基端** (NO 自洽换基让波函数紧凑) 解决。

## L2 改进实验（2026-08-10）：adaptive 换基 + UCJ 采样（两部分均失败）

针对 12,12 (远未收敛, 覆盖根因) 尝试基端 (adaptive NO 换基) 与采样端 (UCJ/Krylov) 改进,
**两部分都在 n2_ccpvdz + 12,12 失败**:

**L2-a `solve_sqd_adaptive` (NO 自洽换基 + active, max_rounds=4)**:
| 体系 | baseline errDirect | adaptive errVar | 差距 |
|---|---|---|---|
| n2_ccpvdz | 2.76e-5 (dim 47961) | 1.26e-3 (dim 13–15k) | 差 45× |
| 12,12 | 7.31e-4 (dim 112896) | 4.97e-3 (dim 25–31k) | 差 7× |
根因: adaptive 换基+选态 (dom_thresh/pt2_floor) 在 max_rounds=4 产生**远小于 active 的子空间**
(13k/27k vs 43k/113k) → 变分能量高。**修正 memory ④ / REVIEW 方向④**: adaptive "修复后相当或更优"
仅 LiH 小体系; 大体系默认参数下显著差于 active (需调 dom_thresh/pt2_floor 或大增 max_rounds, 未做)。

**L2-b UCJ/Krylov 采样 (冻核+冻虚外活性 CCSD t2 → ucj_decomposition → 多 scale (3,5,10,20)
+ 随机旋转电路采样 → active+PT2)**:
| 体系 | baseline random errPT2 | UCJ errPT2 | CCSD |
|---|---|---|---|
| n2_ccpvdz | 8.58e-7 (dim 47961) | 1.61e-5 (dim 33489, n_ucj=240) 差 19× | 不收敛 |
| 12,12 | 2.28e-4 (dim 112896) | 2.40e-4 (dim 34225, n_ucj=315) 略差 | 不收敛 |
t2 形状 `(5,5,5,5)`/`(6,6,6,6)` 正确 (frozen=核+虚外活性), 但根因: ① **R=3.0 强关联 RHF-CCSD
不收敛 (converged=False) → t2 不可靠 → UCJ 电路方向错**; ② n_samples=500 偏少 (recover 后仅
240/315 det); ③ UCJ→active 子空间小。**修正 memory 方向 A**: UCJ 在 N₂/STO-3G 拉伸跑通是因
CCSD 收敛 + n_samples=2000 + UCJ **+include(S+D)** 模式; **cc-pVDZ R=3.0 CCSD 不收敛 → UCJ 失效**。
即 UCJ 适用条件含 "CCSD 收敛"。

**L2 结论**: 12,12 的基端 (NO 换基) 与采样端 (UCJ CCSD) 改进都失败。**evpt2 (L1) 是唯一在 12,12
有正收益的方法 (1.7×), 但仍远落后 SHCI**。12,12 强关联大空间的覆盖问题对当前改进方法 (distill/
adaptive/UCJ) 顽固 — 暗示需更大采样量、不同 Ansatz、或承认 SQD 在该 regime 不如 SHCI 的确定性选态。

## solve_sqd_best / solve_sqd_improved 整合（2026-08-10）

L1/L2 实测后把最优配置固化成库入口 (`integrated.py`):

- **`solve_sqd_improved(h1e,eri,norb,nelec,...)`**: improved SQD 显式入口 = active + PT2
  (`solve_sqd_ev correction="pt2"` 薄封装, 强制 PT2)。原 improved SQD 散在 correction 选项、
  无独立函数 → 本函数整合 (用户观察「improved SQD 没整合好」的修复)。
- **`solve_sqd_best(...)`**: 当前最优 = active + PT2 + **evpt2 多 shots 外推** (`evpt2_scales`
  个不同 shots 各跑 active, `extrapolate_ev_pt2` 外推 E_PT2→0; 互异点<2 退化 = PT2, 永不劣于 PT2)。
  N₂/cc-pVDZ 10o 实测改进 30×。不用 distill/adaptive/UCJ (L1/L2 实测无增益/有害)。
- **`solve_sqd_auto`** 加 `correction` 选项 (`pt2`/`evpt2`/`none` + `correction_used` 字段),
  替代旧 `extrapolate_ev` 布尔 (向后兼容映射)。

**修正层级**: 变分 (active) → +PT2 (improved, 普适) → +evpt2 外推 (best, 近收敛精修)。
**测试**: `tests/test_sqd_active.py` 加 `test_solve_sqd_best_runs` + `test_solve_sqd_auto_correction_option`
(13 passed); `test_sqd_active_trajectory_monotone` 容差 1e-12→1e-11 (davidson 末轮 ~1e-13 数值噪声)。
入口总览见 `docs/solve_sqd_api.md` (补全 10 个 solve_sqd_* 层级表)。

## 方向④：solve_sqd_adaptive 末轮混合基对角化 bug 修复（2026-08-08）

**根因定位** (与方向② "组合版略逊 active" 的 REVIEW 旧结论相关): solve_sqd_adaptive
每轮 ④ 把 ``sub`` 重建到新基 ``B_{last+1}`` (NO 旋转后), 但 ``str_a/str_b`` 仍是
``B_last`` 的 (① 在换基前 recover)。循环后的 ``sub.diag(str_a, str_b)`` 因此是**混合基
对角化** (``B_{last+1}`` 的 H 作用在 ``B_last`` 的 det 上) —— 非合法子空间对角化,
返回值不自洽。与 :func:`solve_sqd_natural_orbitals` 的 F2 同类 (energy/积分差一轮)。

**实证 (LiH/STO-3G 未饱和, 4 轮)**: 旧版返回 ``−8.8745313921``, 而 ``min(各轮 ③ 自洽能量)
= −8.8745316494`` —— 旧版**比真正最优轮差 2.57e-7** (报高, 即更差)。这正是 REVIEW
方向② "adaptive 略逊 active (mean 1.9e-6 vs 4e-7)" 的根因: 不是方法本质不行, 是返回了
**次优**的混合基能量。

**修复**: 跟踪 ``best_E = min(各轮 ③ E_r)`` (每轮 ③ 自洽: ``B_r`` sub + ``B_r`` dets),
循环后**不再** ``sub.diag``, 直接返回 ``best_E``。新增 ``rounds_out`` 出参 (各轮 ③ 能量,
诊断/测试用)。契约测试 ``test_solve_sqd_adaptive_returns_min_per_round_energy`` 锁定
``返回值 == min(rounds_out)``, revert 验证牙齿 (buggy 版差 2.57e-7 → 测试失败)。

**含义**: adaptive 现为**正确的 OO-CI (轨道优化) + active 采样**: 每轮 NO 换基使波函数
更稀疏 + active PT2 选态, 返回真正最优轮 (变分上界)。旧 REVIEW "adaptive 略逊 active,
保留为框架不声称优于" 的措辞**已过时** —— 那是 bug 产物; 修复后 adaptive 应与 active
相当或更优 (多了 NO 换基的稀疏化收益)。完整 N₂ 拉伸对照验证留作慢测试。

## 方向①-A：solve_sci_csf 自旋适配对角化（S² 投影, 2026-08-08）

**交付**: `fermion.solve_sci_csf(ci_strings, h1e, eri, norb, nelec, *, spin_sq, spin_tol=1e-3)`
—— 在 det 子空间内构建 **S² 矩阵** (第 i 列 = `spin_op.contract_ss(basis_i)`), eigh 后取
S²≈`spin_sq` 的本征空间 P, 把 H 投影到该自旋空间再对角化。与"在 CSF 基对角化"等价
(子空间完备时即精确自旋适配), 复用现有 det/contract 基建, 无需 det→CSF 展开表 (路线 B)。

**算法**: ① S² 矩阵 (O(dim) 次 contract_ss) → ② eigh(S²) 取目标自旋本征空间 →
③ H_proj = PᵀHP 对角化 → ④ 基态回 det 基。成本: 两次 O(dim³) eigh, 适合 dim ≲ 2000。

**验证** (H₂/CH/LiH, 均 H₂ 尺度快速):
- H₂ 全空间: singlet (S²=0) = FCI (1e-8), S² 精确 0; triplet (S²=2) 目标亦可达且高于 singlet。
- **CH/STO-3G (4,3)**: M_S=1/2 sector 同时含 doublet (S=1/2) 与 quartet (S=3/2);
  STO-3G 最小基下 **quartet 比 doublet 低 12.8 mHa** (直接 FCI 给 quartet)。CSF 投影
  **精确选择任一自旋**: quartet (S²=3.75) 能量 = FCI (1e-8), doublet (S²=0.75) 是其上的
  自旋纯上界——这是自旋适配价值的直接展示 (det 基 plain 对角化无法做到)。
- LiH 随机恢复的部分 (自旋混合) 子空间: CSF 投影后 S² 精确 = 0 (消污染); 不可达自旋
  (S²=12, 4 电子最大 S=2) raise。

**附带修复 (防 segfault)**: 字符串电子数与 nelec 不符时, pyscf `contract_ss`/`contract_2e`
的 C 层会越界读 → **core dump** (实测: 3 电子字符串 + nelec=(4,3))。`solve_sci_csf` 加
输入一致性校验 (popcount vs nelec), 不符 → ValueError。既有 `solve_sci` 同路径, 风险相同,
未加 (保持范围, 留作后续)。

**后续 (路线 B, 未做)**: 完整 CSF 基 (det→CSF 展开表, pyscf CSF 机器), 维度减半、
准简并根治更彻底; 与 `solve_sqd_active` 的 spin-targeted 采样闭环结合。路线 A 已覆盖
"自旋纯"核心收益, 路线 B 是效率/彻底性增强。

## 方向①：CSQD 聚类恢复落地（2026-08-10，commit f75d817）

对 2025-2026 SQD 文献调研后，把 Cluster-Adaptive SQD（arXiv:2603.09346）实现进库。

**实现**（`configuration_recovery.py`）：
- `_weighted_kmodes`：手写 weighted k-modes（无 sklearn 依赖，纯 numpy；
  按权重随机初始化 → 汉明距离分配 → 加权众数更新 → 空簇重播种）
- `_cluster_reference_occupancies`：每簇加权平均占据，按粒子数归一
- `recover_configurations_clustered(bsm, probs, na, nb, n_clusters=4)`：
  α/β 半串池各分 K 簇，每样本按其所属簇参考做 `_recover_single`（复用），
  合并去重与 `recover_configurations` 完全一致

**集成**（`fermion.py` / `integrated.py`）：
- `diagonalize_fermionic_hamiltonian` / `solve_sqd` 加 `recovery="global"|"clustered"`
  + `n_clusters` 参数（single + iterative 双分支）
- `solve_sqd` 迭代循环 occupancy 更新加 NaN/越界防护（退化子空间偶发）

**实测（N₂/cc-pVDZ (10e,10o) @ R=3.0，强关联基准）**：

| 模式 | 测试点 | 结论 |
|---|---|---|
| single + 聚类 | 12/12 点 | 误差降 **1.3-2.8×**；0% 噪声时全局恢复坍缩到单行列式而聚类不坍缩（dim 46-144 vs 1）|
| iterative + 聚类 | 多点半数以上 | **不占优**（occupancy refinement 单模式收敛抵消聚类多模式保留）|

**架构性结论**：CSQD 的价值在"保留多占据模式"，与迭代 SQD 的"occ 单模式精化"
存在语义张力。**推荐 clustered 配合 `mode="single"`**（已写入两处 docstring）。
这是"恢复侧"的精度改进；后续可探索"聚类参考 + carryover 迭代"的组合（未做）。

## 方向①-A 强化：自旋 λ 惩罚法（2026-08-10，commit 542ed52）

`solve_sci_csf` 加 `method="penalty"`：H_pen = H + λ(S²-spin_sq)²，直接对角化
整个 H_pen（不投影、不 raise），把解连续压向目标自旋。动机：投影法对自旋混合
子空间（无目标自旋本征空间）会 raise，惩罚法提供更稳健的替代（CSQD 论文用法）。

**实测（CH/STO-3G M_S=1/2 sector，混 doublet S²=0.75 / quartet S²=3.75）**：

| λ | S² | E |
|---|---|---|
| 0.00 | 3.750（quartet 基态）| -40.9775 |
| 0.10 | 3.600 | -40.9638 |
| 1.00 | **0.753**（≈doublet 目标）| -40.9573 |

λ 增大连续把解从 quartet 压向 doublet，不 raise。全空间 H₂ 上惩罚法与投影法
一致（=FCI，S²=0）。4 个新测试 + 回归全过。

## OBMP2 + OBDF：完整自洽实现落地（2026-08-10，`tc_sqd.obmp2`）

**背景**：此前"OBDF 实现验证后删除"节的简化公式（v_oo/v_vv）**已被证伪**——
v_oo einsum 切片维度不匹配直接报错、修正后不对称且迹 2.5× 偏离 MP2 相关能，
不是论文的 OBMP2。2026-08-10 从 **Tran 2021（arXiv:2107.11260）** 提取正确
理论并完整实现（见下节对比），落成 `tc_sqd.obmp2` 模块。

**正确理论**（Tran 2021 Eq 3-10）：

    Ĥ_OBMP2 = Ĥ_HF + [Ĥ, Â]_1 + ½[[F̂, Â], Â]_1,   Â = Â_D = ½·T·(â_ij^ab − â_ab^ij)
    V̂_1stBCH = T̄_ij^ab [ f_a^i Ω̂(â_j^b) + g_ab^ip Ω̂(â_j^p) − g_ij^aq Ω̂(â_q^b) ]
    (T̄_ij^ab = T_ij^ab − T_ji^ab; Ω̂(â_q^p) = â_q^p + â_p^q; f_ai = 0 at HF 故首项消失)
    V̂_2ndBCH = 9 项 T·T̄·f 收缩 (见 obmp2.py 函数体注释)
    C'_1stBCH = −2 T̄_ij^ab g_ab^ij

**归一化关键**（破解 10× 符号问题的核心）：`A_D = ½T` → 收缩含 `½` (1st BCH)
与 `½·(½)² = 1/8` (2nd BCH)。施加 **v_1st × ½、v_2nd × (−1/8)** 后：

- **E_OBMP2(0) = E_HF + 2·Tr_occ(v) + C' 精确 = E_MP2**（N₂/H₂O/STO-3G 全吻合
  ≤ 1e-6）。此前 2nd BCH 给 +1.24（应 −0.155，8× 反号）正是缺此因子。
- 自洽 `solve_obmp2` 收敛（16-39 轮）：N₂/STO-3G E=−107.6499 ≈ CCSD（−107.6502，
  差 0.3 mHa）；H₂O −74.878、LiH −7.874，均介于 E_HF 与 FCI（不过校正）。

**OBDF 下折叠**（`obdf_downfold`）：`H_OBDF = H_CAS + scale·v^ext`，v^ext 从
**外部振幅**（≥1 冻结 core/虚指标）构造投影到活性块，仅改 h1e（量子资源不变）。
配套 `from_pyscf(n_core, n_virtual)` 中间区间活性空间（解决旧节"无法折叠虚轨道"）。

**大基组实证**（scale≈0.1 普适，N₂/H₂O/cc-pVDZ × 6-10o）：

| 体系 (活性) | CAS err | **OBDF err** (scale=0.1) | 改善 |
|---|---|---|---|
| N₂/cc-pVDZ (10o,10e) | +0.231 Ha | **+0.006 Ha** | 38× |
| N₂/cc-pVDZ (8o) | +0.245 | **+0.008** | 29× |
| N₂/cc-pVDZ (6o) | +0.300 | **+0.010** | 30× |
| H₂O/cc-pVDZ (6o) | +0.211 | **+0.012** | 18× |

**⚠ 开放问题**：原始 v^ext（A_D 归一化后）对下折叠约 10× 过大——OBMP2 总能量
靠 trace+C' 相消成立，但**元素量级**与下折叠需求差一常数（scale 参数化默认
0.1，普适于两分子/三活性大小）。10× 来源未完全解析，留作开放问题。
STO-3G 小基组确认过校正（旧节警告正确；大基组才是 OBDF 的验证场景）。

**强关联边界（R=3.0 实测，2026-08-10）**：OBDF 在 N₂/cc-pVDZ R=3.0
（近解离强关联）**全面过校正**，scale 需从 R=1.1 的 0.1 降到 ~0.01（几何依赖）。
与 best SQD / SHCI 同活性空间对比（参考 = 活性全空间 FCI）：

| 活性 | 方法 | E | err vs 活性 FCI | 耗时 |
|---|---|---|---|---|
| 10o | SHCI (eps=1e-2, dim=63504) | -108.7594 | **-0.0002** | 205s |
| 10o | best SQD (shots=100) | -108.7611 | **-0.0019** | 373s |
| 10o | **OBDF scale=0.01** | -108.7735 | **-0.0142** | 17s |
| 10o | OBDF scale=0.1 | -108.9131 | -0.154 | 13s |
| 12o | SHCI (eps=1e-2, dim=592900) | -108.7811 | **-0.0002** | 812s |
| 12o | best SQD (shots=100) | -108.7992 | **-0.0182** | 1512s |
| 12o | best SQD (shots=40) | -109.9625 | **-1.18（PT2 崩坏）** | 956s |
| 12o | **OBDF scale=0.01** | -108.7933 | **-0.0123** | 27s |
| 12o | OBDF scale=0.1 | -108.9116 | -0.131 | 18s |

**结论**：
1. **SHCI 是唯一可靠收敛到活性 FCI 的参考级求解器**（两活性 err -0.0002）。
2. **best SQD 在 12o 强关联区不稳定**（shots=40 PT2/evpt2 崩坏 -1.18 Ha；shots=100
   才 -0.018）——印证 REVIEW「12,12 采样覆盖是根本问题」。
3. **OBDF scale=0.01 在 12o 上略优于 best SQD（-0.012 vs -0.018）且快 ~60×**，但两者
   都在活性 FCI 之下（OBDF 加外部相关、SQD 的 PT2 是空间内近似），无全分子参考
   （R=3.0 CCSD(T) 失效、28 轨道 FCI 不可行）无法判定谁更"正确"。
4. **OBDF 机制在强关联仍能折叠外部相关**（10o OBDF scale=0.01 = -108.7735 落在
   10o-FCI(-108.7592) 与 12o-FCI(-108.7809) 之间，捕获 10o→12o 相关缺口 ~66%），
   但 **scale 校准几何依赖、脆弱**，不构成对 best/SHCI 的替代——与论文
   「强关联下 OBDF 退化为 CAS」一致。OBDF 的价值定位在**弱关联区（R≈平衡）**。

**测试**：`tests/test_obmp2.py` 8 项（势对称、E=E_MP2、SCF 介于 HF/FCI、≈CCSD、
active_range 外部限制、OBDF 结构/差异/校验）；`test_molecule.py` 增 4 项中间区间。
全量回归无破坏。

**参考论文**（完整理论链）：
- T. N. Tran et al., "Quantum resource reduction for quantum-centric supercomputing
  via correlated mean-field downfolding framework" (OBDF-SQD), arXiv:2605.08675 —— 主参考。
- L. N. Tran & T. Yanai, "Correlated one-body potential from second-order
  Møller–Plesset perturbation theory", *J. Chem. Phys.* **138**, 224108 (2013) —— OBMP2 奠基。
- L. N. Tran, "Improving perturbation theory for open-shell molecules via
  self-consistency", *J. Phys. Chem. A* **125**, 9242 (2021), arXiv:2107.11260 —— 本库实现依据。
- N. T. Tran et al., arXiv:2310.18154 (O2BMP2); N. T. Tran & L. N. Tran, *J. Chem.
  Phys.* **162** (2025) —— 背景扩展。

## OBDF one-body downfolding：实现验证后删除（2026-08-10，仅留结论与待办）
> **⚠ 已被上节取代**：本节公式（v_oo/v_vv）经证伪，正确实现见上节
> 「OBMP2 + OBDF：完整自洽实现落地」。以下保留历史记录。

调研 arXiv:2605.08675（OBDF-SQD）后曾实现：`_obmp2_correction`（t2 收缩广义
Fock，v_oo = Σ_ikab t_ik^ab ⟨jk‖ab⟩，v_vv = -½ Σ_ijc t_ij^ac ⟨ij‖bc⟩）+
`from_pyscf(downfolding="obmp2")`（仅改 h1e，eri/ecore 不变）+ 4 个测试。
**⚠ 实现不完整，代码已删除（未保留入库）——以下仅留验证结论与待办，仓库当前无 OBDF 代码。**

**验证结论**：
- ✅ 符号正确（+v 使能量降低，-v 升高）、仅改 h1e、v 对称、量级合理（0.1）
- ❌ **STO-3G 小基组过校正**：MP2 在 10 轨道空间"吃光"相关能还过头（MP2 total
  -107.649 < FCI -107.582），OBDF 把 FCI 推到 -108.14（低 0.56 Ha，非物理）
- ❌ **`from_pyscf` 冻结逻辑限制**：只能冻结"前 n_core 个占据轨道"，无法折叠
  **虚轨道**（OBDF 的 v_vv 块正是虚-虚修正）；大基组（N₂/cc-pVDZ 28 轨道）
  上 `n_active=10` 需冻结 18 轨道 > 7 对电子 → 直接报错

**待办**（其他内容接近完成后再续）：重构 `from_pyscf` 支持活性轨道区间
（如 `(n_core, n_virtual)` 参数），在大基组 + 小活性空间上验证 OBDF 收益；
完整 OBMP2（BCH 展开、外部振幅筛选、自洽轨道优化）是论文核心贡献，简化版
收益未证明。

## GPU matrix-free 落地（2026-08-10，`tc_sqd.matrixfree`）

**背景**：早前"CPU 构建稀疏 H + GPU 对角化"方案被否决（下节保留实测），论文 40×
来自 **matrix-free**。2026-08-10 实现**直接 Slater-Condon σ-vector**（绕开逐列
contract_2e），后端无关（numpy/cupy）。

**实现**（`src/tc_sqd/matrixfree.py`）：
- **σ-vector**：`sigma_vector` 枚举每个 CI 字符串的单/双激发连接，矩阵元解析推导
  （对角闭合式、单激发 Fock 型四块预计算、双激发 eri、αβ 交叉）。修复两个 bug：
  ① 单激发符号（源位被错误计入 crossing → 符号反）；② β 双激发用了 α 式收缩
  （应作用在列索引）。
- **算子预计算**：`prepare_sigma_operators`（T 表 + Fock + diag + me 一次算好）+
  `sigma_vector_ops`（后端无关 matvec，numpy/cupy 复用）。
- **GPU 求解**：`eigsh_gpu`（cupyx LinearOperator + eigsh）→ `solve_sci(backend="gpu")`
  基态分支（dim>1000）。

**验证**：
- σ-vector == 稠密 `build_ci_matrix`：H₂ 1.8e-15 / LiH 1.7e-15 / **N₂ dim=14400 8e-13**。
- cupy matvec 正确（diff 5.7e-14），比 numpy 快 **8×**（25ms vs 205ms，dim=14400）。
- `solve_sci(backend="gpu")` 与 CPU/dense 一致（E diff ≤ 1e-13，S²/RDM 一致）。

**性能评估（大维度实测，2026-08-10）**：einsum 版 cupy matvec **标度差于 pyscf C 核**
——C 核是 O(nnz) 近线性，我的 T 表 einsum 是 O(M·na²·nb) 立方标度。N₂/cc-pVDZ
12o 选定子空间实测（pyscf contract_2e 近恒定 ~18-26ms）：

| dim | cupy | pyscf | cupy/pyscf |
|---|---|---|---|
| 10⁴ | 18.7ms | 25.8ms | **0.72×（cupy 胜）** |
| 4×10⁴ | 125ms | 18.2ms | 6.9× |
| 1.6×10⁵ | 1095ms | 21.6ms | **50.7×** |

**结论**：einsum 批量 matmul 版只在极小维度（~10⁴）靠 GPU 启动/并行小胜，大维度
因 O(M·na²·nb) 标度**远落后于 pyscf C 核**。**架构方向正确**（matrix-free 绕开逐列
构建），但 T 表 einsum 未达 C 核性能；`solve_sci(backend="gpu")` 退回 T 表版（子空间
正确，慢）。**linkstr RawKernel 版**（2026-08-11，下节）真正超越 C 核。

**linkstr RawKernel 版**（`sigma_linkstr_gpu`，2026-08-11）：用 linkstr 算法
（RawKernel atomicAdd scatter/gather + 一次 tensordot）替代 batched matmul，
**全空间真正超越 pyscf C 核** direct_spin1.contract_2e：

| dim | linkstr_gpu | pyscf C 核 | 加速 |
|---|---|---|---|
| 1.44×10⁴（N₂/STO-3G 全空间）| 0.88ms | 2.68ms | **3.0×** |
| 10⁴（12o 子空间）| 1.11ms | 15.08ms | **13.6×** |
| 1.6×10⁵ | 14.4ms | 21.7ms | **1.5×** |
| 4.9×10⁵ | 44.7ms | 48.8ms | **1.09×** |

⚠ **linkstr_gpu 仅全空间正确**：linkstr 算法需单激发中间态为全空间（经子空间外
中间态的双激发贡献），子空间（SQD selected-CI）下丢失部分双激发项（与
selected_ci.contract_2e 差 ~2 Ha）。故 `solve_sci(backend="gpu")` 仍用 T 表版
（子空间正确）；`sigma_linkstr_gpu`/`eigsh_linkstr_gpu` 为**独立全空间快速 API**
（FCI 基准，3-13×）。全空间 mid 对大子空间内存爆炸（t1=norb²·na_full·nb），未做
子空间修正。`build_sparse_hamiltonian` 保留为独立 API。

**测试**：`tests/test_matrixfree.py` 3 项（σ==稠密 / ops==直接 / GPU==CPU 全空间，GPU skip 分支）。

**selected_ci 子空间 4-block GPU 移植（2026-08-11 深挖，部分成果 + 卡点记录）**：

动机：`solve_sci` 用 selected_ci.contract_2e（子空间专用，4-block linkstr），linkstr_gpu
（direct_spin1 语义）子空间不匹配。移植 selected_ci 算法可使 solve_sci(backend="gpu")
子空间加速。深挖结论：

- **算法完全解析**（pyscf/lib/mcscf/select_ci.c）：3 个独立 contraction ——
  - **aaaa_α/aaaa_β**（同自旋双）：`des_des` linkstr，intermediate = nelec-2 双消灭去重
    目标集（**含子空间外**，这是子空间正确的关键），antisym eri tril ⟨ij‖ab⟩×2；
  - **bbaa**（αβ 交叉 + h1e）：`cre_des` linkstr（子空间内单激发），eri×2+h_ps restore(4)。
  子空间正确性来自 dd 的 intermediate 维度（nelec-2 去重目标，含子空间外双消灭态）。
- **numpy 参考实现**（验证代数）：bbaa **完全正确**（H2 err 2.7e-17）；aaaa 有残留 bug
  （Be/N2 diag 近对 diff~0.02、off-diag 错，eri 因子/符号/转置扫描均非根因）。sign/eri
  packing/intermediate 逻辑逐一排查均合理，未定位根因（疑似 des_des linkstr 的 sign 语义
  或 eri1_aaaa 的 (des,cre) 对偶索引微妙处）。
- **direct_spin1 子空间不可用**已确认（AssertionError，仅全空间）。
- **状态**：算法理解完整（可指导后续），bbaa 验证（linkstr+eri packing 框架对），aaaa
  需更多调试。cupy 移植未开始（待 numpy 参考正确）。后续可接手：定位 aaaa sign/索引 →
  cupy 3-contraction（scatter RawKernel + eri matmul + gather）→ 接入 solve_sci(backend="gpu")。
- **价值评估**：即使移植成功，pyscf C 核 selected_ci.contract_2e 子空间已快（dim 14k~3ms），
  GPU 优势主要在 linkstr 全空间（已交付）；子空间 GPU 增益有限、内存风险（t1=norb²·ninter·nb）。

**落地（`tc_sqd.selected_ci_gpu`，2026-08-11）**：深挖完成 —— numpy 参考 + cupy 移植
均验证正确，`solve_sci(backend="gpu")` 改用它（子空间正确 + 大维度加速）。
- **调试关键**：numpy 参考初始 err 大，两个根因 —— ① ββ 的 fcivec 须传 `v.T`
  （_aaaa_np 内部按 β 索引取行）；② cupy `v.T` 是转置视图（非连续内存），kernel 线性
  索引错，须 `np.ascontiguousarray(v.T)`。
- **验证**：vs selected_ci.contract_2e —— H2 2.8e-17、Be 7.1e-15、N2/STO-3G 2.3e-13；
  `solve_sci(backend="gpu")` 子空间能量 == cpu（≤2e-13，新增子空间测试锁定）。
- **性能**（N₂/cc-pVDZ 12o 选定子空间，solve_sci 全流程含 eigsh）：dim 10⁴ GPU 慢
  25×（启动开销）；**dim 9×10⁴ GPU 6× 快**（39s→6.4s）；dim 4.9×10⁵ GPU 2× 快。
- **小结**：GPU matrix-free 三版定位 —— T 表 einsum（子空间正确、慢）、linkstr RawKernel
  （全空间快、子空间错）、**selected_ci 3-contraction RawKernel（子空间正确 + 大维度快）**。
  后两者互补：全空间 FCI 基准用 linkstr，SQD 子空间用 selected_ci_gpu。

## GPU 后端：实现验证后搁置（2026-08-10，历史记录）
> **已被上节取代**：`solve_sci(backend="gpu")` 已以 matrix-free 方式重新落地，
> 见「GPU matrix-free 落地」。本节保留"CPU 构建 + GPU 对角化"被否决的实测。

调研 arXiv:2601.16637（GPU 全驻留 matrix-free 选态对角化，35-39×）后实现：
- 环境：cupy-cuda12x 14.1.1 + RTX 5080（WSL CUDA 13.1 驱动）就绪
- `build_sparse_hamiltonian`（**保留，独立 API**）：稀疏 H 构建（COO→CSR，
  含 ecore），内存友好
- `solve_sci(backend="gpu")`：稀疏 H → cupyx csr → cupyx eigsh（已撤销）

**实测（N₂/cc-pVDZ 10o，dim=10⁴）时间分解**：

| 环节 | 耗时 | 占比 |
|---|---|---|
| 稀疏 H 构建（CPU，O(dim) 次 contract_2e）| 29.4s | 95% |
| GPU 传输 | 0.93s | 3% |
| GPU eigsh（cuSPARSE）| 0.56s | 2% |
| CPU eigsh（隐式 LinearOperator matvec）| **0.42s** | — |

**架构性结论**：GPU eigsh 本身极快且精确（diff ≤ 1e-13），但"显式构建稀疏 H"
（O(dim) 次 Python 层调用）占 95%，CPU 隐式 matvec 路径（仅 ~30 次迭代）快
70×。论文的 40× 来自 **matrix-free**（Slater-Condon matvec 直接 GPU 算，
Thrust 核）；"CPU 构建 + GPU 对角化"的简单方案构建成本主导、无收益。

## 后续可选改进（非阻塞）

按 2026-08-10 调研（SURVEY §8.9）更新，剩余可选方向按性价比排序：

- **自旋分辨哈密顿量**（`h_alpha ≠ h_beta`，UHF 式）——需 spin-orbital SQD 后端
- **UCJ 精确对标 ffsim**（完整 J + 多参数 orbital rotation，非简化 SVD）
- **自洽 NO 迭代扩展**（风险 4/5）——`solve_sqd_natural_orbitals` 已是迭代式，
  剩余仅开壳层支持（spin-resolved 1-RDM）。b62d636 否决的是 adaptive 换基
  （状态选择收缩），NO 自洽不做收缩，与失败路径不同
- **ARNN 采样器（AB-SND 路线，arXiv:2508.12724）**（风险 4/5）——Transformer
  ARNN 生成位串替代量子电路采样。论文只测自旋模型未测分子化学；引入 PyTorch
  重依赖。观察归档，除非有量子硬件需求
- **Krylov 广义本征工具**（难度 5/5）——`solve_krylov`（重叠矩阵 S 广义本征 +
  小本征值正则化）。SKQD 作者自评化学近期不可用；greenfield
- ~~**OBDF 下折叠续**~~（**已落地 2026-08-10**）——`from_pyscf(n_core, n_virtual)`
  + `tc_sqd.obmp2`（`solve_obmp2`/`obdf_downfold`），大基组实测见「OBMP2 + OBDF：
  完整自洽实现落地」。剩余：v^ext 的 10× 归一化开放问题 + 完整自洽 OBMP2 的
  2nd-BCH 精细核对
- ~~**GPU matrix-free 重写**~~（**已落地 2026-08-10**）——`tc_sqd.matrixfree` +
  `solve_sci(backend="gpu")`，见「GPU matrix-free 落地」。剩余：大维度性能
  （~300-500 行 RawKernel/RIKEN `sbd` 是进一步优化点）+ T 表内存扩展
- **qDRIFT 随机化**——降 Krylov/演化电路深度（中低优先）
- **>53 轨道支持**——核对 `bitstring_matrix_to_ci_strs` 64-bit 上限
- 配置恢复 tie-breaking 随机性的统计性测试
- 多版本 numpy（1.x / 2.x）CI 矩阵，固化兼容性
- **方向 D 强化**：本征矢重要性采样与 NQS 结合做泛化先验（已落地 distill 闭环，
  见 §"solve_sqd_distill"）
- **方向 D 拓展**：semistochastic PT2 抽样降 E_PT2 估计误差；注意 **σ² 线性
  外推实测过冲（N₂/C₂ 均落 FCI 之下），不作默认**
- **方向 E 强化**：`recommend_sqd_params` 接入真实校准（`calibrate` 拟合的
  KS/KT1 回填）；`noise_impact` 支持 T2/读出噪声类型与多参数安全区扫描

## 方向 C1：尾部发现采样（round_001 部分验证，2026-08-12）

**来源**：SQD-AA（arXiv:2605.02565）振幅抑制的经典模拟版。SQUAD 协作 round_001
全流水线验证（R1 调研→R2 理论→R3 实现→R4 审查→R5 跑分）。

**实现**（commit `6b510b1`，`tc_sqd.tail_sampling`）：
- `suppress_seen_bitstrings`（批内抑制：丢弃恢复后 α∈seen_a ∧ β∈seen_b 的位串）
- `discover_tail_pool`（过抽尾部发现：10× 预算抽新随机位串 → 恢复 → 抑制已见 → 收集新贡献）
- `solve_sqd_active` / `solve_sqd_ev` 加 `tail_suppression` / `tail_max_draw_factor` /
  `tail_n_target_per_round`（默认全关，零行为变化；全库 167 测试通过）
- **distill 边界形式化保证**：API 绝不接受/读取 c2d，只读 (seen_a, seen_b)

**实测**（A/B 对照，同 seed 同 max_strings，差异纯净归因 C1）：

| 体系 | shots | baseline err | C1 err | 比值 | 判定 |
|---|---|---|---|---|---|
| N₂/cc-pVDZ **(12,12)** @100 seed=0 | 100 | 1.958e-4 | **7.979e-5** | 2.45× | 部分 |
| N₂/cc-pVDZ **(12,12)** @100 seed=3 | 100 | 2.493e-4 | 1.362e-4 | 1.83× | 部分（种子依赖）|
| N₂/cc-pVDZ **(10o)** @80 seed=0 | 80 | 9.702e-7 | **1.395e-8**（补满全空间）| **70×** | 通过 |
| P2 诊断（12,12 n_new 末轮）| 100 | 10 / 6 | **41 / 40** | **4.1× / 6.7×** | 机制生效 |

**结论状态：部分（机制成立、未达 3× 目标、种子依赖）**
- **机制端确认**（P2）：C1 每轮新 det 41-80 全程保持 vs baseline 5-10（coupon-collector
  尾部被削），跨 seed 稳健（4.1×/6.7×）。C1 确实改变了采样覆盖。
- **精度改善 ~2.1× 均值**（1.8-2.9×），小于电路模式 SQD-AA 的 3× 目标。
  主因 = **低振幅稀释**（"发现更多 det ≠ 发现更重要的 det"）。
- **10o 补满全空间**（80 shots → err 1.4e-8 = FCI 级），预期外强结果。
- **wall 不倒退**（C1 1009s vs baseline 1094s），内存 <1GB。

**关键实现发现（R5）**：C1 **bootstrap 预算与 shots 解耦** —— tail_suppression=True 时
每轮用 discover_tail_pool 固定预算（10×30=300/轮）替代初始池，初始 bsm 行数（shots）
仅在 discover 返回空池回退时生效。**@500 shots C1 与 @100 shots 逐位相同**：低 shots 端
巨大增益（100 shots → dim 341k），高 shots 端反而落后普通基线（不用多出的 shots）。
这是 bootstrap 模式的结构性局限。

**不按证伪处理**的理由：P2 机制 + P1 不回归 + wall 不倒退 三项独立实证均成立。
"部分"指向改进方向（预算缩放 + 软抑制），不是机制失效。

**后续改进（round_002 候选）**：
1. **预算随 shots 缩放**（最直接杠杆）：discover_tail_pool 总预算与 shots 挂钩
2. **软抑制护栏**（suppression="decay" / prob_floor）：抑制过平引入低振幅 det
3. **电路模式 C1**（需非-CCSD ansatz）：C1 在电路模式才预期 3×+ 纯粹机制

**口径修正**：theory.md 的 `<500 random>` 配方与基线 2.28e-4 不一致（500 shots 实测
5.28e-7 近收敛）；2.28e-4 实际对应 ~100 shots。后续 task 需对齐 shots 数与基线引用。

## 方向 C1-v2：预算随 shots 缩放（round_002 证实，2026-08-12）

**来源**：round_001 R5 发现 C1-v1 "bootstrap 预算与 shots 解耦"局限（@500 = @100 逐位相同）。
round_002 修复命门（cipsi.py:850 的 `n_tgt` 完全不读 shots）。

**实现**（commit `ecc2f65` + `a5e1e39`）：
- `solve_sqd_active` / `solve_sqd_ev` 加 `tail_shots_ref: int = 0`（默认 0 = round_001 C1-v1 行为）
- hook 加 n_tgt 缩放分支：`n_tgt = clip(⌈n_active · n_cur/shots_ref⌉, n_active, 3·n_active)`
  （tail_shots_ref=100 时 @100→30 零回归、@500→90 用上多出 shots；cap=3 物理上限）
- `_n_drawn` 接住（round_001 丢弃的返回值，供诊断 + 后续过抽自适应）
- 全库 173 测试通过（167+6 新）；@100 零回归 L2 锁定

**实测**（A/B/C 三方对照，同 seed 同 max_strings）：

| 条件 | A baseline | B C1-v1 | C C1-v2 | C vs B |
|---|---|---|---|---|
| **12,12 @100** seed=0 | 1.958e-4 | 7.979e-5 | **7.979e-5**（逐位=B）| 1.000×（零回归）|
| **12,12 @500** seed=0 | 5.277e-7 | 7.979e-5（=@100，预算不缩放）| **5.047e-9**（dim 824k ≈98% 全空间）| **15,810×** |

**结论状态：已验证（P0 证实，远超 3× 目标）**
- **P0**：C1-v2 @500 err **5.047e-9**（准 FCI），对 baseline 2.28e-4 改善 **45,170×**；对 C1-v1 @500 好 **15,810×**
- **P1**：@100 与 C1-v1 逐位一致（n_tgt=30 floor）
- **P2**：n_tgt 30→90（缩放生效），n_drawn 比 7.9×（机制确认）
- 10o 不回归：@80 逐位一致；@500 数值一致且快 1.8×

**关键机制**：cap=3 缩放 @500 n_tgt=90 × 10 轮 ≈ 9000 抽 → 把 12,12 子空间推到 908/924
字符串（≈98% 全空间 853,776）→ err 坍缩到准 FCI（5e-9）。**预算缩放把"高 shots 端 C1 退化"
反转成"高 shots 端 C1 准 FCI"**——round_001 的"低振幅稀释"担忧在饱和 regime（≥98% 空间）
下自然消失（空间补全 → 漏掉 det 的振幅总和坍缩）。

**诚实边界**：此强结果部分依赖 cap=3 的物理饱和（12,12 全空间仅 924 字符串/自旋）。
对全空间更大的体系，err 改善未必同样坍缩到 FCI；但"用上多出 shots 的机制"是普适修复
（P2 物证 + @500 行为从退化变改善，与体系大小无关）。大体系验证留 round_005 plot 轮。

**遗留可选项**：cap 敏感性（cap∈{2,3,4}）；过抽自适应 ρ 触发；`n_cur` vs `n_pool`（增量采样时）。

## 方向 GPU-Subspace：_Subspace GPU backend 改造（round_003 部分证伪，2026-08-12）

**来源**：round_003 工程基建轮。`_Subspace.diag`（cipsi.py:103-130）是 5 个 solver 的公共对角化
基建，CPU `contract_2e` + scipy `eigsh`。改造目标：加 GPU backend 加速大子空间对角化。

**实现**（commit `9d8062b`）：
- `_Subspace.__init__` 加 `*, backend="cpu"` + `has_gpu()` 优雅回退
- `diag` 三分支：dim≤1000 始终 CPU numpy eigh（不读 backend）；dim>1000 + GPU 调
  `eigsh_selected_ci_gpu(tol=1e-10)` + try/except 回退；dim>1000 + CPU（默认）scipy eigsh
- 6 solver 透传 backend；全库 179 测试通过（L1 零回归）；P1 正确性 E_diff ≤4.1e-12

**实测**（dim 扫描，N₂/cc-pVDZ 12-MO 窗口）：

| dim | CPU wall | GPU wall | speedup | 判定 |
|---|---|---|---|---|
| 1e4 | 1.7s | 121s（含 85s warm-up）| 0.014× | warm-up 灾难 |
| 5e4 | 84s | 433s | 0.19× | crossover 慢区 |
| **1e5** | 96s | 43s | **2.22×** | 唯一 GPU 受益（<3× 阈值）|
| 5e5 | 137s | 372s | **0.37×** | **大 dim 反而慢 2.7×** |

端到端 12,12 GPU e2e：**GPU 慢于 CPU**（cupyx eigsh 大 dim 卡住）。

**结论状态：部分证伪（正确性通过 + 性能未达 P0 目标）**

**核心矛盾**：`solve_sci(backend="gpu")` 直接调 `eigsh_selected_ci_gpu` 在 dim 5e5 时 **2× 快**
（REVIEW:1334），但 `_Subspace(backend="gpu")` 调同一函数在 dim 5e5 时 **2.7× 慢** ——
同样 GPU 函数、加速比差 ~5×。

**三个根因**：
1. **eri 每次 matvec 重算**：`eigsh_selected_ci_gpu` 内部 `absorb_h1e` + eri packing 每次 diag 重做
   （selected_ci_gpu.py:125/135）；`_Subspace.__init__` 已算 h2e 但 GPU 路径不用它。
   theory §1.4 预判"eri 重算非瓶颈"在大 dim + 多 matvec 下被证伪。
2. **linkstr 双重建**：`_Subspace.diag` 先建 `_all_linkstr_index`（CPU），GPU 分支的
   `eigsh_selected_ci_gpu` 内部又建 4 个 linkstr——大 dim 时 O(norb²·na) 非平凡。
3. **cupyx eigsh 收敛性差于 scipy**：selected-CI 子空间矩阵上 cupyx ARPACK 可能需要更多 matvec。

**改进方向（后续轮）**：
- **方式 B/C 升为必须**（theory 原判 P1/P2 可选）：_Subspace.__init__ 预算 eri1_aaaa/eri1_bbaa
  缓存 + links/kernels 预计算，绕过 eigsh_selected_ci_gpu 包装层重复开销
- 或诊断 cupyx eigsh 的 N_matvec vs scipy（收敛性差异定量）
- 或承认 GPU 仅 solve_sci（单次固定子空间）受益，_Subspace 迭代场景需更深层重构

**对 plot 的影响**：round_005 全量 plot 的 GPU 加速预期下调——当前 _Subspace GPU 端到端
反而慢，plot 应保持 CPU（~10h）或仅 dim>1e5 局部开 GPU。真正的 plot 加速需先解决 eri/linkstr 重复。

**R5 实证补充（2026-08-12，`_probe_eigsh_round003.py` matvec 计数）**：上述「三个根因」中，
**#3（cupyx eigsh 收敛性）被确认为唯一主导项，#1/#2（eri 重算 / linkstr 双重建）为非瓶颈**。
同一子空间下 cupyx `eigsh(which="SA", tol=1e-10)` 的 matvec 次数 = **24,417（dim 5e4）/
6,465（dim 1e5）**，scipy `eigsh` 仅 **741 / 701** → **9-33× 更多迭代**，把 GPU matvec 单次
速度优势（~10-40×）吃光。内存非瓶颈（dim 5e5 t1 峰值 ~4.6 GB << 17 GB）。故「方式 B/C
eri/linkstr 缓存」**不能**解决本问题（开销仅 µs/ms 级）；下轮应优先 **cupyx eigsh 调参
（ncv/maxiter/v0/shift-invert）或加慢回退护栏**（matvec 计数/wall 超时回退 CPU，现只有异常
回退无慢回退）。详见 `docs/rounds/round_003/benchmark.md` §3。

**round_004 R5 实证补充（2026-08-12，方式 B+C 落地后独立进程重测）**：round_004 已实现
「方式 B+C」（eri 缓存 + 内联 cupy LO + 懒构 hop，commit `5fddef4`，默认路径零回归）。R5
独立进程重测（**每点独立进程**消除 round_003 单进程显存碎片 confound）的结论**部分修正
round_003 的「开销仅 µs/ms 级」判断**——B+C 的确定性 per-matvec 收益**实测 ~2.4%**（P0'
缓存/重算 ratio_med 0.977/0.972，方向对但远低于 theory §1.2 的 8-12% 与 1.14× 阈值），
**仍不足以逆转 #3**：

| 项 | round_003（无 B+C，单进程）| round_004（B+C，独立进程）|
|---|---|---|
| dim 1e5 GPU/CPU | 0.45（2.22×，GPU 43s 幸运点）| **1.47（0.68×，GPU 136s）**——幸运点翻转，erratic 实证 |
| dim 5e5 GPU/CPU | 2.71（GPU 372s）| **2.55（GPU 362s，2.6% 改善 ≈ P0' 的 2.4%）** |
| dim 5e4 GPU/CPU | 5.17（GPU 433s stall）| 2.27（GPU 195s，stall 未复发，亦 erratic）|
| cupyx/scipy matvecs | dim 1e5: 6,465/701（9.2×）| **dim 1e5: 7,265/701（10.4×）；dim 5e5: 7,169/811（8.8×）** |

**#3 升级为实证三连**：(a) cupyx matvecs 8.8-10.4× 于 scipy（复现 + 扩至 dim 5e5）；(b) 同
dim 1e5 跨 run N_matvec 差 ~2×（probe 7,265 → 64s vs scan ~15,000 → 136s，cupyx 起始向量
默认随机 → 收敛路径抖动）；(c) 幸运点翻转（2.22×→0.68×）。**B+C 修复了 #1/#2（P0' 证实
~2.4%），但 #3 是主导，round_005 应攻 cupyx 收敛（调参 / GPU-matvec + CPU-scipy-eigsh
混合 / 慢回退护栏）**。P0 dim 5e5 证伪为预期结论（theory §1.5），非 B+C 失败。本轮 4 点 +
2 probe 全收敛无 stall、无 GPU 崩溃。详见 `docs/rounds/round_004/benchmark.md`。

## 方向 hybrid：GPU-matvec + CPU-scipy-eigsh 混合（round_005 突破，2026-08-13）

**来源**：round_003/004 两轮证伪的精确定位——cupyx ARPACK 收敛停滞（matvec 8.8-33× 于
scipy）是 GPU 慢的唯一主导因子（eri 重算仅 2.4%）。round_005 攻 #3：把 `_Subspace.diag`
GPU 分支的本征引擎从 `cupyx.eigsh` 换成 `scipy.eigsh` + GPU matvec（`sigma_selected_ci_gpu`
+ `.get()` 回 CPU）——**GPU 做快 matvec，CPU scipy 做稳收敛**。

**实现**（commit `e3cb9ab`）：`_Subspace` 加 `gpu_eigsh_mode="hybrid"|"cupyx"|"cpu_fallback"`
三模式旋钮（hybrid 为 backend="gpu" 新默认）；cupyx 模式加 maxiter 护栏。全库 187 passed
（+1 xfail +1 xpass）。

**实测**（dim 扫描，每点独立进程）：
- **P0 证实**：dim 1e5 **16.87×**（R3 18.55×）；dim 5e5 **4.86×**（R3 5.13×）
- **P0' 因果锚**：hybrid N_matvec 与 CPU scipy **逐位相等**（701=701 / 811=811）——加速完全
  来自 GPU per-matvec 速度，收敛路径不变
- **crossover 全部移出**：最小 dim 1e4 也 3.27× 快（vs round_003 的 0.014×）
- E_diff ≤1.4e-13（GPU/CPU 代数等价）

**三轮 GPU 探索闭环**：003 诊断根因（cupyx 收敛停滞）→ 004 排除非主因（eri 2.4%）→
005 精准修复（scipy 收敛 + GPU matvec = hybrid，5-18×）。

## round_006：全量 plot + 扫 shots + 端到端加速比（2026-08-13/14）

### max_strings 版 plot（5 体系，C1-v2+best 跨体系验证）

`plot_c1v2_best_vs_shci_*.py`（新建不覆盖旧脚本），4 曲线（C1-v2+best/best/improved/SHCI），
x 轴实际 dim，GPU hybrid，缓存 `plot_cache/round_006/`：

| 体系 | 全空间 | C1-v2+best | best 末点 | SHCI 末点 |
|---|---|---|---|---|
| N₂/STO-3G | 14,400 | 全空间 ~2e-9 | — | — |
| C₂/STO-3G | 44,100 | 全空间 **~1e-13** | 2.0e-9 | 9.3e-11 |
| N₂/cc-pVDZ 10o | 63,504 | 全空间 **~1e-10** | 1.5e-9 | 9.5e-11 |
| C₂/cc-pVDZ 10o | 63,504 | 全空间 **~1.4e-14** | 9.7e-10 | 2.6e-13 |
| **N₂/cc-pVDZ 12,12** | **853,776** | **dim 824k err 2.1e-8** | 1.1e-7@520k | 3.8e-11@830k |

**结论**：C1-v2+best 在 ≤63k 全空间体系全部达准 FCI（err 1e-10~1e-14，500 shots tail 填满
全空间）。12,12 上 SHCI 同 dim 仍优（2.1e-8 vs 3.8e-11），但 C1-v2+best 的价值在量子资源
效率（500 shots 达 97% 全空间）。**问题**：max_strings 对 C1-v2 控制力弱（tail 不受 gate）
→ C1-v2+best 曲线退化为单点。

### 扫 shots 版 plot（5 体系，替代 max_strings 版）

扫 `SHOTS_LIST=[10,30,50,100,300,1000]` + max_strings=None + **自适应 seed**（每 shots 点先
3-seed，max/min err <5× 后单 seed，误差带展示涨落）：

| 体系 | improved err 跨度 | best err 跨度 | C1-v2+best |
|---|---|---|---|
| N₂/STO-3G | ~1 数量级 | 同 | 近全空间饱和 |
| C₂/STO-3G | 1.5 数量级 | 同 | 低 shots 涨落 ratio 8-46× |
| N₂/cc-pVDZ 10o | **6 数量级** | 同 | 低 shots 涨落 ratio **169×** |
| C₂/cc-pVDZ 10o | 2.5-4.5 数量级 | 同 | shots=300 出现 err=0（完美 FCI）|
| **N₂/cc-pVDZ 12,12** | **3.5 数量级**（19k→697k）| **4.5 数量级** | **341k→824k, 1.7e-4→7.4e-9** |

**关键发现**：(1) 扫 shots 让 SQD 三曲线全部有可读分散度；(2) C1-v2+best 低 shots 端极端
涨落（ratio 8-169×，全空间饱和边缘差几个 det 决定 err）——自适应 seed 正确应对；
(3) 12,12 的 C1-v2+best shots 是关键控制变量（低 shots dim 341k err 2e-4 → 高 shots dim
824k err 7.4e-9 准 FCI）。

### GPU hybrid 端到端精确加速比（3 代表性点）

同体系/同参数/同 seed，唯一变量 backend（`bench_round006_speedup_points.py`）：

| 点 | 体系 | dim | CPU | GPU | **加速比** | E_diff |
|---|---|---|---|---|---|---|
| P1 | N₂/STO-3G | ~1.6k | 11.7s | 2.7s | **4.39×** | 4.3e-14 |
| P2 | N₂/cc-pVDZ 10o | ~17k | 10.2s | 2.9s | **3.57×** | 1.8e-14 |
| P3 | N₂/cc-pVDZ 12,12 | ~824k | 1192.5s | 247.7s | **4.81×** | 2.8e-14 |

**端到端加速比稳定 ~3.6-4.8×，无 crossover 慢区**（hybrid 在所有 dim 都有效）。E_diff 机器
精度。12,12 @500 单点 CPU 20min → GPU 4min。修正之前估算（考虑旧版 3-seed 虚高）：真实
端到端加速比 **~4×**。全量 plot wall 实测：max_strings 版 5 体系 ~9h（GPU），CPU 估 ~32h；
12,12 单体系 6.1h（GPU），CPU 估 ~21h。

## 方向剪枝：字符串级 |c|² 边际权重排序剪枝（round_007 部分，2026-08-14）

**来源**：辅助对话方向 B 建议——收窄 12,12 上 C1-v2 vs SHCI 差距（dim 824k err 2.1e-8 vs
SHCI 同 dim 3.8e-11，~550×）。R2 诊断核心洞察：C1 弱点不在变分能量（已准 FCI）而在
**evpt2 外推退化**（低权重字符串稀释使 E_PT2 失去动态范围）。

**实现**（commit `f47958f`）：`solve_sqd_active` 加 `prune_keep: float = 1.0`——末轮 diag 后
final-only，字符串级 |c|² 边际权重排序（闭壳层合并权重保 str_a==str_b），剪后重对角化 +
PT2 自动重算（二阶回补）。ev/best 全链透传。9 新测试 + 全库 182 passed 零回归。

**实测**（12,12 @500 shots C1-v2+best，GPU hybrid，prune_keep sweep）：

| prune_keep | dim | err | wall |
|---|---|---|---|
| 1.0（baseline）| 824k | 6.00e-8 | 1407s |
| 0.8 | 529k | 2.83e-8 | 1511s |
| 0.7 | 404k | 7.02e-7 | 1471s |
| **0.6** | **297k** | **1.61e-8** | 1405s |
| 0.5 | 206k | 6.61e-7 | 1452s |
| 0.4 | 132k | 7.38e-6 | 1422s |

**结论状态：部分**（方向正确、最佳 0.6 改善 3.7×，远低于 20× 目标；wall 不增加）

**关键发现**：
1. **err 非单调**（0.8 改善→0.7 恶化→0.6 最佳→0.5 恶化）——evpt2 外推点少（3 scale×剪枝
   后子空间），随机波动 ~10× 掩盖剪枝平滑信号，单点判定脆弱
2. **12,12 残差主因是缺失高激发 det**（R2 诚实风险声明应验）：0.6 的 1.61e-8 仍比直接变分
   5.05e-9 差 3×——剪枝无法回补 tail 不可达的三激发 det（~16 字符串）
3. **二阶回补机制成立**（R3 锚点：E_V 升 60× 时 |E_PT2| 同步增 60× 回补，E_V+E_PT2 仍准 FCI）
   但净收益有限（3.7×）

**后续启示**：收窄 vs SHCI 差距需补高激发 det（覆盖层：PT2 三激发扩展 / tail 三激发生成），
而非剪枝（质量层）。或接受方法定位差异：C1-v2 = 500 shots 达 97% 全空间 + 1.6e-8（量子
资源效率）；SHCI = 确定性选态大空间精度优势（方法本质差异）。

`prune_keep` 功能保留（默认 1.0 零回归，0.6 为经验甜点但体系依赖）。

## 方向三激发：定向注入 BFS（round_008 部分，2026-08-15）

**来源**：R1 大范围调研 top1（CIPSI-CC(P;Q) 2601.11856：三激发只需 <5%）+ round_007 根因
（缺失 ~16 个高阶字符串）。R2 设计字符串级单激发 BFS（从全部已选字符串生成单激发目标，
激发阶逐层 +1，图连通即补全全空间）。

**实现**（commit `e4c9a4e`）：`_single_excited_strings`（位运算单激发生成器）+
`solve_sqd_active` 末轮独立 pass（BFS → 笛卡尔积 → EN-PT2 打分 → 三重 cap → fixpoint）。
ev/best 透传。10 新测试（LiH 补全到全空间 E=FCI）。全库 0 失败。

**实测**（12,12 @500 C1-v2+best，GPU hybrid）：

| 配置 | err | wall | dim |
|---|---|---|---|
| baseline | 6.00e-8 | 1350s | 824,464（908²）|
| **triples** | **1.11e-8** | **1604s（1.19×）** | **824,464（不变）** |

**结论状态：部分（err 5.4× 改善 + wall 1.19×，但覆盖扩展未实现）**

**关键发现**：
1. **dim 不变**——BFS 未触达缺失的 16 字符串（单激发连通性在 12,12 不完全成立：缺失串与
   已选串的单激发距离 >1，中间节点可能被 cap 过滤断链）。LiH（小体系）上图连通补全全空间成立。
2. **err 5.4× 改善来自 evpt2 外推变化**（注入改变 E_PT2 值 → 外推更准），非覆盖扩展
3. **有效帕累托改进**（wall 1.19× 满足 ≤2× 约束），与 prune（round_007）正交可叠加
4. 残差 ~1.1e-8 仍差 SHCI（3.8e-11）~300×——**采样式 vs 确定性选态的方法本质差异**

**八轮方法线终局**（12,12）：C1-v2 直接变分 5.05e-9（round_002 单点最优）；
C1-v2+best+注入+剪枝叠加大约 ~1e-8~5e-9。与 SHCI 差距是采样路径覆盖 97% 后剩余 3% 的
指数难度——除非有新的覆盖机制（多步 BFS 无 cap / 确定性预置全空间），采样式 SQD 在
12,12 的精度上限 ~1e-9 量级。

## semistochastic PT2：P0a 前提证伪（round_009 快速失败收档，2026-08-15）

**R1 前提**（round_008 调研）："PT2 全枚举是 12,12 墙时主因，semistochastic 后 wall ~0.5-0.8×"。

**R2 成本修正**（glm）：pt2_matrix_elements 的成本在嵌入扩展空间笛卡尔积（缩减候选不缩
主导成本）；wall 重构预测 f_PT2 ≈ 5-12%（对角化才是主因）。设计 M0 决策点：实现前先
cProfile 实测（P0a 先行门，25min 零代码）。

**P0a 实测**（12,12 @500 单次 active，GPU hybrid，369.75s）：

| 分量 | cumtime | 占比 |
|---|---|---|
| **diag（eigsh + GPU matvec）** | **357.3s** | **96.6%** |
| ├─ GPU matvec | 288.9s（43ms/次 × 6681 次）| 78.1% |
| ├─ ARPACK iterate CPU | ~68s | 18.4% |
| PT2 相关 | ~7.1s | **1.9%** |

**结论：前提证伪**——f_PT2 ≈ 1.9% ≪ 4%（R2 预测带下限），semistochastic 即便省 PT2 的
80% 也只省总 wall 的 1.5%。R3 取消，零实现成本收档。

**精确成本画像（附带产出）**：12,12 的 wall 几乎全部在对角化。下一个优化方向是对角化
迭代减少（ARPACK 6681 次迭代 = ~10 轮 × ~600 matvec/轮；warm-start v0 / tol 放宽 /
LOBPCG / 多根并行），而非 PT2 或采样端（合计仅 3.4%）。

## warm-start v0：对角化迭代减少（round_010 证实，2026-08-15）

**来源**：round_009 P0a 成本画像（diag 占 96.6%，ARPACK 6681 迭代因随机 v0 冷启动浪费）。
R2 设计：`_Subspace` 实例缓存上轮 `(sa, sb, c2d)`，下次 diag 投影到新子空间作 `eigsh(v0=)`
（旧字符串索引映射 + 新振幅置零 + 归一化）。生命周期 = 实例（adaptive 换基自动失效）。

**实现**（commit `00e970e`）：`_project_v0` helper + 三处 eigsh `v0=` + `last_n_mv` 仪表 +
warm_start 透传（active/ev/best，默认 False 零回归）。7 新测试 + 全库 199 passed。

**实测**（12,12 @500 单 active C1-v2，GPU hybrid，独立进程 cold vs warm）：

| 模式 | wall | 总 matvec | E |
|---|---|---|---|
| cold（warm_start=False）| 380s | 6,681 | -108.7686853740 |
| **warm（True）** | **117s** | **2,631** | **-108.7686853740（逐位一致）** |

**结论状态：已验证（P0 证实 0.31×，远超 0.6× 阈值）**
- **P0 wall = 证实**：380s → 117s（**3.2× 加速**）
- **P0' matvec = 证实**：6681 → 2631（0.39×，≤3500 阈值）；末轮 diag 仅 **21 次**
  matvec（vs cold 611）——相邻轮解态高重叠（‖v0‖²≈0.95+）的结构性质兑现
- **P1 正确性 = 完美**：E 逐位一致（diff = 0）

**叠加效果**：warm-start（3.2×）× GPU hybrid（4×）= 12,12 单 active 相对纯 CPU 冷启动
**~13× 加速**（1520s → 117s）。C1-v2+best 全链（4× active）从 ~27min 降到 ~8min。

**幂等恢复验证**：R3 网络断连后重新唤起，检查断连前代码完整正确（仅修 2 个测试期望值），
66min 完成收尾——SQUAD 幂等设计的实战验证。

## 自旋分辨积分支持（round_011 已验证，2026-08-16）

**路径**（方案 C，matrixfree 扩展——性价比/风险评估后选定）：`matrixfree.sigma_vector` 是
库自有 Slater-Condon（单激发本分 α/β 通道），扩展为五积分（h_α, h_β, eri_αα, eri_αβ,
eri_ββ）通道分解——**数据结构与 PySCF `direct_uhf` 约定逐字对齐**（免费独立参考锚）。

**实现**（commit `49b0175`）：
- matrixfree：`_split_spin_integrals`（归一化，legacy 同引用零回归）+ `_fock_cross_beta`
  （αβ 基 Fock）+ 六通道换块（einsum 表达式逐字未动）
- fermion：`solve_sci` 自旋分辨分支（稠密/eigsh 走 matrixfree matvec）+ 派发真值表 +
  修 `diagonalize_fermionic_hamiltonian` shape 校验隐性 bug；`compute_ground_state_energy
  (method="fci")` → direct_uhf
- molecule：`from_pyscf` UHF/UKS 分支（五积分 einsum，direct_uhf 约定），`spin_resolved` 属性
- cipsi：5 入口 + _Subspace tuple-eri 守卫（范围外功能对自旋分辨 raise）
- **零新 kwarg**：按输入形状派发，签名不变

**验证**：
- P0：N₂ + CH UHF vs PySCF direct_uhf（conv_tol=1e-12）**≤1e-10**（10 新测试）
- P1：legacy 逐位一致（跨版本 golden 实测 np.array_equal；Fa_ab 分支保护使其结构保证）
- P2：from_pyscf(UHF) 五积分 vs 手工 einsum ≤1e-12
- **端到端**（R5）：stretched N₂ UHF（h_α≠h_β 对称破缺）→ from_pyscf →
  `compute_ground_state_energy(method="fci")` vs direct_uhf 严格收敛 **diff = 0**（逐位）
- 全库 209 passed 0 failed

**口径发现**（PySCF 2.14 坑）：`direct_uhf.contract_1e + contract_2e(raw)` 组合与 kernel
不自洽（差 ~1.2 Ha）；正确参考是 `contract_2e(absorb_h1e(...))`。且默认 kernel 收敛参数
在近简并体系差 ~2e-5——参考须显式 conv_tol=1e-12。

**首期范围外**（文档已标注）：_Subspace（solve_sqd_active/ev/best/HCI/CIPSI）与
selected_ci_gpu/linkstr 对自旋分辨输入 raise——后续如需在 active 闭环中用 UHF 再扩展。

---

## Round 012：BFS 覆盖闭包 coverage_closure（2026-08-20，已验证）

### 问题与根因

round_008 triple_injection=True 但 12,12 dim 不变（824,464 = 908²，缺 16 字符串）。
**根因定位**：`cipsi.py` triple pass 的 `if abs(v) < pt2_floor: break`（默认
`pt2_floor=1e-7`）过滤掉低分**中间父串**，单激发 BFS 链断裂，永远到不了由低分
中间串连接的缺失字符串。round_008 从未扫 `pt2_floor`——这是未验证的缺口。

### P0a 零代码参数门控（4 配置，12,12 @500 warm GPU）

| pt2_floor | dim | n_str | err | sigma² | wall |
|---|---|---|---|---|---|
| off (no triple) | 824,464 | 908 | 3.63e-7 | 6.87e-6 | 114s |
| 1e-7 (=round008) | 824,464 | 908 | 3.63e-7 | 6.87e-6 | 113s |
| **1e-12** | **853,776** | **924** | **2.25e-10** | **0** | 116s |
| **0** | **853,776** | **924** | **2.25e-10** | **0** | 114s |

降 floor 后 BFS 从 908 单激发可达全部 924（缺 16），dim 补全到全空间 FCI。
err 3.63e-7 → 2.25e-10 = **1600×**，wall 1.02×（仅补 16 串 +1-2 次 diag）。

### coverage_closure 特性（commit 325c25d）

把"pt2_floor 调到 0"封装为自描述 API：`coverage_closure=True` 时 ① 强制
`triple_injection=True`；② triple pass 用 `_triple_floor=0`（`abs(v)<0` 永不 break）。
主循环 PT2 选态仍用 `pt2_floor`（两个 floor 解耦）。护栏：`max_strings` 上界
（默认全空间，不超界枚举）。默认 `False` 零回归。透传 active/ev/best/improved。

### R5 跑分（3 seed + 体系矩阵）

| 体系 | seed | dim | err | wall |
|---|---|---|---|---|
| 12,12 baseline | 0 | 824,464 | 3.63e-7 | 115s |
| 12,12 closure | 0/1/2 | 853,776 (全空间) | 2.25e-10 | 124-132s |
| 10o closure | 0 | 63,504 (全空间) | 1.58e-10 | 4.1s |
| STO-3G closure | 0 | 14,400 (全空间) | 4.87e-9 | 11.2s |

- P0：err≤1e-9 ✅、dim=全空间 ✅、wall≤1.5× ✅
- P0' 零回归：R4 全库 228 收集 0 失败（217+9 passed + 1xfail + 1xpass）
- P1：10o/STO-3G 补全到全空间=FCI ✅
- **seed 无关**：3 seed 全部 dim=853,776 + err=2.25e-10（BFS 确定性）

### 认知更新

- 12,12 的全部残余误差（3.63e-7）来自**覆盖缺口**（缺失 16 串），非对角化容差或 PT2。
- closure 补全到全空间 = FCI：是"采样（得 908 高权重串）+ 确定性 BFS 闭包（补 16 串）"
  的混合方案，比冷启动 FCI 快 ~1.4×（warm 增量 diag）。
- **北极星对照**：闭包后 err 2.25e-10 远超 SHCI 同维度（3.8e-11 量级）——但因补全到
  全空间=FCI，这不是纯采样式覆盖突破。纯采样式（不闭包）仍 3.63e-7，采样式 vs SHCI
  确定性选态的本质差距仍在（round_008 结论维持）。
- **证伪更新**：round_008 的"triple BFS 不可补全全空间"结论需修正——不是图不连通，
  是 `pt2_floor` 门控断链；降为 0 后单激发图连通性成立。
