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

## 五轮增强（commit c002d9e → c6d0eb6）

在原有核心基础上连续五轮增强，每轮均带独立测试与验证：

| 轮 | 提交 | 内容 | 关键验证 |
|---|---|---|---|
| 1 | `c002d9e` | 3 修复（density 布局反转 / REM 静默吞错 / kwargs 泄漏）+ `from_pyscf` + `plan_sampling` + `diagnostics` + `optimize_orbitals`→Nelder-Mead | `from_pyscf` 活性 FCI 与受限对角化一致 (1e-8)；轨道优化 5.9s 收敛（原 60 万次对角化不可用）|
| 2 | `f7dc19e` | LUCJ 真机深度预算（`circuit_stats`/`lucj_report`，2Q 门代理）+ `max_dim` 子空间限制 | H2 门数断言；max_dim 裁剪维度 ≤ 限制 |
| 3 | `800945c` | T1 感知恢复 `estimate_true_occupancies` | per-qubit γ 反卷积 RMSE 降 33%（0.116→0.078）；均匀 γ 保序退化（实验证明改翻转决策无效）|
| 4 | `2228552` | 激发态采样策略 `excited_configurations` + 2 全链路示例 | H2 n_roots 精确复现 FCI 4 根 (1e-8)；LiH 激发态误差 ~2e-4 |
| 5 | `c6d0eb6` | 统一采样后端 `sampler`（tc 模拟 / qcloud 真机）| tc 后端驱动 SQD 复现 FCI |

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

## 后续可选改进（非阻塞）

- 完整开壳层（`n_alpha ≠ n_beta` 的独立 alpha/beta CI 空间）与自旋分辨哈密顿量
- 配置恢复 tie-breaking 随机性的统计性测试
- 多版本 numpy（1.x / 2.x）CI 矩阵，固化兼容性
- **SQD + VQE 混合优化**（LUCJ 角度变分，SQD 能量作损失）—— 已规划为下一轮
- UCJ 精确化（t2→SVD→Û/J，对标 ffsim）；GPU CI 对角化（大体系路线）
