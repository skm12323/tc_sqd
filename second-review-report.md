# `tc_sqd` 第二轮代码审查报告

## 总体结论

**修复有效，但修复报告的“严重问题已全部修复”结论不成立。当前项目仍只能判定为部分满足需求。**

确认已修复：

- 配置恢复中的轨道占据顺序
- 单、双电子积分旋转方向一致性
- `return_probabilities=False` 的概率归一化
- `optimize_orbitals` 的四元组解包崩溃本身
- RDM 自旋求和/分辨接口的基础调用方式

仍未满足：目标自旋约束、可靠轨道优化、开壳层、自旋分辨积分、若干迭代 SQD 参数、状态持久化后的 RDM、自旋计算及边界输入处理。

## 动态验证状态

- `python -m compileall -q src tests examples`：**通过**。
- 源码静态诊断：**无诊断项**。
- `PYTHONPATH=src python -m tests.test_h2_sqd`：**未执行到测试逻辑**，当前解释器为 Python 3.14.6，环境缺少 `numpy`，导入阶段报 `ModuleNotFoundError`。
- 当前目录不是 Git 仓库，无法通过 Git 历史独立核对改动范围。

因此，`fix-report.md` 中“17 项全部通过”的结果本轮无法在当前工作区环境复现。

## 上一轮问题复核

| 问题 | 状态 | 结论 |
|---|---|---|
| 轨道优化解包崩溃 | 部分修复 | 四元组解包已正确，但优化过程与返回结果仍不可靠 |
| 配置恢复轨道顺序 | 已修复 | `avg_b[::-1]`、`avg_a[::-1]` 与规范布局对齐，并有直接测试 |
| 目标自旋约束 | 未修复 | `myci.spin` 不能证明在固定 selected-CI 空间中实施了目标 \(S^2\) 投影；SQD/direct 仍忽略该能力 |
| 单双电子积分旋转方向 | 已修复 | 两类积分现在采用一致的 `U_ip` 变换 |
| 开壳层 CI 字符串 | 未修复 | 默认仍会合并 alpha/beta 字符串集合 |
| 自旋分辨一电子积分 | 未修复 | 仍简单平均 alpha/beta 积分 |
| RDM 接口 | 部分修复 | 当前求解结果的基础 RDM 调用已改正，但持久化后元数据丢失等问题仍在 |
| 迭代 SQD 无效参数 | 未修复 | `max_dim`、`include_configurations`、`carryover_threshold` 仍未实现 |
| 均匀概率归一化 | 已修复 | 唯一比特串按 `1/M` 分配概率 |
| 输入边界校验 | 未修复 | 空输入、零和/负概率、奇数位宽等仍缺少检查 |
| NumPy 最低版本 | 未修复 | 仍仅声明 `numpy<2.0`，文档仍声称可低至 1.13 |
| 文档测试命令 | 未修复 | `docs/README.md` 中仍存在与目录结构不匹配的命令 |

## 严重和高优先级问题

### 1. `spin_sq` 仍不能作为可靠的目标自旋约束

`solve_sci` 将 \(S^2\) 换算为 `2S` 后赋给 `myci.spin`，但实际调用的是固定电子数、固定字符串空间的 `selected_ci.kernel_fixed_space`。现有代码没有增加 \(S^2\) 惩罚项，也没有按总自旋投影或筛选本征态。

此外：

- `solve_fermion` 中所谓 spin shift 分支仍是 `pass`。
- `compute_ground_state_energy(method="sqd")` 没有把 `spin_sq` 传入迭代求解器。
- `method="direct"` 完全忽略 `spin_sq`。

位置：

- `src/tc_sqd/fermion.py:379-399`
- `src/tc_sqd/fermion.py:500-510`
- `src/tc_sqd/fermion.py:870-908`

新增测试仅验证 H2 自然基态在 `spin_sq=0` 时仍为单重态；即使约束完全不生效，该测试也通常会通过。应在同一 \(M_S\) 空间中分别请求单重态和三重态，验证能量与 \(S^2\) 均发生预期变化。

### 2. 多电子/选择子空间下的占据数计算可能崩溃

`SCIState.orbital_occupancies` 使用 alpha 电子数 `na` 对振幅矩阵执行：

`self.amplitudes.reshape(na, -1)`

该结果完全未使用，而且 `na` 不是 alpha CI 字符串数量。当振幅元素总数不能被 alpha 电子数整除时，会在真正计算占据数前直接抛出 `ValueError`。

位置：`src/tc_sqd/fermion.py:83-89`。

H2 的 `nelec=(1,1)` 不会触发该问题，当前测试无法发现。

### 3. 轨道优化仍未形成正确、完整的功能

虽然解包错误已修复，但仍存在：

- `num_steps_grad` 从未使用。
- `best_energy` 每轮直接覆盖，不是真正的最优能量。
- 返回的 `k_flat` 是梯度更新后的参数，`energy` 和占据数却是在更新前计算，三者不对应同一轨道基。
- `num_iters=0` 返回 `inf` 和 `None`。
- `k_flat` 长度未校验；过短会越界，过长被静默忽略。
- `tests/test_h2_sqd.py` 没有实际调用 `optimize_orbitals`，与修复报告中的“现可正常调用”缺少对应回归测试。

位置：`src/tc_sqd/fermion.py:693-765`。

### 4. 开壳层及自旋分辨哈密顿量仍不受可靠支持

- `open_shell=False` 时仍把 alpha、beta 字符串取并集后同时使用。当 \(n_\alpha \ne n_\beta\) 时可能混入错误粒子数的字符串。
- 形状为 `(2, norb, norb)` 的一电子积分仍被简单平均，alpha/beta 差异被丢弃。

位置：

- `src/tc_sqd/fermion.py:163-203`
- `src/tc_sqd/fermion.py:310-314`
- `src/tc_sqd/fermion.py:387-393`

`fix-report.md` 将其归类为暂不支持是可以接受的范围决策，但公开 API 和主文档尚未清楚限制为闭壳层，因此当前文档能力声明仍大于实际能力。

### 5. 迭代 SQD 仍存在静默失效参数

以下公开参数仍未参与逻辑：

- `max_dim`
- `include_configurations`
- `carryover_threshold`

Carryover 只有注释，没有阈值筛选或跨迭代配置合并。调用者传入这些参数不会获得预期行为。

另外，`max_iterations<=0` 会使函数返回 `None`；`norb`、积分形状、比特串宽度之间也没有一致性校验。

位置：`src/tc_sqd/fermion.py:520-650`。

## RDM 和状态持久化

当前 RDM 调用方向已有改善，但仍是部分修复：

1. `rank=2, spin_summed=False` 的文档声明四个自旋块，与 PySCF 常见 `make_rdm2s` 返回结构不一致，需要按实际版本确认并测试。
2. selected-CI 振幅携带 CI 字符串元数据；`SCIState.save/load` 通过 `np.savez` 后恢复为普通 `ndarray`，没有重新包装成带字符串元数据的 `SCIvector`。加载后的状态不能保证继续调用 `rdm()` 或 `spin_square()`。
3. 当前测试只检查类型和形状，没有检查粒子数迹、RDM 收缩关系、数值一致性及保存/加载后的行为。

位置：`src/tc_sqd/fermion.py:72-141`。

## 输入鲁棒性问题

### Counts

`counts_dict_to_bitstring_matrix` 仍未处理：

- 空字典
- 总计数为零或负计数
- 超出 `nbits` 的整数键
- 等价键去重合并
- 非法二进制字符串和非正位宽

位置：`src/tc_sqd/counts.py:90-126`。

`sample_from_circuit` 修复后实现为唯一串均匀概率 `1/M`，但 docstring 仍写 `1/n_samples`，文档与实现不一致：`src/tc_sqd/counts.py:157-178`。

### 配置恢复与子采样

仍未检查：

- 奇数比特宽度和空矩阵
- 概率长度、有限性、非负性及总和
- 平均占据数组长度和值域
- 目标电子数范围
- 无放回抽样数是否超过非零概率项数

另外 `_recover_single` 接收 `rng`，但没有真正随机处理并列，占据完全相同的轨道不会按文档进行随机 tie-breaking。

位置：

- `src/tc_sqd/configuration_recovery.py:62-185`
- `src/tc_sqd/subsampling.py:54-101`

## Qubit 模块新增发现

- 非法 Pauli 字符没有 `else` 校验，会被静默当作恒等操作处理。
- `S<=100` 的稠密分支固定只返回一个本征对，忽略调用者传入的 `k` 等 `scipy_kwargs`，与文档“转发给 eigsh”不一致。
- Pauli Y 和复数矩阵元仍无独立测试。

位置：`src/tc_sqd/qubit.py:89-108`、`src/tc_sqd/qubit.py:192-212`。

## 测试质量评估

新增测试确实覆盖了配置恢复顺序、均匀概率归一化、RDM 基础返回类型和积分旋转一致性，但仍存在以下不足：

- 没有调用 `optimize_orbitals`。
- 自旋测试无法区分“约束生效”和“自然基态本来就是单重态”。
- TFIM 参考矩阵继续复用被测的 `matrix_elements_from_pauli`，不是独立参考实现。
- RDM 仅检查类型和形状，不检查数值守恒关系。
- TensorCircuit 采样未固定随机生成器，测试可能波动。
- 缺少多电子、开壳层、Pauli Y、非法输入、大稀疏空间和状态持久化测试。

## 文档与依赖

- `requirements.txt` 应将 NumPy 下限调整为至少 1.17，或移除 `default_rng` 依赖；当前只有 `numpy<2.0`。
- `docs/README.md` 的测试命令仍与 `src/` 布局不匹配。
- 文档应明确当前仅可靠支持闭壳层、共享空间一电子积分，并标记未实现参数。
- 源码依赖 TensorCircuit `_nqubits` 和 PySCF selected-CI 私有接口，建议固定已验证的依赖版本范围并增加安装 CI。

## 最终判定

- 对 **H2/STO-3G 闭壳层演示主路径**：修复后可信度有所提高，但本轮因环境缺少依赖，仍无法独立复现数值测试。
- 对公开声明的 **目标自旋、开壳层、自旋分辨积分、轨道优化、完整 RDM/状态持久化和迭代控制参数**：仍不满足。

建议下一轮优先处理：

1. 删除占据数计算中的错误 `reshape`，增加多电子选择子空间测试。
2. 重新设计或明确拒绝不支持的 `spin_sq` 分支。
3. 完整修复并直接测试 `optimize_orbitals`。
4. 明确闭壳层范围，或实现开壳层与自旋分辨积分。
5. 实现或删除无效迭代参数。
6. 补齐输入校验、状态持久化和独立数值测试。
