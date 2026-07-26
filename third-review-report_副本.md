# `tc_sqd` 第三轮代码审查报告

## 总体结论

**第三轮修改有明显进步，但项目仍只能判定为“部分满足需求”。**

本轮确认新增修复包括：输入校验增强、开壳层错误用法保护、自旋分辨积分显式拒绝、部分迭代参数处理、Pauli 非法字符校验、Pauli Y 独立测试、NumPy 下限和文档测试命令修正、TFIM 参考实现独立化。

仍有三个核心阻塞项：

1. `SCIState.orbital_occupancies` 的错误 `reshape` 仍在，多电子子空间可能直接崩溃。
2. `spin_sq` 的目标自旋约束仍未真正实现，只有部分分支改为显式拒绝。
3. `optimize_orbitals` 虽已能保持返回能量与参数一致，但仍缺参数校验、最优参数保留和直接回归测试。

## 动态验证状态

- `python -m compileall -q src tests examples`：**通过**。
- 全部源码和测试 AST 解析：**通过**。
- 静态诊断：**无诊断项**。
- `PYTHONPATH=src python -m tests.test_h2_sqd`：当前 Python 3.14.6 环境仍缺少 `numpy`，导入阶段报 `ModuleNotFoundError`，因此完整数值测试未能独立复现。

## 已确认修复

### 配置恢复轨道顺序

`avg_b[::-1]` 和 `avg_a[::-1]` 已与 `[beta_(n-1)...beta_0 | alpha_(n-1)...alpha_0]` 对齐，测试明确验证了具体恢复轨道。

位置：`src/tc_sqd/configuration_recovery.py:176-188`；测试：`tests/test_h2_sqd.py:369-387`。

### 积分旋转方向

一电子积分与二电子积分现在使用一致的 `U_ip` 基变换，且有独立表达式回归测试。

位置：`src/tc_sqd/fermion.py:752-760`；测试：`tests/test_h2_sqd.py:342-358`。

### Counts 和采样校验

已增加空字典、非法位宽、越界整数、负计数、零总计数及非正采样数检查；均匀概率文档也已同步为 `1/M`。

位置：`src/tc_sqd/counts.py:113-142`、`185-202`。

### 配置恢复与子采样校验

已增加二维/偶数位宽、概率长度、非负有限概率、正概率总和、占据数组长度等校验；子采样会根据非零概率项数量限制无放回样本数。

位置：`src/tc_sqd/configuration_recovery.py:140-174`；`src/tc_sqd/subsampling.py:94-132`。

### 开壳层与自旋分辨输入保护

- 当 `open_shell=False` 且 alpha/beta 电子数不等时，现在显式报错。
- 当 `(2,norb,norb)` 的 alpha/beta 一电子积分不相同时，现在显式拒绝，不再静默平均算错。

位置：`src/tc_sqd/fermion.py:509-515`、`310-318`、`395-405`。

这属于“正确拒绝不支持能力”，不是完整实现开壳层或自旋分辨哈密顿量。

### Qubit 校验与测试独立性

- 非法 Pauli 字符现在抛出 `ValueError`。
- 新增 Pauli Y 的独立 Kronecker 参考测试。
- TFIM 参考矩阵不再复用被测函数，测试可信度明显提高。

位置：`src/tc_sqd/qubit.py:46-55`；测试：`tests/test_h2_sqd.py:148-170`、`516-538`。

### 依赖与文档命令

- NumPy 依赖已调整为 `numpy>=1.17,<2.0`。
- README 和 docs 中的测试命令已统一为 `PYTHONPATH=src python -m tests.test_h2_sqd`。

## 未修复的高优先级问题

### 1. 多电子占据数计算仍可能崩溃

`SCIState.orbital_occupancies` 仍执行：

`self.amplitudes.reshape(na, -1)`

其中 `na` 是 alpha 电子数，不是 alpha CI 字符串数量；该变量后续完全未使用。振幅元素数不能被 `na` 整除时会无意义地触发 `ValueError`。

位置：`src/tc_sqd/fermion.py:83-89`。

新增占据数测试仍只使用 `nelec=(1,1)`，不能覆盖该缺陷：`tests/test_h2_sqd.py:458-462`。

### 2. `spin_sq` 目标自旋约束仍未真正实现

`solve_sci` 仍只是把换算后的 `2S` 赋给 `myci.spin`，随后调用固定 selected-CI 空间求解；代码没有总自旋投影、按 \(S^2\) 选态或惩罚项。

`solve_fermion` 中的 spin shift 分支仍是 `pass`：`src/tc_sqd/fermion.py:520-526`。

本轮已改善之处是 `compute_ground_state_energy` 的 `sqd` 和 `direct` 分支会拒绝非空 `spin_sq`，避免静默忽略。但 FCI/`solve_sci` 路径的约束能力仍未经证明。现有测试只请求 H2 自然单重态，约束不生效也会通过：`tests/test_h2_sqd.py:311-327`。

### 3. `optimize_orbitals` 仍只是部分可用

已改善：

- `num_steps_grad` 已进入循环。
- 返回前会在最终 `k_flat` 上重新计算能量，能量、参数和占据数现在相互对应。
- `num_iters=0` 会执行最终求值，而不是返回 `inf/None`。

仍存在：

- 未校验 `k_flat` 长度必须为 `norb*(norb-1)//2`；过短越界、过长静默忽略。
- `num_steps_grad<=0` 被 `max(1, num_steps_grad)` 静默改成一步，参数语义不清。
- 未校验 `num_iters`、学习率和有限数值。
- 不保存最低能量对应的参数；最后一步可能比中间点更差。
- 每个外层循环内容完全相同，`num_iters*num_steps_grad` 实际只是总梯度步数，双层 API 语义可简化。
- 测试文件仍没有直接调用 `optimize_orbitals`。

位置：`src/tc_sqd/fermion.py:764-832`。

## 迭代 SQD：部分修复但语义仍有缺陷

### `max_dim`

已从静默忽略改为显式 `NotImplementedError`，这是正确的防误用方式，但功能仍未实现：`src/tc_sqd/fermion.py:585-589`。

### `include_configurations`

现在会合并进入采样池，但随后仍由概率无放回抽样。低概率的强制配置不保证出现在任一批次，因此参数名称暗示的“必须包含”语义没有实现。

位置：`src/tc_sqd/fermion.py:603-614`、`648-658`。

### `carryover_threshold`

现在从 CI 振幅筛选 determinant 并加入下一轮采样池，有实质实现；但仍只是概率加入，不保证被抽样。同时没有校验阈值非负或合理范围。负阈值会生成负概率并在下一轮失败。

位置：`src/tc_sqd/fermion.py:638-652`、`688-704`。

### 其他边界

- `max_iterations<=0` 仍返回 `None`。
- `include_configurations` 仅检查列数，未检查二维之外的高维输入、粒子数或概率语义。
- `norb` 与比特串宽度、积分形状之间缺少系统性一致性检查。
- 没有测试 `include_configurations`、carryover 或 `max_dim`。

## RDM 与状态持久化仍是部分修复

`SCIState.save/load` 测试只比较保存字段，没有在加载后调用：

- `loaded.rdm(...)`
- `loaded.spin_square()`
- `loaded.orbital_occupancies()`

selected-CI 振幅携带 CI 字符串元数据；加载后是普通 `ndarray`，当前方法没有用 `ci_strs_a/ci_strs_b` 重新包装为 `SCIvector`，因此 RDM 和自旋方法仍不能保证可用。

位置：`src/tc_sqd/fermion.py:72-141`；测试：`tests/test_h2_sqd.py:447-456`。

另外 `rank=2, spin_summed=False` 的具体返回结构仍没有测试。

## 其他残留问题

### 配置恢复随机并列处理未实现

`_recover_single` 接收 `rng`，文档声称用于 tie-breaking，但函数没有使用它；`deviation` 变量也计算后未使用。相同平均占据的候选轨道始终按原索引顺序选择。

位置：`src/tc_sqd/configuration_recovery.py:62-100`。

### 配置恢复仍缺少部分校验

尚未检查：

- 平均占据数组是否为一维、有限且位于 `[0,1]`
- `num_elec_a/b` 是否在 `[0,norb]`
- 空比特串矩阵
- `n>=64` 的整数去重溢出风险

### Counts 等价键不会合并

例如字符串键 `"01"` 与整数键 `1` 会生成重复行，而文档声明输出唯一比特串。当前实现没有按整数值聚合计数。

位置：`src/tc_sqd/counts.py:118-143`。

### TensorCircuit qubit 数回退不可靠

缺少 `_nqubits` 时通过最大采样整数的 `bit_length` 推断位数；若最高位在此次采样中始终为 0，会低估 qubit 数。没有可靠元数据时应要求调用方显式提供位数，而不是静默推断。

位置：`src/tc_sqd/counts.py:194-199`。

### Qubit 稠密分支仍忽略 `k`

`solve_qubit` 文档称 `scipy_kwargs` 转发给 `eigsh`，但 `S<=100` 时固定返回一个最低本征对。调用 `k=3` 不会得到三个本征对。

位置：`src/tc_sqd/qubit.py:199-209`。

### Qubit 测试描述与实际覆盖不一致

`test_pauli_y_and_validation` docstring 写有“large sparse branch”，但测试只构造 2-qubit、4 维矩阵，没有覆盖 `S>100` 的 `eigsh` 分支：`tests/test_h2_sqd.py:510-547`。

## 测试质量结论

本轮测试质量较上一轮明显提高：

- TFIM 使用独立参考矩阵。
- Pauli Y 使用独立 Kronecker 参考。
- 增加配置恢复、子采样、开壳层错误用法和自旋分辨输入保护测试。

仍缺少能够证明以下能力的测试：

- 多电子占据数
- 真正目标自旋选择
- `optimize_orbitals`
- 加载后 RDM/自旋
- `include_configurations`、carryover、`max_dim`
- Qubit `k>1` 和 `S>100` 稀疏分支
- 配置恢复随机 tie-breaking

## 最终判定

- 对 **H2/STO-3G 闭壳层演示和基础 Qubit 子空间求解**：静态上已接近可用，测试设计也更可信；但当前环境缺依赖，未能独立复跑数值结果。
- 对 **多电子一般体系、目标自旋、可靠轨道优化、完整状态持久化及严格迭代控制**：仍不满足。

下一轮建议优先修复：

1. 删除 `orbital_occupancies` 中无意义且危险的 `reshape`，增加 `n_alpha>1` 测试。
2. 正确实现目标自旋选态，或从 `solve_sci`/FCI API 中移除并显式拒绝 `spin_sq`。
3. 为 `optimize_orbitals` 增加参数校验、best-so-far 参数保留和直接测试。
4. 让 `include_configurations` 真正强制进入每个或指定批次，并校验 carryover 阈值。
5. 加载状态后重建 `SCIvector`，验证 RDM 和自旋数值。
6. 修复 Counts 去重聚合、Qubit `k` 语义和剩余边界校验。
