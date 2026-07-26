# `tc_sqd` 代码审查结论

## 总体判断

**当前代码仅部分满足需求，不能判定为完整满足。**

H2/STO-3G 闭壳层、CI 矩阵、统一能量入口和 TFIM 已有对应实现及测试；全部 Python 文件通过语法编译。但配置恢复、轨道旋转、轨道优化和目标自旋存在确定性缺陷，公开 API 的完整能力尚不可交付。

## 验证状态

- 已审查 `README.md`、`docs/*.md`、`src/tc_sqd/*.py`、`tests/test_h2_sqd.py`、示例和依赖文件。
- `python -m compileall -q src tests examples`：通过。
- 源码静态诊断：无诊断项。
- 自带测试未能进入测试逻辑：当前 Python 环境缺少 `numpy`，导入即报 `ModuleNotFoundError`。因此文档声明的数值结果本次未动态复核。

## 严重问题

1. **轨道优化调用即崩溃**  
   `solve_fermion` 返回 4 项，但 `optimize_orbitals` 按 5 项解包。`num_steps_grad` 也未使用。  
   位置：`src/tc_sqd/fermion.py:491-496`、`src/tc_sqd/fermion.py:716-740`。

2. **配置恢复的轨道占据顺序错误**  
   比特布局是 `[beta_(n-1)...beta_0 | alpha_(n-1)...alpha_0]`，平均占据数组却按轨道 `0...n-1` 直接拼接，没有反转，导致恢复时翻错轨道。  
   位置：`src/tc_sqd/configuration_recovery.py:150-164`。

3. **`spin_sq` 没有形成有效目标自旋约束**  
   `solve_sci` 把数值赋给 `spin0`；FCI 分支创建的约束求解器没有被使用；SQD 和 direct 分支未正确处理该参数。  
   位置：`src/tc_sqd/fermion.py:360-378`、`src/tc_sqd/fermion.py:833-876`。

4. **轨道旋转的单双电子积分变换不一致**  
   一电子积分使用 `U.T @ h @ U`，二电子积分却使用相反索引方向的 `"pi,qj,rk,sl"`。非零旋转后的哈密顿量不对应同一轨道基。  
   位置：`src/tc_sqd/fermion.py:650-667`。

## 其他重要问题

- `open_shell=False` 会合并 alpha/beta CI 字符串；当电子数不等时会混入错误汉明权重的字符串：`src/tc_sqd/fermion.py:144-184`。
- 自旋分辨的一电子积分被简单平均，不能正确处理 alpha/beta 不同的哈密顿量：`src/tc_sqd/fermion.py:291-306`、`366-375`。
- `SCIState.rdm` 对已经自旋求和的 RDM 再次按切片相加，返回维度和数值语义错误：`src/tc_sqd/fermion.py:104-118`。
- `max_dim`、`include_configurations`、`carryover_threshold` 出现在迭代 SQD 签名中，但未真正实现：`src/tc_sqd/fermion.py:499-519`、`611-613`。
- `sample_from_circuit(return_probabilities=False)` 返回的概率通常不归一化：`src/tc_sqd/counts.py:168-179`。
- 空计数字典、零和/负概率、奇数位宽、概率长度不匹配等边界输入缺少统一校验。
- 文档声称支持 NumPy `>=1.13`，但 `np.random.default_rng` 要求至少 1.17。
- `docs/README.md` 的测试命令与当前 `src/` 目录结构不匹配。

## 测试覆盖缺口

现有测试覆盖 H2 闭壳层、基础转换、X/Z TFIM 和三种统一入口，但缺少：

- 配置恢复及子采样的直接测试
- 轨道旋转和轨道优化测试
- 开壳层、目标自旋、自旋分辨积分测试
- RDM、状态保存/加载测试
- Pauli `Y`、非法 Pauli、大稀疏空间测试
- 独立构造的 Qubit 参考矩阵；当前 TFIM 参考矩阵复用了被测函数
- 安装测试和多版本 CI

## 最终判定

- 若需求仅限 **H2/STO-3G 闭壳层演示**：实现结构接近满足，但当前环境依赖缺失，尚未动态确认数值结果。
- 若需求包括文档公开声明的 **配置恢复、开壳层、目标自旋、RDM、轨道旋转和轨道优化**：**当前不满足**。

建议优先修复：轨道优化崩溃 → 配置恢复顺序 → 积分旋转 → 自旋约束 → 开壳层/RDM → 参数与输入校验，并补齐对应回归测试。
