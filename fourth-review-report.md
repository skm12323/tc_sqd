# `tc_sqd` 第四轮验证报告

> 验证日期：2026-07-27
> 关键区别：**首次在真实环境中完成完整动态运行**

## 一、与前三轮的本质区别

前三轮审查（`code-review-report.md` / `second-review-report.md` /
`third-review-report.md`）均止步于**静态审查**——报告原文反复出现：

> 当前 Python 3.14.6 环境仍缺少 `numpy`，导入阶段报 `ModuleNotFoundError`，
> 因此完整数值测试未能独立复现。

因此三轮的"部分满足需求"判定中，数值正确性始终是**未经动态证实**的假设。
本轮在 `WSL Ubuntu-22.04 + conda tc` 环境中实际运行了编译检查、完整测试套件
和示例，首次给出动态证据。

## 二、运行环境

| 组件 | 版本 |
|---|---|
| Python | 3.10.20 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| pyscf | 2.14.0 |
| tensorcircuit | 0.12.0 |

> 注：实际 numpy 为 **2.2.6**，而 `requirements.txt` 此前声明 `numpy<2.0`。
> 本轮据实测结果放宽了下界（见第五节）。

## 三、动态验证结果

### 3.1 编译

```
python -m compileall -q src tests examples   →  通过（COMPILE_OK）
```

### 3.2 测试套件

```
PYTHONPATH=src python -m tests.test_h2_sqd   →  All tests passed!
```

**9 个测试函数、约 50 项断言全部通过：**

| # | 测试函数 | 覆盖 | 结果 |
|---|---|---|---|
| 1 | `test_fermion_sqd` | H2 全子空间 SQD、`build_ci_matrix`、迭代 SQD | PASS |
| 2 | `test_qubit_sqd` | TFIM 子空间、变分界、去重 | PASS |
| 3 | `test_counts` | 比特串↔整数、计数字典 | PASS |
| 4 | `test_compute_ground_state_energy` | fci / direct / sqd 三入口、非法输入 | PASS |
| 5 | `test_bugfixes` | 占据顺序、概率归一化、spin_sq、RDM、旋转一致性 | PASS |
| 6 | `test_recovery_and_subsampling` | 配置恢复、汉明后选择、子采样校验 | PASS |
| 7 | `test_state_io_and_open_shell` | save/load 后 RDM/自旋可用、开壳层/自旋分辨守卫 | PASS |
| 8 | `test_pauli_y_and_validation` | Pauli Y、非法字符、稠密 k>1 | PASS |
| 9 | `test_third_review_fixes` | 多电子占据数、目标自旋选态、轨道优化、include/carryover、counts 合并、稀疏分支 | PASS |

### 3.3 示例

```
PYTHONPATH=src python examples/h2_sqd_demo.py   →  通过
```

### 3.4 关键数值结果

**H2 / STO-3G（norb=2, nelec=(1,1)）** —— 四条独立路径互相吻合：

```
E(HF)                 = -1.11675931
E(FCI, PySCF)         = -1.13728383
E(SQD, full subspace) = -1.13728383   ✓
E(build_ci_matrix)    = -1.13728383   ✓
E(SQD, iterative)     = -1.13728383   ✓
E(direct)             = -1.13728383   ✓
```

**TFIM 3-qubit**：SQD 基态 −2.403212，与独立 Kronecker 参考一致 ✓

**多电子 (norb=3, nelec=(2,1))**：`orbital_occupancies` 给出
`occ_a.sum()=2`、`occ_b.sum()=1`，**不再崩溃** ✓

**目标自旋选态**：`spin_sq=0.75` 在 (2,1) 系统中选中 S=1/2 根
（E=−5.044396 ≥ 自由基态 E=−7.493671）；不可达自旋（S²=42）抛出
`ValueError` 而非静默返回错误自旋态 ✓

**Qubit 稀疏分支（S=256）**：本征值与独立参考吻合 ✓

## 四、第三轮 6 项优先建议 —— 逐项动态确认

第三轮报告末尾建议下一轮优先修复的 6 项，现全部已实现并经回归测试动态证实：

| # | 第三轮建议 | 当前状态 | 证据 |
|---|---|---|---|
| 1 | 删除 `orbital_occupancies` 中危险的 `reshape`，补 `n_alpha>1` 测试 | 已删，按 CI 字符串数 reshape | `fermion.py:83`；Test 9 (1) PASS |
| 2 | 正确实现目标自旋选态，或显式拒绝 | 多根对角化 + S² 匹配选根，不可达则 raise | `fermion.py:425-456`；Test 9 (2) PASS |
| 3 | `optimize_orbitals` 参数校验 + best-so-far + 直接测试 | k_flat 长度/步数/学习率校验 + best-so-far 跟踪 | `fermion.py:893-941`；Test 9 (3) PASS |
| 4 | `include_configurations` 真正强制包含 + carryover 校验 | 强制并入每批 CI 字符串；carryover 范围校验 | `fermion.py:726-746`；Test 9 (4) PASS |
| 5 | 加载后重建 `SCIvector`，验证 RDM / 自旋数值 | `_as_scivector` 重附 `_strs` 元数据 | `fermion.py:146-162`；Test 7 PASS |
| 6 | Counts 去重聚合 / Qubit k 语义 / 边界校验 | 等价键按整数值合并；稠密分支尊重 k；稀疏分支验证 | `counts.py:118-145`、`qubit.py:195-225`；Test 8/9 PASS |

## 五、本轮依赖与文档修正

| 文件 | 修改 |
|---|---|
| `requirements.txt` | `numpy>=1.17,<2.0` → `numpy>=1.17`（实测兼容 numpy 2.2.6）；移除**未被任何代码引用**的 `h5py>=2.7`（持久化使用 `np.savez`） |
| `docs/README.md`、`docs/usage.md` | 安装命令 `"numpy<2.0"` → `numpy` |
| `docs/usage.md` | 测试命令 `cd ~/Desktop/tc_sqd` → `cd tc_sqd`（与目录无关） |
| `README.md` | 目录结构补 `docs/usage.md`；`numpy 1.x` → `numpy 1.x/2.x` |
| `docs/README.md` | `numpy 1.x` → `numpy 1.x/2.x` |

> numpy 上界放宽的依据：在 numpy **2.2.6** 下，编译、9 个测试函数、示例均通过，
> 且全项目无 `np.bool_` / `np.int0` / `np.product` 等 numpy 2.x 已移除 API 的使用
> （`np.integer` 是仍受支持的抽象基类）。

## 六、最终判定

相比第三轮"部分满足需求"，本轮经**动态运行**确认：

- **H2/STO-3G 闭壳层主路径**：fci / direct / sqd 三种方法数值精确匹配 PySCF FCI。
- **此前被判定"仍不满足"的扩展能力**——多电子占据数、目标自旋选态、可靠轨道优化、
  `include_configurations` 强制包含、状态保存/加载后的 RDM 与自旋、迭代控制参数校验、
  Counts 等价键合并、Qubit 稠密 k 与稀疏分支——**均已实现并通过对应回归测试**。

第三轮所列待满足期待，本轮经动态验证视为满足。

## 七、后续可选改进（非阻塞）

- 完整开壳层（`n_alpha ≠ n_beta` 下的独立 alpha/beta CI 空间）与自旋分辨哈密顿量
  （目前为"正确拒绝"而非完整实现）。
- `max_dim` 子空间维度限制（目前显式 `NotImplementedError`）。
- 配置恢复 tie-breaking 随机性的统计性测试（当前 `_recover_single` 已用
  Fisher-Yates 打破并列，单次确定性测试已通过）。
- 多版本 numpy（1.x / 2.x）CI 矩阵，固化本轮观察到的兼容性。
