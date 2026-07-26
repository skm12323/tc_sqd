# tc_sqd

**Sample-based Quantum Diagonalization (SQD) for TensorCircuit**

一个适配 TensorCircuit + numpy 1.x/2.x + PySCF 的轻量级 SQD 包，参考
[`qiskit-addon-sqd`](https://github.com/qiskit/qiskit-addon-sqd) 设计，但**不依赖
numpy≥2 / jax 的硬性要求**。

## 特性

- 统一入口 `compute_ground_state_energy`，支持 `fci` / `sqd` / `direct` 三种方法
- 比特串矩阵 ↔ 整数互转，TensorCircuit 采样适配
- 基于平均占据数的配置恢复（纠正噪声导致的粒子数违例）
- 批量子采样、汉明权重后选择
- CI 矩阵构造（Slater–Condon）、子空间对角化、迭代 SQD、轨道优化
- Pauli 哈密顿量在比特串子空间的投影与对角化（非费米子问题，如 QAOA-MaxCut）

## 安装

```bash
conda create -n tc python=3.10
conda activate tc
pip install -r requirements.txt
```

依赖：`tensorcircuit`、`numpy>=1.17`（实测兼容至 2.2.6）、`scipy`、`pyscf`。

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

## API 速查

| 模块 | 函数 | 作用 |
|---|---|---|
| counts | `bitarray_to_int(bsm)` | 比特串矩阵 → 整数数组 |
| counts | `int_to_bitarray(vals, nbits)` | 整数 → 比特串矩阵 |
| counts | `counts_dict_to_bitstring_matrix(counts, nbits)` | 计数字典 → (比特串矩阵, 概率)，等价键自动合并 |
| counts | `sample_from_circuit(circuit, n_samples)` | 从 TC 电路采样 → (比特串矩阵, 概率) |
| configuration_recovery | `recover_configurations(bsm, probs, avg_occ, na, nb)` | 基于平均占据数的配置恢复 |
| configuration_recovery | `postselect_by_hamming_weight(bsm, *, hamming_right, hamming_left)` | 按汉明权重筛选 |
| subsampling | `subsample(bsm, probs, samples_per_batch, num_batches)` | 按概率无放回批量子采样 |
| subsampling | `postselect_by_hamming_right_and_left(bsm, probs, ...)` | 汉明权重后选择 + 重归一化 |
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

```bash
PYTHONPATH=src python -m tests.test_h2_sqd      # 9 个测试函数，约 50 项断言
PYTHONPATH=src python examples/h2_sqd_demo.py    # H2 完整演示
```

## 限制与已知边界

- **闭壳层**：仅可靠支持 `n_alpha == n_beta`；`open_shell=False` 下电子数不等会显式报错。
- **一电子积分**：需为单个 `(norb, norb)`（或两块相同的 `(2, norb, norb)`）；`h_alpha ≠ h_beta` 的自旋分辨积分会显式拒绝。
- **`max_dim`**：未实现（显式 `NotImplementedError`）；子空间维度由 `samples_per_batch` / `num_batches` 控制。
- **`spin_sq`**：在 `solve_sci` / `fci` 路径通过多根 S² 匹配实现真正的目标自旋选态（不可达时 raise）；`sqd` / `direct` 路径显式拒绝。
- **状态持久化**：`SCIState.save/load` 后通过 `_as_scivector` 重建 PySCF `SCIvector` 元数据，`rdm` / `spin_square` 在加载后仍可用。

## 目录结构

```
tc_sqd/
├── README.md                 # 本文件
├── REVIEW.md                 # 代码审查与验证历史
├── requirements.txt
├── src/tc_sqd/               # counts, configuration_recovery, subsampling, fermion, qubit
├── tests/test_h2_sqd.py      # 9 个测试函数
└── examples/h2_sqd_demo.py   # H2 完整演示
```

> 审查与验证历史（4 轮）见 [`REVIEW.md`](REVIEW.md)。
