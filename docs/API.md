# tc_sqd API 参考

## 模块总览

| 模块 | 文件 | 功能 |
|---|---|---|
| counts | `counts.py` | 比特串矩阵 ↔ 整数互转，TensorCircuit 采样适配 |
| configuration_recovery | `configuration_recovery.py` | 基于平均占据数的配置恢复 |
| subsampling | `subsampling.py` | 批量子采样、汉明权重后选择 |
| fermion | `fermion.py` | CI 矩阵、SQD 对角化、轨道优化、基态能量计算 |
| qubit | `qubit.py` | Pauli 哈密顿量在比特串子空间上的投影与对角化 |

## 比特串约定

每个比特串矩阵 `bitstring_matrix` 是 `np.ndarray`，dtype 为 `bool`，每行为一个采样比特串。

**自旋布局**（与 qiskit-addon-sqd 一致）：
```
[ beta_{norb-1} ... beta_0 | alpha_{norb-1} ... alpha_0 ]
  ^------- 左半 (beta) -----^  ^----- 右半 (alpha) -----^
```

---

## 1. counts 模块

### `bitarray_to_int(bitstring_matrix) -> ndarray`
比特串矩阵 → 整数数组（最左列 = 最高有效位）。

```python
bsm = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=bool)
vals = tc_sqd.bitarray_to_int(bsm)  # array([0, 1, 2, 3])
```

### `int_to_bitarray(int_values, nbits) -> ndarray`
整数 → 比特串矩阵。

```python
tc_sqd.int_to_bitarray([0, 1, 2, 3], 2)  # shape (4, 2)
```

### `counts_dict_to_bitstring_matrix(counts, nbits) -> (bsm, probs)`
TC/qiskit 计数字典 → 比特串矩阵 + 归一化概率。

```python
counts = {3: 100, 12: 50}
bsm, probs = tc_sqd.counts_dict_to_bitstring_matrix(counts, 4)
```

### `sample_from_circuit(circuit, n_samples=1000) -> (bsm, probs)`
从 TensorCircuit 电路采样，返回去重后的比特串矩阵和概率。

```python
c = tc.Circuit(4)
# ... 构建电路 ...
bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=2000)
```

---

## 2. configuration_recovery 模块

### `recover_configurations(bsm, probs, avg_occupancies, num_elec_a, num_elec_b, rand_seed=None) -> (bsm, probs)`
基于平均占据数 n̄ᵢ 修正违反粒子数的比特串，返回去重后的恢复矩阵。

```python
occ_a = np.array([1., 0.])  # alpha 平均占据
occ_b = np.array([1., 0.])  # beta  平均占据
recovered, rec_probs = tc_sqd.recover_configurations(
    bsm, probs, (occ_a, occ_b), 1, 1, rand_seed=42,
)
```

### `postselect_by_hamming_weight(bsm, *, hamming_right, hamming_left) -> mask`
返回满足左右半汉明权重的行索引布尔掩码。

---

## 3. subsampling 模块

### `subsample(bsm, probs, samples_per_batch, num_batches, rand_seed=None) -> list[ndarray]`
按概率无放回子采样，返回多个批次的比特串矩阵列表。

### `postselect_by_hamming_right_and_left(bsm, probs, *, hamming_right, hamming_left) -> (bsm, probs)`
按汉明权重后选择，概率重新归一化。

---

## 4. fermion 模块

### `compute_ground_state_energy(h1e, eri, norb, nelec, *, ecore=0.0, method="fci", ...) -> float`
**统一入口**，从哈密顿量积分计算基态能量。

| method | 说明 | 额外参数 |
|---|---|---|
| `"fci"` | 精确 Full-CI 对角化（基准） | `spin_sq` |
| `"sqd"` | 迭代 SQD，需采样比特串 | `bitstring_matrix`, `probabilities`, `samples_per_batch`, `max_iterations` |
| `"direct"` | 显式构造 CI 矩阵 + numpy 对角化 | 适合小体系 |

```python
# FCI 精确解
e = tc_sqd.compute_ground_state_energy(h1e, eri, 2, (1,1), ecore=ecore, method="fci")

# SQD 采样解
e = tc_sqd.compute_ground_state_energy(
    h1e, eri, 2, (1,1), ecore=ecore, method="sqd",
    bitstring_matrix=bsm, probabilities=probs,
)

# 直接矩阵对角化
e = tc_sqd.compute_ground_state_energy(h1e, eri, 2, (1,1), ecore=ecore, method="direct")
```

### `build_ci_matrix(ci_strs_a, ci_strs_b, h1e, eri, norb, nelec, ecore=0.0) -> ndarray`
Slater-Condon 规则构造 CI 哈密顿矩阵（shape `(na*nb, na*nb)`）。

### `bitstring_matrix_to_ci_strs(bsm, open_shell=False) -> (ci_strs_a, ci_strs_b)`
比特串矩阵 → PySCF CI 字符串元组。

### `solve_sci(ci_strings, h1e, eri, norb, nelec, *, spin_sq=None) -> SCIResult`
在 CI 字符串定义的子空间中对角化哈密顿量。

### `solve_fermion(bitstring_matrix, hcore, eri, *, open_shell=False, spin_sq=None, ...) -> (energy, state, occ, s2)`
从比特串矩阵出发的 SQD 求解。

### `diagonalize_fermionic_hamiltonian(h1e, eri, bit_array, samples_per_batch, norb, nelec, ...) -> SCIResult`
迭代 SQD 循环：恢复 → 采样 → 对角化 → 更新占据数。

### `optimize_orbitals(bsm, hcore, eri, k_flat, *, nelec=None, ...) -> (energy, k_flat, occ)`
轨道优化（梯度下降）。

### `rotate_integrals(hcore, eri, k_flat) -> (hcore_rot, eri_rot)`
应用轨道旋转 U=exp(K) 到积分。

### 数据类

- `SCIState`：SQD 波函数（振幅 + CI 字符串 + norb + nelec）
- `SCIResult`：SQD 结果（energy + sci_state + avg_orb_occupancies + spin_square）

---

## 5. qubit 模块

### `solve_qubit(bsm, hamiltonian, *, verbose=False, **scipy_kwargs) -> (eigenvalues, eigenvectors)`
在比特串子空间中对角化 Pauli 哈密顿量。

```python
hamiltonian = [("ZZI", -1.0), ("IZZ", -1.0), ("XII", -0.5)]
vals, vecs = tc_sqd.solve_qubit(bsm, hamiltonian)
```

### `project_operator_to_subspace(bsm, hamiltonian, *, verbose=False) -> spmatrix`
将 Pauli 哈密顿量投影到子空间，返回稀疏矩阵。

### `matrix_elements_from_pauli(bsm, pauli) -> (amps, rows, cols)`
单个 Pauli 算符在子空间中的稀疏矩阵元。

### `sort_and_remove_duplicates(bsm) -> ndarray`
排序 + 去重（solve_qubit 前置处理）。

---

## 与 qiskit-addon-sqd 的区别

| 特性 | qiskit-addon-sqd | tc_sqd |
|---|---|---|
| numpy 要求 | >= 2.0 | >= 1.17 |
| 量子电路 | qiskit QuantumCircuit | tensorcircuit.Circuit |
| 采样接口 | BitArray (qiskit) | 直接 numpy 数组 |
| CI 矩阵后端 | jax 加速 | PySCF selected_ci |
| jax 依赖 | 必须 | 不需要 |
| 基态能量入口 | 无统一入口 | `compute_ground_state_energy` |

---

## 测试验证

H2/STO-3G 基态能量（所有方法一致）：

```
E(HF)     = -1.11675931
E(FCI)    = -1.13728383
E(SQD)    = -1.13728383  ✓
E(direct) = -1.13728383  ✓
```

TFIM 3-qubit 测试也全部通过。
