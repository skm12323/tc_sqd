# tc_sqd

**Sample-based Quantum Diagonalization for TensorCircuit**

A lightweight SQD package adapted for TensorCircuit + numpy 1.x/2.x + PySCF, inspired by [`qiskit-addon-sqd`](https://github.com/qiskit/qiskit-addon-sqd) but free of the numpy>=2 / jax hard dependencies.

## 安装

```bash
conda create -n tc python=3.10
conda activate tc
pip install tensorcircuit numpy pyscf scipy
```

## 从 tex 文档中识别的 SQD 函数

阅读 `main-polish.tex` 后，确认以下 SQD 相关函数被使用：

| tex 中的引用 | tc_sqd 中的函数 | 说明 |
|---|---|---|
| `recover_configurations(bitstrings, nelec, nq)` | `tc_sqd.recover_configurations` | 基于平均占据数的配置恢复 |
| `build_ci_matrix(basis, h1e, eri, ecore)` | `tc_sqd.build_ci_matrix` | Slater-Condon 规则构建 CI 矩阵 |
| `c.sample(batch=1000, format="count_dict_int")` | `tc_sqd.sample_from_circuit` | TC 电路采样适配器 |
| 配置恢复 (平均占据数 n̄_i) | `tc_sqd.recover_configurations` | 翻转违反粒子数的比特 |
| 子空间对角化 | `tc_sqd.solve_sci` / `tc_sqd.solve_fermion` | CI 子空间精确对角化 |
| 迭代 SQD 循环 | `tc_sqd.diagonalize_fermionic_hamiltonian` | 恢复→采样→对角化→更新 |
| LUCJ 采样 | TC 电路直接构建 | 通过 `tc.Circuit` 构建电路 |
| 能量重构 | 用户公式 `E_MF + Σ(E_f - E_MF,f)` | 在示例中展示 |

## API 概览

### counts 模块
```python
from tc_sqd import bitarray_to_int, int_to_bitarray, counts_dict_to_bitstring_matrix, sample_from_circuit
```
- `bitarray_to_int(bsm)` — 比特串矩阵 → 整数数组
- `int_to_bitarray(vals, nbits)` — 整数 → 比特串矩阵
- `counts_dict_to_bitstring_matrix(counts, nbits)` — 计数字典 → (比特串矩阵, 概率)
- `sample_from_circuit(circuit, n_samples)` — 从 TC 电路采样 → (比特串矩阵, 概率)

### configuration_recovery 模块
```python
from tc_sqd import recover_configurations, postselect_by_hamming_weight
```
- `recover_configurations(bsm, probs, avg_occ, na, nb)` — 配置恢复
- `postselect_by_hamming_weight(bsm, hamming_right, hamming_left)` — 汉明权重后选择

### subsampling 模块
```python
from tc_sqd import subsample, postselect_by_hamming_right_and_left
```

### fermion 模块
```python
from tc_sqd import (
    bitstring_matrix_to_ci_strs, build_ci_matrix,
    solve_sci, solve_fermion, diagonalize_fermionic_hamiltonian,
    optimize_orbitals, rotate_integrals,
    SCIState, SCIResult,
)
```
- `bitstring_matrix_to_ci_strs(bsm)` — 比特串 → CI 字符串
- `build_ci_matrix(ci_strs_a, ci_strs_b, h1e, eri, norb, nelec, ecore)` — 构建 CI 矩阵
- `solve_sci(ci_strs, h1e, eri, norb, nelec)` — 子空间对角化
- `solve_fermion(bsm, hcore, eri)` — SQD 求解
- `diagonalize_fermionic_hamiltonian(h1e, eri, bsm, ...)` — 迭代 SQD 循环

### qubit 模块
```python
from tc_sqd import solve_qubit, project_operator_to_subspace, sort_and_remove_duplicates
```

## 快速示例

```python
import numpy as np
import tensorcircuit as tc
from pyscf import gto, scf, fci
import tc_sqd

# PySCF: H2 + RHF
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
mf = scf.RHF(mol).run()
mo = mf.mo_coeff
h1e = mo.T @ mf.get_hcore() @ mo
eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", mol.intor("int2e_sph"), mo, mo, mo, mo)
ecore = mf.energy_nuc()
norb, nelec = mol.nao_nr(), (mol.nelectron // 2,) * 2

# TC: 采样
c = tc.Circuit(2 * norb)
c.x(0); c.x(norb)
c.ry(0, theta=0.8); c.cnot(0, 1); c.ry(0, theta=-0.8)
c.ry(norb, theta=0.8); c.cnot(norb, norb + 1); c.ry(norb, theta=-0.8)
bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=2000)

# SQD: 迭代求解
result = tc_sqd.diagonalize_fermionic_hamiltonian(
    h1e, eri, (bsm, probs), samples_per_batch=200,
    norb=norb, nelec=nelec, max_iterations=5, seed=42,
)
print(f"E(SQD) = {result.energy + ecore:.8f}")
print(f"E(FCI) = {fci.FCI(mf).run().e_tot:.8f}")
```

## 运行测试

```bash
PYTHONPATH=src python -m tests.test_h2_sqd
```

## 与 qiskit-addon-sqd 的区别

| 特性 | qiskit-addon-sqd | tc_sqd |
|---|---|---|
| numpy 要求 | >= 2.0 | >= 1.17 |
| 量子电路 | qiskit QuantumCircuit | tensorcircuit.Circuit |
| 采样接口 | BitArray (qiskit) | 直接 numpy 数组 |
| CI 矩阵 | 内部 (jax 加速) | PySCF selected_ci |
| jax 依赖 | 必须 | 不需要 |
