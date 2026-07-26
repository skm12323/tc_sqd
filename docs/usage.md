# tc_sqd 简要用法介绍

## 1. 环境准备

```bash
conda create -n tc python=3.10
conda activate tc
pip install tensorcircuit numpy pyscf scipy
```

## 2. 三步完成 SQD 计算

```python
import numpy as np
import tensorcircuit as tc
from pyscf import gto, scf, fci
import tc_sqd

# ── Step 1: PySCF 构建分子哈密顿量 ──
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
mf  = scf.RHF(mol).run()
mo  = mf.mo_coeff
h1e  = mo.T @ mf.get_hcore() @ mo
eri  = np.einsum("pqrs,pi,qj,rk,sl->ijkl", mol.intor("int2e_sph"), mo, mo, mo, mo)
ecore = mf.energy_nuc()
norb, nelec = mol.nao_nr(), (mol.nelectron // 2, mol.nelectron // 2)

# ── Step 2: TensorCircuit 采样比特串 ──
c = tc.Circuit(2 * norb)
c.x(0); c.x(norb)              # HF 初态
c.ry(0, theta=0.8); c.cnot(0, 1); c.ry(0, theta=-0.8)   # 纠缠门
c.ry(norb, theta=0.8); c.cnot(norb, norb+1); c.ry(norb, theta=-0.8)
bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=2000)

# ── Step 3: SQD 求解基态能量 ──
e = tc_sqd.compute_ground_state_energy(
    h1e, eri, norb, nelec, ecore=ecore, method="sqd",
    bitstring_matrix=bsm, probabilities=probs,
)
print(f"E(SQD) = {e:.8f}")
```

输出：
```
E(SQD) = -1.13728383
```

## 3. 三种求解方法

`compute_ground_state_energy` 支持三种模式，输入相同、一键切换：

```python
e1 = tc_sqd.compute_ground_state_energy(h1e, eri, norb, nelec, ecore=ecore, method="fci")
e2 = tc_sqd.compute_ground_state_energy(h1e, eri, norb, nelec, ecore=ecore, method="direct")
e3 = tc_sqd.compute_ground_state_energy(
    h1e, eri, norb, nelec, ecore=ecore, method="sqd",
    bitstring_matrix=bsm, probabilities=probs,
)
```

| method | 用途 | 特点 |
|---|---|---|
| `"fci"` | 精确基准 | 枚举全部行列式，PySCF Davidson 对角化 |
| `"direct"` | 小体系 / 教学 | 显式构造 CI 矩阵 + `numpy.linalg.eigvalsh` |
| `"sqd"` | 量子采样 | 迭代 SQD：恢复→采样→对角化→更新占据数 |

## 4. 分步使用（进阶）

需要更细粒度控制时，可单独调用各环节：

```python
# 配置恢复：修正违反粒子数的比特串
occ_a = np.zeros(norb); occ_a[:nelec[0]] = 1.0
occ_b = np.zeros(norb); occ_b[:nelec[1]] = 1.0
recovered, rec_probs = tc_sqd.recover_configurations(
    bsm, probs, (occ_a, occ_b), nelec[0], nelec[1], rand_seed=42,
)

# 转换为 CI 字符串
ci_strs_a, ci_strs_b = tc_sqd.bitstring_matrix_to_ci_strs(recovered)

# 子空间对角化
result = tc_sqd.solve_sci(
    (ci_strs_a, ci_strs_b), h1e, eri, norb, nelec,
)
print(f"E = {result.energy + ecore:.8f}, S² = {result.spin_square:.4f}")

# 显式构造 CI 矩阵（可导出 / 检查）
H = tc_sqd.build_ci_matrix(ci_strs_a, ci_strs_b, h1e, eri, norb, nelec, ecore=ecore)
```

## 5. 非费米子问题（Qubit 哈密顿量）

适用于 QAOA-MaxCut 等 Pauli 哈密顿量问题：

```python
hamiltonian = [("ZZI", -1.0), ("IZZ", -1.0), ("XII", -0.5), ("IXI", -0.5)]
vals, vecs = tc_sqd.solve_qubit(bsm, hamiltonian)
print(f"Ground energy = {vals[0]:.6f}")
```

## 6. 关键 API 速查

| 函数 | 作用 |
|---|---|
| `sample_from_circuit(circuit, n_samples)` | 从 TC 电路采样比特串 |
| `recover_configurations(bsm, probs, avg_occ, na, nb)` | 配置恢复 |
| `bitstring_matrix_to_ci_strs(bsm)` | 比特串 → CI 行列式 |
| `build_ci_matrix(...)` | 构造显式 CI 矩阵 |
| `solve_sci(ci_strs, h1e, eri, norb, nelec)` | 子空间对角化 |
| `diagonalize_fermionic_hamiltonian(...)` | 迭代 SQD 循环 |
| `compute_ground_state_energy(...)` | **统一入口**，支持 fci / sqd / direct |
| `solve_qubit(bsm, hamiltonian)` | Pauli 哈密顿量子空间求解 |

## 7. 运行测试

```bash
cd tc_sqd            # 进入项目根目录
PYTHONPATH=src python -m tests.test_h2_sqd
```

预期输出：
```
E(HF)  = -1.11675931
E(FCI) = -1.13728383
E(SQD) = -1.13728383    ✓ PASS
All tests passed!
```
