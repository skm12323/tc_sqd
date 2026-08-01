# tc_sqd

**Sample-based Quantum Diagonalization (SQD) for TensorCircuit**

一个适配 TensorCircuit + numpy 1.x/2.x + PySCF 的轻量级 SQD 包，参考
[`qiskit-addon-sqd`](https://github.com/qiskit/qiskit-addon-sqd) 设计，但**不依赖
numpy≥2 / jax 的硬性要求**。

## 特性

- 统一入口 `compute_ground_state_energy`，支持 `fci` / `sqd` / `direct` 三种方法
- 比特串矩阵 ↔ 整数互转，TensorCircuit 采样适配
- 基于平均占据数的配置恢复（纠正噪声导致的粒子数违例）+
  **T1 感知恢复**（`estimate_true_occupancies`：从观测位串反卷积真实平均占据，
  喂回 recover / `initial_occupancies`；per-qubit γ 不均匀时 RMSE 降 ~30%）
- 批量子采样、汉明权重后选择、**`max_dim` 子空间维度限制**（int / (na, nb)）
- CI 矩阵构造（Slater–Condon）、子空间对角化、迭代 SQD、轨道优化
- CCSD 振幅驱动的 LUCJ ansatz 电路构造（量子态制备侧）+
  **真机深度预算报告**（`circuit_stats` / `lucj_report`：1Q/2Q 门统计，2Q 门数作保守深度代理）
- Pauli 哈密顿量在比特串子空间的投影与对角化（非费米子问题，如 QAOA-MaxCut）
- **激发态**：`solve_sci(..., n_roots=k)` 取前 k 个本征值（基态 + 低激发态）
- **密度矩阵噪声模拟**（`noise`）：退相干/振幅阻尼/去极化 Kraus 通道，cupy GPU 可选
- **噪声容限预测器**（`predict`）：输入 T₁/电路/shots → 预测 SQD 基态/激发态精度；
  `depth_budget` 结构化深度预算；`plan_sampling` 自动找最优 (shots, depth) 采样方案
- **一键分子接口**（`molecule`）：`from_pyscf(mol_or_mf)` 自动算 MO 基
  h1e/eri/ecore/norb/nelec，支持活性空间（冻结 core，含 core 平均场修正）
- **采样诊断**（`diagnostics`）：采样熵 / 子空间维度 / 配置分布 / 能量随 shots 收敛曲线
- **真机一站式**（`hardware`）：腾讯 qcloud 校准加载 / 选最优 qubit 子图 / 真机采样 / SQD 后处理

## 安装

支持 Python **3.10–3.12**（实测 3.10.20；3.11/3.12 下 tensorcircuit/pyscf/numpy/scipy 均有 wheel）。

```bash
conda create -n tc python=3.10   # 3.10 / 3.11 / 3.12 均可
conda activate tc
pip install -e .          # editable 安装, 之后任何目录都可 import tc_sqd
```

`pip install -e .` 以 editable 模式安装 tc_sqd（源码改动即时生效，无需 PYTHONPATH）。
也可仅装依赖：`pip install -r requirements.txt`，但运行范例时需手动设置
`PYTHONPATH=/mnt/d/tc_sqd/src`。

依赖：`tensorcircuit==0.12.0`、`numpy>=1.17`、`scipy>=1.10`、`pyscf>=2.0`。

> **numpy 版本（tensorcircuit 0.12 兼容）**：tensorcircuit 0.12.0 用了 numpy 在 2.x
> 中搬走/移除的 `np.ComplexWarning` 与 `np.reshape(newshape=)`。两条路径：
>
> - **路径 A（省心）**：固定 `numpy<2.0`（如 1.26.4）+ `scipy<1.14`，无需任何 patch。
> - **路径 B（用 numpy 2.x，例如要与 Vayesta 等倾向 numpy 2 的库共存）**：装标准
>   numpy 2.x，再启用 tc_sqd 的兼容补丁。最省事的是写入 sitecustomize（一劳永逸）：
>   ```bash
>   python -m tc_sqd._compat install
>   ```
>   之后该环境任何脚本 `import tensorcircuit` 前都自动 patch。也可在脚本里
>   `import tc_sqd`（导入即 patch）后再 `import tensorcircuit`。

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

## LUCJ ansatz（CCSD 振幅驱动）

从 PySCF CCSD 双激发振幅 t2 构造 LUCJ 电路，采样后交给 SQD（替代上文的手写纠缠电路）：

```python
c = tc_sqd.build_lucj_circuit(mf, norb, nelec, ccsd_scale=1.0)
bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=3000)
e = tc_sqd.compute_ground_state_energy(
    h1e, eri, norb, nelec, ecore=ecore, method="sqd",
    bitstring_matrix=bsm, probabilities=probs)
# H2: e = -1.13728383 (= FCI)；LiH: 误差 ~7.5e-4 vs FCI
```

关键：必须由 t2（而非 t1）驱动 —— H2/STO-3G 的 t1≈0（Brillouin 定理），相关能几乎全来自 t2 双激发。

## API 速查

| 模块 | 函数 | 作用 |
|---|---|---|
| counts | `bitarray_to_int(bsm)` | 比特串矩阵 → 整数数组 |
| counts | `int_to_bitarray(vals, nbits)` | 整数 → 比特串矩阵 |
| counts | `counts_dict_to_bitstring_matrix(counts, nbits)` | 计数字典 → (比特串矩阵, 概率)，等价键自动合并 |
| counts | `sample_from_circuit(circuit, n_samples)` | 从 TC 电路采样 → (比特串矩阵, 概率) |
| configuration_recovery | `recover_configurations(bsm, probs, avg_occ, na, nb)` | 基于平均占据数的配置恢复 |
| configuration_recovery | `estimate_true_occupancies(bsm, na, nb, t1_gamma)` | T1 反卷积估计真实平均占据（per-qubit γ；喂 recover / initial_occupancies）|
| configuration_recovery | `postselect_by_hamming_weight(bsm, *, hamming_right, hamming_left)` | 按汉明权重筛选 |
| subsampling | `subsample(bsm, probs, samples_per_batch, num_batches)` | 按概率无放回批量子采样 |
| subsampling | `postselect_by_hamming_right_and_left(bsm, probs, ...)` | 汉明权重后选择 + 重归一化 |
| subsampling | `limit_subspace(bsm, max_dim, norb, *, probabilities)` | 按概率裁剪子空间（int=总行列式数 / tuple=(na, nb)）|
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
| lucj | `get_ccsd_amplitudes(mf)` | 跑 RHF-CCSD，返回 (t1, t2, mycc) |
| lucj | `build_lucj_circuit(mf, norb, nelec, *, ccsd_scale)` | 从 CCSD t2 构造简化 LUCJ 电路（HF + 占据-空 Givens） |
| lucj | `circuit_stats(circuit)` | 门统计：n_1q / n_2q / n_multi / n_gates |
| lucj | `lucj_report(mf, norb, nelec, *, max_excitations, max_depth)` | 真机深度预算：2Q 门数代理 / within_budget / max_entries |
| fermion | `solve_sci(..., n_roots=k)` | 激发态：n_roots>1 返回前 k 个本征态 list[SCIResult] |
| noise | `statevector_to_density(psi)` | 纯态 → 密度矩阵 ρ=\|ψ⟩⟨ψ\| |
| noise | `apply_dephasing(rho, p, nq)` / `apply_amp_damping(rho, γ, nq)` / `apply_depolarizing(rho, p, nq)` | 退相干(T₂)/振幅阻尼(T₁)/去极化 Kraus 通道（gpu=True 走 cupy）|
| noise | `density_to_bitstring_matrix(diag, norb, n_samples)` | 密度矩阵 diag → 采样 bsm（接 recover_configurations）|
| predict | `gamma_T1(depth, t_gate_ns, T1_us)` | 真机振幅阻尼率 γ = 1−exp(−depth·t_gate/T₁) |
| predict | `predict_sqd_error(T1, depth, t_gate, shots, n_excited)` | 预测 SQD 基态/激发态误差（退相干免疫，T₁ 主导）|
| predict | `depth_budget(T1, t_gate, shots, target, excited)` | 结构化深度预算（`DepthBudget`：max_depth/status/reason）|
| predict | `max_depth_for_accuracy(T1, t_gate, shots, target, excited)` | 反向预测达目标精度的 depth 上限（int 薄封装）|
| predict | `plan_sampling(T1, t_gate, *, target, excited, ...)` | 采样预算分配：枚举 (shots, depth) 网格，按成本排序可行方案 |
| molecule | `from_pyscf(mf_or_mol, *, n_active)` | 一键构建 SQD 输入（MO 积分 + 核能 + 电子数，活性空间冻结 core）|
| molecule | `MolecularData.solve(method, ...)` | 一键求基态能量（fci/sqd/direct）|
| diagnostics | `sampling_report(h1e, eri, norb, nelec, bsm, ...)` | 采样质量综合报告（熵/维度/配置/收敛曲线）|
| diagnostics | `energy_convergence(...)` | 能量随 shots 收敛曲线 |
| diagnostics | `shannon_entropy(probs)` / `subspace_dimension(bsm)` | 采样熵 / 子空间维度 |
| hardware | `load_calibration(device_name)` | 从 tc qcloud 读校准快照（T₁/T₂/读出/CZ/拓扑）|
| hardware | `select_qubits(calibration, nq)` | 多起点贪心选最优 nq 物理 qubit 子图（min T₂ 最大化）|
| hardware | `bitstring_matrix_to_energy(bsm, h1e, eri, norb, nelec, ecore)` | 采样 bsm → recover → 子空间对角化 → 能量 |
| hardware | `sample_on_hw(device, circuit, physical_qubits, ...)` | 真机采样（编译+submit_task+REM+字节序自校准）|

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
PYTHONPATH=src python -m tests.test_noise        # noise 模块 8 个测试
PYTHONPATH=src python -m tests.test_predict      # predict 模块 7 个测试
PYTHONPATH=src python -m tests.test_molecule     # molecule 模块 5 个测试
PYTHONPATH=src python -m tests.test_diagnostics  # diagnostics 模块 4 个测试
PYTHONPATH=src python -m tests.test_lucj         # lucj 模块 4 个测试
PYTHONPATH=src python -m tests.test_subsampling  # subsampling 模块 5 个测试
PYTHONPATH=src python -m tests.test_t1_recovery  # T1 感知恢复 3 个测试
PYTHONPATH=src python examples/h2_sqd_demo.py    # H2 完整演示
```

## 限制与已知边界

- **闭壳层**：仅可靠支持 `n_alpha == n_beta`；`open_shell=False` 下电子数不等会显式报错。
- **一电子积分**：需为单个 `(norb, norb)`（或两块相同的 `(2, norb, norb)`）；`h_alpha ≠ h_beta` 的自旋分辨积分会显式拒绝。
- **`max_dim`**：已实现（`limit_subspace` 按概率贪心裁剪；int=总行列式数、tuple=(na, nb)）。`include_configurations` / carryover 强制配置不受裁剪。
- **`spin_sq`**：在 `solve_sci` / `fci` 路径通过多根 S² 匹配实现真正的目标自旋选态（不可达时 raise）；`sqd` / `direct` 路径显式拒绝。
- **状态持久化**：`SCIState.save/load` 后通过 `_as_scivector` 重建 PySCF `SCIvector` 元数据，`rdm` / `spin_square` 在加载后仍可用。
- **LUCJ**：`build_lucj_circuit` 为简化实现（t2 范数驱动 Givens，未做 ffsim 的精确 SVD + 对角 Coulomb Jastrow）；仅支持闭壳层。H₂ 精确复现 FCI，LiH 误差 ~7.5e-4。
- **`optimize_orbitals`**：基于 scipy Nelder-Mead 无导数优化（旧版数值梯度每梯度分量一次 SQD 对角化，实际不可用）。`learning_rate` 仅保留兼容，不再使用。
- **`predict` 校准常数**：KS/KT1 来自 H₄/STO-3G 拟合，跨体系只作数量级参考；`plan_sampling` / `depth_budget` 的误差界在同一近似下成立。
- **`from_pyscf` 冻结 core**：frozen-core 近似冻结 core-valence 关联（~2e-4 Ha 量级，对 LiH），活性 FCI 与"core 严格双占据受限对角化"精确一致。

## 目录结构

```
tc_sqd/
├── README.md                 # 本文件
├── REVIEW.md                 # 代码审查与验证历史
├── requirements.txt
├── src/tc_sqd/               # counts, configuration_recovery, subsampling, fermion, qubit, lucj, noise, predict, hardware, molecule, diagnostics, _compat
├── tests/                    # test_h2_sqd, test_noise, test_predict, test_molecule, test_diagnostics, test_lucj, test_subsampling, test_t1_recovery
└── examples/h2_sqd_demo.py   # H2 完整演示
```

> 审查与验证历史（4 轮）见 [`REVIEW.md`](REVIEW.md)。
