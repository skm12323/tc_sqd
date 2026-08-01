# `solve_sqd` API 文档

`tc_sqd.solve_sqd` 是量子 SQD（Sample-based Quantum Diagonalization）算法的**端到端入口**：
把"积分准备 → 采样 → 配置恢复 → CI 字符串 → 子空间对角化"整套流程打包成一个调用，
并通过 `mode` 参数在**单次运算** (`single`) 与**多次迭代** (`iterative`) 之间切换。

> **与 `compute_ground_state_energy` 的分工**（详见 §9）：
> - `solve_sqd`（本模块）＝ 端到端，接受**电路或位串**、可选迭代，返回 `SCIResult`（含状态/占据）；
> - `compute_ground_state_energy`（`tc_sqd.fermion`）＝ 积分→能量，接受**外部采样的位串**，返回 `float`。
> 快速拿一个能量数字用后者；要状态/迭代/从电路出发用前者。

---

## 0. 比特串布局约定（先读这条）

tc_sqd **全库一致**采用：

```
[ β_{n-1} ... β_0 | α_{n-1} ... α_0 ]
  ^----- 左半 β -----^  ^----- 右半 α -----^
```

- **右半为 α**（自旋向上），**左半为 β**（自旋向下）；
- 每一半内部**轨道降序**：列 `0` = β 最高轨道，列 `norb-1` = β 最低轨道，
  列 `norb` = α 最高轨道，列 `2*norb-1` = α 最低轨道。

参考实现：`fermion.py:42`、`configuration_recovery.py:215-218`、`lucj.py:24-27`。

> ⚠️ 过去版本本文档 §3.1 把布局写反（"前 norb 位为 α"），**会导致静默算错**。已修正。

---

## 1. 函数签名

```python
solve_sqd(
    h1e, eri, norb, nelec, *,
    ecore=0.0,
    bitstring_matrix=None,
    probabilities=None,
    circuit=None,
    n_samples=2000,
    mode="iterative",
    samples_per_batch=None,
    num_batches=1,
    max_iterations=5,
    seed=None,
    rand_seed=None,
    include_configurations=None,
    carryover_threshold=0.0,
    avg_occupancy=None,
    spin_sq=None,
    verbose=False,
    **solver_kwargs,
) -> SCIResult
```

前 4 个为**位置必填参数**，其余均为关键字参数（有合理默认值）。

---

## 2. 必填参数与获取方式

| 参数 | 类型 | 含义 | 一般如何获得 |
|------|------|------|--------------|
| `h1e` | `ndarray (norb, norb)` | 分子轨道（MO）基单电子积分 | `from_pyscf` / RHF + MO 变换 |
| `eri` | `ndarray (norb, norb, norb, norb)` | MO 基双电子积分（化学记号） | 同上 |
| `norb` | `int` | 空间轨道数 | `mol.nao_nr()` |
| `nelec` | `(n_alpha, n_beta)` | 电子数（自旋分辨） | `(mol.nelectron//2, mol.nelectron//2)` |

### 2.1 推荐：用 `from_pyscf` 一键拿到全部（不易错）

```python
from pyscf import gto
import tc_sqd

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
data = tc_sqd.from_pyscf(mol)          # 自动 RHF + MO 基积分 + 核排斥
h1e, eri, ecore = data.h1e, data.eri, data.ecore
norb, nelec = data.norb, data.nelec
```

`from_pyscf`（`molecule.py:115`）内部完成 MO 变换与核排斥，比手写 `einsum` 不易出错。

### 2.2 手写（等价，旧方式）

```python
import numpy as np
from pyscf import gto, scf

mol = gto.Mole(); mol.atom = "H 0 0 0; H 0 0 0.74"; mol.basis = "sto-3g"; mol.build()
mf = scf.RHF(mol).run(verbose=0)
mo = mf.mo_coeff
h1e = mo.T @ mf.get_hcore() @ mo
eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", mol.intor("int2e_sph"), mo, mo, mo, mo)
norb, nelec, ecore = mol.nao_nr(), (mol.nelectron // 2,) * 2, mf.energy_nuc()
```

---

## 3. 采样输入（二选一）

SQD 需要一组"比特串 + 概率"作为子空间候选。两种方式：

### 3.1 直接传入已采样的比特串（推荐用于调参/复用）

```python
from tc_sqd import sample_from_circuit

bsm, probs = sample_from_circuit(circuit, n_samples=3000)   # bsm: bool(S, 2*norb)
result = solve_sqd(h1e, eri, norb, nelec,
                   bitstring_matrix=bsm, probabilities=probs, mode="single")
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `bitstring_matrix` | `ndarray (S, 2*norb)`, `bool` | 每行 `[β_{n-1}..β_0 | α_{n-1}..α_0]`（§0 布局，**右半 α**）|
| `probabilities` | `ndarray (S,)`, 可选 | 对应概率，可非归一化；省略则按**均匀分布**处理 |

### 3.2 直接传入电路（真正的"一个函数"端到端）

```python
result = solve_sqd(h1e, eri, norb, nelec,
                   circuit=circ, n_samples=3000, mode="single")
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `circuit` | `tensorcircuit.Circuit` | 含 `2 * norb` 个量子比特（布局见 §0；`build_lucj_circuit` 已按此布局）|
| `n_samples` | `int`, 默认 `2000` | 仅当使用 `circuit` 时生效，控制采样数 |

### 3.3 电路从哪来

最常用的是 **LUCJ 电路**（用 CCSD 振幅初始化）：

```python
from tc_sqd import build_lucj_circuit

circ = build_lucj_circuit(mf, norb, nelec, ccsd_scale=0.5)   # mf 是上面的 RHF 对象
```

`build_lucj_circuit` 真实签名（`lucj.py:156`）：

```python
build_lucj_circuit(mf, norb, nelec, *, ccsd_scale=1.0,
                   max_excitations=None, angle_multiplier=2.0, theta_list=None)
```

- `mf`：PySCF SCF 对象（内部自动跑 CCSD 取 t2；结果缓存，变分多次调用不重跑）。
- `max_excitations`：只取强度最大的前 K 个 occ-vir 对（控制深度）。
- `theta_list`：变分入口，覆盖 t2 推导角度（配合 `optimize_ansatz_parameters`）。

---

## 4. 控制与迭代参数

| 参数 | 默认 | 作用 | 如何选 |
|------|------|------|--------|
| `mode` | `"iterative"` | `"single"` 恢复+对角化一次；`"iterative"` 反复更新占据数 | 单次足够则选 `single`；欲逼近更好子空间选 `iterative` |
| `ecore` | `0.0` | 核排斥能；总能量 = `result.energy + ecore` | 从 `data.ecore` / `mf.energy_nuc()` 取 |
| `max_iterations` | `5` | 迭代轮数（`iterative` 模式） | 5–10 通常足够；能量不再变化即收敛 |
| `num_batches` | `1` | 子采样批数；`>1` 启用批量子采样 | 内存/采样受限时增大 |
| `samples_per_batch` | `None` | 每批子采样比特串数；`num_batches>1` 时必填且 `>0` | 视采样预算定 |
| `seed` | `None` | 配置恢复（`recover_configurations`）的随机种子 | 固定可复现 |
| `rand_seed` | `None` | 批量子采样（`subsample`）的随机种子 | 固定可复现 |
| `include_configurations` | `None` | 强制纳入子空间的确定性比特串（如 HF determinant / 单双激发配置）| 想确保某行列式必在子空间时 |
| `carryover_threshold` | `0.0` | 批量子采样时不同迭代间子空间的"遗忘"比例（`0`=仅留当前批）| 一般保持 `0.0`；需扩大子空间时调大 |
| `avg_occupancy` | `None` | 初始平均占据 `(occ_a, occ_b)`；省略则退化为 **HF 占据** | 有更好初猜（如上一轮结果）时传入 |
| `spin_sq` | `None` | 目标 ⟨S²⟩ 约束（如单重态传 `0.0`） | 已知自旋态时传入可去壳层污染 |
| `verbose` | `False` | 打印每轮能量/子空间维度 | 调试/观察收敛时开 |

> **`single` 模式** 会忽略 `max_iterations / num_batches / samples_per_batch / carryover_threshold`，
> 只做"恢复一次 + 对角化一次"。

---

## 5. `**solver_kwargs` 透传

多余的关键字参数会原样传递给底层 PySCF SCI 对角化器（`solve_sci` → `kernel_fixed_space`），
常用于控制求根与收敛：

```python
result = solve_sqd(..., which="SA", k=1, tol=1e-10, max_cycle=200)
```

常用键：

- `which="SA"`、`k=1`：求最小本征值（单次 SQD 通常只要基态，默认值已够用）。
- `tol` / `max_cycle`：对角化收敛容差与最大迭代。
- `nroots`：若需要多个根（注意此时返回结构会变化）。

> ⚠️ 不要在此透传 `mode`、`open_shell` 等不属于 PySCF `eig` 的关键字，
> 否则会触发 `davidson1() got an unexpected keyword argument ...` 一类的错误。

---

## 6. 返回值 `SCIResult`

`solve_sqd` 返回与 `solve_sci` 一致的 `SCIResult` 对象：

| 属性/方法 | 含义 |
|-----------|------|
| `result.energy` | **电子**能量（不含核排斥） |
| `result.energy + ecore` | **总**基态能量（与 FCI 比较时用这个） |
| `result.spin_square` | 解出的 ⟨S²⟩（验证自旋纯度） |
| `result.sci_state` | selected-CI 态对象 |
| `result.sci_state.make_rdm1()` | 一阶约化密度矩阵（轨道占据、偶极等） |
| `result.sci_state.make_rdm2()` | 二阶约化密度矩阵 |
| `result.sci_state.orbital_occupancies()` | `(occ_a, occ_b)` 轨道平均占据，可作为下一轮 `avg_occupancy` |

```python
e_tot   = result.energy + ecore
s2      = result.spin_square
rdm1_a, rdm1_b = result.sci_state.make_rdm1()
occ_a, occ_b   = result.sci_state.orbital_occupancies()
```

---

## 7. 典型工作流（端到端示例，H₂ 三步，实跑通过）

```python
import numpy as np
from pyscf import gto
import tensorcircuit as tc
import tc_sqd

tc.set_backend("numpy")

# --- 1) 积分（from_pyscf 一键） ---
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
data = tc_sqd.from_pyscf(mol)
h1e, eri, ecore = data.h1e, data.eri, data.ecore
norb, nelec = data.norb, data.nelec

# --- 2) 电路（LUCJ, CCSD 振幅驱动） ---
circ = tc_sqd.build_lucj_circuit(data.mf, norb, nelec, ccsd_scale=1.0)

# --- 3) 单次运算 ---
res_single = tc_sqd.solve_sqd(h1e, eri, norb, nelec, ecore=ecore,
                              circuit=circ, n_samples=3000, mode="single")
print("E(SQD, single)  =", res_single.energy + ecore)   # -1.13728383 (= FCI)

# --- 4) 多次迭代（verbose 观察收敛） ---
res_iter = tc_sqd.solve_sqd(h1e, eri, norb, nelec, ecore=ecore,
                            circuit=circ, n_samples=3000, mode="iterative",
                            max_iterations=5, seed=42, verbose=True)
print("E(SQD, iter)    =", res_iter.energy + ecore)
print("⟨S²⟩             =", res_iter.spin_square)
```

---

## 8. 单次 vs 多次迭代：怎么选

- **`single`**：计算便宜、确定（不依赖迭代历史），适合流水线打通、协议验证、
  或采样预算极低时。一次恢复+对角化，子空间来自当前比特串。
- **`iterative`**：用上一轮解出的占据数去**修正配置恢复**，从而把采样预算
  集中到"更有物理意义"的行列式上，子空间质量通常更好，更接近 FCI；
  代价是多次对角化 + 重新采样。

经验法则：先用 `single` 确认流程正确并接近 FCI，再切到 `iterative` 追求更高精度；
若 `single` 已足够，则无需迭代。

---

## 9. `solve_sqd` vs `compute_ground_state_energy`

两者职责不同，**互不替代**：

| | `solve_sqd`（本模块） | `compute_ground_state_energy`（`tc_sqd.fermion`） |
|---|---|---|
| 输入 | `circuit` **或** `bitstring_matrix` | `bitstring_matrix`（必须外部提供）|
| 模式 | `single` / `iterative` | `method="fci" / "sqd" / "direct"`（后者只做一次对角化）|
| 返回 | `SCIResult`（含状态/占据/RDM） | `float`（能量） |
| 典型用途 | 端到端、要状态、迭代收敛 | 快速拿能量、对比三种方法 |

```python
# 快速能量（推荐）：
e = tc_sqd.compute_ground_state_energy(h1e, eri, norb, nelec, ecore=ecore,
                                       method="sqd", bitstring_matrix=bsm,
                                       probabilities=probs)

# 端到端（含采样/迭代/状态）：
res = tc_sqd.solve_sqd(h1e, eri, norb, nelec, ecore=ecore,
                       circuit=circ, mode="iterative")
```

两者内部都复用同一套 `recover_configurations → bitstring_matrix_to_ci_strs → solve_sci`，
保证口径一致。
