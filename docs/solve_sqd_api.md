# `solve_sqd` API 文档

`tc_sqd.solve_sqd` 是量子 SQD（Sample-based Quantum Diagonalization）算法的**单一入口函数**：
把"积分准备 → 采样 → 配置恢复 → CI 字符串 → 子空间对角化"整套流程打包成一个调用，
并通过 `mode` 参数在 **单次运算** (`single`) 与 **多次迭代** (`iterative`) 之间切换。

```python
from tc_sqd import solve_sqd

result = solve_sqd(h1e, eri, norb, nelec, ecore=ecore, bitstring_matrix=bsm,
                   probabilities=probs, mode="single")
e_total = result.energy + ecore      # 总能量（含核排斥）
```

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
| `h1e` | `ndarray (norb, norb)` | 分子轨道（MO）基单电子积分 | PySCF RHF + MO 系数变换 |
| `eri` | `ndarray (norb, norb, norb, norb)` | MO 基双电子积分（化学记号） | 同上 |
| `norb` | `int` | 空间轨道数 | `mol.nao_nr()` |
| `nelec` | `(n_alpha, n_beta)` | 电子数（自旋分辨） | `(mol.nelectron//2, mol.nelectron//2)` |

### 2.1 从 PySCF 得到 `h1e / eri / norb / nelec / ecore`

这是最常见的来源。约定与 `examples/h2_sqd_demo.py`、`tests/test_h2_sqd.py` 完全一致：

```python
from pyscf import gto, scf

mol = gto.Mole()
mol.atom = "H 0 0 0; H 0 0 0.74"
mol.basis = "sto-3g"
mol.build()

mf = scf.RHF(mol).run(verbose=0)        # 先做 HF 得到 MO 系数
mo = mf.mo_coeff

# 单电子积分：h1e = Cᵀ · h_core · C
h1e = mo.T @ mf.get_hcore() @ mo

# 双电子积分：eri(ij|kl) = Σ Cᵢₚ Cⱼ_q Cₖᵣ Cₗₛ (pq|rs)
eri_ao = mol.intor("int2e")             # 或 "int2e_sph"
eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, mo, mo, mo, mo)

norb  = mol.nao_nr()                    # 空间轨道数
nelec = (mol.nelectron // 2, mol.nelectron // 2)   # 闭壳层示例
ecore = mf.energy_nuc()                 # 核排斥能（可选，默认 0.0）
```

> **旋转后的轨道（推荐）**：SQD 在"自然轨道"或某种旋转基上表现更好。可先对角化 `h1e`
> 得到旋转矩阵 `U`，再令 `h1e_rot = Uᵀ h1e U`、`eri_rot = einsum(..., U)`，
> 并把同一 `U` 应用到电路里的轨道基。具体旋转方式由你自己决定，
> `solve_sqd` 只认最终的 `h1e / eri`。

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
| `bitstring_matrix` | `ndarray (S, 2*norb)`, `bool` | 每一行是 `[α₀…α_{n-1}, β₀…β_{n-1}]` 的占据比特串 |
| `probabilities` | `ndarray (S,)`, 可选 | 对应概率，可非归一化；省略则按**均匀分布**处理 |

比特串的顺序必须与 `norb` 一致：前 `norb` 位为 α 自旋，后 `norb` 位为 β 自旋。

### 3.2 直接传入电路（真正的"一个函数"端到端）

若提供 `circuit`（TensorCircuit 电路），`solve_sqd` 会在内部调用
`sample_from_circuit` 完成采样，无需你提前采样：

```python
result = solve_sqd(h1e, eri, norb, nelec,
                   circuit=circ, n_samples=3000, mode="single")
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `circuit` | `tensorcircuit.Circuit` | 含 `2 * norb` 个量子比特的电路（前 `norb` 为 α，后 `norb` 为 β） |
| `n_samples` | `int`, 默认 `2000` | 仅当使用 `circuit` 时生效，控制采样数 |

### 3.3 电路从哪来

最常用的是 **LUCJ 电路**（用 CCSD 振幅初始化）：

```python
from tc_sqd import get_ccsd_amplitudes, build_lucj_circuit

t1, t2, _ = get_ccsd_amplitudes(mf)             # mf 是上面的 RHF 对象
circ = build_lucj_circuit(norb, nelec, t1, t2, nlayers=1)
```

更轻量的做法是手写一个 HF 初态 + 纠缠门电路（见 `examples/h2_sqd_demo.py`，
也用于 `tests.py`），适合快速验证 `solve_sqd` 本身。

---

## 4. 控制与迭代参数

| 参数 | 默认 | 作用 | 如何选 |
|------|------|------|--------|
| `mode` | `"iterative"` | `"single"` 恢复+对角化一次；`"iterative"` 反复更新占据数 | 单次足够则选 `single`；欲逼近更好子空间选 `iterative` |
| `ecore` | `0.0` | 核排斥能；总能量 = `result.energy + ecore` | 从 `mf.energy_nuc()` 取 |
| `max_iterations` | `5` | 迭代轮数（`iterative` 模式） | 5–10 通常足够；能量不再变化即收敛 |
| `num_batches` | `1` | 子采样批数；`>1` 启用批量子采样 | 内存/采样受限时增大 |
| `samples_per_batch` | `None` | 每批子采样比特串数；`num_batches>1` 时必填且 `>0` | 视采样预算定 |
| `seed` | `None` | 配置恢复（`recover_configurations`）的随机种子 | 固定可复现 |
| `rand_seed` | `None` | 批量子采样（`subsample`）的随机种子 | 固定可复现 |
| `include_configurations` | `None` | 强制纳入子空间的确定性比特串（如 HF determinant） | 想确保某行列式必在子空间时 |
| `carryover_threshold` | `0.0` | 批量子采样时不同迭代间子空间的"遗忘"比例（`0`=仅留当前批，`1`=保留全部历史） | 一般保持 `0.0`；需扩大子空间时调大 |
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
| `result.sci_state` | PySCF 的 selected-CI 态对象 |
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

## 7. 典型工作流（端到端示例）

```python
import numpy as np
from pyscf import gto, scf
import tensorcircuit as tc
from tc_sqd import (solve_sqd, get_ccsd_amplitudes, build_lucj_circuit)

tc.set_backend("numpy")

# --- 1) 积分（PySCF RHF） ---
mol = gto.Mole(); mol.atom = "H 0 0 0; H 0 0 0.74"; mol.basis = "sto-3g"; mol.build()
mf = scf.RHF(mol).run(verbose=0)
mo = mf.mo_coeff
h1e = mo.T @ mf.get_hcore() @ mo
eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", mol.intor("int2e"), mo, mo, mo, mo)
norb, nelec, ecore = mol.nao_nr(), (mol.nelectron//2,)*2, mf.energy_nuc()

# --- 2) 电路（LUCJ） ---
t1, t2, _ = get_ccsd_amplitudes(mf)
circ = build_lucj_circuit(norb, nelec, t1, t2, nlayers=1)

# --- 3) 单次运算 ---
res_single = solve_sqd(h1e, eri, norb, nelec, ecore=ecore,
                       circuit=circ, n_samples=3000, mode="single")

# --- 4) 多次迭代（verbose 观察收敛） ---
res_iter = solve_sqd(h1e, eri, norb, nelec, ecore=ecore,
                     circuit=circ, n_samples=3000, mode="iterative",
                     max_iterations=5, seed=42, verbose=True)

print("E(SQD)   =", res_iter.energy + ecore)
print("⟨S²⟩     =", res_iter.spin_square)
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
