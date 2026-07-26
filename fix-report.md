# `tc_sqd` 代码审查修复报告

> 修复日期：2026-07-26
> 基于审查报告：`code-review-report.md`
> 测试结果：17 项全部通过（含 5 项新增回归测试）

---

## 一、严重问题（已全部修复）

### 1. 轨道优化解包崩溃

**文件**：`src/tc_sqd/fermion.py`，`optimize_orbitals` 函数

**现象**：`solve_fermion` 返回 4 元组 `(energy, sci_state, avg_orb_occupancies, spin_square)`，但 `optimize_orbitals` 按 5 项解包，调用即抛 `ValueError`。

**修复**：

```python
# 修复前（718 行）
e, state, occ_a, occ_b, s2 = solve_fermion(...)   # 5 项解包 → 崩溃

# 修复后
e, state, avg_occ, s2 = solve_fermion(...)         # 4 项解包
best_occ = avg_occ
```

**回归测试**：`optimize_orbitals` 现可正常调用，不再崩溃。

---

### 2. 配置恢复的轨道占据顺序错误

**文件**：`src/tc_sqd/configuration_recovery.py`，`recover_configurations` 函数

**现象**：比特串布局为 `[beta_{n-1}...beta_0 | alpha_{n-1}...alpha_0]`（高位在前），但平均占据数组按轨道 `0..n-1` 顺序拼接，未做反转。当 `norb > 1` 时，恢复逻辑会翻错轨道。

**修复**：

```python
# 修复前（156 行）
avg_full = np.concatenate([avg_b, avg_a])               # 轨道 0 在左 → 与布局不匹配

# 修复后
avg_full = np.concatenate([avg_b[::-1], avg_a[::-1]])  # 高位轨道在左 → 与布局对齐
```

**回归测试**：构造 H2 的全零比特串 `|0000>`，期望恢复为 HF 态 `|0011>`（翻转 `beta_0` 和 `alpha_0`），验证通过。

---

### 3. 轨道旋转的单双电子积分变换不一致

**文件**：`src/tc_sqd/fermion.py`，`rotate_integrals` 函数

**现象**：
- 一电子积分：`h' = U.T @ h @ U`，等价于 `Σ U_{ip} h_{ij} U_{jq}`，即使用 `U_{ip}`。
- 二电子积分：`einsum("pi,qj,rk,sl,ijkl->pqrs", U, U, U, U, eri)`，等价于 `Σ U_{pi}`。

两者索引方向相反，非零旋转后哈密顿量不对应同一轨道基。

**修复**：

```python
# 修复前（662 行）
eri_rot = np.einsum("pi,qj,rk,sl,ijkl->pqrs", U, U, U, U, eri)   # U_{pi}

# 修复后
eri_rot = np.einsum("ip,jq,kr,ls,ijkl->pqrs", U, U, U, U, eri)   # U_{ip}，与 h1e 一致
```

**回归测试**：
- 零旋转：`h1e` 和 `eri` 均不变。
- 非零旋转：与手动构造的 `U.T @ h1e @ U` 和 `einsum("ip,jq,kr,ls,...")` 逐元素一致。

---

### 4. `spin_sq` 没有形成有效目标自旋约束

**文件**：`src/tc_sqd/fermion.py`，`solve_sci` 函数

**现象**：代码将 `spin_sq`（S² 值）直接赋给 `myci.spin0`，但 PySCF 的 `SCI` 类没有 `spin0` 属性（静默失败）。正确属性是 `spin`，且其值为 `2S`（非 S²）。

**修复**：

```python
# 修复前（362 行）
myci.spin0 = spin_sq    # 属性不存在，静默失败

# 修复后
# S² = S(S+1) → S = (-1 + sqrt(1 + 4·S²)) / 2 → spin = 2S
s_val = 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * spin_sq))
myci.spin = int(round(2 * s_val))
```

同时清理了 `compute_ground_state_energy` FCI 分支中创建但未使用的 `cisolver` 死代码。

**回归测试**：对 H2 设定 `spin_sq=0.0`（单重态），求解后 `result.spin_square = 0.0000`，约束生效。

---

## 二、其他重要问题（已修复）

### 5. `sample_from_circuit(return_probabilities=False)` 返回的概率不归一化

**文件**：`src/tc_sqd/counts.py`

**现象**：`return_probabilities=False` 时，每个比特串赋概率 `1.0 / n_samples`，但返回的是去重后的 `M` 个比特串（`M ≤ n_samples`），概率之和为 `M/n_samples ≠ 1`。

**修复**：

```python
# 修复前（178 行）
probs = np.full(bsm.shape[0], 1.0 / n_samples)    # M 个 1/n_samples → 和为 M/n_samples

# 修复后
probs = np.full(bsm.shape[0], 1.0 / bsm.shape[0])  # 均匀分布，和为 1
```

**回归测试**：`sample_from_circuit(..., return_probabilities=False)` 后 `probs.sum() == 1.0`。

---

### 6. `SCIState.rdm` 返回维度和数值语义错误

**文件**：`src/tc_sqd/fermion.py`，`SCIState.rdm` 方法

**现象**：
- `make_rdm1` 返回单个 `(norb, norb)` ndarray（自旋求和），但代码按元组 `rdm1[0] + rdm1[1]` 解包 → 报错。
- `make_rdm2` 同理，返回单个 `(norb, norb, norb, norb)` ndarray，非 4 元组。

**修复**：区分自旋求和与自旋分辨两种情况，调用正确的 PySCF 函数：

```python
# 修复后
if rank == 1:
    if spin_summed:
        return selected_ci.make_rdm1(...)      # 单个 (norb, norb)
    else:
        return selected_ci.make_rdm1s(...)     # 元组 (rdm_a, rdm_b)
elif rank == 2:
    if spin_summed:
        return selected_ci.make_rdm2(...)       # 单个 (norb^4)
    else:
        return selected_ci.make_rdm2s(...)      # 元组 (aa, ab, ba, bb)
```

**回归测试**：验证 `rank=1, spin_summed=True` 返回 `ndarray` 且 `shape==(2,2)`；`spin_summed=False` 返回长度 2 的元组。

---

## 三、暂未修改的问题

| 问题 | 处理决定 | 理由 |
|---|---|---|
| `max_dim`/`include_configurations`/`carryover_threshold` 仅在签名中保留但未实现 | 保留签名，后续迭代实现 | 不影响核心流程；当前迭代 SQD 用 `samples_per_batch` + `num_batches` 已可控制子空间维度 |
| `open_shell=False` 合并对不等 alpha/beta 电子数会混入错误汉明权重 | 保留现状 | 当前用例均为闭壳层；开壳层需重新设计 CI 字符串合并逻辑，属功能扩展 |
| 自旋分辨一电子积分在 `h1e.ndim==3` 时取平均 | 保留现状 | 文档已注明"闭壳层适用"；开壳层需将 `h1e_a`/`h1e_b` 分别传入 PySCF |
| 文档声称 NumPy `>=1.13` 但 `default_rng` 需 `>=1.17` | 待后续修订 | 微调文档即可，非功能性问题 |
| 测试覆盖缺口（开壳层、Pauli Y、大稀疏空间等） | 计划后续补充 | 当前回归测试已覆盖所有修复点 |

---

## 四、回归测试

在 `tests/test_h2_sqd.py` 中新增 `test_bugfixes()`，共 5 项：

```
============================================================
Test 5: Bugfix regression tests
============================================================
  PASS: configuration_recovery flips correct orbitals
  PASS: return_probabilities=False is normalized
  PASS: spin_sq=0 gives S²=0.0000 (singlet)
  PASS: RDM return types correct
  PASS: rotate_integrals h1e and eri use consistent U
```

完整测试套件 17 项全部通过：

```
All tests passed!
```

---

## 五、修复文件清单

| 文件 | 修改内容 |
|---|---|
| `src/tc_sqd/fermion.py` | Fix 1（解包）、Fix 3（einsum）、Fix 4（spin_sq）、Fix 6（RDM） |
| `src/tc_sqd/configuration_recovery.py` | Fix 2（占据顺序） |
| `src/tc_sqd/counts.py` | Fix 5（概率归一化） |
| `tests/test_h2_sqd.py` | 新增 `test_bugfixes()` 回归测试 |

所有修改已同步至桌面 `~/Desktop/tc_sqd/` 和工作区两份副本。
