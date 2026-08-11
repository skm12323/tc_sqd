"""tail_sampling --- C1 尾部发现采样 (SQD-AA 经典模拟版, round_001).

在**采样端**对已进入子空间的 determinant 做抑制 (拒绝/重加权), 迫使采样器每轮
贡献**新** determinant, 延缓 "每 shot 新 det 数随 shots 趋零" 的 coupon-collector
+ 振幅衰减饱和 (theory.md §1.1/§1.2)。

**与 distill (:func:`tc_sqd.cipsi.eigenvector_importance_sample`) 的硬边界**:
distill 按解态 ``|c|^(2/T)`` **重采、学分布** (12,12 已证伪: 远未收敛 |Ψ⟩ → 学错
分布); 本模块**不学任何分布** —— 代码路径**绝不接受/读取 ``c2d``**, 只读子空间
字符串集合 ``(seen_a, seen_b)``, 对已见 determinant 做硬掩蔽。无 "学错" 路径
(theory.md §0/§"给 R3 的关键叮嘱"#5)。

**抑制判据** (theory.md §1.4 关键观察②): tc_sqd 子空间 = ``str_a × str_b`` (笛卡尔
积; 闭壳层 ``str_a == str_b``)。一位串恢复得 ``(α, β)``:
  - ``α ∈ seen_a`` **且** ``β ∈ seen_b`` → 对乘积子空间零新增 → **抑制**;
  - 否则 (α 或 β 新) → **保留** (至少扩了一个字符串集合)。
键空间 = CI 字符串整数 (轨道 0 = LSB), 与
:func:`tc_sqd.fermion.bitstring_matrix_to_ci_strs` 一致 (本模块自备逐行 packing,
**不修改 fermion.py**)。

参考: SQD-AA (arXiv:2605.02565) amplitude amplification 的经典模拟版。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .configuration_recovery import recover_configurations

__all__ = ["suppress_seen_bitstrings", "discover_tail_pool"]


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _bsm_to_ci_strs_per_row(
    bsm: np.ndarray, norb: int
) -> Tuple[np.ndarray, np.ndarray]:
    """逐行给出 ``(α_str, β_str)`` 整数 (轨道 0 = LSB), 保留行对应关系。

    与 :func:`tc_sqd.fermion.bitstring_matrix_to_ci_strs` 同位 packing, 但**不去重
    /不合并**, 返回形状 ``(S,)`` 的 ``int64`` 数组, 与输入行一一对应 (抑制判据需逐行
    判定)。bitstring 布局: ``[β_{n-1}..β_0 | α_{n-1}..α_0]`` (左 β 右 α)。
    """
    alpha_bits = bsm[:, norb:]            # 右半 = α, [orb n-1 .. orb 0] 左→右
    beta_bits = bsm[:, :norb]             # 左半 = β
    powers = (1 << np.arange(norb, dtype=np.uint64)).astype(np.int64)
    a_strs = (alpha_bits[:, ::-1].astype(np.int64) @ powers).ravel()
    b_strs = (beta_bits[:, ::-1].astype(np.int64) @ powers).ravel()
    return a_strs, b_strs


def _validate_seen(name: str, seen) -> set:
    """校验 seen_a/seen_b 为 int 的 set/frozenset, 返回 python int 的 set。"""
    if not isinstance(seen, (set, frozenset)):
        raise TypeError(
            f"{name} 必须是 int 的 set/frozenset, got {type(seen).__name__}."
        )
    out: set = set()
    for x in seen:
        if isinstance(x, (bool,)):                       # bool 是 int 子类, 排除
            raise TypeError(f"{name} 的元素须为 int, got bool.")
        if not isinstance(x, (int, np.integer)):
            raise TypeError(
                f"{name} 的元素须为 int, got {type(x).__name__}."
            )
        out.add(int(x))
    return out


def _validate_avg_occ(
    avg_occupancies, norb: int
) -> Tuple[np.ndarray, np.ndarray]:
    """校验 avg_occupancies (形状/有限/范围), 返回 (occ_a, occ_b) float64 数组。"""
    if not isinstance(avg_occupancies, tuple) or len(avg_occupancies) != 2:
        raise ValueError("avg_occupancies 必须是长度 2 的 tuple (occ_a, occ_b).")
    occ_a, occ_b = avg_occupancies
    occ_a = np.asarray(occ_a, dtype=np.float64)
    occ_b = np.asarray(occ_b, dtype=np.float64)
    if occ_a.ndim != 1 or occ_b.ndim != 1:
        raise ValueError("avg_occupancies entries must be 1-D.")
    if occ_a.shape[0] != norb or occ_b.shape[0] != norb:
        raise ValueError(
            f"avg_occupancies entries must have length norb={norb}, "
            f"got {occ_a.shape[0]} and {occ_b.shape[0]}."
        )
    for name, arr in (("avg_occupancies[0]", occ_a), ("avg_occupancies[1]", occ_b)):
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError(f"{name} must contain finite occupancies in [0, 1].")
    return occ_a, occ_b


# --------------------------------------------------------------------------- #
# 公开 API
# --------------------------------------------------------------------------- #
def suppress_seen_bitstrings(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    avg_occupancies: Tuple[np.ndarray, np.ndarray],
    num_elec_a: int,
    num_elec_b: int,
    *,
    seen_a: set,
    seen_b: set,
    rand_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """C1 批内抑制: 丢弃恢复后不引入新 α/β 字符串的位串 (SQD-AA 经典模拟)。

    对每位串做 :func:`recover_configurations` 得 ``(α, β)``; 若 ``α ∈ seen_a`` 且
    ``β ∈ seen_b`` (对乘积子空间 ``str_a × str_b`` 零新增) 则丢弃, 否则保留并重归一化
    概率。等效采样分布 ``p'(x) ∝ p(x)·w(x)``, ``w(x)=0`` 当 ``recover(x) ∈ S_measured``,
    否则 1。

    **不学任何分布** (与 distill/``eigenvector_importance_sample`` 的本质区别: 后者按
    解态 ``|c|^(2/T)`` 重采、学分布; 本函数只对已见 det 做硬掩蔽, 代码路径不读
    ``c2d``)。适合电路模式 (池固定、概率非平凡)。自举模式下需配合
    :func:`discover_tail_pool` 过抽新位串 —— 单独用本函数在固定随机池上第 2 轮起会
    几乎全丢 (theory.md §1.5)。

    输入校验对齐 :func:`recover_configurations` 风格 (形状/非负/概率和>0/occ 范围/
    seen 类型)。**空 bsm 不崩**: 返回空池 ``(shape (0, 2*norb))``。

    Parameters
    ----------
    bitstring_matrix : ndarray, shape (S, 2*norb)
        采样位串 (每行一个 bitstring)。
    probabilities : ndarray, shape (S,)
        对应概率/权重 (有限、非负、和>0)。
    avg_occupancies : tuple(ndarray, ndarray)
        ``(avg_occ_alpha, avg_occ_beta)``, 各长 ``norb``, 值 ∈ [0, 1]。
    num_elec_a, num_elec_b : int
        α/β 电子数 (须 ∈ [0, norb])。
    seen_a, seen_b : set of int
        当前子空间 α/β CI 字符串集合 (键 = 整数, 轨道 0 = LSB)。抑制判据:
        ``α∈seen_a ∧ β∈seen_b``。
    rand_seed : int | None
        恢复 tie-breaking 种子。

    Returns
    -------
    bsm_new, probs_new : ndarray
        仅含 "贡献新字符串" 的恢复后位串, 概率重归一化。若无任何新贡献, 返回空池
        ``(shape (0, 2*norb))`` —— 调用方应回退原池避免采样端饿死。
    """
    bsm = np.asarray(bitstring_matrix, dtype=bool)
    if bsm.ndim != 2:
        raise ValueError(f"bitstring_matrix must be 2-D, got ndim={bsm.ndim}.")
    n = bsm.shape[1]
    if n % 2 != 0:
        raise ValueError(
            f"bitstring width must be even (alpha/beta halves), got {n}."
        )
    # seen 类型校验 (先于 recover, 确保 distill 边界锁认知)
    seen_a = _validate_seen("seen_a", seen_a)
    seen_b = _validate_seen("seen_b", seen_b)
    norb = n // 2

    # 空 bsm: 不崩, 直接返回空池
    if bsm.shape[0] == 0:
        return np.empty((0, n), dtype=bool), np.empty(0, dtype=np.float64)

    # 其余校验 (概率/occ/电子数) + 恢复 交给 recover_configurations
    probs = np.asarray(probabilities, dtype=np.float64)
    rec, rec_probs = recover_configurations(
        bsm, probs, avg_occupancies, num_elec_a, num_elec_b, rand_seed=rand_seed
    )

    # 逐行 (α, β), 抑制判据: α∈seen_a ∧ β∈seen_b → 丢弃
    a_strs, b_strs = _bsm_to_ci_strs_per_row(rec, norb)
    mask = np.fromiter(
        ((int(a) not in seen_a) or (int(b) not in seen_b)
         for a, b in zip(a_strs, b_strs)),
        dtype=bool,
        count=rec.shape[0],
    )
    if not mask.any():
        return np.empty((0, n), dtype=bool), np.empty(0, dtype=np.float64)
    keep = rec[mask]
    keep_probs = rec_probs[mask]
    s = float(keep_probs.sum())
    if s <= 0.0:
        return np.empty((0, n), dtype=bool), np.empty(0, dtype=np.float64)
    return keep, keep_probs / s


def discover_tail_pool(
    avg_occupancies: Tuple[np.ndarray, np.ndarray],
    num_elec_a: int,
    num_elec_b: int,
    norb: int,
    *,
    seen_a: set,
    seen_b: set,
    n_target_new: int,
    base_distribution: str = "bootstrap",
    circuit=None,
    max_draw_factor: int = 10,
    rand_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """C1 尾部发现: 迭代抽样 → 恢复 → 抑制已见, 直到收集 ``n_target_new`` 个 "贡献
    新字符串" 的位串, 或耗尽 ``max_draw_factor × n_target_new`` 原始抽样预算。

    自举模式 (12,12 实际路径, theory.md §1.5): base_sampler = 均匀随机位串生成器;
    过抽 ``max_draw_factor×`` 随机位串, 恢复后去重, 仅保留新贡献者 → 把 "更多 shots
    全花在新 det 上"。这是 12,12 P0 的有效实现路径 (固定池上单用
    :func:`suppress_seen_bitstrings` 不够: 同一批随机位串每轮恢复到同一批 det, 第 2 轮
    起抑制会全丢)。

    **distill 边界**: 本函数不读 ``c2d``, 只读 ``(seen_a, seen_b)``。

    输入校验对齐 :func:`recover_configurations` 风格。

    Parameters
    ----------
    avg_occupancies : tuple(ndarray, ndarray)
        当前平均占据 (采样偏置), 各长 ``norb``, ∈ [0, 1]。
    num_elec_a, num_elec_b : int
        α/β 电子数 (∈ [0, norb])。
    norb : int
        空间轨道数。
    seen_a, seen_b : set of int
        当前子空间 α/β CI 字符串集合 (键 = 整数)。
    n_target_new : int
        目标收集的 "新贡献" 位串数 (>0)。
    base_distribution : {"bootstrap", "circuit"}
        base 采样分布。``"bootstrap"`` (默认) = 均匀随机位串; ``"circuit"`` =
        ansatz ``|ψ|²`` (P1, round_001 暂未实现, 走 :func:`sample_from_circuit`)。
    circuit : object | None
        ``base_distribution="circuit"`` 时必给 (P1)。
    max_draw_factor : int
        原始抽样预算 = ``max_draw_factor × n_target_new`` (默认 10×, 自举过抽)。
    rand_seed : int | None
        随机种子 (整个发现过程用一个 Generator, 决定性可复现)。

    Returns
    -------
    bsm, probs : ndarray
        收集到的新位串 (恢复后, 供 vstack/喂给 solve_sqd_active 的 ① 恢复池)。无新
        贡献时返回空池 ``(shape (0, 2*norb))`` —— 调用方应回退原池。
    n_drawn : int
        实际原始抽样数 (诊断: ``n_drawn / n_target_new`` → 恢复映像饱和度)。
    """
    # ---- 校验 ----
    occ_a, occ_b = _validate_avg_occ(avg_occupancies, norb)
    if not (0 <= num_elec_a <= norb) or not (0 <= num_elec_b <= norb):
        raise ValueError(
            f"Electron counts must be in [0, {norb}]; got "
            f"num_elec_a={num_elec_a}, num_elec_b={num_elec_b}."
        )
    if base_distribution not in ("bootstrap", "circuit"):
        raise ValueError(
            f"base_distribution 须为 'bootstrap' | 'circuit', got {base_distribution!r}."
        )
    if base_distribution == "circuit":
        if circuit is None:
            raise ValueError("base_distribution='circuit' 需要传 circuit=...")
        # P1 (theory.md §4 #7): round_001 P0 仅实现 bootstrap。
        raise NotImplementedError(
            "circuit base_distribution 为 round_001 P1, 当前仅实现 bootstrap。"
        )
    seen_a = _validate_seen("seen_a", seen_a)
    seen_b = _validate_seen("seen_b", seen_b)
    if not isinstance(n_target_new, (int, np.integer)) or isinstance(n_target_new, bool):
        raise TypeError(f"n_target_new 须为正整数, got {type(n_target_new).__name__}.")
    if int(n_target_new) <= 0:
        raise ValueError(f"n_target_new 须为正整数, got {n_target_new}.")
    if not isinstance(max_draw_factor, (int, np.integer)) or isinstance(max_draw_factor, bool):
        raise TypeError(f"max_draw_factor 须为 >=1 的整数, got {type(max_draw_factor).__name__}.")
    if int(max_draw_factor) < 1:
        raise ValueError(f"max_draw_factor 须为 >=1 的整数, got {max_draw_factor}.")

    n_target_new = int(n_target_new)
    max_draw_factor = int(max_draw_factor)
    width = 2 * norb
    rng = np.random.default_rng(rand_seed)
    total_budget = max_draw_factor * n_target_new
    batch_size = n_target_new

    n_drawn = 0
    collected_bsm: list = []
    collected_probs: list = []
    collected_pairs: set = set()        # 局部去重 (跨 batch, (α,β) int 元组)

    while n_drawn < total_budget:
        cur = min(batch_size, total_budget - n_drawn)
        # base_sampler (bootstrap): 均匀随机位串
        bsm_batch = rng.random((cur, width)) > 0.5
        probs_batch = np.full(cur, 1.0 / cur)
        n_drawn += cur
        # 恢复 (用共享 rng 推进 tie-breaking 状态)
        rec, rec_probs = recover_configurations(
            bsm_batch, probs_batch, (occ_a, occ_b),
            num_elec_a, num_elec_b, rand_seed=rng,
        )
        # 逐行抑制 + 局部去重
        a_strs, b_strs = _bsm_to_ci_strs_per_row(rec, norb)
        for i in range(rec.shape[0]):
            ai, bi = int(a_strs[i]), int(b_strs[i])
            if ai in seen_a and bi in seen_b:
                continue                       # 乘积子空间零新增 → 抑制
            if (ai, bi) in collected_pairs:
                continue                       # 本池已收 → 去重
            collected_pairs.add((ai, bi))
            collected_bsm.append(rec[i])
            collected_probs.append(float(rec_probs[i]))
        if len(collected_bsm) >= n_target_new:
            break

    if not collected_bsm:
        return (np.empty((0, width), dtype=bool),
                np.empty(0, dtype=np.float64), n_drawn)
    bsm_out = np.vstack(collected_bsm).astype(bool)
    probs_out = np.asarray(collected_probs, dtype=np.float64)
    s = float(probs_out.sum())
    if s <= 0.0:
        return (np.empty((0, width), dtype=bool),
                np.empty(0, dtype=np.float64), n_drawn)
    return bsm_out, probs_out / s, n_drawn
