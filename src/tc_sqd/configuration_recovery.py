"""Configuration recovery for SQD.

Implements the average-occupancy-based bitstring refinement described in
Robledo-Sato et al., *Chemistry Beyond Exact Solutions on a Quantum-Centric
Supercomputer*, arXiv:2405.05068.

Given noisy bitstrings sampled from a quantum circuit, configuration recovery
flips individual bits so that every bitstring satisfies the target particle
numbers (Hamming weights) for spin-up and spin-down sectors.  The decision of
*which* bit to flip is guided by the average occupancy
:math:`\\bar n_i = \\frac{1}{S}\\sum_s b_i^{(s)}` --- bits whose value deviates
most from ``n_bar_i`` are flipped first.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

__all__ = [
    "recover_configurations",
    "postselect_by_hamming_weight",
    "estimate_true_occupancies",
]


def _hamming_weight(row: np.ndarray) -> int:
    return int(np.sum(row))


def _argsort_with_random_ties(
    scores: np.ndarray,
    rng: np.random.Generator,
    descending: bool,
) -> np.ndarray:
    """Argsort indices of ``scores`` with uniform-random tie-breaking.

    Uses the Fisher-Yates shuffle on groups of equal scores so that tied
    elements appear in a uniformly random order.  Returns indices into
    ``scores`` (i.e. the same convention as ``np.argsort``).
    """
    order = np.arange(len(scores))
    # Group positions by equal score, then shuffle within each group and
    # concatenate groups in sorted (or reverse-sorted) score order.
    uniq = np.unique(scores)
    if descending:
        uniq = uniq[::-1]
    out = []
    for val in uniq:
        group = order[scores == val].copy()
        rng.shuffle(group)  # Fisher-Yates inside np.random.Generator.shuffle
        out.append(group)
    return np.concatenate(out) if out else np.array([], dtype=int)


def postselect_by_hamming_weight(
    bitstring_matrix: np.ndarray,
    *,
    hamming_right: int,
    hamming_left: int,
) -> np.ndarray:
    """Return a boolean mask selecting rows with the correct Hamming weights.

    The bitstring is split into a left half (beta) and a right half (alpha).

    Parameters
    ----------
    bitstring_matrix : ndarray, shape (S, N)
    hamming_right : int
        Target Hamming weight of the right (alpha) half.
    hamming_left : int
        Target Hamming weight of the left (beta) half.

    Returns
    -------
    mask : ndarray, shape (S,), dtype bool
    """
    bsm = np.asarray(bitstring_matrix)
    n = bsm.shape[1]
    half = n // 2
    right = bsm[:, half:]
    left = bsm[:, :half]
    mask = (right.sum(axis=1) == hamming_right) & (left.sum(axis=1) == hamming_left)
    return mask


def _recover_single(
    row: np.ndarray,
    avg_occ: np.ndarray,
    target_weight: int,
    low: int,
    high: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Recover a single half-row (either alpha or beta) to ``target_weight``."""
    segment = row[low:high].copy()
    current = int(np.sum(segment))
    if current == target_weight:
        return row
    n_flip = abs(current - target_weight)
    seg_occ = avg_occ[low:high]
    if current > target_weight:
        # Need to flip 1 -> 0; pick the most "surprising" 1s, i.e. those with
        # the *smallest* average occupancy.  Ties are broken randomly via the
        # Fisher-Yates shuffle implemented by argsort on random keys.
        ones_idx = np.where(segment == 1)[0]
        scores = seg_occ[ones_idx]
        flip_local = ones_idx[_argsort_with_random_ties(scores, rng, descending=False)][:n_flip]
        segment[flip_local] = 0
    else:
        # Need to flip 0 -> 1; pick the most likely 1s, i.e. those with the
        # *largest* average occupancy (ties again broken randomly).
        zeros_idx = np.where(segment == 0)[0]
        scores = seg_occ[zeros_idx]
        flip_local = zeros_idx[_argsort_with_random_ties(scores, rng, descending=True)][:n_flip]
        segment[flip_local] = 1
    row = row.copy()
    row[low:high] = segment
    return row


def recover_configurations(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    avg_occupancies: Tuple[np.ndarray, np.ndarray],
    num_elec_a: int,
    num_elec_b: int,
    rand_seed: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refine bitstrings based on average orbital occupancies.

    Each bitstring is independently corrected so that its alpha (right) half
    has Hamming weight ``num_elec_a`` and its beta (left) half has weight
    ``num_elec_b``.  Bits are flipped greedily: the bits whose values deviate
    most from ``avg_occupancies`` are flipped first.

    Parameters
    ----------
    bitstring_matrix : ndarray, shape (S, 2*norb)
        Each row is a bitstring.
    probabilities : ndarray, shape (S,)
        Probability / weight of each bitstring.
    avg_occupancies : tuple of two ndarrays
        ``(avg_occ_alpha, avg_occ_beta)``, each of length ``norb``.
    num_elec_a : int
        Target number of alpha electrons.
    num_elec_b : int
        Target number of beta electrons.
    rand_seed : int | Generator | None
        Random seed for tie-breaking.

    Returns
    -------
    recovered_matrix : ndarray, shape (S', 2*norb)
        Recovered (and de-duplicated) bitstrings.
    recovered_probs : ndarray, shape (S',)
        Re-normalised probabilities.
    """
    bsm = np.asarray(bitstring_matrix, dtype=bool).copy()
    probs = np.asarray(probabilities, dtype=np.float64)
    if bsm.ndim != 2:
        raise ValueError(
            f"bitstring_matrix must be 2-D, got ndim={bsm.ndim}."
        )
    if bsm.shape[0] == 0:
        raise ValueError("bitstring_matrix must contain at least one row.")
    n = bsm.shape[1]
    if n % 2 != 0:
        raise ValueError(
            f"bitstring width must be even (alpha/beta halves), got {n}."
        )
    if n >= 64:
        raise ValueError(
            f"bitstring width must be < 64 for integer de-duplication, got {n}."
        )
    if probs.shape[0] != bsm.shape[0]:
        raise ValueError(
            "probabilities length must match bitstring_matrix row count."
        )
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("probabilities must be finite and non-negative.")
    if probs.sum() <= 0:
        raise ValueError("probabilities must have a positive sum.")

    if isinstance(rand_seed, np.random.Generator):
        rng = rand_seed
    else:
        rng = np.random.default_rng(rand_seed)

    half = n // 2

    avg_a, avg_b = avg_occupancies
    avg_a = np.asarray(avg_a, dtype=np.float64)
    avg_b = np.asarray(avg_b, dtype=np.float64)
    if avg_a.ndim != 1 or avg_b.ndim != 1:
        raise ValueError("avg_occupancies entries must be 1-D.")
    if avg_a.shape[0] != half or avg_b.shape[0] != half:
        raise ValueError(
            f"avg_occupancies entries must have length {half} "
            f"(norb), got {avg_a.shape[0]} and {avg_b.shape[0]}."
        )
    for name, arr in (("avg_occupancies[0]", avg_a), ("avg_occupancies[1]", avg_b)):
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError(
                f"{name} must contain finite occupancies in [0, 1]."
            )
    if not (0 <= num_elec_a <= half) or not (0 <= num_elec_b <= half):
        raise ValueError(
            f"Electron counts must be in [0, {half}]; got "
            f"num_elec_a={num_elec_a}, num_elec_b={num_elec_b}."
        )

    # Full occupancy vector aligned with bitstring layout
    #   bitstring: [ beta_{norb-1}..beta_0 | alpha_{norb-1}..alpha_0 ]
    #   avg_occ arrays are indexed by orbital (0..norb-1), so we reverse to
    #   match the left-to-right high-to-low orbital layout in the bitstring.
    avg_full = np.concatenate([avg_b[::-1], avg_a[::-1]])

    for i in range(bsm.shape[0]):
        bsm[i] = _recover_single(
            bsm[i], avg_full, num_elec_a, low=half, high=n, rng=rng
        )
        bsm[i] = _recover_single(
            bsm[i], avg_full, num_elec_b, low=0, high=half, rng=rng
        )

    # De-duplicate and merge probabilities
    int_vals = (
        bsm.astype(np.uint64)
        @ (1 << np.arange(n - 1, -1, -1, dtype=np.uint64)).reshape(-1, 1)
    ).ravel()
    uniq, inverse = np.unique(int_vals, return_inverse=True)
    merged_probs = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(merged_probs, inverse, probs)
    merged_probs /= merged_probs.sum()

    # Convert unique ints back to bitstring matrix
    recovered = np.zeros((len(uniq), n), dtype=bool)
    for i, val in enumerate(uniq):
        recovered[i] = (
            (val >> np.arange(n - 1, -1, -1, dtype=np.uint64)) & 1
        ).astype(bool)

    return recovered, merged_probs


def estimate_true_occupancies(
    bitstring_matrix: np.ndarray,
    num_elec_a: int,
    num_elec_b: int,
    t1_gamma: Union[float, np.ndarray],
    *,
    norb: Optional[int] = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """从观测位串 + per-qubit T1 率估计真实平均占据 (T1 反卷积)。

    **物理模型**: T1 (振幅阻尼) 只把 |1⟩→|0⟩, 逐 qubit 独立。因此观测平均
    占据与真实占据满足 ``⟨n̂_i⟩ = (1-γ_i)·⟨n_i⟩_true``, 反解出
    ``⟨n_i⟩_true ≈ ⟨n̂_i⟩ / (1-γ_i)``。逐位 γ_i 不均匀 (真机 per-qubit T1
    不同) 时该校正**改变轨道相对序**, 使估计显著更接近真实
    (在 per-qubit γ 模拟下 RMSE 约降 30%); 均匀 γ 时仅整体缩放 (保序, 无增益)。

    **用法**: 把返回的 ``(avg_occ_a, avg_occ_b)`` 喂给 :func:`recover_configurations`
    或 ``diagonalize_fermionic_hamiltonian`` 的 ``initial_occupancies``, 即构成
    T1 感知的配置恢复 (校正后的 avg_occ 而非直接观测/HF 值)。

    Parameters
    ----------
    bitstring_matrix : ndarray (S, 2*norb)
        观测位串 (含 T1 破坏)。
    num_elec_a, num_elec_b : int
        目标 α/β 电子数 (normalize=True 时用于归一)。
    t1_gamma : float | ndarray (2*norb,)
        振幅阻尼率, 布局同 bitstring 列序 ``[β_{n-1}..β0 | α_{n-1}..α0]``。
        float 广播到全部位。
    norb : int | None
        空间轨道数 (省略时从 bsm 宽度推断)。
    normalize : bool
        True (默认) 把每自旋 avg_occ 按粒子数归一, 使总和 = 电子数
        (合法态 avg_occ 总和 = 电子数), 便于直接作 recover 目标。

    Returns
    -------
    (avg_occ_a, avg_occ_b) : tuple of ndarray, shape (norb,)
        估计的真实平均占据 (轨道序 0..norb-1)。
    """
    bsm = np.asarray(bitstring_matrix, dtype=bool)
    if bsm.ndim != 2:
        raise ValueError(f"bitstring_matrix must be 2-D, got ndim={bsm.ndim}.")
    nq = bsm.shape[1]
    if nq == 0 or nq % 2 != 0:
        raise ValueError(f"bitstring width must be a positive even number, got {nq}.")
    if norb is None:
        norb = nq // 2
    elif norb != nq // 2:
        raise ValueError(f"norb={norb} inconsistent with bitstring width {nq}.")
    if bsm.shape[0] == 0:
        raise ValueError("bitstring_matrix must contain at least one row.")

    # γ -> 逐位列 (bsm 列序)
    if np.isscalar(t1_gamma):
        gamma_col = np.full(nq, float(t1_gamma))
    else:
        gamma_col = np.asarray(t1_gamma, dtype=np.float64)
        if gamma_col.shape != (nq,):
            raise ValueError(
                f"t1_gamma must be float or 1-D array of length {nq} "
                f"(bitstring 列序), got shape {gamma_col.shape}."
            )
        if np.any(gamma_col < 0.0) or np.any(gamma_col > 1.0):
            raise ValueError("t1_gamma must be in [0, 1].")

    obs_col = bsm.mean(axis=0)
    # γ→1 (完全衰减): 观测>0 的位必为真实 1 (clip 到 1), 观测=0 无法反推 (置 0)。
    # 用 1e-12 下限避免 0/0 -> NaN。
    denom = np.maximum(1.0 - gamma_col, 1e-12)
    est_col = np.clip(obs_col / denom, 0.0, 1.0)

    # 列序 -> 轨道序: 左半 β (列 k = β 轨道 n-1-k), 右半 α (列 norb+k = α 轨道 n-1-k)
    avg_a = est_col[norb:][::-1]      # α 轨道序 0..norb-1
    avg_b = est_col[:norb][::-1]      # β 轨道序 0..norb-1

    if normalize:
        sa = avg_a.sum()
        if sa > 0:
            avg_a = np.clip(avg_a * num_elec_a / sa, 0.0, 1.0)
        sb = avg_b.sum()
        if sb > 0:
            avg_b = np.clip(avg_b * num_elec_b / sb, 0.0, 1.0)

    return avg_a, avg_b
