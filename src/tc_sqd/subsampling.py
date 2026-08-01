"""Batch subsampling and Hamming-weight postselection utilities."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from .counts import bitarray_to_int

__all__ = [
    "subsample",
    "postselect_by_hamming_right_and_left",
    "limit_subspace",
]


def postselect_by_hamming_right_and_left(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    *,
    hamming_right: int,
    hamming_left: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Postselect bitstrings by Hamming weight of right and left halves.

    Parameters
    ----------
    bitstring_matrix : ndarray, shape (S, N)
    probabilities : ndarray, shape (S,)
    hamming_right : int
        Target Hamming weight of the right (alpha) half.
    hamming_left : int
        Target Hamming weight of the left (beta) half.

    Returns
    -------
    bitstring_matrix : ndarray, shape (S', N)
    probabilities : ndarray, shape (S',)
        Re-normalised.
    """
    bsm = np.asarray(bitstring_matrix)
    probs = np.asarray(probabilities, dtype=np.float64)
    if bsm.ndim != 2:
        raise ValueError(
            f"bitstring_matrix must be 2-D, got ndim={bsm.ndim}."
        )
    n = bsm.shape[1]
    if n % 2 != 0:
        raise ValueError(
            f"bitstring width must be even (alpha/beta halves), got {n}."
        )
    if hamming_right < 0 or hamming_left < 0:
        raise ValueError("Hamming weights must be non-negative.")
    if probs.shape[0] != bsm.shape[0]:
        raise ValueError(
            "probabilities length must match bitstring_matrix row count."
        )
    half = n // 2
    mask = (bsm[:, half:].sum(1) == hamming_right) & (bsm[:, :half].sum(1) == hamming_left)
    bsm_new = bsm[mask]
    probs_new = probs[mask]
    total = probs_new.sum()
    if total > 0:
        probs_new = probs_new / total
    return bsm_new, probs_new


def subsample(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    samples_per_batch: int,
    num_batches: int,
    rand_seed: Optional[Union[int, np.random.Generator]] = None,
) -> list:
    """Subsample *batches* of bitstrings from the input matrix.

    Each batch is drawn **without replacement** from ``bitstring_matrix``
    according to ``probabilities``.  After a batch is drawn, samples are
    replaced, so different batches may overlap.

    Parameters
    ----------
    bitstring_matrix : ndarray, shape (S, N)
    probabilities : ndarray, shape (S,)
    samples_per_batch : int
    num_batches : int
    rand_seed : int | Generator | None

    Returns
    -------
    list of ndarray
        ``num_batches`` bitstring matrices, each of shape ``(<= samples_per_batch, N)``.
    """
    bsm = np.asarray(bitstring_matrix)
    if bsm.ndim != 2:
        raise ValueError(
            f"bitstring_matrix must be 2-D, got ndim={bsm.ndim}."
        )
    if samples_per_batch <= 0 or num_batches <= 0:
        raise ValueError("samples_per_batch and num_batches must be positive.")
    if len(probabilities) != bsm.shape[0]:
        raise ValueError(
            "probabilities length must match bitstring_matrix row count."
        )

    probs = np.asarray(probabilities, dtype=np.float64)
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite.")
    if np.any(probs < 0):
        raise ValueError("probabilities must be non-negative.")
    total = probs.sum()
    if total <= 0:
        raise ValueError("probabilities must have a positive sum.")
    # Sampling without replacement requires at least ``spb`` non-zero
    # probability entries.
    n_nonzero = int(np.count_nonzero(probs > 0))
    spb = min(samples_per_batch, n_nonzero)
    if spb == 0:
        raise ValueError("No bitstrings with non-zero probability to sample.")

    if isinstance(rand_seed, np.random.Generator):
        rng = rand_seed
    else:
        rng = np.random.default_rng(rand_seed)

    probs = probs / total
    S = bsm.shape[0]

    batches = []
    for _ in range(num_batches):
        idx = rng.choice(S, size=spb, replace=False, p=probs)
        batches.append(bsm[idx])
    return batches


def limit_subspace(bitstring_matrix: np.ndarray, max_dim, norb: int, *,
                   probabilities: Optional[np.ndarray] = None) -> np.ndarray:
    """按概率降序贪心裁剪 bitstring, 使唯一 α/β 字符串满足 ``max_dim`` 限制。

    子空间维度 = 唯一 α 字符串数 × 唯一 β 字符串数; 高维时对角化成本暴涨,
    此函数用于把子空间限制在预算内 (真机大 shots / 大轨道场景)。

    Parameters
    ----------
    bitstring_matrix : ndarray (S, 2*norb)
    max_dim : int | tuple(int,int)
        ``int``      —— 总行列式数 ``na*nb ≤ max_dim``;
        ``tuple``    —— ``na ≤ max_dim[0]`` 且 ``nb ≤ max_dim[1]``。
    norb : int
        空间轨道数。
    probabilities : ndarray (S,) | None
        按概率降序优先保留 (None = 输入顺序, 等价均匀)。

    Returns
    -------
    ndarray
        裁剪后的 bitstring matrix (概率高的优先; 一旦新字符串使维度超限即停)。
    """
    bsm = np.asarray(bitstring_matrix, dtype=bool)
    n = bsm.shape[0]
    if n == 0 or max_dim is None:
        return bsm
    if isinstance(max_dim, (int, np.integer)):
        limit_na = limit_nb = None
        limit_prod = int(max_dim)
    else:
        limit_na, limit_nb = int(max_dim[0]), int(max_dim[1])
        limit_prod = None

    # α/β 字符串 int (与 bitstring_matrix_to_ci_strs 的列序约定一致)
    powers = (1 << np.arange(norb, dtype=np.int64))
    a_ints = (bsm[:, norb:][:, ::-1].astype(np.int64) @ powers).ravel()
    b_ints = (bsm[:, :norb][:, ::-1].astype(np.int64) @ powers).ravel()

    if probabilities is None:
        order = range(n)
    else:
        order = np.argsort(-np.asarray(probabilities, dtype=np.float64))

    sel_a: set = set()
    sel_b: set = set()
    keep = []
    for idx in order:
        a, b = a_ints[idx], b_ints[idx]
        na = len(sel_a) + (0 if a in sel_a else 1)
        nb = len(sel_b) + (0 if b in sel_b else 1)
        if limit_prod is not None:
            if na * nb > limit_prod:
                break
        else:
            if na > limit_na or nb > limit_nb:
                break
        sel_a.add(a)
        sel_b.add(b)
        keep.append(idx)

    if len(keep) == n:
        return bsm
    return bsm[np.array(keep, dtype=int)]
