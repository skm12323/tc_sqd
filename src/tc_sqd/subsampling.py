"""Batch subsampling and Hamming-weight postselection utilities."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from .counts import bitarray_to_int

__all__ = [
    "subsample",
    "postselect_by_hamming_right_and_left",
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
