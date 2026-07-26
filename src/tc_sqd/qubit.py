"""Qubit-subspace SQD: project a Pauli Hamiltonian onto a bitstring subspace
and diagonalise.

This module is useful for non-fermionic problems (e.g. QAOA-MaxCut) where the
Hamiltonian is given directly as a sum of Pauli strings.
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

from .counts import bitarray_to_int, int_to_bitarray

__all__ = [
    "sort_and_remove_duplicates",
    "matrix_elements_from_pauli",
    "project_operator_to_subspace",
    "solve_qubit",
]

# Pauli character codes
_I = ord("I")
_X = ord("X")
_Y = ord("Y")
_Z = ord("Z")


def sort_and_remove_duplicates(bitstring_matrix: np.ndarray) -> np.ndarray:
    """Sort bitstrings by integer value and remove duplicates."""
    bsm = np.asarray(bitstring_matrix, dtype=bool)
    if bsm.ndim == 1:
        bsm = bsm.reshape(1, -1)
    int_vals = bitarray_to_int(bsm)
    uniq = np.unique(int_vals)
    nbits = bsm.shape[1]
    result = np.zeros((len(uniq), nbits), dtype=bool)
    for i, val in enumerate(uniq):
        result[i] = ((val >> np.arange(nbits - 1, -1, -1, dtype=np.uint64)) & 1).astype(bool)
    return result


def _pauli_to_codes(pauli_str: str) -> np.ndarray:
    """Convert a Pauli string like 'IXZY' to an int8 array of codes."""
    valid = {"I", "X", "Y", "Z"}
    bad = set(pauli_str.upper()) - valid
    if bad:
        raise ValueError(
            f"Illegal Pauli character(s) {sorted(bad)} in '{pauli_str}'; "
            "allowed: I, X, Y, Z."
        )
    return np.array([ord(c) for c in pauli_str.upper()], dtype=np.int8)


def matrix_elements_from_pauli(
    bitstring_matrix: np.ndarray,
    pauli: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the sparse matrix elements of a single Pauli operator in the
    subspace spanned by ``bitstring_matrix`` rows.

    Parameters
    ----------
    bitstring_matrix : ndarray, shape (S, N)
        Must be **sorted and de-duplicated** (use ``sort_and_remove_duplicates``).
    pauli : str
        Pauli string of length N, e.g. ``"IIXZ"``.

    Returns
    -------
    amplitudes : ndarray of complex128
    rows : ndarray of int
    cols : ndarray of int
    """
    bsm = np.asarray(bitstring_matrix, dtype=bool)
    S, N = bsm.shape
    if len(pauli) != N:
        raise ValueError(
            f"Pauli string length {len(pauli)} != bitstring length {N}."
        )
    if N >= 64:
        raise ValueError("Bitstring length must be < 64 for integer indexing.")

    codes = _pauli_to_codes(pauli)
    int_vals = bitarray_to_int(bsm)
    # Build a lookup: integer -> row index
    int_to_idx = {int(v): i for i, v in enumerate(int_vals)}

    amps = []
    rows = []
    cols = []

    for i in range(S):
        bits = bsm[i]
        # Compute the target bitstring after applying the Pauli
        target = bits.copy()
        phase = 1.0 + 0.0j
        for j in range(N):
            c = codes[j]
            if c == _I:
                pass
            elif c == _Z:
                if bits[j]:
                    phase *= -1.0
            elif c == _X:
                target[j] = not target[j]
            elif c == _Y:
                if bits[j]:
                    phase *= 1.0j  # Y|1> = -i|0>
                else:
                    phase *= -1.0j  # Y|0> = i|1>
                target[j] = not target[j]
        target_int = int(bitarray_to_int(target.reshape(1, -1))[0])
        if target_int in int_to_idx:
            j = int_to_idx[target_int]
            amps.append(phase)
            rows.append(i)
            cols.append(j)

    return (
        np.array(amps, dtype=np.complex128),
        np.array(rows, dtype=np.int64),
        np.array(cols, dtype=np.int64),
    )


def project_operator_to_subspace(
    bitstring_matrix: np.ndarray,
    hamiltonian: List[Tuple[str, float]],
    *,
    verbose: bool = False,
) -> csr_matrix:
    """Project a Hamiltonian (list of ``(pauli_str, coeff)``) onto the subspace.

    Parameters
    ----------
    bitstring_matrix : ndarray, shape (S, N)
        Sorted and de-duplicated.
    hamiltonian : list of (str, float)
        Each entry is ``(pauli_string, coefficient)``.

    Returns
    -------
    spmatrix : scipy.sparse.csr_matrix, shape (S, S)
    """
    bsm = np.asarray(bitstring_matrix, dtype=bool)
    S = bsm.shape[0]
    data = []
    row_ind = []
    col_ind = []

    for pauli_str, coeff in hamiltonian:
        amps, rows, cols = matrix_elements_from_pauli(bsm, pauli_str)
        if len(amps) == 0:
            continue
        data.extend((coeff * amps).tolist())
        row_ind.extend(rows.tolist())
        col_ind.extend(cols.tolist())
        if verbose:
            print(f"  {pauli_str}: {len(amps)} non-zero elements")

    return coo_matrix(
        (data, (row_ind, col_ind)), shape=(S, S), dtype=np.complex128
    ).tocsr()


def solve_qubit(
    bitstring_matrix: np.ndarray,
    hamiltonian: List[Tuple[str, float]],
    *,
    verbose: bool = False,
    **scipy_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalise a Pauli Hamiltonian in the subspace defined by bitstrings.

    Parameters
    ----------
    bitstring_matrix : ndarray, shape (S, N)
    hamiltonian : list of (str, float)
    **scipy_kwargs
        Forwarded to ``scipy.sparse.linalg.eigsh``.

    Returns
    -------
    eigenvalues : ndarray, shape (k,)
    eigenvectors : ndarray, shape (S, k)
    """
    bsm = sort_and_remove_duplicates(bitstring_matrix)
    H = project_operator_to_subspace(bsm, hamiltonian, verbose=verbose)

    S = H.shape[0]
    k_requested = int(scipy_kwargs.get("k", 1))
    if S == 1:
        # Trivial case
        return np.array([H[0, 0].real]), np.array([[1.0]])

    # If the matrix is small, use dense diagonalisation
    if S <= 100:
        H_dense = H.toarray()
        # Ensure Hermitian
        H_dense = 0.5 * (H_dense + H_dense.conj().T)
        vals, vecs = np.linalg.eigh(
            H_dense.real if np.allclose(H_dense.imag, 0) else H_dense
        )
        k = min(k_requested, S)
        return vals[:k], vecs[:, :k]

    # Default kwargs for eigsh
    kwargs = {"k": 1, "which": "SA"}
    kwargs.update(scipy_kwargs)
    # scipy eigsh requires k < N - 1; clamp defensively.
    kwargs["k"] = min(int(kwargs["k"]), max(1, S - 2))

    # Ensure Hermitian
    H = 0.5 * (H + H.getH())

    # Check if matrix is real
    if np.allclose(H.data.imag, 0):
        H = H.real.asfptype()

    vals, vecs = eigsh(H, **kwargs)
    return vals, vecs
