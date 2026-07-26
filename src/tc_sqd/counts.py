"""Bitstring-matrix / integer conversion utilities and TensorCircuit sampling adapter.

The canonical *bitstring matrix* used throughout ``tc_sqd`` is a 2-D ``np.ndarray``
of dtype ``bool`` / ``uint8`` where **each row is one sampled bitstring** and
``bitstring_matrix[i, j]`` is the value of qubit *j* in sample *i*.

Spin ordering convention (same as qiskit-addon-sqd)
---------------------------------------------------
For a molecule with *n* spatial orbitals the bitstring has ``2n`` bits laid out as::

    [ b_{n-1} ... b_0  |  a_{n-1} ... a_0 ]
      ^------- beta ---^  ^----- alpha ----^

i.e. the **right** half encodes spin-up (alpha) occupations and the **left**
half encodes spin-down (beta) occupations.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "bitarray_to_int",
    "int_to_bitarray",
    "counts_dict_to_bitstring_matrix",
    "sample_from_circuit",
]


# --------------------------------------------------------------------------- #
#  Core conversions
# --------------------------------------------------------------------------- #
def bitarray_to_int(bitstring_matrix: np.ndarray) -> np.ndarray:
    """Convert each row of a bitstring matrix to its unsigned integer value.

    The leftmost column is the most-significant bit.

    Parameters
    ----------
    bitstring_matrix : ndarray, shape (S, N)
        2-D boolean / uint8 array.

    Returns
    -------
    ndarray, shape (S,)
        1-D ``uint64`` array of integer representations.
    """
    bitstring_matrix = np.asarray(bitstring_matrix)
    if bitstring_matrix.ndim == 1:
        bitstring_matrix = bitstring_matrix.reshape(1, -1)
    nbits = bitstring_matrix.shape[1]
    if nbits >= 64:
        raise ValueError(
            f"Bitstring length {nbits} >= 64; cannot represent as int64."
        )
    # powers: 2**(N-1), ..., 2**0
    powers = (1 << np.arange(nbits - 1, -1, -1, dtype=np.uint64)).reshape(1, -1)
    return (bitstring_matrix.astype(np.uint64) @ powers.T).ravel()


def int_to_bitarray(int_values: Union[int, Sequence[int], np.ndarray],
                    nbits: int) -> np.ndarray:
    """Convert integer(s) to a bitstring matrix.

    Parameters
    ----------
    int_values : int | Sequence[int] | ndarray
        Integer representation(s) of bitstring(s).
    nbits : int
        Number of bits.

    Returns
    -------
    ndarray, shape (S, nbits), dtype bool
        Bitstring matrix.
    """
    scalar = np.isscalar(int_values)
    arr = np.atleast_1d(np.asarray(int_values, dtype=np.uint64))
    out = np.zeros((arr.size, nbits), dtype=bool)
    for col in range(nbits):
        out[:, col] = (arr >> (nbits - 1 - col)) & 1
    return out[0] if scalar else out


# --------------------------------------------------------------------------- #
#  Counts-dict helpers
# --------------------------------------------------------------------------- #
def counts_dict_to_bitstring_matrix(
    counts: Dict[Union[int, str], int],
    nbits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a TC / qiskit counts dict to ``(bitstring_matrix, probabilities)``.

    Accepts keys that are either ``int`` (e.g. from TC ``count_dict_int``)
    or binary strings (e.g. from TC ``count_dict_bin``).

    Parameters
    ----------
    counts : dict
        ``{bitstring: count}`` mapping.
    nbits : int
        Number of qubits.

    Returns
    -------
    bitstring_matrix : ndarray, shape (M, nbits), dtype bool
        Unique bitstrings (one per row).
    probabilities : ndarray, shape (M,), dtype float64
        Normalised probability for each bitstring.
    """
    if not isinstance(counts, dict) or len(counts) == 0:
        raise ValueError("counts must be a non-empty dict.")
    if nbits <= 0:
        raise ValueError(f"nbits must be positive, got {nbits}.")

    # Aggregate equivalent keys first: a binary-string key (e.g. "01") and an
    # integer key (e.g. 1) can denote the same bitstring and must be merged.
    merged = {}
    for key, cnt in counts.items():
        if isinstance(key, str):
            # binary string, possibly padded
            try:
                intval = int(key, 2)
            except ValueError as exc:
                raise ValueError(f"Invalid binary-string key '{key}'.") from exc
        else:
            intval = int(key)
        if intval < 0 or intval >= (1 << nbits):
            raise ValueError(
                f"Bitstring value {intval} out of range for nbits={nbits}."
            )
        if cnt < 0:
            raise ValueError(f"Negative count {cnt} for bitstring {key}.")
        merged[intval] = merged.get(intval, 0) + cnt

    int_vals = np.array(sorted(merged), dtype=np.uint64)
    counts_arr = np.array([merged[int(v)] for v in int_vals], dtype=np.float64)
    bsm = int_to_bitarray(int_vals, nbits).astype(bool)
    total = counts_arr.sum()
    if total <= 0:
        raise ValueError("Sum of counts must be positive.")
    probs = counts_arr / total
    return bsm, probs


# --------------------------------------------------------------------------- #
#  TensorCircuit sampling adapter
# --------------------------------------------------------------------------- #
def sample_from_circuit(
    circuit,
    n_samples: int = 1000,
    *,
    nbits: Optional[int] = None,
    allow_state: bool = True,
    readout_error=None,
    random_generator=None,
    return_probabilities: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample bitstrings from a TensorCircuit ``Circuit`` and return them in the
    ``tc_sqd`` canonical format.

    Parameters
    ----------
    circuit : tensorcircuit.Circuit
        A (compiled) TensorCircuit object.
    n_samples : int
        Number of shots.
    nbits : int, optional
        Number of qubits.  If ``None`` (default), inferred from the circuit's
        ``_nqubits`` attribute; when that attribute is unavailable the value
        **must** be supplied explicitly — no guessing from sampled bitstrings
        is performed, since that silently under-pads leading zeros.
    allow_state : bool
        Forwarded to ``circuit.sample``.  ``True`` samples from the full state
        vector (faster for small circuits).
    readout_error : optional
        Forwarded to ``circuit.sample``.
    random_generator : optional
        Forwarded to ``circuit.sample``.
    return_probabilities : bool
        If ``True`` (default) also return the normalised probability of each
        *unique* bitstring.  If ``False`` probabilities are uniform ``1/M``.

    Returns
    -------
    bitstring_matrix : ndarray, shape (M, N), dtype bool
        Unique bitstrings, one per row.  ``M <= n_samples``.
    probabilities : ndarray, shape (M,), dtype float64
        Probability of each unique bitstring.

    Raises
    ------
    ValueError
        If the qubit count cannot be determined from the circuit and ``nbits``
        was not provided.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    if nbits is None:
        # TensorCircuit exposes the qubit count only via the private
        # ``_nqubits`` attribute.  Never infer from sampled integers: that
        # silently under-pads leading zeros.
        nbits = getattr(circuit, "_nqubits", None)
    if nbits is None:
        raise ValueError(
            "Cannot determine the number of qubits from the circuit "
            "(`_nqubits` missing); pass `nbits` explicitly."
        )
    if nbits <= 0:
        raise ValueError(f"nbits must be positive, got {nbits}.")
    result = circuit.sample(
        batch=n_samples,
        allow_state=allow_state,
        format="count_dict_int",
        readout_error=readout_error,
        random_generator=random_generator,
    )
    bsm, probs = counts_dict_to_bitstring_matrix(result, nbits)
    if not return_probabilities:
        probs = np.full(bsm.shape[0], 1.0 / bsm.shape[0])
    return bsm, probs
