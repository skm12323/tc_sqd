"""Fermionic SQD: CI-matrix construction, subspace diagonalisation, and the
iterative SQD loop.

The heavy lifting (Slater-Condon matrix elements and Davidson diagonalisation)
is delegated to ``pyscf.fci.selected_ci``, exactly as in
``qiskit-addon-sqd``.  This module wraps it in a numpy-1.x / TensorCircuit
friendly API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.linalg import expm

from pyscf.fci import cistring, selected_ci
from pyscf.fci import spin_op

from .configuration_recovery import recover_configurations
from .subsampling import subsample

__all__ = [
    "SCIState",
    "SCIResult",
    "bitstring_matrix_to_ci_strs",
    "enlarge_batch_from_transitions",
    "build_ci_matrix",
    "solve_sci",
    "solve_sci_batch",
    "solve_fermion",
    "diagonalize_fermionic_hamiltonian",
    "optimize_orbitals",
    "rotate_integrals",
    "compute_ground_state_energy",
]

# Spin-convention constants
#   bitstring layout:  [ beta_{norb-1..0}  alpha_{norb-1..0} ]
#   right half  -> alpha
#   left half   -> beta


# --------------------------------------------------------------------------- #
#  Data containers
# --------------------------------------------------------------------------- #
@dataclass
class SCIState:
    """An SQD wavefunction: amplitudes on a set of determinants.

    Attributes
    ----------
    amplitudes : ndarray, shape (n_a_strs, n_b_strs)
        CI coefficient matrix.
    ci_strs_a : ndarray of int
        Alpha determinant strings (bit-encoded).
    ci_strs_b : ndarray of int
        Beta determinant strings (bit-encoded).
    norb : int
        Number of spatial orbitals.
    nelec : tuple(int, int)
        ``(n_alpha, n_beta)``.
    """

    amplitudes: np.ndarray
    ci_strs_a: np.ndarray
    ci_strs_b: np.ndarray
    norb: int
    nelec: Tuple[int, int]

    def save(self, filename: str) -> None:
        np.savez(filename, amplitudes=self.amplitudes,
                 ci_strs_a=self.ci_strs_a, ci_strs_b=self.ci_strs_b,
                 norb=self.norb, nelec=np.array(self.nelec))

    @classmethod
    def load(cls, filename: str) -> "SCIState":
        d = np.load(filename, allow_pickle=True)
        return cls(d["amplitudes"], d["ci_strs_a"], d["ci_strs_b"],
                   int(d["norb"]), tuple(int(x) for x in d["nelec"]))

    def orbital_occupancies(self) -> Tuple[np.ndarray, np.ndarray]:
        """Average orbital occupation numbers for alpha and beta."""
        amps = np.asarray(self.amplitudes)
        if amps.ndim == 1:
            amps = amps.reshape(len(self.ci_strs_a), len(self.ci_strs_b))
        # |c_{ab}|^2 gives probability of (a_str, b_str)
        probs = np.abs(amps) ** 2
        probs /= probs.sum()
        occ_a = np.zeros(self.norb)
        occ_b = np.zeros(self.norb)
        for i, sa in enumerate(self.ci_strs_a):
            bits = _int_to_bits(sa, self.norb)
            for b in range(self.norb):
                if bits[b]:
                    occ_a[b] += probs[i, :].sum()
        for j, sb in enumerate(self.ci_strs_b):
            bits = _int_to_bits(sb, self.norb)
            for b in range(self.norb):
                if bits[b]:
                    occ_b[b] += probs[:, j].sum()
        return occ_a, occ_b

    def rdm(self, rank: int = 1, spin_summed: bool = False) -> np.ndarray:
        """Compute the reduced density matrix via PySCF.

        Parameters
        ----------
        rank : int
            1 for 1-RDM, 2 for 2-RDM.
        spin_summed : bool
            If ``True``, return the spin-traced (spatial) RDM.
            If ``False``, return spin-separated RDMs as a tuple.

        Returns
        -------
        ndarray | tuple of ndarray
            For ``spin_summed=True``: single array of shape ``(norb, norb)``
            (rank 1) or ``(norb, norb, norb, norb)`` (rank 2).
            For ``spin_summed=False``: tuple ``(rdm_a, rdm_b)`` for rank 1,
            or the 3 independent spin blocks ``(rdm_aa, rdm_ab, rdm_bb)``
            for rank 2 (PySCF ``make_rdm2s`` convention).
        """
        civec = self._as_scivector()
        if rank == 1:
            if spin_summed:
                # make_rdm1 returns the spin-summed (norb, norb) array directly
                return selected_ci.make_rdm1(civec, self.norb, self.nelec)
            else:
                # make_rdm1s returns (rdm_a, rdm_b) as a tuple
                return selected_ci.make_rdm1s(civec, self.norb, self.nelec)
        elif rank == 2:
            if spin_summed:
                # make_rdm2 returns the spin-summed (norb, norb, norb, norb) array
                return selected_ci.make_rdm2(civec, self.norb, self.nelec)
            else:
                return selected_ci.make_rdm2s(civec, self.norb, self.nelec)
        else:
            raise ValueError("Only rank 1 and 2 RDMs are supported.")

    def spin_square(self) -> float:
        s2 = selected_ci.spin_square(self._as_scivector(), self.norb, self.nelec)[0]
        return float(s2)

    def _as_scivector(self):
        """Return amplitudes as a PySCF ``SCIvector`` carrying ``_strs``.

        After :meth:`load`, ``amplitudes`` is a plain ``ndarray`` without the
        ``_strs`` metadata that PySCF's ``make_rdm*`` / ``spin_square`` rely
        on; this helper re-attaches it so all state methods work on both
        freshly-computed and reloaded states.
        """
        amps = np.asarray(self.amplitudes)
        if hasattr(amps, "_strs"):
            return amps
        if amps.ndim == 2:
            amps = amps.ravel()
        return selected_ci._as_SCIvector(
            amps,
            (np.asarray(self.ci_strs_a), np.asarray(self.ci_strs_b)),
        )


@dataclass
class SCIResult:
    """Result of an SQD diagonalisation."""

    energy: float
    sci_state: SCIState
    avg_orb_occupancies: Tuple[np.ndarray, np.ndarray]
    spin_square: float = 0.0
    eci: float = 0.0  # correlation energy


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _int_to_bits(val: int, nbits: int) -> np.ndarray:
    """Convert integer to a bit array of length *nbits* (LSB = orbital 0)."""
    return np.array([(val >> i) & 1 for i in range(nbits)], dtype=bool)


def bitstring_matrix_to_ci_strs(
    bitstring_matrix: np.ndarray,
    open_shell: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a bitstring matrix to ``(ci_strs_a, ci_strs_b)``.

    Each row is split into left (beta) and right (alpha) halves.  The bits in
    each half are packed into an integer (orbital 0 = LSB).

    Parameters
    ----------
    bitstring_matrix : ndarray, shape (S, 2*norb)
    open_shell : bool
        If ``False`` (default), merge alpha and beta strings into a single
        unique set used for both spin sectors.

    Returns
    -------
    ci_strs_a : ndarray of int
    ci_strs_b : ndarray of int
    """
    bsm = np.asarray(bitstring_matrix, dtype=bool)
    n = bsm.shape[1]
    norb = n // 2
    # right half = alpha, left half = beta
    alpha_bits = bsm[:, norb:]  # shape (S, norb), orbital norb-1 .. 0 left-to-right
    beta_bits = bsm[:, :norb]

    # Pack bits into integers: bit 0 (LSB) = orbital 0
    powers = (1 << np.arange(norb, dtype=np.uint64))
    a_strs = (alpha_bits[:, ::-1].astype(np.int64) @ powers.astype(np.int64)).ravel()
    b_strs = (beta_bits[:, ::-1].astype(np.int64) @ powers.astype(np.int64)).ravel()

    a_unique = np.unique(a_strs)
    b_unique = np.unique(b_strs)

    if not open_shell:
        merged = np.union1d(a_unique, b_unique)
        a_unique = b_unique = merged

    return a_unique, b_unique


def enlarge_batch_from_transitions(
    bitstring_matrix: np.ndarray,
    transition_operators: np.ndarray,
) -> np.ndarray:
    """Apply transition operators (I/+/-/n) to bitstrings, augmenting the set.

    ``transition_operators`` is a 1-D or 2-D array of character codes where
    each entry is one of ``'I'`` (identity), ``'+'`` (creation),
    ``'-'`` (annihilation), ``'n'`` (number).

    Returns
    -------
    ndarray
        Augmented bitstring matrix (original + generated, de-duplicated).
    """
    bsm = np.asarray(bitstring_matrix, dtype=bool).copy()
    if transition_operators.ndim == 1:
        transition_operators = transition_operators.reshape(1, -1)
    n = bsm.shape[1]
    new_rows = []
    for row in bsm:
        for op_row in transition_operators:
            new = row.copy()
            valid = True
            for col, code in enumerate(op_row):
                c = chr(int(code)) if isinstance(code, (int, np.integer)) else code
                if c == 'I':
                    pass
                elif c == '+':
                    if new[col]:
                        valid = False
                        break
                    new[col] = True
                elif c == '-':
                    if not new[col]:
                        valid = False
                        break
                    new[col] = False
                elif c == 'n':
                    if not new[col]:
                        valid = False
                        break
            if valid:
                new_rows.append(new)
    if new_rows:
        all_rows = np.vstack([bsm, np.array(new_rows, dtype=bool)])
    else:
        all_rows = bsm
    # de-duplicate
    int_vals = (all_rows.astype(np.uint64) @ (1 << np.arange(n - 1, -1, -1, dtype=np.uint64))).ravel()
    uniq = np.unique(int_vals)
    result = np.zeros((len(uniq), n), dtype=bool)
    for i, val in enumerate(uniq):
        result[i] = ((val >> np.arange(n - 1, -1, -1, dtype=np.uint64)) & 1).astype(bool)
    return result


# --------------------------------------------------------------------------- #
#  CI matrix construction
# --------------------------------------------------------------------------- #
def build_ci_matrix(
    ci_strs_a: np.ndarray,
    ci_strs_b: np.ndarray,
    h1e: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    ecore: float = 0.0,
) -> np.ndarray:
    """Build the CI Hamiltonian matrix in the subspace spanned by
    ``(ci_strs_a, ci_strs_b)`` using Slater-Condon rules (via PySCF).

    The matrix element ``H[i,j]`` corresponds to the determinant pair
    ``(ci_strs_a[i] x ci_strs_b[i])`` and ``(ci_strs_a[j] x ci_strs_b[j])``.

    Parameters
    ----------
    ci_strs_a : ndarray of int
        Alpha determinant strings.
    ci_strs_b : ndarray of int
        Beta determinant strings.
    h1e : ndarray, shape (norb, norb)
        One-body integrals in the spin-orbital basis? No --- **spatial-orbital**
        basis, PySCF convention ``(h1e[alpha] + h1e[beta])``.
    eri : ndarray, shape (norb, norb, norb, norb)
        Two-electron repulsion integrals, chemist's notation, **spatial**.
    norb : int
    nelec : tuple(int, int)
        ``(n_alpha, n_beta)``.
    ecore : float
        Core energy (nuclear repulsion + frozen-core).

    Returns
    -------
    H : ndarray, shape (na*nb, na*nb)
        Full CI matrix in the subspace.
    """
    na_strs = np.asarray(ci_strs_a, dtype=np.int64)
    nb_strs = np.asarray(ci_strs_b, dtype=np.int64)
    na_e, nb_e = nelec
    myci = selected_ci.SCI()
    # Set the CI strings on the solver (required by contract_2e / contract_1e)
    myci._strs = (na_strs, nb_strs)

    # PySCF's direct_spin1 expects h1e as a single (norb, norb) matrix
    if h1e.ndim == 3:
        if not np.allclose(h1e[0], h1e[1]):
            raise ValueError(
                "Spin-resolved one-body integrals (shape (2, norb, norb) with "
                "h_alpha != h_beta) are not supported; pass a single "
                "(norb, norb) closed-shell h1e."
            )
        h1e_single = np.asarray(h1e[0], dtype=np.float64)
    else:
        h1e_single = np.asarray(h1e, dtype=np.float64)

    h2e = np.asarray(eri, dtype=np.float64)
    na_len = len(na_strs)
    nb_len = len(nb_strs)
    dim = na_len * nb_len
    H = np.zeros((dim, dim), dtype=np.float64)

    # Absorb h1e into h2e (same as kernel_fixed_space does)
    from pyscf.fci import direct_spin1
    from pyscf import ao2mo
    h2e_abs = direct_spin1.absorb_h1e(h1e_single, h2e, norb, nelec, 0.5)
    h2e_abs = ao2mo.restore(1, h2e_abs, norb)

    # Generate linkstr index for the given CI strings
    link_index = selected_ci._all_linkstr_index(
        (na_strs, nb_strs), norb, nelec
    )

    # Build the full matrix by applying H to each basis vector
    for i in range(dim):
        ia, ib = divmod(i, nb_len) if nb_len > 0 else (i, 0)
        vec = np.zeros((na_len, nb_len), dtype=np.float64)
        vec[ia, ib] = 1.0
        vec_flat = vec.ravel()
        # Wrap as SCIvector so contract_2e can find ci_strs
        scivec = selected_ci._as_SCIvector(vec_flat, (na_strs, nb_strs))
        hvec = myci.contract_2e(h2e_abs, scivec, norb, nelec, link_index)
        H[:, i] = np.asarray(hvec).ravel()

    H += ecore * np.eye(dim)
    return H


# --------------------------------------------------------------------------- #
#  Subspace diagonalisation
# --------------------------------------------------------------------------- #
def solve_sci(
    ci_strings: Tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    spin_sq: Optional[float] = None,
    **kwargs,
) -> SCIResult:
    """Diagonalise the Hamiltonian in the subspace defined by ``ci_strings``.

    Parameters
    ----------
    ci_strings : tuple (ci_strs_a, ci_strs_b)
    one_body_tensor : ndarray, shape (norb, norb) or (2, norb, norb)
    two_body_tensor : ndarray, shape (norb, norb, norb, norb)
    norb : int
    nelec : tuple(int, int)
    spin_sq : float | None
        Target S².  ``None`` = no constraint.
    **kwargs
        Forwarded to ``pyscf.fci.selected_ci.kernel_fixed_space``.

    Returns
    -------
    SCIResult
    """
    ci_strs_a, ci_strs_b = ci_strings
    myci = selected_ci.SCI()

    # Prepare h1e in PySCF format: direct_spin1 expects a single (norb, norb)
    # matrix (not spin-separated).
    if one_body_tensor.ndim == 3:
        # (2, norb, norb): accept only if alpha/beta blocks are identical
        # (closed-shell); otherwise the Hamiltonian is genuinely spin-dependent
        # and the spin-orbital backend here cannot represent it.
        if not np.allclose(one_body_tensor[0], one_body_tensor[1]):
            raise ValueError(
                "Spin-resolved one-body integrals (shape (2, norb, norb) with "
                "h_alpha != h_beta) are not supported; pass a single "
                "(norb, norb) closed-shell h1e."
            )
        h1e = np.asarray(one_body_tensor[0])
    else:
        h1e = np.asarray(one_body_tensor)

    if spin_sq is not None:
        # Genuine target-spin selection: diagonalise several roots, evaluate
        # S^2 on each, then keep the lowest-energy root whose S^2 matches the
        # target within tolerance.  This is a real constraint (unlike merely
        # setting solver attributes), but it only works when the subspace
        # contains states of the requested spin.
        n_spin_roots = kwargs.pop("n_spin_roots", 10)
        spin_tol = kwargs.pop("spin_tol", 1e-2)
        e_roots, c_roots = selected_ci.kernel_fixed_space(
            myci, h1e, two_body_tensor, norb, nelec,
            (np.asarray(ci_strs_a), np.asarray(ci_strs_b)),
            nroots=min(n_spin_roots, len(ci_strs_a) * len(ci_strs_b)),
            **kwargs,
        )
        e_roots = np.atleast_1d(e_roots)
        best_idx = None
        best_err = np.inf
        for i in range(len(e_roots)):
            s2_i = float(selected_ci.spin_square(c_roots[i], norb, nelec)[0])
            err = abs(s2_i - spin_sq)
            if err < best_err:
                best_err = err
                best_idx = i
        if best_err > spin_tol:
            raise ValueError(
                f"No subspace state matches target S^2={spin_sq} within "
                f"tolerance {spin_tol}; closest root has S^2 error "
                f"{best_err:.4f}."
            )
        e_tot = float(e_roots[best_idx])
        civec = c_roots[best_idx]
        s2 = float(selected_ci.spin_square(civec, norb, nelec)[0])
    else:
        e_tot, civec = selected_ci.kernel_fixed_space(
            myci, h1e, two_body_tensor, norb, nelec,
            (np.asarray(ci_strs_a), np.asarray(ci_strs_b)),
            **kwargs,
        )
        e_tot = float(e_tot)
        # Compute spin
        try:
            s2 = float(selected_ci.spin_square(civec, norb, nelec)[0])
        except Exception:
            s2 = 0.0

    state = SCIState(
        amplitudes=civec,
        ci_strs_a=np.asarray(ci_strs_a),
        ci_strs_b=np.asarray(ci_strs_b),
        norb=norb,
        nelec=nelec,
    )
    occ_a, occ_b = state.orbital_occupancies()
    return SCIResult(
        energy=float(e_tot),
        sci_state=state,
        avg_orb_occupancies=(occ_a, occ_b),
        spin_square=s2,
    )


def solve_sci_batch(
    ci_strings_list: List[Tuple[np.ndarray, np.ndarray]],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    spin_sq: Optional[float] = None,
    **kwargs,
) -> List[SCIResult]:
    """Diagonalise in multiple subspaces (batch version)."""
    results = []
    for ci_strs in ci_strings_list:
        results.append(
            solve_sci(ci_strs, one_body_tensor, two_body_tensor,
                      norb, nelec, spin_sq=spin_sq, **kwargs)
        )
    return results


# --------------------------------------------------------------------------- #
#  High-level SQD entry points
# --------------------------------------------------------------------------- #
def solve_fermion(
    bitstring_matrix,
    hcore,
    eri,
    *,
    open_shell: bool = False,
    spin_sq: Optional[float] = None,
    shift: float = 0.1,
    **kwargs,
) -> Tuple[float, SCIState, Tuple[np.ndarray, np.ndarray], float]:
    """Solve SQD given a bitstring matrix and molecular integrals.

    Parameters
    ----------
    bitstring_matrix : ndarray | tuple
        Either a 2-D bool array (bitstrings) or a tuple
        ``(ci_strs_a, ci_strs_b)``.
    hcore : ndarray, shape (norb, norb)
        One-body integrals (spatial orbital basis).
    eri : ndarray, shape (norb, norb, norb, norb)
        Two-body integrals (chemist notation, spatial).
    open_shell : bool
    spin_sq : float | None
        Target S².  ``None`` = no constraint.
    shift : float
        Level shift for spin penalty (unused if ``spin_sq`` is None).
    **kwargs
        Forwarded to ``solve_sci``.

    Returns
    -------
    energy : float
    sci_state : SCIState
    avg_orb_occupancies : tuple(ndarray, ndarray)
    spin_sq : float
    """
    # Determine if we have a bitstring matrix or CI strings
    if isinstance(bitstring_matrix, tuple):
        ci_strs_a, ci_strs_b = bitstring_matrix
        norb = hcore.shape[-1]
        nelec = kwargs.pop("_nelec", None)
        if nelec is None:
            raise ValueError("nelec must be provided when passing CI strings directly.")
    else:
        bsm = np.asarray(bitstring_matrix, dtype=bool)
        norb = bsm.shape[1] // 2
        nelec = kwargs.pop("_nelec", None)
        if nelec is None:
            raise ValueError("nelec must be provided.")
        if not open_shell and nelec[0] != nelec[1]:
            raise ValueError(
                "open_shell=False merges alpha/beta CI strings and is only "
                "valid for n_alpha == n_beta. Pass open_shell=True for "
                f"unequal electron counts (got {nelec})."
            )
        ci_strs_a, ci_strs_b = bitstring_matrix_to_ci_strs(bsm, open_shell=open_shell)

    # Infer norb from hcore if available
    norb = hcore.shape[-1]

    # Spin selection is handled inside ``solve_sci`` via multi-root S^2
    # matching; the ``shift`` parameter is retained for API compatibility but
    # no longer applies a penalty (the penalty approach was never implemented).
    result = solve_sci(
        (ci_strs_a, ci_strs_b), np.asarray(hcore), eri, norb, nelec,
        spin_sq=spin_sq, **kwargs,
    )
    return (
        result.energy,
        result.sci_state,
        result.avg_orb_occupancies,
        result.spin_square,
    )


def diagonalize_fermionic_hamiltonian(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    bit_array,
    samples_per_batch: int,
    norb: int,
    nelec: Tuple[int, int],
    *,
    num_batches: int = 1,
    energy_tol: float = 1e-8,
    occupancies_tol: float = 1e-5,
    max_iterations: int = 100,
    sci_solver: Optional[Callable] = None,
    symmetrize_spin: bool = False,
    max_dim: Optional[Union[int, Tuple[int, int]]] = None,
    include_configurations=None,
    initial_occupancies: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    carryover_threshold: float = 1e-4,
    callback: Optional[Callable] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> SCIResult:
    """Run the iterative SQD algorithm.

    Iterates: configuration recovery -> subsampling -> SCI diagonalisation ->
    update average occupancies -> repeat until convergence.

    Parameters
    ----------
    one_body_tensor, two_body_tensor : ndarray
        Molecular integrals (spatial-orbital basis, chemist's notation for 2e).
    bit_array : ndarray | tuple
        Either a 2-D bool bitstring matrix, or ``(bitstring_matrix, probabilities)``.
    samples_per_batch : int
    norb : int
    nelec : tuple(int, int)
    num_batches : int
    energy_tol, occupancies_tol : float
    max_iterations : int
    carryover_threshold : float
    seed : int | Generator | None

    Returns
    -------
    SCIResult
    """
    if max_dim is not None:
        raise NotImplementedError(
            "max_dim subspace-dimension limits are not implemented; "
            "control the subspace via samples_per_batch / num_batches."
        )
    if max_iterations < 1:
        raise ValueError(
            f"max_iterations must be >= 1, got {max_iterations}."
        )
    # Consistency checks between norb, integrals and bitstring width.
    if one_body_tensor.shape[0] != norb:
        raise ValueError(
            f"norb={norb} does not match one_body_tensor shape "
            f"{one_body_tensor.shape}."
        )

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    # Unpack bit_array
    if isinstance(bit_array, tuple):
        bsm, probs = bit_array
    else:
        bsm = np.asarray(bit_array, dtype=bool)
        probs = np.ones(bsm.shape[0]) / bsm.shape[0]
    if bsm.shape[1] != 2 * norb:
        raise ValueError(
            f"bit_array width must be 2*norb = {2 * norb}, got {bsm.shape[1]}."
        )

    if not (0.0 <= carryover_threshold <= 1.0):
        raise ValueError(
            f"carryover_threshold must be in [0, 1], got {carryover_threshold}."
        )

    # Normalise include_configurations to a bitstring matrix (or None).
    include_bsm = None
    if include_configurations is not None:
        inc = np.asarray(include_configurations, dtype=bool)
        if inc.ndim == 1:
            inc = inc.reshape(1, -1)
        if inc.ndim != 2:
            raise ValueError(
                f"include_configurations must be 1-D or 2-D, got ndim={inc.ndim}."
            )
        if inc.shape[1] != 2 * norb:
            raise ValueError(
                f"include_configurations must have {2 * norb} columns, "
                f"got {inc.shape[1]}."
            )
        include_bsm = inc

    na_e, nb_e = nelec

    if initial_occupancies is not None:
        occ_a, occ_b = initial_occupancies
    else:
        # Start from HF guess: occupy lowest orbitals
        occ_a = np.zeros(norb)
        occ_a[:na_e] = 1.0
        occ_b = np.zeros(norb)
        occ_b[:nb_e] = 1.0

    best_result = None
    prev_energy = np.inf
    prev_occ = (occ_a.copy(), occ_b.copy())
    carryover_bsm = None

    for iteration in range(max_iterations):
        # 1. Configuration recovery
        recovered_bsm, recovered_probs = recover_configurations(
            bsm, probs, (occ_a, occ_b), na_e, nb_e, rand_seed=rng,
        )

        # 1b. Merge carryover configurations into the sampling pool.
        if carryover_bsm is not None and carryover_bsm.shape[0] > 0:
            carry_probs = np.full(
                carryover_bsm.shape[0],
                carryover_threshold * recovered_probs.max()
                if recovered_probs.size else carryover_threshold,
            )
            recovered_bsm = np.vstack([recovered_bsm, carryover_bsm])
            recovered_probs = np.concatenate([recovered_probs, carry_probs])
            recovered_probs /= recovered_probs.sum()

        # 2. Subsample into batches
        batches = subsample(
            recovered_bsm, recovered_probs,
            samples_per_batch, num_batches, rand_seed=rng,
        )

        # 3. Convert to CI strings and solve.  ``include_configurations`` and
        #    carryover determinants are *force-appended* to every batch's CI
        #    strings so they are always present in the diagonalised subspace,
        #    regardless of the probabilistic subsampling above.
        forced_ci_a, forced_ci_b = None, None
        forced_rows = []
        if include_bsm is not None:
            forced_rows.append(include_bsm)
        if carryover_bsm is not None and carryover_bsm.shape[0] > 0:
            forced_rows.append(carryover_bsm)
        if forced_rows:
            forced_bsm = np.vstack(forced_rows)
            forced_ci_a, forced_ci_b = bitstring_matrix_to_ci_strs(
                forced_bsm, open_shell=not symmetrize_spin,
            )

        all_ci_strs = []
        for batch in batches:
            ci_strs_a, ci_strs_b = bitstring_matrix_to_ci_strs(
                batch, open_shell=not symmetrize_spin,
            )
            if forced_ci_a is not None:
                ci_strs_a = np.union1d(ci_strs_a, forced_ci_a)
                ci_strs_b = np.union1d(ci_strs_b, forced_ci_b)
            all_ci_strs.append((ci_strs_a, ci_strs_b))

        if sci_solver is not None:
            results = sci_solver(
                all_ci_strs, one_body_tensor, two_body_tensor, norb, nelec,
            )
        else:
            results = solve_sci_batch(
                all_ci_strs, one_body_tensor, two_body_tensor,
                norb, nelec,
            )

        # 4. Pick best result
        best_in_batch = min(results, key=lambda r: r.energy)
        if best_result is None or best_in_batch.energy < best_result.energy:
            best_result = best_in_batch

        # 5. Update occupancies
        new_occ_a, new_occ_b = best_in_batch.avg_orb_occupancies
        occ_a = 0.5 * (occ_a + new_occ_a)
        occ_b = 0.5 * (occ_b + new_occ_b)

        # 6. Carryover: keep determinants whose CI weight exceeds the
        #    threshold for the next iteration.
        amps = np.abs(best_in_batch.sci_state.amplitudes)
        if amps.size and amps.max() > 0:
            keep = amps >= carryover_threshold * amps.max()
            st = best_in_batch.sci_state
            ia, ib = np.nonzero(keep)
            carry_rows = []
            for a_i, b_i in zip(ia, ib):
                bits_a = _int_to_bits(int(st.ci_strs_a[a_i]), norb)[::-1]
                bits_b = _int_to_bits(int(st.ci_strs_b[b_i]), norb)[::-1]
                carry_rows.append(np.concatenate([bits_b, bits_a]))
            carryover_bsm = (
                np.array(carry_rows, dtype=bool) if carry_rows else None
            )
        else:
            carryover_bsm = None

        # 7. Convergence check
        energy_diff = abs(best_in_batch.energy - prev_energy)
        occ_diff = max(
            np.max(np.abs(occ_a - prev_occ[0])),
            np.max(np.abs(occ_b - prev_occ[1])),
        )
        prev_energy = best_in_batch.energy
        prev_occ = (occ_a.copy(), occ_b.copy())

        if callback is not None:
            callback(results)

        if energy_diff < energy_tol and occ_diff < occupancies_tol:
            break

    return best_result


# --------------------------------------------------------------------------- #
#  Orbital optimisation
# --------------------------------------------------------------------------- #
def rotate_integrals(
    hcore: np.ndarray,
    eri: np.ndarray,
    k_flat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply orbital rotation ``U = exp(K)`` to the integrals.

    ``k_flat`` is the upper-triangular (row-major) of the anti-Hermitian
    generator ``K``.

    Returns
    -------
    hcore_rot : ndarray
    eri_rot : ndarray
    """
    norb = hcore.shape[0]
    # Reconstruct K
    K = np.zeros((norb, norb), dtype=np.float64)
    idx = 0
    for i in range(norb):
        for j in range(i + 1, norb):
            K[i, j] = k_flat[idx]
            K[j, i] = -k_flat[idx]
            idx += 1
    U = expm(K)
    # Orbital rotation: φ'_p = Σ_i φ_i U_{ip}
    #   h'_{pq} = Σ_{ij} U_{ip} h_{ij} U_{jq}  =  (U^T h U)_{pq}
    hcore_rot = U.T @ hcore @ U
    #   g'_{pqrs} = Σ_{ijkl} U_{ip} U_{jq} U_{kr} U_{ls} g_{ijkl}
    eri_rot = np.einsum(
        "ip,jq,kr,ls,ijkl->pqrs",
        U, U, U, U, eri,
        optimize=True,
    )
    return hcore_rot, eri_rot


def optimize_orbitals(
    bitstring_matrix,
    hcore,
    eri,
    k_flat,
    *,
    open_shell: bool = False,
    spin_sq: float = 0.0,
    num_iters: int = 10,
    num_steps_grad: int = 10000,
    learning_rate: float = 0.01,
    nelec: Optional[Tuple[int, int]] = None,
    **kwargs,
) -> Tuple[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Optimise orbital rotation parameters to minimise SQD energy.

    Uses simple gradient descent on ``k_flat`` (numerical gradient).

    Parameters
    ----------
    bitstring_matrix : ndarray | tuple
    hcore, eri : ndarray
    k_flat : ndarray
        Initial anti-Hermitian parameters (upper-triangular, row-major).
        Length must be ``norb * (norb - 1) // 2``.
    nelec : tuple(int, int)
        Must be provided.
    num_iters : int
        Outer SQD iterations.  Must be >= 0.
    num_steps_grad : int
        Gradient steps per outer iteration.  Must be >= 1.
    learning_rate : float
        Gradient-descent step size.  Must be positive and finite.

    Returns
    -------
    energy : float
        Energy at the best-so-far parameters (not necessarily the last step).
    k_flat : ndarray
        Best-so-far orbital-rotation parameters.
    avg_orb_occupancies : tuple
        Occupancies at the best-so-far parameters.

    Notes
    -----
    The optimiser tracks the best-so-far parameters across *all* gradient
    steps and returns them, so the returned energy is always consistent with
    the returned ``k_flat``.
    """
    if nelec is None:
        raise ValueError("nelec must be provided for optimize_orbitals.")
    norb = hcore.shape[0]
    k_flat = np.asarray(k_flat, dtype=np.float64).copy()

    expected_len = norb * (norb - 1) // 2
    if k_flat.shape[0] != expected_len:
        raise ValueError(
            f"k_flat length must be norb*(norb-1)//2 = {expected_len}, "
            f"got {k_flat.shape[0]}."
        )
    if num_iters < 0:
        raise ValueError(f"num_iters must be >= 0, got {num_iters}.")
    if num_steps_grad < 1:
        raise ValueError(f"num_steps_grad must be >= 1, got {num_steps_grad}.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be positive and finite, got {learning_rate}."
        )

    def _energy(k: np.ndarray) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
        h1_r, eri_r = rotate_integrals(hcore, eri, k)
        e, _, occ, _ = solve_fermion(
            bitstring_matrix, h1_r, eri_r,
            open_shell=open_shell, spin_sq=spin_sq,
            _nelec=nelec, **kwargs,
        )
        return e, occ

    # Track the best-so-far parameters so a bad final step cannot be returned.
    best_energy, best_occ = _energy(k_flat)
    best_k = k_flat.copy()

    eps = 1e-4
    total_steps = num_iters * num_steps_grad
    for _ in range(total_steps):
        e, occ = _energy(k_flat)
        grad = np.zeros_like(k_flat)
        for i in range(len(k_flat)):
            k_plus = k_flat.copy()
            k_plus[i] += eps
            ep, _ = _energy(k_plus)
            grad[i] = (ep - e) / eps
        k_flat -= learning_rate * grad
        e_new, occ_new = _energy(k_flat)
        if e_new < best_energy:
            best_energy, best_occ, best_k = e_new, occ_new, k_flat.copy()

    return best_energy, best_k, best_occ


# --------------------------------------------------------------------------- #
#  Unified ground-state energy entry point
# --------------------------------------------------------------------------- #
def compute_ground_state_energy(
    h1e: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    ecore: float = 0.0,
    method: str = "fci",
    bitstring_matrix: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    samples_per_batch: int = 200,
    num_batches: int = 1,
    max_iterations: int = 5,
    spin_sq: Optional[float] = None,
    verbose: bool = False,
    **kwargs,
) -> float:
    """Compute the ground-state energy from fermionic Hamiltonian integrals.

    A single high-level entry point that dispatches to the appropriate solver
    based on ``method``:

    - ``"fci"``      : exact Full-CI diagonalisation (ground truth / benchmark).
    - ``"sqd"``      : iterative SQD using sampled bitstrings.
    - ``"direct"``   : build the full CI matrix explicitly and diagonalise it
                       with ``numpy.linalg.eigvalsh`` (useful for small systems
                       and for inspecting the matrix).

    Parameters
    ----------
    h1e : ndarray, shape (norb, norb) or (2, norb, norb)
        One-body integrals in the spatial-orbital (MO) basis.
    eri : ndarray, shape (norb, norb, norb, norb)
        Two-electron repulsion integrals, chemist's notation, spatial basis.
    norb : int
        Number of spatial orbitals.
    nelec : tuple(int, int)
        ``(n_alpha, n_beta)``.
    ecore : float
        Core energy offset (nuclear repulsion + frozen-core).  Default 0.
    method : {"fci", "sqd", "direct"}
        Solver to use.
    bitstring_matrix : ndarray, optional
        **Required when ``method="sqd"``**.  Sampled bitstrings, shape (S, 2*norb).
    probabilities : ndarray, optional
        Probability of each bitstring (``method="sqd"``).  If ``None``,
        uniform weights are assumed.
    samples_per_batch : int
        SQD subsampling parameter.
    num_batches : int
        SQD subsampling parameter.
    max_iterations : int
        SQD outer-loop iteration cap.
    spin_sq : float | None
        Target spin squared S².  ``None`` = no constraint.
    verbose : bool
        Print progress information.
    **kwargs
        Extra keyword arguments forwarded to the underlying solver.

    Returns
    -------
    energy : float
        Ground-state energy (including ``ecore``).

    Examples
    --------
    >>> # Exact FCI
    >>> e = compute_ground_state_energy(h1e, eri, norb, nelec, ecore=ecore,
    ...                                  method="fci")
    >>>
    >>> # SQD from TensorCircuit samples
    >>> bsm, probs = sample_from_circuit(circuit, n_samples=2000)
    >>> e = compute_ground_state_energy(h1e, eri, norb, nelec, ecore=ecore,
    ...                                  method="sqd", bitstring_matrix=bsm,
    ...                                  probabilities=probs)
    >>>
    >>> # Direct matrix diagonalisation (small systems only)
    >>> e = compute_ground_state_energy(h1e, eri, norb, nelec, ecore=ecore,
    ...                                  method="direct")
    """
    h1e_arr = np.asarray(h1e, dtype=np.float64)
    eri_arr = np.asarray(eri, dtype=np.float64)
    na_e, nb_e = nelec

    # ---- FCI: exact diagonalisation via PySCF -----------------------------
    if method == "fci":
        # Build the full determinant list
        ci_strs_a = cistring.make_strings(range(norb), na_e)
        ci_strs_b = cistring.make_strings(range(norb), nb_e)
        if verbose:
            print(f"[FCI] Full space: {len(ci_strs_a)} x {len(ci_strs_b)} "
                  f"= {len(ci_strs_a) * len(ci_strs_b)} determinants")
        result = solve_sci(
            (ci_strs_a, ci_strs_b), h1e_arr, eri_arr, norb, nelec,
            spin_sq=spin_sq, **kwargs,
        )
        return result.energy + ecore

    # ---- SQD: iterative sample-based diagonalisation ----------------------
    elif method == "sqd":
        if bitstring_matrix is None:
            raise ValueError(
                "method='sqd' requires `bitstring_matrix` (sampled bitstrings)."
            )
        if spin_sq is not None:
            raise ValueError(
                "method='sqd' does not support spin_sq constraints; "
                "use method='fci' or method='direct' instead."
            )
        bsm = np.asarray(bitstring_matrix, dtype=bool)
        if probabilities is None:
            probs = np.full(bsm.shape[0], 1.0 / bsm.shape[0])
        else:
            probs = np.asarray(probabilities, dtype=np.float64)
        if verbose:
            print(f"[SQD] {bsm.shape[0]} bitstrings, "
                  f"batch={samples_per_batch}, batches={num_batches}, "
                  f"iters={max_iterations}")
        result = diagonalize_fermionic_hamiltonian(
            h1e_arr, eri_arr, (bsm, probs),
            samples_per_batch=samples_per_batch,
            norb=norb,
            nelec=nelec,
            num_batches=num_batches,
            max_iterations=max_iterations,
            **kwargs,
        )
        return result.energy + ecore

    # ---- Direct: build explicit CI matrix and diagonalise -----------------
    elif method == "direct":
        if spin_sq is not None:
            raise ValueError(
                "method='direct' does not support spin_sq constraints; "
                "use method='fci' instead."
            )
        ci_strs_a = cistring.make_strings(range(norb), na_e)
        ci_strs_b = cistring.make_strings(range(norb), nb_e)
        dim = len(ci_strs_a) * len(ci_strs_b)
        if verbose:
            print(f"[direct] Building {dim} x {dim} CI matrix ...")
        H = build_ci_matrix(
            ci_strs_a, ci_strs_b, h1e_arr, eri_arr,
            norb, nelec, ecore=ecore,
        )
        eigs = np.linalg.eigvalsh(H)
        return float(eigs[0])

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from 'fci', 'sqd', 'direct'."
        )
