"""Tests for cluster-adaptive configuration recovery (CSQD-style).

Covers ``recover_configurations_clustered`` and the internal ``_weighted_kmodes``.
Run with: ``PYTHONPATH=src python -m tests.test_recovery_clustered``
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pyscf
import pyscf.scf
import tc_sqd
from tc_sqd.configuration_recovery import _weighted_kmodes


def test_kmodes_convergence():
    """``_weighted_kmodes`` recovers two known clusters on a synthetic dataset."""
    rng = np.random.default_rng(42)
    # Cluster A: orbital 0,1 always occupied; cluster B: orbital 2,3 always.
    base_a = np.array([[1, 1, 0, 0]] * 20)
    base_b = np.array([[0, 0, 1, 1]] * 20)
    # Add light noise so it's not trivially separable.
    noise_a = rng.random(base_a.shape) < 0.05
    noise_b = rng.random(base_b.shape) < 0.05
    data = np.vstack([base_a ^ noise_a, base_b ^ noise_b])
    w = np.ones(data.shape[0])

    labels, centroids = _weighted_kmodes(data, w, k=2, max_iter=20, rng=rng)

    # The two recovered centroids must span the two distinct modes (up to label
    # permutation).  Each centroid should be dominated by one of {0,1} vs {2,3}.
    modes = [tuple(c > 0.5) for c in centroids]
    assert ((1, 1, 0, 0) in modes and (0, 0, 1, 1) in modes) or \
           ((1, 1, 0, 0) in modes and (0, 0, 1, 1) in modes), \
        f"k-modes centroids {modes} do not match the two planted modes"
    # Samples within each planted group should share a single label.
    assert len(np.unique(labels[:20])) == 1, "cluster A split across labels"
    assert len(np.unique(labels[20:])) == 1, "cluster B split across labels"
    assert labels[0] != labels[20], "the two planted clusters collapsed into one"
    print("  PASS: _weighted_kmodes recovers two planted clusters")


def test_clustered_recovers_particle_number():
    """Recovered bitstrings all satisfy the target alpha/beta Hamming weights."""
    rng = np.random.default_rng(0)
    norb, na, nb = 4, 2, 2
    # Build a bimodal pool: half the strings prefer orbitals {0,1}, half {2,3},
    # with random bit-flips that break particle-number symmetry.
    n = 200
    bsm = np.zeros((n, 2 * norb), dtype=bool)
    for i in range(n):
        if i < n // 2:
            base = [1, 1, 0, 0]   # beta
            base2 = [1, 1, 0, 0]  # alpha
        else:
            base = [0, 0, 1, 1]
            base2 = [0, 0, 1, 1]
        # Inject ~1 random flip per half to violate particle number.
        for arr in (base, base2):
            j = rng.integers(0, norb)
            arr[j] ^= 1
        bsm[i, :norb] = base
        bsm[i, norb:] = base2
    probs = rng.random(n) + 0.01

    rec, rec_probs = tc_sqd.recover_configurations_clustered(
        bsm, probs, na, nb, n_clusters=2, rand_seed=0,
    )
    assert rec.shape[0] > 0
    assert np.all(rec[:, norb:].sum(axis=1) == na), \
        "alpha half must all have weight == num_elec_a"
    assert np.all(rec[:, :norb].sum(axis=1) == nb), \
        "beta half must all have weight == num_elec_b"
    assert abs(rec_probs.sum() - 1.0) < 1e-9, "probabilities must renormalise"
    print(f"  PASS: all {rec.shape[0]} recovered rows satisfy particle numbers")


def test_clustered_matches_plain_on_single_mode():
    """With k=1 (or a unimodal pool), clustered recovery reduces to global."""
    # Unimodal pool: all strings centred on orbital 0,1 for both spins.
    rng = np.random.default_rng(1)
    norb, na, nb = 3, 1, 1
    n = 80
    bsm = np.zeros((n, 2 * norb), dtype=bool)
    for i in range(n):
        bsm[i, norb - 1] = True      # beta orbital 0
        bsm[i, -1] = True            # alpha orbital 0
        if rng.random() < 0.3:       # mild noise
            bsm[i, rng.integers(0, 2 * norb)] ^= True
    probs = np.ones(n)

    # Clustered with k=1 should behave like the global-average recovery.
    occ_global = np.asarray(bsm, dtype=float).mean(axis=0)
    occ_a = occ_global[norb:][::-1]
    occ_b = occ_global[:norb][::-1]
    rec_plain, _ = tc_sqd.recover_configurations(
        bsm, probs, (occ_a, occ_b), na, nb, rand_seed=0,
    )
    rec_clust, _ = tc_sqd.recover_configurations_clustered(
        bsm, probs, na, nb, n_clusters=1, rand_seed=0,
    )
    # Compare as sets of integer-encoded bitstrings.
    def _to_ints(mat):
        return np.sort(
            mat.astype(np.uint64)
            @ (1 << np.arange(2 * norb - 1, -1, -1, dtype=np.uint64)).reshape(-1, 1)
        ).ravel()
    assert np.array_equal(_to_ints(rec_plain), _to_ints(rec_clust)), \
        "k=1 clustered recovery must equal global-average recovery"
    print("  PASS: k=1 clustered recovery matches global-average recovery")


def test_clustered_beats_global_on_strongly_correlated():
    """On a real strongly-correlated system, clustered recovery preserves more
    determinant families than global averaging, yielding a richer subspace and
    lower energy.

    System: N2 / cc-pVDZ, 10 electrons in 10 orbitals (active space).  This is
    the repo's standard strong-correlation stress test (see SURVEY).  We sample
    from a shallow entangled circuit, inject T1-like bit-flip noise, and compare
    the two recovery strategies.  The global average occupancy is uninformative
    in this regime (near-uniform), so global recovery collapses the determinant
    pool, while clustered (k>=4) recovery keeps multiple families alive.
    """
    from pyscf import mcscf, ao2mo
    import tensorcircuit as tc2

    mol = pyscf.M(atom="N 0 0 -1.5; N 0 0 1.5", basis="cc-pVDZ",
                  spin=0, verbose=0)
    mf = pyscf.scf.RHF(mol).run()
    ncas, nelecas = 10, 10
    cas = mcscf.CASCI(mf, ncas, nelecas)
    h1e, ecore = cas.h1e_for_cas()
    ncore = int((mf.mo_occ.sum() - nelecas) // 2)
    eri = ao2mo.full(mol, mf.mo_coeff[:, ncore:ncore + ncas],
                     aosym="1").reshape([ncas] * 4)
    norb, na, nb = ncas, nelecas // 2, nelecas // 2

    # --- Sample from a shallow entangled circuit + inject noise.
    c = tc2.Circuit(2 * norb)
    for i in range(na):
        c.x(norb + i)
    for i in range(nb):
        c.x(i)
    for k in range(3):           # mild entangler → some determinant diversity
        c.ry(k % norb, theta=0.3)
        c.cnot(k % norb, (k + 1) % norb)
    bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=3000)
    rng = np.random.default_rng(1)
    flips = rng.random(bsm.shape) < 0.06
    bsm_noisy = bsm ^ flips

    # --- Global recovery.
    occ_col = bsm_noisy.mean(axis=0)
    occ_a, occ_b = occ_col[norb:][::-1], occ_col[:norb][::-1]
    rec_g, _ = tc_sqd.recover_configurations(
        bsm_noisy, probs, (occ_a, occ_b), na, nb, rand_seed=0,
    )

    # --- Clustered recovery (k=8).
    rec_c, _ = tc_sqd.recover_configurations_clustered(
        bsm_noisy, probs, na, nb, n_clusters=8, rand_seed=0,
    )

    def _energy(bsm_rec):
        ci_a, ci_b = tc_sqd.bitstring_matrix_to_ci_strs(bsm_rec)
        res = tc_sqd.solve_sci((ci_a, ci_b), h1e, eri, norb, (na, nb))
        return res.energy + ecore

    e_global = _energy(rec_g)
    e_clustered = _energy(rec_c)

    print(f"  [N2/cc-pVDZ 10o] E(global)  = {e_global:.6f}  "
          f"(subspace dim = {rec_g.shape[0]})")
    print(f"  [N2/cc-pVDZ 10o] E(cluster) = {e_clustered:.6f}  "
          f"(subspace dim = {rec_c.shape[0]})")

    # Core claim: clustered preserves more distinct determinants AND yields a
    # lower (better) variational energy than global on this strong-correlation
    # benchmark.
    assert rec_c.shape[0] > rec_g.shape[0], (
        f"clustered subspace ({rec_c.shape[0]}) not larger than global "
        f"({rec_g.shape[0]})"
    )
    assert e_clustered < e_global - 1e-3, (
        f"clustered energy {e_clustered:.6f} not meaningfully better than "
        f"global {e_global:.6f} (need ΔE < -1e-3 Ha)"
    )
    print(f"  PASS: clustered +{rec_c.shape[0] - rec_g.shape[0]} dets, "
          f"ΔE = {e_clustered - e_global:+.4f} Ha vs global")


def test_solve_sqd_single_clustered():
    """solve_sqd(mode='single', recovery='clustered') produces the same result
    as calling recover_configurations_clustered + solve_sci directly."""
    mol = pyscf.M(atom="H 0 0 0; H 0 0 0.74; H 0 0 1.5; H 0 0 2.2",
                  basis="sto-3g", verbose=0)
    mf = pyscf.scf.RHF(mol).run()
    mo = mf.mo_coeff
    norb = mol.nao_nr()
    na = nb = mol.nelectron // 2
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", mol.intor("int2e_sph"),
                    mo, mo, mo, mo)
    ecore = mf.energy_nuc()

    rng = np.random.default_rng(3)
    bsm = rng.random((200, 2 * norb)) > 0.5
    probs = rng.random(200) + 0.01
    probs /= probs.sum()

    # Direct path
    from tc_sqd.configuration_recovery import recover_configurations_clustered
    rec, _ = recover_configurations_clustered(
        bsm, probs, na, nb, n_clusters=3, rand_seed=0)
    ci_a, ci_b = tc_sqd.bitstring_matrix_to_ci_strs(rec)
    e_direct = tc_sqd.solve_sci(
        (ci_a, ci_b), h1e, eri, norb, (na, nb)).energy + ecore

    # Via solve_sqd
    res = tc_sqd.solve_sqd(
        h1e, eri, norb, (na, nb), ecore=ecore,
        bitstring_matrix=bsm, probabilities=probs,
        mode="single", recovery="clustered", n_clusters=3, seed=0,
    )
    e_via = res.energy + ecore

    assert abs(e_direct - e_via) < 1e-9, (
        f"solve_sqd single+clustered ({e_via:.10f}) != direct path "
        f"({e_direct:.10f})"
    )
    print(f"  PASS: solve_sqd(single, clustered) matches direct path "
          f"(E = {e_via:.6f})")


def test_solve_sqd_rejects_invalid_recovery():
    """Invalid recovery value raises ValueError in both solve_sqd and
    diagonalize_fermionic_hamiltonian."""
    bsm = np.random.default_rng(0).random((10, 6)) > 0.5
    h1e = np.zeros((3, 3))
    eri = np.zeros((3, 3, 3, 3))

    try:
        tc_sqd.solve_sqd(h1e, eri, 3, (1, 1), bitstring_matrix=bsm,
                         mode="single", recovery="bogus")
        assert False, "should have raised"
    except ValueError:
        pass

    try:
        tc_sqd.diagonalize_fermionic_hamiltonian(
            h1e, eri, bsm, 10, 3, (1, 1),
            max_iterations=1, recovery="also-bogus")
        assert False, "should have raised"
    except ValueError:
        pass
    print("  PASS: invalid recovery value rejected by both entry points")


def test_diagonalize_clustered_runs():
    """diagonalize_fermionic_hamiltonian(recovery='clustered') runs end-to-end
    and returns a valid SCIResult (energy >= FCI)."""
    mol = pyscf.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = pyscf.scf.RHF(mol).run()
    mo = mf.mo_coeff
    norb = mol.nao_nr()
    na = nb = 1
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", mol.intor("int2e_sph"),
                    mo, mo, mo, mo)
    ecore = mf.energy_nuc()
    e_fci = tc_sqd.compute_ground_state_energy(
        h1e, eri, norb, (na, nb), ecore=ecore, method="fci")

    rng = np.random.default_rng(5)
    bsm = rng.random((100, 2 * norb)) > 0.5

    res = tc_sqd.diagonalize_fermionic_hamiltonian(
        h1e, eri, bsm, 50, norb, (na, nb),
        max_iterations=3, recovery="clustered", n_clusters=2, seed=0,
    )
    e = res.energy + ecore
    assert e >= e_fci - 1e-9, f"energy {e} below FCI {e_fci} (not variational)"
    assert np.isfinite(e), "energy must be finite"
    print(f"  PASS: diagonalize_fermionic_hamiltonian(clustered) -> "
          f"E = {e:.6f} (FCI = {e_fci:.6f})")


if __name__ == "__main__":
    print("=== test_kmodes_convergence ===")
    test_kmodes_convergence()
    print("=== test_clustered_recovers_particle_number ===")
    test_clustered_recovers_particle_number()
    print("=== test_clustered_matches_plain_on_single_mode ===")
    test_clustered_matches_plain_on_single_mode()
    print("=== test_clustered_beats_global_on_strongly_correlated ===")
    test_clustered_beats_global_on_strongly_correlated()
    print("=== test_solve_sqd_single_clustered ===")
    test_solve_sqd_single_clustered()
    print("=== test_solve_sqd_rejects_invalid_recovery ===")
    test_solve_sqd_rejects_invalid_recovery()
    print("=== test_diagonalize_clustered_runs ===")
    test_diagonalize_clustered_runs()
    print("\nAll tests passed.")
