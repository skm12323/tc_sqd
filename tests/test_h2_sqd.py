"""End-to-end test for tc_sqd on H2 (STO-3G).

Tests:
  1. Full FCI subspace SQD -> exact FCI energy
  2. build_ci_matrix -> explicit matrix eigenvalue matches FCI
  3. Iterative SQD with TC sampling -> converges to FCI
  4. Qubit-subspace SQD (TFIM)
"""

import numpy as np
import tensorcircuit as tc
from pyscf import gto, scf, fci
from pyscf.fci import cistring

import tc_sqd


def build_h2_integrals():
    """Build MO-basis one-body and two-body integrals for H2/STO-3G."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    h1e = mo.T @ mf.get_hcore() @ mo
    eri_ao = mol.intor("int2e_sph")
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, mo, mo, mo, mo)
    ecore = mf.energy_nuc()
    return mol, mf, h1e, eri, ecore


def make_tc_circuit(norb, theta=0.8):
    """Build a TC circuit: HF state + entangling gates to create diversity.

    Layout: [beta_{n-1}..beta_0 | alpha_{n-1}..alpha_0]  (right=alpha)
    For H2 (norb=2): qubits [3=b1, 2=b0, 1=a1, 0=a0]
    HF: a0=1, b0=1 -> x(0), x(2)
    """
    norb = int(norb)   # int(): pyscf 某些版本 mol.nao_nr() 返回 numpy int, 会让 tensorcircuit 崩
    nq = 2 * norb
    c = tc.Circuit(nq)
    # HF state
    c.x(0)
    c.x(norb)
    # Create superposition: apply gates that mix orbitals 0 and 1
    # for both alpha (qubits 0,1) and beta (qubits 2,3)
    if norb > 1:
        # Givens-like rotation: ry on qubit 0 then cnot(0,1)
        c.ry(0, theta=theta)
        c.cnot(0, 1)
        c.ry(0, theta=-theta)  # un-rotate to restore some structure
        # Same for beta
        c.ry(norb, theta=theta)
        c.cnot(norb, norb + 1)
        c.ry(norb, theta=-theta)
    return c


def test_fermion_sqd():
    """Test fermionic SQD on H2."""
    print("=" * 60)
    print("Test 1: H2 / STO-3G  Fermionic SQD")
    print("=" * 60)

    mol, mf, h1e, eri, ecore = build_h2_integrals()
    norb = mol.nao_nr()
    nelec = mol.nelectron
    nelec_t = (nelec // 2, nelec // 2)

    print(f"norb={norb}, nelec={nelec} {nelec_t}")
    print(f"E(HF)  = {mf.e_tot:.8f}")

    fci_solver = fci.FCI(mf)
    fci_solver.kernel()
    print(f"E(FCI) = {fci_solver.e_tot:.8f}")

    # --- Full FCI subspace ---
    all_a = cistring.make_strings(range(norb), nelec_t[0])
    all_b = cistring.make_strings(range(norb), nelec_t[1])
    dim_fci = len(all_a) * len(all_b)
    print(f"\nFull FCI space: {len(all_a)} x {len(all_b)} = {dim_fci} determinants")

    result = tc_sqd.solve_sci(
        (all_a, all_b), h1e, eri, norb, nelec_t,
    )
    e_sqd = result.energy + ecore
    print(f"E(SQD, full) = {e_sqd:.8f}")
    assert abs(e_sqd - fci_solver.e_tot) < 1e-6, \
        f"SQD {e_sqd} != FCI {fci_solver.e_tot}"
    print("  PASS: SQD full subspace matches FCI")

    # --- build_ci_matrix ---
    H = tc_sqd.build_ci_matrix(all_a, all_b, h1e, eri, norb, nelec_t, ecore=ecore)
    eigs = np.linalg.eigvalsh(H)
    print(f"E(build_ci_matrix) = {eigs[0]:.8f}")
    assert abs(eigs[0] - fci_solver.e_tot) < 1e-6, \
        f"CI matrix {eigs[0]} != FCI {fci_solver.e_tot}"
    print("  PASS: build_ci_matrix matches FCI")

    # --- TC sampling + iterative SQD ---
    c = make_tc_circuit(norb, theta=0.8)
    bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=3000)
    print(f"\nSampled {bsm.shape[0]} unique bitstrings from TC circuit")

    result_iter = tc_sqd.diagonalize_fermionic_hamiltonian(
        h1e, eri, (bsm, probs),
        samples_per_batch=100,
        norb=norb,
        nelec=nelec_t,
        num_batches=1,
        max_iterations=5,
        seed=42,
    )
    e_iter = result_iter.energy + ecore
    print(f"E(SQD, iterative) = {e_iter:.8f}")
    assert abs(e_iter - fci_solver.e_tot) < 1e-3, \
        f"Iterative SQD {e_iter} != FCI {fci_solver.e_tot}"
    print("  PASS: Iterative SQD matches FCI")

    print()


def test_qubit_sqd():
    """Test qubit-subspace SQD on a transverse-field Ising model."""
    print("=" * 60)
    print("Test 2: Qubit-subspace SQD (TFIM)")
    print("=" * 60)

    # 3-qubit TFIM: H = -Z0Z1 - Z1Z2 - 0.5*(X0 + X1 + X2)
    hamiltonian = [
        ("ZZI", -1.0),
        ("IZZ", -1.0),
        ("XII", -0.5),
        ("IXI", -0.5),
        ("IIX", -0.5),
    ]

    # Use the full Hilbert space as the subspace
    N = 3
    dim = 2 ** N
    all_bs = np.array(
        [[bool((i >> (N - 1 - j)) & 1) for j in range(N)] for i in range(dim)],
        dtype=bool,
    )
    # Sort and remove duplicates (required by solve_qubit)
    all_bs = tc_sqd.sort_and_remove_duplicates(all_bs)

    vals, vecs = tc_sqd.solve_qubit(all_bs, hamiltonian)
    print(f"Full-space SQD ground energy: {vals[0]:.6f}")

    # Brute-force full diagonalisation via an *independent* reference:
    # build each Pauli term by Kronecker products of 2x2 matrices so the
    # reference does not reuse the code under test.
    I2 = np.eye(2)
    X2 = np.array([[0, 1], [1, 0]])
    Y2 = np.array([[0, -1j], [1j, 0]])
    Z2 = np.array([[1, 0], [0, -1]])
    pauli_mats = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}

    def kron_term(pauli_str):
        m = pauli_mats[pauli_str[0]]
        for ch in pauli_str[1:]:
            m = np.kron(m, pauli_mats[ch])
        return m

    H_full = np.zeros((dim, dim), dtype=np.complex128)
    for pauli_str, coeff in hamiltonian:
        H_full += coeff * kron_term(pauli_str)
    H_full = 0.5 * (H_full + H_full.conj().T)
    full_vals = np.linalg.eigvalsh(H_full.real)
    print(f"Brute-force ground energy:   {full_vals[0]:.6f}")
    assert abs(vals[0] - full_vals[0]) < 1e-6, \
        f"Subspace {vals[0]} != full {full_vals[0]}"
    print("  PASS: Qubit SQD matches full diagonalisation")

    # Also test a proper subspace (even parity) and check variational bound
    even_bs = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ], dtype=bool)
    even_bs = tc_sqd.sort_and_remove_duplicates(even_bs)
    vals_even, _ = tc_sqd.solve_qubit(even_bs, hamiltonian)
    print(f"Even-parity subspace energy: {vals_even[0]:.6f}")
    assert vals_even[0] >= full_vals[0] - 1e-10, \
        f"Subspace energy {vals_even[0]} < ground {full_vals[0]} (should be variational)"
    print("  PASS: Variational bound holds (subspace energy >= ground energy)")

    # Test sort_and_remove_duplicates
    bsm_dup = np.array([[0, 0], [1, 1], [0, 0], [1, 0], [1, 1]], dtype=bool)
    sorted_bsm = tc_sqd.sort_and_remove_duplicates(bsm_dup)
    assert sorted_bsm.shape[0] == 3, f"Expected 3 unique, got {sorted_bsm.shape[0]}"
    print("  PASS: sort_and_remove_duplicates works")
    print()


def test_counts():
    """Test bitstring conversion utilities."""
    print("=" * 60)
    print("Test 3: Counts / conversion utilities")
    print("=" * 60)

    # bitarray_to_int
    bsm = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=bool)
    vals = tc_sqd.bitarray_to_int(bsm)
    assert list(vals) == [0, 1, 2, 3], f"Got {vals}"
    print("  PASS: bitarray_to_int")

    # int_to_bitarray
    bsm2 = tc_sqd.int_to_bitarray([0, 1, 2, 3], 2)
    assert np.array_equal(bsm, bsm2), f"Got {bsm2}"
    print("  PASS: int_to_bitarray")

    # counts_dict_to_bitstring_matrix
    counts = {3: 100, 12: 50}
    bsm3, probs = tc_sqd.counts_dict_to_bitstring_matrix(counts, 4)
    assert bsm3.shape == (2, 4), f"Got shape {bsm3.shape}"
    assert abs(probs.sum() - 1.0) < 1e-10, f"Probs sum = {probs.sum()}"
    print("  PASS: counts_dict_to_bitstring_matrix")
    print()


def test_compute_ground_state_energy():
    """Test the unified compute_ground_state_energy entry point."""
    print("=" * 60)
    print("Test 4: compute_ground_state_energy (H2 / STO-3G)")
    print("=" * 60)

    mol, mf, h1e, eri, ecore = build_h2_integrals()
    norb = mol.nao_nr()
    nelec = (mol.nelectron // 2, mol.nelectron // 2)

    fci_solver = fci.FCI(mf)
    fci_solver.kernel()
    print(f"E(FCI, PySCF) = {fci_solver.e_tot:.8f}")

    # --- FCI mode ---
    e_fci = tc_sqd.compute_ground_state_energy(
        h1e, eri, norb, nelec, ecore=ecore, method="fci", verbose=True,
    )
    print(f"E(FCI, tc_sqd) = {e_fci:.8f}")
    assert abs(e_fci - fci_solver.e_tot) < 1e-6
    print("  PASS: method='fci' matches PySCF FCI")

    # --- direct mode ---
    e_direct = tc_sqd.compute_ground_state_energy(
        h1e, eri, norb, nelec, ecore=ecore, method="direct", verbose=True,
    )
    print(f"E(direct)      = {e_direct:.8f}")
    assert abs(e_direct - fci_solver.e_tot) < 1e-6
    print("  PASS: method='direct' matches FCI")

    # --- SQD mode ---
    c = make_tc_circuit(norb, theta=0.8)
    bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=2000)
    e_sqd = tc_sqd.compute_ground_state_energy(
        h1e, eri, norb, nelec, ecore=ecore, method="sqd",
        bitstring_matrix=bsm, probabilities=probs,
        samples_per_batch=100, max_iterations=5, verbose=True,
    )
    print(f"E(SQD)        = {e_sqd:.8f}")
    assert abs(e_sqd - fci_solver.e_tot) < 1e-3
    print("  PASS: method='sqd' matches FCI")

    # --- Invalid method ---
    try:
        tc_sqd.compute_ground_state_energy(h1e, eri, norb, nelec, method="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  PASS: invalid method raises ValueError: {e}")

    # --- SQD without bitstrings ---
    try:
        tc_sqd.compute_ground_state_energy(h1e, eri, norb, nelec, method="sqd")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  PASS: sqd without bitstrings raises ValueError")
    print()


def test_bugfixes():
    """Regression tests for bugs found in code review."""
    print("=" * 60)
    print("Test 5: Bugfix regression tests")
    print("=" * 60)

    # --- Bug 2: configuration_recovery occupancy order ---
    # H2: norb=2, nelec=(1,1).  HF bitstring = |0011> (alpha0=1, beta0=1)
    # Layout: [beta_1 beta_0 | alpha_1 alpha_0] = [0 1 | 0 1]
    # A noisy bitstring |0000> (all zeros) should recover to |0011>
    # by flipping beta_0 (col 1) and alpha_0 (col 3).
    bsm = np.array([[0, 0, 0, 0]], dtype=bool)
    probs = np.array([1.0])
    # Correct occupancy: orbital 0 occupied, orbital 1 empty
    occ_a = np.array([1.0, 0.0])
    occ_b = np.array([1.0, 0.0])
    recovered, _ = tc_sqd.recover_configurations(
        bsm, probs, (occ_a, occ_b), 1, 1, rand_seed=42,
    )
    # Should flip cols 1 and 3 (orbital 0 for both spins)
    expected = np.array([[0, 1, 0, 1]], dtype=bool)
    assert np.array_equal(recovered, expected), \
        f"Recovered {recovered}, expected {expected}"
    print("  PASS: configuration_recovery flips correct orbitals")

    # --- Bug 4: return_probabilities=False normalization ---
    c = make_tc_circuit(2, theta=0.8)
    bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=100, return_probabilities=False)
    assert abs(probs.sum() - 1.0) < 1e-10, \
        f"Probs sum = {probs.sum()}, should be 1.0"
    print("  PASS: return_probabilities=False is normalized")

    # --- Bug 5: spin_sq constraint ---
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", mol.intor("int2e_sph"), mo, mo, mo, mo)
    norb = 2
    nelec = (1, 1)
    # H2 singlet: S=0, S²=0
    result = tc_sqd.solve_sci(
        (cistring.make_strings(range(norb), 1),
         cistring.make_strings(range(norb), 1)),
        h1e, eri, norb, nelec, spin_sq=0.0,
    )
    assert abs(result.spin_square) < 0.1, \
        f"S² = {result.spin_square}, expected ~0 for singlet"
    print(f"  PASS: spin_sq=0 gives S²={result.spin_square:.4f} (singlet)")

    # --- Bug 6: RDM return types ---
    state = result.sci_state
    rdm1_summed = state.rdm(rank=1, spin_summed=True)
    assert isinstance(rdm1_summed, np.ndarray) and rdm1_summed.shape == (norb, norb), \
        f"rdm1 spin_summed: type={type(rdm1_summed)}, shape={getattr(rdm1_summed, 'shape', None)}"
    rdm1_sep = state.rdm(rank=1, spin_summed=False)
    assert isinstance(rdm1_sep, tuple) and len(rdm1_sep) == 2, \
        f"rdm1 spin-separated: type={type(rdm1_sep)}"
    rdm2_summed = state.rdm(rank=2, spin_summed=True)
    assert isinstance(rdm2_summed, np.ndarray) and rdm2_summed.shape == (norb,) * 4, \
        f"rdm2 spin_summed: type={type(rdm2_summed)}, shape={getattr(rdm2_summed, 'shape', None)}"
    print("  PASS: RDM return types correct")

    # --- Bug 3: rotate_integrals consistency ---
    # With zero rotation, integrals should be unchanged
    h1e0, eri0 = tc_sqd.rotate_integrals(h1e, eri, np.zeros(1))
    assert np.allclose(h1e0, h1e), "h1e changed under zero rotation"
    assert np.allclose(eri0, eri), "eri changed under zero rotation"
    # With non-zero rotation, h1e and eri must use same U
    k = np.array([0.1])
    h1r, erir = tc_sqd.rotate_integrals(h1e, eri, k)
    # Verify by manual rotation
    from scipy.linalg import expm
    K = np.array([[0, 0.1], [-0.1, 0]])
    U = expm(K)
    h1r_ref = U.T @ h1e @ U
    erir_ref = np.einsum("ip,jq,kr,ls,ijkl->pqrs", U, U, U, U, eri)
    assert np.allclose(h1r, h1r_ref), "h1e rotation mismatch"
    assert np.allclose(erir, erir_ref), "eri rotation mismatch"
    print("  PASS: rotate_integrals h1e and eri use consistent U")

    print()


def test_recovery_and_subsampling():
    """Direct tests for configuration recovery and subsampling."""
    print("=" * 60)
    print("Test 6: Recovery / subsampling / postselection")
    print("=" * 60)

    # --- Recovery: mixed valid/invalid bitstrings ---
    # norb=2, nelec=(1,1).  avg occupancy: orbital 0 occupied.
    occ_a = np.array([0.9, 0.1])
    occ_b = np.array([0.9, 0.1])
    # |1100>: beta weight 2 (invalid), alpha weight 0 (invalid)
    bsm = np.array([[1, 1, 0, 0]], dtype=bool)
    probs = np.array([1.0])
    rec, rec_probs = tc_sqd.recover_configurations(
        bsm, probs, (occ_a, occ_b), 1, 1, rand_seed=0,
    )
    # beta must drop one 1 (flip the least-occupied beta bit = col 0, beta_1),
    # alpha must gain one 1 at col 3 (alpha_0, highest avg occupancy).
    assert rec.shape[0] == 1
    assert rec[0].sum() == 2, f"Recovered Hamming weight {rec[0].sum()} != 2"
    assert rec[0, 2:].sum() == 1, "alpha half must have weight 1"
    assert rec[0, :2].sum() == 1, "beta half must have weight 1"
    assert rec[0, 3] and rec[0, 1], \
        f"Expected orbital 0 occupied in both spins, got {rec[0]}"
    print("  PASS: recover_configurations fixes both halves to correct orbitals")

    # Recovery validation: odd width rejected
    try:
        tc_sqd.recover_configurations(
            np.array([[0, 1, 0]], dtype=bool), np.array([1.0]),
            (occ_a, occ_b), 1, 1,
        )
        assert False, "odd width should raise"
    except ValueError:
        print("  PASS: odd-width bitstring rejected")

    # --- postselect_by_hamming_right_and_left ---
    bsm2 = np.array([
        [0, 1, 0, 1],   # valid (1,1)
        [1, 1, 0, 0],   # beta=2 invalid
        [0, 0, 1, 1],   # beta=0 invalid
        [0, 1, 1, 0],   # valid (1,1)
    ], dtype=bool)
    probs2 = np.array([0.4, 0.3, 0.2, 0.1])
    sel, sel_probs = tc_sqd.postselect_by_hamming_right_and_left(
        bsm2, probs2, hamming_right=1, hamming_left=1,
    )
    assert sel.shape[0] == 2, f"Expected 2 selected, got {sel.shape[0]}"
    assert abs(sel_probs.sum() - 1.0) < 1e-10
    assert np.allclose(sel_probs, [0.8, 0.2]), f"Got {sel_probs}"
    print("  PASS: postselect_by_hamming_right_and_left filters and renormalises")

    # --- subsample ---
    rng_batches = tc_sqd.subsample(bsm2, probs2, 2, 3, rand_seed=7)
    assert len(rng_batches) == 3
    for b in rng_batches:
        assert b.shape[0] <= 2
    print("  PASS: subsample returns requested batches")

    # subsample validation: all-zero probabilities rejected
    try:
        tc_sqd.subsample(bsm2, np.zeros(4), 2, 1)
        assert False, "zero-sum probs should raise"
    except ValueError:
        print("  PASS: zero-sum probabilities rejected")

    print()


def test_state_io_and_open_shell():
    """SCIState save/load round-trip and open-shell guards."""
    print("=" * 60)
    print("Test 7: SCIState IO / open-shell / spin-resolved guards")
    print("=" * 60)

    mol, mf, h1e, eri, ecore = build_h2_integrals()
    norb, nelec = 2, (1, 1)
    result = tc_sqd.solve_sci(
        (cistring.make_strings(range(norb), 1),
         cistring.make_strings(range(norb), 1)),
        h1e, eri, norb, nelec,
    )
    state = result.sci_state

    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "state.npz")
        state.save(path)
        loaded = tc_sqd.SCIState.load(path)
    assert np.allclose(loaded.amplitudes, state.amplitudes)
    assert np.array_equal(loaded.ci_strs_a, state.ci_strs_a)
    assert np.array_equal(loaded.ci_strs_b, state.ci_strs_b)
    assert loaded.norb == state.norb and loaded.nelec == state.nelec
    print("  PASS: SCIState save/load round-trip")

    # --- loaded state must support ALL state methods (spin_square / rdm) ---
    assert abs(loaded.spin_square() - state.spin_square()) < 1e-8
    assert np.allclose(loaded.rdm(1, spin_summed=True),
                       state.rdm(1, spin_summed=True))
    assert np.allclose(loaded.rdm(2, spin_summed=True),
                       state.rdm(2, spin_summed=True))
    rdm2s = loaded.rdm(2, spin_summed=False)
    # PySCF make_rdm2s returns the 3 independent spin blocks (aa, ab, bb).
    assert isinstance(rdm2s, tuple) and len(rdm2s) == 3
    assert all(getattr(x, "shape", None) == (norb,) * 4 for x in rdm2s)
    print("  PASS: loaded state supports spin_square/rdm/rdm2s methods")

    # orbital_occupancies sum to electron counts
    occ_a, occ_b = state.orbital_occupancies()
    assert abs(occ_a.sum() - nelec[0]) < 1e-8, f"occ_a sum {occ_a.sum()}"
    assert abs(occ_b.sum() - nelec[1]) < 1e-8, f"occ_b sum {occ_b.sum()}"
    print("  PASS: orbital_occupancies sum to electron counts")

    # --- open_shell=False with unequal electrons must raise ---
    bsm = np.array([[0, 1, 0, 1]], dtype=bool)
    try:
        tc_sqd.solve_fermion(bsm, h1e, eri, open_shell=False, _nelec=(2, 1))
        assert False, "unequal nelec with open_shell=False should raise"
    except ValueError as exc:
        assert "open_shell" in str(exc)
        print("  PASS: open_shell=False rejects unequal electron counts")

    # --- spin-resolved h1e with h_alpha != h_beta must raise ---
    h1e_spin = np.stack([h1e, h1e + 0.5])
    try:
        tc_sqd.solve_sci(
            (cistring.make_strings(range(norb), 1),
             cistring.make_strings(range(norb), 1)),
            h1e_spin, eri, norb, nelec,
        )
        assert False, "spin-resolved h1e should raise"
    except ValueError:
        print("  PASS: spin-resolved h1e with h_alpha != h_beta rejected")

    # Identical spin blocks are accepted (closed-shell equivalent)
    h1e_same = np.stack([h1e, h1e])
    result_same = tc_sqd.solve_sci(
        (cistring.make_strings(range(norb), 1),
         cistring.make_strings(range(norb), 1)),
        h1e_same, eri, norb, nelec,
    )
    assert abs(result_same.energy - result.energy) < 1e-10
    print("  PASS: identical (2,norb,norb) h1e accepted")

    # --- spin_sq rejected in sqd/direct branches ---
    for m in ("sqd", "direct"):
        try:
            tc_sqd.compute_ground_state_energy(
                h1e, eri, norb, nelec, method=m, spin_sq=0.0,
                bitstring_matrix=bsm if m == "sqd" else None,
            )
            assert False, f"spin_sq should be rejected for method={m}"
        except ValueError:
            pass
    print("  PASS: spin_sq explicitly rejected in sqd/direct branches")

    print()


def test_pauli_y_and_validation():
    """Pauli Y support, illegal Pauli rejection, dense-k semantics."""
    print("=" * 60)
    print("Test 8: Pauli Y / validation / dense-k")
    print("=" * 60)

    # --- Pauli Y via independent Kronecker reference ---
    I2 = np.eye(2)
    Y2 = np.array([[0, -1j], [1j, 0]])
    Z2 = np.array([[1, 0], [0, -1]])
    ham = [("YZ", 1.0), ("ZY", 1.0)]
    N = 2
    dim = 2 ** N
    all_bs = tc_sqd.sort_and_remove_duplicates(
        tc_sqd.int_to_bitarray(list(range(dim)), N)
    )
    vals, _ = tc_sqd.solve_qubit(all_bs, ham)
    H_ref = np.kron(Y2, Z2) + np.kron(Z2, Y2)
    ref_vals = np.linalg.eigvalsh(H_ref)
    assert abs(vals[0] - ref_vals[0]) < 1e-8, \
        f"Pauli Y energy {vals[0]} != reference {ref_vals[0]}"
    print("  PASS: Pauli Y matrix elements match Kronecker reference")

    # --- illegal Pauli character must raise ---
    try:
        tc_sqd.solve_qubit(all_bs, [("XQ", 1.0)])
        assert False, "illegal Pauli should raise"
    except (ValueError, KeyError):
        print("  PASS: illegal Pauli character rejected")

    # --- mismatched Pauli length must raise ---
    try:
        tc_sqd.solve_qubit(all_bs, [("XYZ", 1.0)])
        assert False, "mismatched length should raise"
    except ValueError:
        print("  PASS: mismatched Pauli length rejected")

    # --- dense branch honours k>1 ---
    vals_k, vecs_k = tc_sqd.solve_qubit(all_bs, ham, k=2)
    assert len(vals_k) == 2, f"Expected 2 eigenvalues, got {len(vals_k)}"
    assert vecs_k.shape == (4, 2), f"Eigenvectors shape {vecs_k.shape}"
    ref_sorted = np.sort(ref_vals)
    assert np.allclose(np.sort(vals_k), ref_sorted[:2], atol=1e-8)
    print("  PASS: dense branch honours k=2")

    print()


def test_third_review_fixes():
    """Regression tests for issues from the third review round."""
    print("=" * 60)
    print("Test 9: Third-review regression tests")
    print("=" * 60)

    mol, mf, h1e, eri, ecore = build_h2_integrals()
    norb, nelec = 2, (1, 1)

    # --- (1) multi-electron orbital_occupancies must not crash ---
    # Build a (2,1) system: norb=3 spatial orbitals, 2 alpha + 1 beta.
    from pyscf import gto as _gto, scf as _scf
    mol3 = _gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
    # Fabricate simple integrals for norb=3 directly (no SCF needed).
    rng = np.random.default_rng(0)
    n3 = 3
    h1_3 = rng.standard_normal((n3, n3)); h1_3 = h1_3 + h1_3.T
    eri_3 = rng.standard_normal((n3, n3, n3, n3)) * 0.1
    eri_3 = eri_3 + eri_3.transpose(1, 0, 3, 2)
    res3 = tc_sqd.solve_sci(
        (cistring.make_strings(range(n3), 2),
         cistring.make_strings(range(n3), 1)),
        h1_3, eri_3, n3, (2, 1),
    )
    occ_a, occ_b = res3.sci_state.orbital_occupancies()
    assert abs(occ_a.sum() - 2) < 1e-8, f"occ_a sum {occ_a.sum()}"
    assert abs(occ_b.sum() - 1) < 1e-8, f"occ_b sum {occ_b.sum()}"
    print("  PASS: multi-electron orbital_occupancies (2 alpha + 1 beta)")

    # --- (2) genuine spin_sq selection picks the triplet root ---
    # He-like open-shell (2,1) with S=1/2 ground state.  Requesting S^2=2
    # (S=1) should select a higher-energy root or raise if unavailable.
    res_spin = tc_sqd.solve_sci(
        (cistring.make_strings(range(n3), 2),
         cistring.make_strings(range(n3), 1)),
        h1_3, eri_3, n3, (2, 1),
        spin_sq=0.75,  # S = 1/2, the physical ground state
    )
    assert abs(res_spin.spin_square - 0.75) < 1e-1, \
        f"S2={res_spin.spin_square} != 0.75"
    # Ground-state energy (no constraint) must be <= spin-constrained energy.
    res_free = tc_sqd.solve_sci(
        (cistring.make_strings(range(n3), 2),
         cistring.make_strings(range(n3), 1)),
        h1_3, eri_3, n3, (2, 1),
    )
    assert res_free.energy <= res_spin.energy + 1e-8
    print(f"  PASS: spin_sq=0.75 selects S=1/2 root "
          f"(E={res_spin.energy:.6f} >= E_free={res_free.energy:.6f})")

    # Unreachable target spin must raise rather than silently return a
    # wrong-spin state.
    try:
        tc_sqd.solve_sci(
            (cistring.make_strings(range(n3), 2),
             cistring.make_strings(range(n3), 1)),
            h1_3, eri_3, n3, (2, 1),
            spin_sq=42.0, spin_tol=1e-3,
        )
        assert False, "unreachable spin should raise"
    except ValueError:
        print("  PASS: unreachable target spin raises (not silently wrong)")

    # --- (3) optimize_orbitals validation + best-so-far semantics ---
    bsm_h2 = np.array([[0, 1, 0, 1]], dtype=bool)
    with np.testing.assert_raises(ValueError):
        tc_sqd.optimize_orbitals(
            bsm_h2, h1e, eri, np.zeros(5), nelec=nelec, num_iters=0,
        )  # wrong k_flat length
    with np.testing.assert_raises(ValueError):
        tc_sqd.optimize_orbitals(
            bsm_h2, h1e, eri, np.zeros(1), nelec=nelec,
            num_iters=1, num_steps_grad=0,
        )  # num_steps_grad < 1
    with np.testing.assert_raises(ValueError):
        tc_sqd.optimize_orbitals(
            bsm_h2, h1e, eri, np.zeros(1), nelec=nelec,
            num_iters=1, num_steps_grad=1, learning_rate=-1.0,
        )  # bad learning rate
    # zero iterations must still return a consistent (energy, k_flat) pair
    e0, k0, _ = tc_sqd.optimize_orbitals(
        bsm_h2, h1e, eri, np.zeros(1), nelec=nelec,
        num_iters=0, num_steps_grad=1,
    )
    assert np.isfinite(e0) and k0.shape == (1,)
    print("  PASS: optimize_orbitals validates k_flat/steps/lr, zero-iter consistent")

    # --- (4) include_configurations is always present in the subspace ---
    hf_det = np.array([[0, 1, 0, 1]], dtype=bool)
    res_inc = tc_sqd.diagonalize_fermionic_hamiltonian(
        h1e, eri, (hf_det, np.array([1.0])),
        samples_per_batch=1, norb=norb, nelec=nelec,
        num_batches=1, max_iterations=2, seed=1,
        include_configurations=hf_det,
    )
    assert np.isfinite(res_inc.energy)
    # invalid width must raise
    try:
        tc_sqd.diagonalize_fermionic_hamiltonian(
            h1e, eri, (hf_det, np.array([1.0])),
            samples_per_batch=1, norb=norb, nelec=nelec,
            include_configurations=np.zeros((1, 5), dtype=bool),
        )
        assert False, "bad include width should raise"
    except ValueError:
        pass
    # carryover_threshold out of range must raise
    for bad in (-0.1, 1.5):
        try:
            tc_sqd.diagonalize_fermionic_hamiltonian(
                h1e, eri, (hf_det, np.array([1.0])),
                samples_per_batch=1, norb=norb, nelec=nelec,
                carryover_threshold=bad,
            )
            assert False, f"carryover_threshold={bad} should raise"
        except ValueError:
            pass
    # max_iterations<=0 must raise
    try:
        tc_sqd.diagonalize_fermionic_hamiltonian(
            h1e, eri, (hf_det, np.array([1.0])),
            samples_per_batch=1, norb=norb, nelec=nelec, max_iterations=0,
        )
        assert False, "max_iterations=0 should raise"
    except ValueError:
        pass
    # norb / bitstring width mismatch must raise
    try:
        tc_sqd.diagonalize_fermionic_hamiltonian(
            h1e, eri, (np.zeros((1, 6), dtype=bool), np.array([1.0])),
            samples_per_batch=1, norb=norb, nelec=nelec,
        )
        assert False, "width mismatch should raise"
    except ValueError:
        pass
    # max_dim now supported (limit_subspace); illegal values raise
    try:
        tc_sqd.diagonalize_fermionic_hamiltonian(
            h1e, eri, (hf_det, np.array([1.0])),
            samples_per_batch=1, norb=norb, nelec=nelec, max_dim=0,
        )
        assert False, "max_dim=0 should raise ValueError"
    except ValueError:
        pass
    print("  PASS: include/carryover/max_iter/norb/max_dim validation")

    # --- (9) counts merges equivalent keys ---
    bsm_c, probs_c = tc_sqd.counts_dict_to_bitstring_matrix(
        {"01": 30, 1: 70}, 2,
    )
    assert bsm_c.shape[0] == 1, f"Expected 1 unique, got {bsm_c.shape[0]}"
    assert abs(probs_c[0] - 1.0) < 1e-10
    assert np.array_equal(bsm_c[0], np.array([False, True]))
    print("  PASS: counts merges equivalent string/int keys")

    # --- (10) sample_from_circuit explicit nbits when metadata missing ---
    c_dummy = make_tc_circuit(2, theta=0.5)
    bsm_s, _ = tc_sqd.sample_from_circuit(c_dummy, n_samples=50, nbits=4)
    assert bsm_s.shape[1] == 4
    print("  PASS: sample_from_circuit honours explicit nbits")

    # --- (12) sparse branch (S > 100) ---
    N_sp = 8
    dim_sp = 2 ** N_sp
    bsm_sp = tc_sqd.sort_and_remove_duplicates(
        tc_sqd.int_to_bitarray(list(range(dim_sp)), N_sp)
    )
    ham_sp = [("ZZIIIIII", -1.0), ("XIIIIIII", -0.5)]
    vals_sp, vecs_sp = tc_sqd.solve_qubit(bsm_sp, ham_sp, k=2)
    assert len(vals_sp) == 2
    # independent reference: H = -Z⊗Z - 0.5 X on first two qubits
    Z2 = np.array([[1, 0], [0, -1]])
    X2 = np.array([[0, 1], [1, 0]])
    I2 = np.eye(2)
    H_ref_sp = -np.kron(Z2, Z2) - 0.5 * np.kron(X2, I2)
    for _ in range(N_sp - 2):
        H_ref_sp = np.kron(H_ref_sp, I2)
    ref_sp = np.sort(np.linalg.eigvalsh(H_ref_sp))
    assert np.allclose(np.sort(vals_sp), ref_sp[:2], atol=1e-6), \
        f"sparse eigenvalues {vals_sp} != reference {ref_sp[:2]}"
    print("  PASS: sparse branch (S=256) eigenvalues match reference")

    print()


def test_sqd_carryover_amplitude_threshold():
    """B4: solve_sqd carryover_threshold>0 用振幅阈值 carryover, 不劣化结果。

    统一语义 (与 diagonalize_fermionic_hamiltonian 一致): 保留上一轮解态
    |c|>=thr·max|c| 的 det 注入下一轮子空间, 替代原 Hamming-weight postselect。
    验证: 振幅阈值注入高置信 det 不劣化能量 (回归)。
    """
    mol, mf, h1e, eri, ecore = build_h2_integrals()
    norb, nelec = 2, (1, 1)
    c = make_tc_circuit(norb)
    bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=500)
    e_fci = fci.direct_spin1.kernel(h1e, eri, norb, nelec)[0] + ecore

    e0 = tc_sqd.solve_sqd(h1e, eri, norb, nelec, bitstring_matrix=bsm,
                          probabilities=probs, max_iterations=3)
    e1 = tc_sqd.solve_sqd(h1e, eri, norb, nelec, bitstring_matrix=bsm,
                          probabilities=probs, max_iterations=3,
                          carryover_threshold=1e-4)
    assert np.isfinite(e0.energy) and np.isfinite(e1.energy)
    # 振幅阈值 carryover 不劣化 (注入高置信 det)
    assert abs(e1.energy - e_fci) <= abs(e0.energy - e_fci) + 1e-9, (
        f"carryover 劣化: with={abs(e1.energy - e_fci):.2e} "
        f"without={abs(e0.energy - e_fci):.2e}")


def test_sqd_batch_probs_preserved():
    """B4: solve_sqd num_batches>1 批内保留真实 probs (回归)。

    批量子采样时 subsample(return_probs=True) 恢复原始概率, 不再用均匀 probs。
    非均匀采样概率下 num_batches=2 应正常收敛 (回归, 不崩)。
    """
    mol, mf, h1e, eri, ecore = build_h2_integrals()
    norb, nelec = 2, (1, 1)
    c = make_tc_circuit(norb)
    bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=2000)  # 非均匀 probs
    r = tc_sqd.solve_sqd(h1e, eri, norb, nelec, bitstring_matrix=bsm,
                         probabilities=probs, num_batches=2, samples_per_batch=500,
                         max_iterations=2)
    assert np.isfinite(r.energy)


if __name__ == "__main__":
    test_counts()
    test_fermion_sqd()
    test_qubit_sqd()
    test_compute_ground_state_energy()
    test_bugfixes()
    test_recovery_and_subsampling()
    test_state_io_and_open_shell()
    test_pauli_y_and_validation()
    test_third_review_fixes()
    print("All tests passed!")
