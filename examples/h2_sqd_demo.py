"""Example: Full SQD pipeline on H2 using tc_sqd + TensorCircuit + PySCF.

Reproduces the pseudocode from the tex document (Problem 2, minimal example):

    PySCF RHF -> MO integrals -> TC circuit (HF + entangling) -> sample
    -> configuration recovery -> CI matrix -> SQD energy -> compare FCI.
"""

import numpy as np
import tensorcircuit as tc
from pyscf import gto, scf, fci

import tc_sqd


def main():
    # Step 1: PySCF -- H2 + RHF -> MO-basis integrals
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    h1e = mo.T @ mf.get_hcore() @ mo
    eri_ao = mol.intor("int2e_sph")
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, mo, mo, mo, mo)
    ecore = mf.energy_nuc()

    norb = int(mol.nao_nr())   # int(): pyscf 2.7 等返回 np.int64 会让 tensorcircuit 的 c.x 崩
    nelec = (mol.nelectron // 2, mol.nelectron // 2)
    nq = 2 * norb

    print(f"norb={norb}, nq={nq}, nelec={nelec}")
    print(f"E(HF)  = {mf.e_tot:.8f}")

    # FCI reference
    fci_solver = fci.FCI(mf)
    fci_solver.kernel()
    print(f"E(FCI) = {fci_solver.e_tot:.8f}")

    # Step 2: TC -- prepare HF state + entangling gates, sample
    c = tc.Circuit(nq)
    c.x(0)           # alpha orbital 0
    c.x(norb)        # beta  orbital 0
    # Entangling gates (mimics LUCJ with small theta)
    c.ry(0, theta=0.8)
    c.cnot(0, 1)
    c.ry(0, theta=-0.8)
    c.ry(norb, theta=0.8)
    c.cnot(norb, norb + 1)
    c.ry(norb, theta=-0.8)

    bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=2000)
    print(f"Sampled {bsm.shape[0]} unique bitstrings")

    # Step 3: Configuration recovery
    occ_a = np.zeros(norb); occ_a[:nelec[0]] = 1.0
    occ_b = np.zeros(norb); occ_b[:nelec[1]] = 1.0
    recovered, rec_probs = tc_sqd.recover_configurations(
        bsm, probs, (occ_a, occ_b), nelec[0], nelec[1], rand_seed=42,
    )
    print(f"After recovery: {recovered.shape[0]} unique bitstrings")

    # Step 4: SQD diagonalisation (one-shot)
    ci_strs_a, ci_strs_b = tc_sqd.bitstring_matrix_to_ci_strs(recovered)
    result = tc_sqd.solve_sci(
        (ci_strs_a, ci_strs_b), h1e, eri, norb, nelec,
    )
    e_sqd = result.energy + ecore
    print(f"E(SQD) = {e_sqd:.8f}")

    # Step 5: Iterative SQD (full loop)
    result_iter = tc_sqd.diagonalize_fermionic_hamiltonian(
        h1e, eri, (bsm, probs),
        samples_per_batch=200,
        norb=norb, nelec=nelec,
        num_batches=1, max_iterations=5, seed=42,
    )
    e_iter = result_iter.energy + ecore
    print(f"E(SQD, iterative) = {e_iter:.8f}")

    # Step 6: Unified entry point -- compute_ground_state_energy
    print("\n--- compute_ground_state_energy ---")
    e_fci = tc_sqd.compute_ground_state_energy(
        h1e, eri, norb, nelec, ecore=ecore, method="fci", verbose=True,
    )
    print(f"E(FCI)    = {e_fci:.8f}")

    e_direct = tc_sqd.compute_ground_state_energy(
        h1e, eri, norb, nelec, ecore=ecore, method="direct", verbose=True,
    )
    print(f"E(direct) = {e_direct:.8f}")

    e_sqd = tc_sqd.compute_ground_state_energy(
        h1e, eri, norb, nelec, ecore=ecore, method="sqd",
        bitstring_matrix=bsm, probabilities=probs,
        samples_per_batch=200, max_iterations=5, verbose=True,
    )
    print(f"E(SQD)    = {e_sqd:.8f}")


if __name__ == "__main__":
    main()
