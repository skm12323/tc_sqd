"""Excited-state SQD end-to-end demo — LiH/STO-3G ground + low-lying states.

Shows the tc_sqd excited-state pipeline:
  1. ``from_pyscf`` one-call molecular data
  2. ``build_lucj_circuit`` sampling (ground-state-like)
  3. ``recover_configurations`` fix particle numbers (LUCJ Givens breaks them)
  4. ``excited_configurations`` force-include single excitations
     (excited-state sampling strategy: guarantees variational bound on n_roots)
  5. ``solve_sci(n_roots)`` ground + excited energies
  6. ``plan_sampling(excited=True)`` shot-budget recommendation (3x T1 sensitivity)

Run:  PYTHONPATH=src python examples/excited_sqd_demo.py
"""

import numpy as np
from pyscf import fci, gto

import tc_sqd


def main() -> None:
    # 1. One-call molecular data (LiH/STO-3G, 6 MO / 4 e)
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    norb, nelec = data.norb, data.nelec
    print(f"LiH: norb={norb}, nelec={nelec}, ecore={data.ecore:.6f}")

    # 2. FCI reference (ground + 2 excited roots)
    e_fci = fci.direct_spin1.kernel(data.h1e, data.eri, norb, nelec,
                                    nroots=3)[0] + data.ecore
    print(f"FCI reference   : {np.round(e_fci, 6)}")

    # 3. Sample from the LUCJ circuit (CCSD-t2 driven)
    mf = data.mf
    c = tc_sqd.build_lucj_circuit(mf, norb, nelec, ccsd_scale=0.5)
    bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=3000)

    # 4. Recover particle numbers (LUCJ Givens gates are not number-conserving)
    occ_a = np.zeros(norb); occ_a[:nelec[0]] = 1.0
    occ_b = np.zeros(norb); occ_b[:nelec[1]] = 1.0
    bsm_rec, _ = tc_sqd.recover_configurations(
        bsm, probs, (occ_a, occ_b), nelec[0], nelec[1], rand_seed=42)

    # 5. Excited-state sampling strategy: force-include single excitations
    exc = tc_sqd.excited_configurations(norb, nelec, max_excitations=1)
    print(f"Force-including {len(np.unique(tc_sqd.bitarray_to_int(exc)))} "
          f"excited configurations")

    # 6. Ground + excited energies via n_roots diagonalisation
    ci_a, ci_b = tc_sqd.bitstring_matrix_to_ci_strs(np.vstack([bsm_rec, exc]))
    results = tc_sqd.solve_sci((ci_a, ci_b), data.h1e, data.eri,
                               norb, nelec, n_roots=3)
    e_sqd = np.array([r.energy for r in results]) + data.ecore
    print(f"SQD ground+exc  : {np.round(e_sqd, 6)}")
    print(f"|error| vs FCI  : {np.round(np.abs(e_sqd - e_fci), 6)}")

    # 7. Shot-budget recommendation (excited states ~3x T1-sensitive)
    plan = tc_sqd.plan_sampling(T1_us=15, t_gate_ns=30,
                                target=1.6e-3, excited=True)
    if plan["best"] is not None:
        b = plan["best"]
        print(f"Shot plan (T1=15us): shots={b.shots}, depth={b.depth}, "
              f"pred_err={b.error:.2e}")


if __name__ == "__main__":
    main()
