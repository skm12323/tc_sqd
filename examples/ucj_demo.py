"""UCJ (Unitary Cluster Jastrow) demo — LiH/STO-3G.

Shows the full UCJ pipeline (P2-2):
  1. ``ucj_decomposition`` — CCSD t2 -> SVD -> multi-layer (kappa, J)
  2. ``ucj_subspace_energy`` — deterministic SQD on UCJ-supported dets (H2 = FCI)
  3. ``build_ucj_circuit`` — UCJ circuit (U Givens), sample -> SQD
  4. compare vs simplified LUCJ

Run:  PYTHONPATH=src python examples/ucj_demo.py
"""

import numpy as np
from pyscf import gto

import tc_sqd


def main() -> None:
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    norb, nelec = data.norb, data.nelec
    nocc = nelec[0]
    e_fci = data.solve(method="fci")
    e_hf = data.mf.e_tot
    print(f"LiH: HF={e_hf:.6f}  FCI={e_fci:.6f}\n")

    t1, t2, _ = tc_sqd.get_ccsd_amplitudes(data.mf)

    # 1. UCJ decomposition
    layers = tc_sqd.ucj_decomposition(t2, norb, nocc, nlayers=2, scale=5)
    print(f"[1] UCJ decomposition: {len(layers)} layers, "
          f"kappa anti-Hermitian, J symmetric")

    # 2. Deterministic SQD (matrix-level)
    e_sub = tc_sqd.ucj_subspace_energy(layers, data.h1e, data.eri,
                                       norb, nelec) + data.ecore
    print(f"[2] UCJ subspace (deterministic SQD): {e_sub:.6f}  "
          f"(err {e_sub - e_fci:+.2e} vs FCI)")

    # 3. UCJ circuit -> sample -> SQD
    circ = tc_sqd.build_ucj_circuit(data.mf, norb, nelec, nlayers=2, scale=5)
    stats = tc_sqd.circuit_stats(circ)
    bsm, probs = tc_sqd.sample(circ, 3000)
    e_ucj = data.solve(method="sqd", bitstring_matrix=bsm, probabilities=probs,
                       max_iterations=3)
    print(f"[3] UCJ circuit SQD     : {e_ucj:.6f}  "
          f"(err {e_ucj - e_fci:+.2e}, 2Q gates={stats['n_2q']})")

    # 4. Simplified LUCJ baseline
    c_l = tc_sqd.build_lucj_circuit(data.mf, norb, nelec, ccsd_scale=1.0)
    bsm_l, probs_l = tc_sqd.sample(c_l, 3000)
    e_lucj = data.solve(method="sqd", bitstring_matrix=bsm_l,
                        probabilities=probs_l, max_iterations=3)
    print(f"[4] simplified LUCJ SQD : {e_lucj:.6f}  "
          f"(err {e_lucj - e_fci:+.2e})")


if __name__ == "__main__":
    main()
