"""SQD + VQE hybrid optimization demo — LiH/STO-3G.

VQE's variational principle (parameter search) + SQD's subspace
diagonalisation (error absorption) complement each other: the LUCJ Givens
angles are optimised with the *sampled SQD total energy* as loss.

  1. ``from_pyscf`` one-call molecular data
  2. fixed CCSD-LUCJ baseline (SQD energy)
  3. ``optimize_ansatz_parameters`` — Nelder-Mead, SQD energy as loss,
     multi-seed average (``n_seeds``) to avoid single-seed overfitting
  4. **error-optimisation key finding**: force-include single+double
     excitations (``excited_configurations``) -> deterministic coverage of
     the correlation space -> SQD reaches FCI exactly, zero shot noise
  5. compare: fixed / optimised / include-excitations vs FCI

Run:  PYTHONPATH=src python examples/ansatz_opt_demo.py
"""

import time

import numpy as np
from pyscf import gto

import tc_sqd


def main() -> None:
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    norb, nelec = data.norb, data.nelec

    e_fci = data.solve(method="fci")
    print(f"FCI reference        : {e_fci:.6f}")

    c0 = tc_sqd.build_lucj_circuit(data.mf, norb, nelec, ccsd_scale=0.5,
                                   max_excitations=6)

    # Fixed CCSD-LUCJ baseline
    bsm, probs = tc_sqd.sample(c0, 3000, backend="tc")
    e_fixed = data.solve(method="sqd", bitstring_matrix=bsm,
                         probabilities=probs, max_iterations=3)
    print(f"Fixed CCSD-LUCJ SQD  : {e_fixed:.6f}  "
          f"(err {e_fixed - e_fci:+.2e})")

    # SQD+VQE variational optimisation (multi-seed to avoid overfitting)
    t0 = time.time()
    res = tc_sqd.optimize_ansatz_parameters(
        data.mf, data.h1e, data.eri, norb, nelec,
        ecore=data.ecore, n_samples=2000, max_excitations=6,
        num_restarts=3, maxiter=25, seed=42, n_seeds=3, verbose=True)
    dt = time.time() - t0
    e = res["energy"]
    print(f"SQD+VQE optimised    : {e:.6f}  "
          f"(err {e - e_fci:+.2e})  [{dt:.0f}s, {res['n_params']} params]")
    print(f"Improvement vs fixed : {e_fixed - e:+.6f} Ha")

    # Error-optimisation: force-include singles+doubles (classical coverage)
    exc = tc_sqd.excited_configurations(norb, nelec, max_excitations=2)
    bsm, probs = tc_sqd.sample(c0, 1000, backend="tc")
    e_hyb = data.solve(method="sqd", bitstring_matrix=bsm,
                       probabilities=probs, max_iterations=3,
                       include_configurations=exc)
    print(f"+include S+D config : {e_hyb:.6f}  "
          f"(err {e_hyb - e_fci:+.2e}, 1000 shots, zero shot noise)")


if __name__ == "__main__":
    main()
