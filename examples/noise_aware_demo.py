"""Noise-aware SQD demo — T1 deconvolution, error prediction, sampling plan.

Shows the tc_sqd noise pipeline (no real hardware needed):
  1. ``predict_sqd_error``  / ``gamma_T1``        — error forecast from T1/depth/shots
  2. ``plan_sampling``                             — optimal (shots, depth) budget search
  3. ``estimate_true_occupancies``                 — T1 deconvolution of avg occupancy
  4. ``diagnostics.sampling_report``               — sample-quality report
  5. ``lucj_report``                               — hardware depth budget of the circuit

Run:  PYTHONPATH=src python examples/noise_aware_demo.py
"""

import numpy as np
from pyscf import gto

import tc_sqd


def main() -> None:
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    norb, nelec = data.norb, data.nelec

    # 1. Error forecast: T2-immune, T1-dominated
    r = tc_sqd.predict_sqd_error(T1_us=15, depth=200, t_gate_ns=30,
                                 shots=8000, n_excited=2)
    print(f"[predict] gamma_T1={r['gamma_T1']:.4f}, eps_sample={r['eps_sample']:.2e}")
    print(f"          ground err={r['ground']:.2e} (chemical: {r['ground_chemical']})")
    print(f"          excited errs={np.round(r['excited'], 2)} (3x T1 sensitivity)")
    print(f"          dominant={r['dominant']}")

    # 2. Sampling plan: cheapest (shots, depth) meeting chemical accuracy
    plan = tc_sqd.plan_sampling(15, 30, target=1.6e-3)
    b = plan["best"]
    print(f"[plan] best: shots={b.shots}, depth={b.depth}, err={b.error:.2e}, "
          f"cost={b.cost:.0f}, dominant={b.dominant}")

    # 3. T1 deconvolution of average occupancies
    #    Synthetic data: mostly-HF bitstrings + some double excitation, then T1.
    hf = np.array([[0, 1, 0, 1]], dtype=bool)
    de = np.array([[1, 0, 1, 0]], dtype=bool)
    real = np.vstack([np.repeat(hf, 1800, axis=0), np.repeat(de, 200, axis=0)])
    gamma = np.array([0.05, 0.15, 0.02, 0.20])        # per-qubit (nq = 4)
    rng = np.random.default_rng(0)
    obs = real.copy()
    for col in range(2 * norb):
        obs[(rng.random(obs.shape[0]) < gamma[col]) & obs[:, col], col] = False
    est_a, est_b = tc_sqd.estimate_true_occupancies(
        obs, nelec[0], nelec[1], gamma, norb=norb)
    #  Reference: HF-dominated -> alpha occupancy near (0.9, 0.1)
    naive_a = obs[:, norb:].mean(0)[::-1]
    print(f"[t1-deconv] true-alpha-occ ~ (0.90, 0.10)")
    print(f"            naive observed : {np.round(naive_a, 3)}  "
          f"(T1-damped)")
    print(f"            deconvolved    : {np.round(est_a, 3)}  "
          f"(closer to true)")

    # 4. Sample-quality report (entropy / subspace / energy-vs-shots)
    rep = tc_sqd.sampling_report(data.h1e, data.eri, norb, nelec, obs,
                                 ecore=data.ecore, max_iterations=2)
    conv = rep["energy_convergence"]
    print(f"[diagnostics] n_unique={rep['n_unique']}, "
          f"subspace_dim={rep['subspace_dim']}, "
          f"entropy={rep['entropy_nat']:.3f} nat")
    print(f"              E(conv)={conv['converged_energy']:.6f} vs "
          f"FCI={data.solve(method='fci'):.6f}")

    # 5. Hardware depth budget of the LUCJ circuit (2Q-gate proxy)
    mf = data.mf
    lr = tc_sqd.lucj_report(mf, norb, nelec, max_depth=1500)
    print(f"[hardware] LUCJ 2Q depth proxy={lr['depth_proxy']}, "
          f"within 1500 budget: {lr['within_budget']}")


if __name__ == "__main__":
    main()
