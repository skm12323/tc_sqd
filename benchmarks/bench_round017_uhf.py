"""round_017 R5: UHF active 闭环扩展验收跑分（CH (4,3) + N2 (7,7)）。

可证伪预测（docs/rounds/round_017/theory.md §3）:
  P0 : CH/STO-3G UHF (4,3) active 全空间闭环, err ≤1e-8 vs direct_uhf
  P0': N2/STO-3G R=2.5 UHF (7,7) @500 shots, err ≤1e-6
  P2 : CH (4,3) coverage_closure=True → dim=300 全空间, err ≤1e-9
  契约: backend="gpu" + 三元组 → NotImplementedError

参考: pyscf fci.direct_uhf.kernel(conv_tol=1e-12)。全部 CPU。
"""
import os, sys, time, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
import tc_sqd
from tc_sqd import solve_sqd_active
from pyscf import gto, scf, fci
from pyscf.fci import cistring


def uhf_system(atom, nelec):
    spin = nelec[0] - nelec[1]  # 2S = nα-nβ
    mf = scf.UHF(gto.M(atom=atom, basis="sto-3g", spin=spin,
                       verbose=0))
    mf.conv_tol = 1e-12
    mf = mf.run()
    assert not np.allclose(mf.mo_coeff[0], mf.mo_coeff[1]), "假 UHF"
    d = tc_sqd.from_pyscf(mf)
    # 与 tests/test_spin_resolved.py 同口径: e_ref 不含 ecore,
    # solve_sqd_active 也不传 ecore。
    e_ref = float(fci.direct_uhf.kernel(
        (d.h1e[0], d.h1e[1]), tuple(d.eri), d.norb, d.nelec,
        conv_tol=1e-12, max_cycle=1000)[0])
    full = int(cistring.num_strings(d.norb, nelec[0])) * \
        int(cistring.num_strings(d.norb, nelec[1]))
    return d.h1e, d.eri, d.norb, d.nelec, d.ecore, e_ref, full


res = {}

# ---- P0: CH (4,3) 全空间闭环 ----
h1e, eri, norb, nelec, ec, eref, full = uhf_system("C 0 0 0; H 0 0 1.12", (4, 3))
traj = []
t0 = time.perf_counter()
e = solve_sqd_active(
    h1e, eri, norb, nelec,
    bitstring_matrix=np.random.default_rng(0).random((2000, 2 * norb)) > 0.5,
    n_active_per_round=50, max_rounds=10, rand_seed=0, trajectory=traj)
w = time.perf_counter() - t0
dim = traj[-1]["dim"]
res["ch_p0"] = {"E": float(e), "err": abs(e - eref), "dim": dim,
                "full": full, "wall": round(w, 1)}
print(f"P0 CH(4,3): dim={dim}/{full} err={res['ch_p0']['err']:.2e} "
      f"wall={w:.1f}s", flush=True)

# ---- P2: CH (4,3) coverage_closure ----
nA, nB = int(cistring.num_strings(norb, nelec[0])), \
    int(cistring.num_strings(norb, nelec[1]))
traj = []
t0 = time.perf_counter()
e = solve_sqd_active(
    h1e, eri, norb, nelec,
    bitstring_matrix=np.random.default_rng(0).random((30, 2 * norb)) > 0.5,
    max_strings=max(nA, nB), n_active_per_round=5, max_rounds=3, rand_seed=0,
    coverage_closure=True, trajectory=traj)
w = time.perf_counter() - t0
dim = traj[-1]["dim"]
res["ch_p2_closure"] = {"E": float(e), "err": abs(e - eref), "dim": dim,
                        "full": full, "wall": round(w, 1)}
print(f"P2 CH closure: dim={dim}/{full} err={res['ch_p2_closure']['err']:.2e} "
      f"wall={w:.1f}s", flush=True)

# ---- P0': N2 (7,7) R=2.5 @500 ----
h1e, eri, norb, nelec, ec, eref, full = uhf_system("N 0 0 0; N 0 0 2.5", (7, 7))
traj = []
t0 = time.perf_counter()
e = solve_sqd_active(
    h1e, eri, norb, nelec,
    bitstring_matrix=np.random.default_rng(0).random((500, 2 * norb)) > 0.5,
    n_active_per_round=50, max_rounds=10, rand_seed=0, trajectory=traj)
w = time.perf_counter() - t0
dim = traj[-1]["dim"]
res["n2_p0p"] = {"E": float(e), "err": abs(e - eref), "dim": dim,
                 "full": full, "wall": round(w, 1)}
print(f"P0' N2(7,7) @500: dim={dim}/{full} err={res['n2_p0p']['err']:.2e} "
      f"wall={w:.1f}s", flush=True)

p0 = res["ch_p0"]["err"] <= 1e-8 and res["ch_p0"]["dim"] == res["ch_p0"]["full"]
p0p = res["n2_p0p"]["err"] <= 1e-6
p2 = (res["ch_p2_closure"]["err"] <= 1e-9
      and res["ch_p2_closure"]["dim"] == res["ch_p2_closure"]["full"])
print(f"\nP0: {'PASS' if p0 else 'FAIL'}  P0': {'PASS' if p0p else 'FAIL'}  "
      f"P2: {'PASS' if p2 else 'FAIL'}")
out = os.path.join(BASE, "benchmarks", "_round017_uhf.json")
with open(out, "w") as f:
    json.dump(res, f, indent=2)
print(f"saved {out}")
