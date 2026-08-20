"""round_012 R5: coverage_closure 跑分验收（3 seed + 体系矩阵）。

验收 P0/P1 (vs round_012 theory):
  - P0: 12,12 @500, 3 seed, coverage_closure=True → dim=853776(全空间 FCI),
        err ≤1e-9 (vs e_ref), wall ≤1.5× baseline
  - P1: 10o/cc-pVDZ + N2/STO-3G @500, coverage_closure=True → 不回归
        (closure 补全到全空间 = FCI, err≈0; 体系不崩)

固定输入 (禁现跑 SCF): 12,12 与 10o 用存盘 npz; N2/STO-3G 小体系 from_pyscf
(C(10,7)=120 串, 全 dim 14400, direct_spin1 算 e_ref, 秒级)。
GPU 计时显式 synchronize, 独立进程口径 (本脚本单进程顺序跑, 12,12 warm GPU)。
"""
import os
import sys
import time
import json
import math
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

import tc_sqd
from tc_sqd.cipsi import solve_sqd_active, _Subspace


_ORIG_DIAG = _Subspace.diag  # 模块级真原始, 避免重复包装


def _instrument_diag():
    """计 n_mv_total (同 bench_round010 口径); 每次返回独立计数器。"""
    n_mv = [0]

    def wrapped(self, sa, sb):
        r = _ORIG_DIAG(self, sa, sb)
        n = getattr(self, "last_n_mv", 0)
        if n:
            n_mv[0] += n
        return r
    _Subspace.diag = wrapped
    return n_mv


def _restore_diag():
    _Subspace.diag = _ORIG_DIAG


def run_active(h1e, eri, norb, nelec, ecore, shots, seed, *,
               coverage_closure=False, warm_start=True, backend="gpu",
               n_active=30, max_strings=None):
    n_mv = _instrument_diag()
    traj = []
    kw = dict(ecore=ecore, max_strings=max_strings,
              n_active_per_round=n_active, rand_seed=seed,
              tail_suppression=True, tail_shots_ref=100,
              warm_start=warm_start, backend=backend,
              verbose=False, trajectory=traj)
    if coverage_closure:
        kw["coverage_closure"] = True
    t0 = time.perf_counter()
    E = solve_sqd_active(h1e, eri, norb, nelec,
                         bitstring_matrix=np.random.default_rng(seed).random(
                             (shots, 2 * norb)) > 0.5,
                         probabilities=np.full(shots, 1.0 / shots), **kw)
    wall = time.perf_counter() - t0
    _restore_diag()
    final = traj[-1] if traj else {}
    return {"E": float(E), "wall": round(wall, 1),
            "dim": int(final.get("dim", -1)),
            "sigma2": float(final.get("sigma2", -1)),
            "n_mv": n_mv[0]}


results = {"systems": {}}

# ---- P0: 12,12 @500, 3 seed, coverage_closure=True (GPU warm) ----
print("\n" + "=" * 70)
print("P0: N2/cc-pVDZ (12,12) @500, coverage_closure=True, 3 seed (GPU warm)")
print("=" * 70)
npz = np.load(os.path.join(BASE, "_n2_1212_ints.npz"))
h1e, eri = npz["h1e"], npz["eri"]
ecore = float(npz["ecore"])
e_ref = float(npz["e_ref"])
NCAS, NELEC = 12, (6, 6)
sys12 = {"e_ref": e_ref, "full_dim": 924 * 924, "seeds": []}
for seed in [0, 1, 2]:
    r = run_active(h1e, eri, NCAS, NELEC, ecore, 500, seed,
                   coverage_closure=True, warm_start=True, backend="gpu")
    r["seed"] = seed
    r["err"] = abs(r["E"] - e_ref)
    r["n_str"] = int(round(math.sqrt(r["dim"]))) if r["dim"] > 0 else -1
    sys12["seeds"].append(r)
    print(f"  seed={seed}: E={r['E']:.10f} err={r['err']:.2e} dim={r['dim']} "
          f"n_str={r['n_str']} sigma2={r['sigma2']:.2e} wall={r['wall']}s", flush=True)
# baseline seed 0 (no closure) for wall ratio
rb = run_active(h1e, eri, NCAS, NELEC, ecore, 500, 0,
                coverage_closure=False, warm_start=True, backend="gpu")
rb["err"] = abs(rb["E"] - e_ref)
sys12["baseline_seed0"] = rb
print(f"  baseline seed0: E={rb['E']:.10f} err={rb['err']:.2e} dim={rb['dim']} "
      f"wall={rb['wall']}s", flush=True)
results["systems"]["n2_12_12"] = sys12

# ---- P1a: 10o/cc-pVDZ @500, coverage_closure=True (GPU warm) ----
print("\n" + "=" * 70)
print("P1a: N2/cc-pVDZ (10o) @500, coverage_closure=True (GPU warm)")
print("=" * 70)
npz10 = np.load(os.path.join(BASE, "_n2_ccpvdz_10o_ints.npz"))
h1e10, eri10 = npz10["h1e"], npz10["eri"]
ecore10 = float(npz10["ecore"])
e_ref10 = float(npz10["e_ref"])
r10 = run_active(h1e10, eri10, 10, (5, 5), ecore10, 500, 0,
                 coverage_closure=True, warm_start=True, backend="gpu")
r10["err"] = abs(r10["E"] - e_ref10)
r10["n_str"] = int(round(math.sqrt(r10["dim"]))) if r10["dim"] > 0 else -1
sys10 = {"e_ref": e_ref10, "full_dim": 252 * 252, "result": r10}
print(f"  E={r10['E']:.10f} err={r10['err']:.2e} dim={r10['dim']} "
      f"n_str={r10['n_str']} wall={r10['wall']}s", flush=True)
results["systems"]["n2_10o"] = sys10

# ---- P1b: N2/STO-3G @500, coverage_closure=True (CPU, 小体系) ----
print("\n" + "=" * 70)
print("P1b: N2/STO-3G (10o,7e) @500, coverage_closure=True (CPU)")
print("=" * 70)
from pyscf import gto
from pyscf.fci import direct_spin1
mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
data = tc_sqd.from_pyscf(mol)
h1e_s, eri_s = data.h1e, data.eri
norb_s, nelec_s = data.norb, data.nelec
ecore_s = data.ecore
e_ref_s = direct_spin1.kernel(h1e_s, eri_s, norb_s, nelec_s)[0]
rs = run_active(h1e_s, eri_s, norb_s, nelec_s, ecore_s, 500, 0,
                coverage_closure=True, warm_start=True, backend="cpu")
rs["err"] = abs(rs["E"] - (e_ref_s + ecore_s))
rs["n_str"] = int(round(math.sqrt(rs["dim"]))) if rs["dim"] > 0 else -1
sys_s = {"e_ref": e_ref_s + ecore_s, "full_dim": 120 * 120,
         "norb": norb_s, "nelec": list(nelec_s), "result": rs}
print(f"  E={rs['E']:.10f} err={rs['err']:.2e} dim={rs['dim']} "
      f"n_str={rs['n_str']} wall={rs['wall']}s", flush=True)
results["systems"]["n2_sto3g"] = sys_s

# ---- 汇总 + 判定 ----
print("\n" + "=" * 70)
print("=== R5 汇总 ===")
print("=" * 70)
s12 = results["systems"]["n2_12_12"]
errs = [s["err"] for s in s12["seeds"]]
dims = [s["dim"] for s in s12["seeds"]]
walls = [s["wall"] for s in s12["seeds"]]
print(f"P0 (12,12 @500 closure, 3 seed):")
print(f"  err: {min(errs):.2e} ~ {max(errs):.2e}  (目标 ≤1e-9)")
print(f"  dim: {min(dims)} ~ {max(dims)}  (全空间 {s12['full_dim']})")
print(f"  wall: {min(walls)} ~ {max(walls)}s  (baseline seed0 {s12['baseline_seed0']['wall']}s)")
p0_err = max(errs) <= 1e-9
p0_dim = all(d == s12["full_dim"] for d in dims)
p0_wall = max(walls) <= 1.5 * s12["baseline_seed0"]["wall"]
print(f"  P0 err≤1e-9: {'PASS' if p0_err else 'FAIL'} | "
      f"dim=全空间: {'PASS' if p0_dim else 'FAIL'} | "
      f"wall≤1.5×base: {'PASS' if p0_wall else 'FAIL'}")
s10 = results["systems"]["n2_10o"]
print(f"\nP1a (10o closure): err={s10['result']['err']:.2e} "
      f"dim={s10['result']['dim']} (全 {s10['full_dim']})")
ss = results["systems"]["n2_sto3g"]
print(f"P1b (STO-3G closure): err={ss['result']['err']:.2e} "
      f"dim={ss['result']['dim']} (全 {ss['full_dim']})")
p1 = (s10["result"]["dim"] == s10["full_dim"]
      and ss["result"]["dim"] == ss["full_dim"])
print(f"  P1 体系补全到全空间(=FCI): {'PASS' if p1 else 'FAIL'}")

results["verdict"] = {
    "p0_err_le_1e-9": p0_err, "p0_dim_full": p0_dim, "p0_wall_le_1.5x": p0_wall,
    "p1_systems_full_fci": p1,
}
out = os.path.join(BASE, "benchmarks", "_round012_r5_results.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved {out}")
