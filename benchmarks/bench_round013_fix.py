"""round_013 修复补跑：M1（CPU tol 显式 0.0 可复现）+ M2（GPU 联合配方实测）。

M1: CPU 10o closure, eigsh_tol 用**显式** 0.0/1e-10/1e-8（不用 None，
避免 "None=默认" 在改默认后语义漂移导致不可复现）。
M2: GPU 12,12 closure 联合配方 baseline(默认) vs recipe(1e-6 + n_active=90)
—— 实测而非乘积外推。
"""
import os, sys, time, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
from tc_sqd.cipsi import solve_sqd_active, _Subspace
_ORIG = _Subspace.diag

def run(h1e, eri, norb, nelec, ecore, shots, seed, **kw):
    n_mv = [0]
    def wrapped(self, sa, sb):
        r = _ORIG(self, sa, sb)
        n = getattr(self, "last_n_mv", 0)
        if n: n_mv[0] += n
        return r
    _Subspace.diag = wrapped
    try:
        traj = []
        base_kw = dict(ecore=ecore, max_strings=None, n_active_per_round=30,
                       rand_seed=seed, tail_suppression=True, tail_shots_ref=100,
                       warm_start=True, verbose=False, trajectory=traj,
                       coverage_closure=True)
        base_kw.update(kw)
        t0 = time.perf_counter()
        E = solve_sqd_active(
            h1e, eri, norb, nelec,
            bitstring_matrix=np.random.default_rng(seed).random((shots, 2*norb)) > 0.5,
            probabilities=np.full(shots, 1.0/shots), **base_kw)
        wall = time.perf_counter() - t0
    finally:
        _Subspace.diag = _ORIG
    final = traj[-1] if traj else {}
    return {"E": float(E), "wall": round(wall,1), "dim": int(final.get("dim",-1)),
            "n_mv": n_mv[0]}

res = {}
# M1: CPU 10o closure, 显式 tol
print("="*70, "\nM1: CPU 10o closure, eigsh_tol 显式 (可复现)", "\n"+"="*70, flush=True)
z10 = np.load(os.path.join(BASE, "_n2_ccpvdz_10o_ints.npz"))
h1e10, eri10 = z10["h1e"], z10["eri"]
ec10 = float(z10["ecore"]); eref10 = float(z10["e_ref"])
rows = []
for label, tol in [("tol=0.0(显式)", 0.0), ("tol=1e-10", 1e-10), ("tol=1e-8", 1e-8)]:
    r = run(h1e10, eri10, 10, (5,5), ec10, 500, 0, backend="cpu", eigsh_tol=tol)
    r.update(label=label, err=abs(r["E"]-eref10)); rows.append(r)
    print(f"  {label}: E={r['E']:.10f} err={r['err']:.2e} dim={r['dim']} "
          f"n_mv={r['n_mv']} wall={r['wall']}s", flush=True)
res["cpu_10o_explicit_tol"] = {"e_ref": eref10, "rows": rows}

# M2: GPU 12,12 closure, baseline vs recipe(1e-6 + n_active=90)
print("\n"+"="*70, "\nM2: GPU 12,12 closure, baseline vs recipe(1e-6+n_active=90)",
      "\n"+"="*70, flush=True)
z = np.load(os.path.join(BASE, "_n2_1212_ints.npz"))
h1e, eri = z["h1e"], z["eri"]
ec = float(z["ecore"]); eref = float(z["e_ref"])
r_base = run(h1e, eri, 12, (6,6), ec, 500, 0, backend="gpu")
r_base.update(label="baseline(默认)", err=abs(r_base["E"]-eref))
print(f"  {r_base['label']}: E={r_base['E']:.10f} err={r_base['err']:.2e} "
      f"dim={r_base['dim']} n_mv={r_base['n_mv']} wall={r_base['wall']}s", flush=True)
r_rec = run(h1e, eri, 12, (6,6), ec, 500, 0, backend="gpu",
            eigsh_tol=1e-6, n_active_per_round=90)
r_rec.update(label="recipe(1e-6+n_active=90)", err=abs(r_rec["E"]-eref))
print(f"  {r_rec['label']}: E={r_rec['E']:.10f} err={r_rec['err']:.2e} "
      f"dim={r_rec['dim']} n_mv={r_rec['n_mv']} wall={r_rec['wall']}s", flush=True)
print(f"  实测联合比: n_mv {r_rec['n_mv']}/{r_base['n_mv']}={r_rec['n_mv']/r_base['n_mv']:.2f}x "
      f"wall {r_rec['wall']}/{r_base['wall']}={r_rec['wall']/r_base['wall']:.2f}x", flush=True)
res["gpu_12_12_combined"] = {"e_ref": eref, "baseline": r_base, "recipe": r_rec}

out = os.path.join(BASE, "benchmarks", "_round013_fix.json")
with open(out, "w") as f: json.dump(res, f, indent=2)
print(f"\nsaved {out}")
