"""round_007 R5: 12,12 prune_keep sweep（P0 验证 + P2 诊断分解）。

C1-v2+best @500 shots + prune_keep ∈ {1.0, 0.8, 0.7, 0.6, 0.5, 0.4}
P0: ∃ prune_keep 使 err ≤ 1e-9（≥20× vs 2.1e-8），wall 不增加
P2: 分解 (E_V, E_PT2, evpt2 r2, dim) 前后变化
"""
import os
import time
import json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.join(BASE, "src"))

from tc_sqd.integrated import solve_sqd_best

npz = np.load(os.path.join(BASE, "_n2_1212_ints.npz"))
h1e, eri = npz["h1e"], npz["eri"]
NCAS, NELEC = 12, (6, 6)
ecore = float(npz["ecore"]) if "ecore" in npz else 0.0
E_FCI = -108.7686857  # round_004 库全空间对角化

SHOTS = 500
results = []

for pk in [1.0, 0.8, 0.7, 0.6, 0.5, 0.4]:
    print(f"\n=== prune_keep={pk} ===", flush=True)
    t0 = time.perf_counter()
    out = solve_sqd_best(
        h1e, eri, NCAS, NELEC, ecore=ecore, n_shots=SHOTS,
        max_strings=None, rand_seed=0,
        tail_suppression=True, tail_shots_ref=100,
        prune_keep=pk, backend="gpu",
        return_details=True, verbose=False)
    wall = time.perf_counter() - t0
    err = abs(float(out["energy"]) - E_FCI)
    row = {
        "prune_keep": pk, "wall_s": round(wall, 1),
        "err": float(err), "dim": out.get("dim"),
        "E_pt2": out.get("E_pt2"), "E_evpt2": out.get("E_evpt2"),
        "evpt2_fit": out.get("evpt2"),
    }
    results.append(row)
    print(f"  wall={wall:.0f}s dim={row['dim']} err={err:.2e} "
          f"E_pt2={row['E_pt2']}", flush=True)

# 汇总
print("\n=== P0 汇总 ===")
best = min(results, key=lambda r: r["err"])
print(f"best prune_keep={best['prune_keep']} err={best['err']:.2e}")
base = results[0]
print(f"baseline(1.0) err={base['err']:.2e}")
print(f"improvement = {base['err']/best['err']:.1f}x")
p0 = "证实" if best["err"] <= 1e-9 else ("部分" if best["err"] < base["err"] else "证伪")
print(f"P0 判定: {p0}")

out = os.path.join(BASE, "benchmarks", "_round007_prune_sweep.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"saved {out}")
