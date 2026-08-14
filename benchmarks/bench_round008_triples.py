"""round_008 R5: 12,12 三激发注入验证（决胜一击）。

C1-v2+best @500 shots：baseline（无注入）vs triple_injection=True（无 cap 补全）。
P0: err ≤1e-9（vs baseline ~1.6-6e-8 → ≥16× 改善），wall ≤1.5×
P2: 注入后 dim 是否达全空间 924 字符串（853776）
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
E_FCI = -108.7686857

results = []
for label, kwargs in [
    ("baseline", dict(triple_injection=False)),
    ("triples_fixpoint", dict(triple_injection=True, n_triples_per_round=0)),
]:
    print(f"\n=== {label} ===", flush=True)
    t0 = time.perf_counter()
    out = solve_sqd_best(
        h1e, eri, NCAS, NELEC, ecore=ecore, n_shots=500,
        max_strings=None, rand_seed=0,
        tail_suppression=True, tail_shots_ref=100,
        backend="gpu", return_details=True, verbose=False, **kwargs)
    wall = time.perf_counter() - t0
    err = abs(float(out["energy"]) - E_FCI)
    row = {"label": label, "wall_s": round(wall, 1), "err": float(err),
           "dim": out.get("dim"), "E_pt2": out.get("E_pt2"),
           "E_evpt2": out.get("E_evpt2")}
    results.append(row)
    print(f"  wall={wall:.0f}s dim={row['dim']} err={err:.2e}", flush=True)

b, t = results[0], results[1]
imp = b["err"] / t["err"] if t["err"] > 0 else float("inf")
print(f"\n=== P0 ===")
print(f"baseline err={b['err']:.2e} -> triples err={t['err']:.2e}  ({imp:.1f}x)")
print(f"wall: {b['wall_s']}s -> {t['wall_s']}s  ({t['wall_s']/b['wall_s']:.2f}x)")
print(f"dim: {b['dim']} -> {t['dim']}  (全空间 853776)")
verdict = "证实" if t["err"] <= 1e-9 else ("部分" if imp >= 3 else "证伪")
print(f"P0 判定: {verdict}")

out = os.path.join(BASE, "benchmarks", "_round008_triples.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"saved {out}")
