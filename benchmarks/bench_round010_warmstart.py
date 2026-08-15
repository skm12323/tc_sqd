"""round_010 R5: 12,12 warm-start wall 验证。

单次 solve_sqd_active @500 shots：cold（warm_start=False）vs warm（True），同 seed 独立进程。
P0: 总 wall ≤0.6×；P0': 总 matvec ≤3500（vs cold 6681）；P1: E diff ≤1e-10。
"""
import os
import sys
import time
import json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

warm = "--warm" in sys.argv
npz = np.load(os.path.join(BASE, "_n2_1212_ints.npz"))
h1e, eri = npz["h1e"], npz["eri"]
ecore = float(npz["ecore"]) if "ecore" in npz else 0.0

shots = 500
bsm = np.random.default_rng(0).random((shots, 24)) > 0.5
probs = np.full(shots, 1.0 / shots)

from tc_sqd.cipsi import solve_sqd_active, _Subspace
orig_diag = _Subspace.diag
n_mv_total = [0]

def instrumented(self, sa, sb):
    self._n_mv_before = getattr(self, "last_n_mv", 0)
    r = orig_diag(self, sa, sb)
    n = getattr(self, "last_n_mv", 0)
    if n:
        n_mv_total[0] += n
        print(f"  [diag] n_mv={n} (cum {n_mv_total[0]})", flush=True)
    return r

_Subspace.diag = instrumented
t0 = time.perf_counter()
E = solve_sqd_active(
    h1e, eri, 12, (6, 6), ecore=ecore, bitstring_matrix=bsm,
    probabilities=probs, max_strings=None, n_active_per_round=30,
    rand_seed=0, tail_suppression=True, tail_shots_ref=100,
    backend="gpu", warm_start=warm, verbose=False)
wall = time.perf_counter() - t0

out = {"mode": "warm" if warm else "cold", "wall_s": round(wall, 1),
       "E": float(E), "n_mv_total": n_mv_total[0]}
print(f"\n=== {out['mode']} === wall={wall:.0f}s E={E:.10f} n_mv={n_mv_total[0]}")

f = os.path.join(BASE, "benchmarks", f"_round010_{'warm' if warm else 'cold'}.json")
with open(f, "w") as fo:
    json.dump(out, fo, indent=2)
print(f"saved {f}")
