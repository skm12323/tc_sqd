"""round_009 P0a: cProfile 12,12 单 active，实测 f_PT2（R2 成本模型验证，零代码改动）。"""
import os
import cProfile
import pstats
import io
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.join(BASE, "src"))

from tc_sqd.cipsi import solve_sqd_active

npz = np.load(os.path.join(BASE, "_n2_1212_ints.npz"))
h1e, eri = npz["h1e"], npz["eri"]
ecore = float(npz["ecore"]) if "ecore" in npz else 0.0

shots = 500
bsm = np.random.default_rng(0).random((shots, 24)) > 0.5
probs = np.full(shots, 1.0 / shots)

def run():
    solve_sqd_active(
        h1e, eri, 12, (6, 6), ecore=ecore, bitstring_matrix=bsm,
        probabilities=probs, max_strings=None, n_active_per_round=30,
        rand_seed=0, tail_suppression=True, tail_shots_ref=100,
        backend="gpu", verbose=False)

prof = cProfile.Profile()
prof.enable()
run()
prof.disable()

s = io.StringIO()
ps = pstats.Stats(prof, stream=s).sort_stats("cumulative")
ps.print_stats(40)

# 提取 PT2 相关函数 cumtime
lines = s.getvalue().splitlines()
total = None
pt2_time = 0.0
for ln in lines:
    if "pt2_matrix_elements" in ln or "_excited_dets" in ln:
        parts = ln.split()
        try:
            cum = float(parts[4])
            pt2_time += cum
        except (ValueError, IndexError):
            pass
    if total is None and "built-in method builtins.exec" in ln:
        parts = ln.split()
        try:
            total = float(parts[4])
        except (ValueError, IndexError):
            pass

if total is None:
    total = ps.total_tt

f_pt2 = pt2_time / total if total else -1
print(f"\n=== P0a 结果 ===")
print(f"total time: {total:.1f}s")
print(f"PT2-related cumtime: {pt2_time:.1f}s")
print(f"f_PT2 = {f_pt2:.3f}")
print(f"R2 预测带 [0.04, 0.20]; >=0.30 则 R2 模型证伪 (P0 可行)")
verdict = "带内(R2 模型证实)" if 0.04 <= f_pt2 <= 0.20 else ("R2 模型证伪(P0 可行)" if f_pt2 >= 0.30 else "PT2 更无关")
print(f"判定: {verdict}")

with open(os.path.join(BASE, "benchmarks", "_round009_p0a_profile.txt"), "w") as f:
    f.write(s.getvalue())
    f.write(f"\n=== f_PT2 = {f_pt2:.3f} ({verdict}) ===\n")
