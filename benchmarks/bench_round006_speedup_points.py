"""代表性点 CPU vs GPU hybrid 精确计时（round_006 加速比验证）。

3 个代表性点覆盖小/中/大 dim，同体系/同参数/同 seed，唯一变量 backend。
输出 wall 对照 + 加速比。参考 _n2_1212_ints.npz（12,12，只读）。
"""
import os
import time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.join(BASE, "src"))

import tc_sqd
from tc_sqd import from_pyscf
from tc_sqd.cipsi import solve_sqd_active

results = []

def bench_point(name, h1e, eri, norb, nelec, shots, max_strings, seed=0):
    bsm = np.random.default_rng(seed).random((shots, 2 * norb)) > 0.5
    probs = np.full(shots, 1.0 / shots)
    row = {"name": name, "shots": shots, "max_strings": max_strings}
    for backend in ("cpu", "gpu"):
        t0 = time.perf_counter()
        E = solve_sqd_active(
            h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
            max_strings=max_strings, n_active_per_round=30, rand_seed=seed,
            backend=backend, verbose=False)
        row[f"t_{backend}"] = time.perf_counter() - t0
        row[f"E_{backend}"] = float(E)
    row["speedup"] = row["t_cpu"] / row["t_gpu"]
    row["E_diff"] = abs(row["E_cpu"] - row["E_gpu"])
    results.append(row)
    print(f"[{name}] cpu={row['t_cpu']:.1f}s gpu={row['t_gpu']:.1f}s "
          f"speedup={row['speedup']:.2f}x E_diff={row['E_diff']:.2e}", flush=True)
    return row

# P1: N2/STO-3G 小 dim（crossover 慢区）
from pyscf import gto
mol1 = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
d1 = from_pyscf(mol1)
bench_point("P1_N2STO3G_dim1.6k", d1.h1e, d1.eri, d1.norb, d1.nelec, shots=30, max_strings=40)

# P2: N2/cc-pVDZ 10o 中 dim（过渡区）
mol2 = gto.M(atom="N 0 0 0; N 0 0 3.0", basis="cc-pvdz", verbose=0)
d2 = from_pyscf(mol2, n_core=2, n_virtual=16)  # 10o 活性
bench_point("P2_N2ccpvdz10o_dim17k", d2.h1e, d2.eri, d2.norb, d2.nelec, shots=100, max_strings=130)

# P3: N2/cc-pVDZ 12,12 大 dim（GPU 受益区）
npz = np.load(os.path.join(BASE, "_n2_1212_ints.npz"))
bench_point("P3_N2ccpvdz1212_dim824k", npz["h1e"], npz["eri"], 12, (6, 6), shots=500, max_strings=None)

# 汇总
print("\n=== 汇总 ===")
for r in results:
    print(f"{r['name']}: dim~{r.get('max_strings', 'none')} shots={r['shots']} "
          f"cpu={r['t_cpu']:.1f}s gpu={r['t_gpu']:.1f}s speedup={r['speedup']:.2f}x "
          f"E_diff={r['E_diff']:.2e}")

out = os.path.join(BASE, "benchmarks", "_round006_speedup_points.json")
import json
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"saved {out}")
