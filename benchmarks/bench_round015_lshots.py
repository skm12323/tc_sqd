"""round_015: coverage_closure 低 shots 稳健性。

问题: closure 配方在 @500 已验证 (853,776 全空间 FCI)。低 shots 下采样覆盖
更低 (@100 可能只得 ~600-800 串), BFS 闭包需补更多串——是否仍能补全到
全空间? err/wall 随 shots 如何变化? 配方适用边界在哪?

设计 (12,12, 配方 config: coverage_closure=True + eigsh_tol=1e-6 + n_active=90):
  shots 扫 {500(已知锚), 100, 50, 20}: closure vs baseline(无闭包)
  记录: dim / n_str / err / wall / n_mv / diag 次数 (closure 爬阶层级)
10o 旁证 (CPU, 便宜): shots {100, 50, 20} closure。

可证伪预测:
  P0: @100 与 @50 closure 仍达 dim=853,776 且 err ≤1e-9
  P1: @20 closure 达全空间 (或明确记录断点); 全空间处 err 与 shots 无关 (=FCI)
  P2: wall 随 shots 下降 (active 循环更短) 直到被 closure 的全空间 diag 主导
"""
import os, sys, time, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
from tc_sqd.cipsi import solve_sqd_active, _Subspace
_ORIG = _Subspace.diag


def run(h1e, eri, norb, nelec, ecore, shots, seed=0, *,
        coverage_closure=True, eigsh_tol=1e-6, n_active=90,
        warm_start=True, backend="gpu"):
    n_mv = [0]
    n_diag = [0]
    str_seq = []

    def wrapped(self, sa, sb):
        str_seq.append(len(sa))
        n_diag[0] += 1
        r = _ORIG(self, sa, sb)
        n = getattr(self, "last_n_mv", 0)
        if n:
            n_mv[0] += n
        return r
    _Subspace.diag = wrapped
    try:
        traj = []
        kw = dict(ecore=ecore, max_strings=None, n_active_per_round=n_active,
                  rand_seed=seed, tail_suppression=True, tail_shots_ref=100,
                  warm_start=warm_start, backend=backend, verbose=False,
                  trajectory=traj, coverage_closure=coverage_closure)
        if eigsh_tol is not None:
            kw["eigsh_tol"] = eigsh_tol
        t0 = time.perf_counter()
        E = solve_sqd_active(
            h1e, eri, norb, nelec,
            bitstring_matrix=np.random.default_rng(seed).random(
                (shots, 2 * norb)) > 0.5,
            probabilities=np.full(shots, 1.0 / shots), **kw)
        wall = time.perf_counter() - t0
    finally:
        _Subspace.diag = _ORIG
    final = traj[-1] if traj else {}
    import math
    dim = int(final.get("dim", -1))
    return {"E": float(E), "wall": round(wall, 1), "dim": dim,
            "n_str": int(round(math.sqrt(dim))) if dim > 0 else -1,
            "n_mv": n_mv[0], "n_diag": n_diag[0],
            "str_seq_head": str_seq[:6], "str_seq_tail": str_seq[-4:]}


res = {}
# ---- 12,12 主实验 ----
z = np.load(os.path.join(BASE, "_n2_1212_ints.npz"))
h1e, eri = z["h1e"], z["eri"]
ec = float(z["ecore"]); eref = float(z["e_ref"])
rows12 = []
for shots in [500, 100, 50, 20]:
    print(f"\n=== 12,12 @{shots} closure(配方) ===", flush=True)
    r = run(h1e, eri, 12, (6, 6), ec, shots)
    r.update(shots=shots, err=abs(r["E"] - eref)); rows12.append(r)
    print(f"  E={r['E']:.10f} err={r['err']:.2e} dim={r['dim']} n_str={r['n_str']} "
          f"n_mv={r['n_mv']} n_diag={r['n_diag']} wall={r['wall']}s", flush=True)
    print(f"  str_seq: head={r['str_seq_head']} tail={r['str_seq_tail']}", flush=True)
res["n2_12_12_closure"] = {"e_ref": eref, "rows": rows12}

# baseline (无闭包) 同 shots —— 采样覆盖对照
rowsb = []
for shots in [500, 100, 50, 20]:
    r = run(h1e, eri, 12, (6, 6), ec, shots, coverage_closure=False,
            eigsh_tol=None, n_active=30)
    r.update(shots=shots, err=abs(r["E"] - eref)); rowsb.append(r)
    print(f"  baseline @{shots}: dim={r['dim']} n_str={r['n_str']} "
          f"err={r['err']:.2e} wall={r['wall']}s", flush=True)
res["n2_12_12_baseline"] = {"rows": rowsb}

# ---- 10o 旁证 (CPU 便宜) ----
z10 = np.load(os.path.join(BASE, "_n2_ccpvdz_10o_ints.npz"))
h1e10, eri10 = z10["h1e"], z10["eri"]
ec10 = float(z10["ecore"]); eref10 = float(z10["e_ref"])
rows10 = []
for shots in [100, 50, 20]:
    r = run(h1e10, eri10, 10, (5, 5), ec10, shots, backend="cpu",
            eigsh_tol=1e-8)
    r.update(shots=shots, err=abs(r["E"] - eref10)); rows10.append(r)
    print(f"  10o @{shots}: dim={r['dim']} n_str={r['n_str']} err={r['err']:.2e} "
          f"wall={r['wall']}s", flush=True)
res["n2_10o_closure"] = {"e_ref": eref10, "rows": rows10}

# ---- 汇总 ----
print("\n" + "=" * 74)
print("=== round_015 低 shots closure 稳健性汇总 (12,12, 配方) ===")
print("=" * 74)
print(f"{'shots':>6} {'base_dim':>9} {'base_err':>10} | {'clos_dim':>9} "
      f"{'clos_err':>10} {'clos_wall':>9} {'n_diag':>6}")
FULL = 853776
for rb, rc in zip(rowsb, rows12):
    print(f"{rb['shots']:>6} {rb['dim']:>9} {rb['err']:>10.2e} | "
          f"{rc['dim']:>9} {rc['err']:>10.2e} {rc['wall']:>8.1f}s {rc['n_diag']:>6}")
p0 = all(r["dim"] == FULL and r["err"] <= 1e-9 for r in rows12 if r["shots"] in (100, 50))
p1 = all(r["dim"] == FULL and r["err"] <= 1e-9 for r in rows12 if r["shots"] == 20)
p10 = all(r["dim"] == 252 * 252 for r in rows10)
print(f"\nP0 (@100,@50 补全+err≤1e-9): {'PASS' if p0 else 'FAIL'}")
print(f"P1 (@20 补全+err≤1e-9):       {'PASS' if p1 else 'FAIL'}")
print(f"P1' (10o 全部补全):           {'PASS' if p10 else 'FAIL'}")

out = os.path.join(BASE, "benchmarks", "_round015_lshots.json")
with open(out, "w") as f:
    json.dump(res, f, indent=2)
print(f"\nsaved {out}")
