"""round_013 P0a/P1: eigsh_tol 消融 + active 批量消融。

问题: round_012 closure 后 12,12 @500 = ~130s (GPU warm)。时间画像:
active 循环 ~115s (~10 次增量 diag) + closure ~15s。GPU hybrid 的 tol 硬编码
1e-10; CPU 分支 tol=0。round_005 实证 CPU tol=1e-10 vs 0 快 1.46× (E diff~1e-13)。

消融 (固定输入, 12,12 @500 closure, warm GPU):
  P0a-GPU: eigsh_tol ∈ {None(=1e-10), 1e-8, 1e-6} → wall/n_mv/err 权衡
  P1-GPU:  n_active_per_round ∈ {30(基线), 60, 90} (tol=1e-10) → 轮次/批量权衡
  P0a-CPU: 10o @500 closure, eigsh_tol ∈ {None(=0), 1e-10, 1e-8} → CPU 路径收益

判定口径:
  - tol 档: err vs e_ref 反映 tol 精度代价; wall/n_mv 反映收益
  - 批量档: E 不劣化 (变分) + wall 是否下降
"""
import os
import sys
import time
import json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

from tc_sqd.cipsi import solve_sqd_active, _Subspace

_ORIG_DIAG = _Subspace.diag


def _make_runner():
    """返回 (run, ...) —— 每次调用独立 n_mv 计数器 (包装真原始 diag)。"""
    n_mv = [0]

    def wrapped(self, sa, sb):
        r = _ORIG_DIAG(self, sa, sb)
        n = getattr(self, "last_n_mv", 0)
        if n:
            n_mv[0] += n
        return r
    return wrapped, n_mv


def run_case(h1e, eri, norb, nelec, ecore, shots, seed, *,
             eigsh_tol=None, n_active=30, warm_start=True, backend="gpu",
             coverage_closure=True):
    wrapped, n_mv = _make_runner()
    _Subspace.diag = wrapped
    try:
        traj = []
        t0 = time.perf_counter()
        kw = dict(ecore=ecore, max_strings=None,
                  n_active_per_round=n_active, rand_seed=seed,
                  tail_suppression=True, tail_shots_ref=100,
                  warm_start=warm_start, backend=backend,
                  verbose=False, trajectory=traj,
                  coverage_closure=coverage_closure)
        if eigsh_tol is not None:
            kw["eigsh_tol"] = eigsh_tol
        E = solve_sqd_active(
            h1e, eri, norb, nelec,
            bitstring_matrix=np.random.default_rng(seed).random(
                (shots, 2 * norb)) > 0.5,
            probabilities=np.full(shots, 1.0 / shots), **kw)
        wall = time.perf_counter() - t0
    finally:
        _Subspace.diag = _ORIG_DIAG
    final = traj[-1] if traj else {}
    return {"E": float(E), "wall": round(wall, 1),
            "dim": int(final.get("dim", -1)),
            "n_mv": n_mv[0]}


def main():
    results = {}

    # ============ P0a-GPU: 12,12 @500 closure, tol 扫描 ============
    print("\n" + "=" * 72)
    print("P0a-GPU: N2/cc-pVDZ (12,12) @500 closure, eigsh_tol 扫描 (warm GPU)")
    print("=" * 72, flush=True)
    npz = np.load(os.path.join(BASE, "_n2_1212_ints.npz"))
    h1e, eri = npz["h1e"], npz["eri"]
    ecore = float(npz["ecore"])
    e_ref = float(npz["e_ref"])
    NCAS, NELEC = 12, (6, 6)
    tol_rows = []
    for label, tol in [("tol_1e-10(默认)", None), ("tol_1e-8", 1e-8),
                       ("tol_1e-6", 1e-6)]:
        r = run_case(h1e, eri, NCAS, NELEC, ecore, 500, 0, eigsh_tol=tol)
        r.update(label=label, tol=tol, err=abs(r["E"] - e_ref))
        tol_rows.append(r)
        print(f"  {label}: E={r['E']:.10f} err={r['err']:.2e} dim={r['dim']} "
              f"n_mv={r['n_mv']} wall={r['wall']}s", flush=True)
    results["gpu_12_12_tol"] = {"e_ref": e_ref, "rows": tol_rows}

    # ============ P1-GPU: 12,12 @500 closure, n_active 批量扫描 ============
    print("\n" + "=" * 72)
    print("P1-GPU: N2/cc-pVDZ (12,12) @500 closure, n_active 扫描 (tol=1e-10)")
    print("=" * 72, flush=True)
    act_rows = []
    for n_active in [30, 60, 90]:
        r = run_case(h1e, eri, NCAS, NELEC, ecore, 500, 0,
                     eigsh_tol=None, n_active=n_active)
        r.update(label=f"n_active={n_active}", n_active=n_active,
                 err=abs(r["E"] - e_ref))
        act_rows.append(r)
        print(f"  n_active={n_active}: E={r['E']:.10f} err={r['err']:.2e} "
              f"dim={r['dim']} n_mv={r['n_mv']} wall={r['wall']}s", flush=True)
    results["gpu_12_12_n_active"] = {"rows": act_rows}

    # ============ P0a-CPU: 10o @500 closure, tol 扫描 ============
    print("\n" + "=" * 72)
    print("P0a-CPU: N2/cc-pVDZ (10o) @500 closure, eigsh_tol 扫描 (warm CPU)")
    print("=" * 72, flush=True)
    npz10 = np.load(os.path.join(BASE, "_n2_ccpvdz_10o_ints.npz"))
    h1e10, eri10 = npz10["h1e"], npz10["eri"]
    ecore10 = float(npz10["ecore"])
    e_ref10 = float(npz10["e_ref"])
    cpu_rows = []
    for label, tol in [("tol_0(默认)", None), ("tol_1e-10", 1e-10),
                       ("tol_1e-8", 1e-8)]:
        r = run_case(h1e10, eri10, 10, (5, 5), ecore10, 500, 0,
                     eigsh_tol=tol, backend="cpu")
        r.update(label=label, tol=tol, err=abs(r["E"] - e_ref10))
        cpu_rows.append(r)
        print(f"  {label}: E={r['E']:.10f} err={r['err']:.2e} dim={r['dim']} "
              f"n_mv={r['n_mv']} wall={r['wall']}s", flush=True)
    results["cpu_10o_tol"] = {"e_ref": e_ref10, "rows": cpu_rows}

    # ============ 汇总 ============
    print("\n" + "=" * 72)
    print("=== round_013 P0a/P1 汇总 ===")
    print("=" * 72)
    base = tol_rows[0]
    print("\nGPU 12,12 closure tol 扫描 (基线 = tol 1e-10):")
    print(f"{'label':<16} {'err':>10} {'n_mv':>6} {'wall':>7} {'wall比':>7}")
    for r in tol_rows:
        print(f"{r['label']:<16} {r['err']:>10.2e} {r['n_mv']:>6} "
              f"{r['wall']:>7.1f} {r['wall']/base['wall']:>7.2f}x")
    bact = act_rows[0]
    print("\nGPU 12,12 closure n_active 扫描 (基线 = 30):")
    print(f"{'label':<14} {'err':>10} {'n_mv':>6} {'wall':>7} {'wall比':>7}")
    for r in act_rows:
        print(f"{r['label']:<14} {r['err']:>10.2e} {r['n_mv']:>6} "
              f"{r['wall']:>7.1f} {r['wall']/bact['wall']:>7.2f}x")
    bcpu = cpu_rows[0]
    print("\nCPU 10o closure tol 扫描 (基线 = tol 0):")
    print(f"{'label':<14} {'err':>10} {'n_mv':>6} {'wall':>7} {'wall比':>7}")
    for r in cpu_rows:
        print(f"{r['label']:<14} {r['err']:>10.2e} {r['n_mv']:>6} "
              f"{r['wall']:>7.1f} {r['wall']/bcpu['wall']:>7.2f}x")

    out = os.path.join(BASE, "benchmarks", "_round013_tol_batch.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
