"""R5 round_003: _Subspace GPU backend 跑分 (P0 dim scan + P1 正确性 + 端到端加速)。

theory.md §3 三锚点:
- **P0 性能**: 固定子空间 (N2/cc-pVDZ 12-MO 窗口), dim ∈ {1e4,5e4,1e5,5e5},
  _Subspace(...,backend="gpu").diag vs "cpu" 单次 wall 对照; dim>1e5 时 GPU/CPU ratio ≤0.33 (≥3×)。
  记录 crossover (dim ∈ (1e4,5e4) 可能 GPU 慢)。
- **P1 正确性**: dim>1e5 子空间 GPU vs CPU E diff ≤1e-10 (R3 已测 2.27e-13, 确认一致)。
- **端到端附加测**: 12,12 solve_sqd_improved(backend="gpu") vs "cpu" 整条 wall 对照
  (seed=0, shots=500, max_strings=1e5, C1-v2: tail_suppression=True, tail_shots_ref=100)。

口径:
- CPU/GPU 同进程跑 (避免环境差异)。先 CPU 后 GPU, 记录 GPU warm-up (进程级一次性)。
- 同一 sa/sb/h1e/eri (积分复用 _n2_1212_ints.npz)。
- 结果增量写 JSON (benchmarks/_round003_gpu_results.json, 幂等可续)。

用法:
  python bench_round003_gpu.py --phase all          # 全部 (默认)
  python bench_round003_gpu.py --phase scan         # 只 P0+P1 dim 扫描
  python bench_round003_gpu.py --phase e2e          # 只端到端
  python bench_round003_gpu.py --phase scan --only  # 不落盘 (调试)
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
from pyscf.fci import cistring

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import tc_sqd  # noqa: E402
from tc_sqd.cipsi import _Subspace  # noqa: E402
from tc_sqd.noise import has_gpu  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "benchmarks", "_round003_gpu_results.json")

# 12,12 N2/cc-pVDZ R=3.0 (round_002 口径)
INTS = os.path.join(BASE, "_n2_1212_ints.npz")
E_REF_1212 = -108.7686857

# P0 dim 扫描: n_str → dim = n_str² (全 C(12,6)=924 字符串取前 n_str)
SCAN_NSTR = [100, 224, 317, 708]   # dim ≈ 1e4 / 5e4 / 1e5 / 5e5


def have_gpu():
    try:
        import cupy  # noqa: F401
        if not cupy.cuda.runtime.getDeviceCount():
            return False
        return True
    except Exception:
        return False


def load_1212():
    d = np.load(INTS)
    return (d["h1e"], d["eri"], float(d["ecore"]), float(d["e_ref"]))


def make_subspace(n_str):
    """N2/cc-pVDZ 12-MO 窗口固定子空间: 前 n_str 个 alpha/beta 字符串。"""
    h1e, eri, _, _ = load_1212()
    norb, nelec = 12, (6, 6)
    full = np.array(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    sa = full[:n_str]
    sb = full[:n_str]
    return h1e, eri, norb, nelec, sa, sb


def diag_timed(sub, sa, sb):
    t0 = time.perf_counter()
    E, c2d, _, _ = sub.diag(sa, sb)
    wall = time.perf_counter() - t0
    return E, wall


def run_scan(only=False, verbose=True):
    """P0 dim 扫描 + P1 正确性。返回结果 dict。"""
    if not have_gpu():
        print("[scan] SKIP: 无 cupy/GPU", flush=True)
        return {"scan": {"skipped": True}}

    rows = []
    # ---- CPU 全部 dim (先跑, 免 GPU warm-up 污染) ----
    cpu = {}
    for n_str in SCAN_NSTR:
        h1e, eri, norb, nelec, sa, sb = make_subspace(n_str)
        dim = len(sa) * len(sb)
        sub = _Subspace(h1e, eri, norb, nelec, backend="cpu")
        E, t = diag_timed(sub, sa, sb)
        cpu[n_str] = {"dim": dim, "E": E, "wall": t}
        if verbose:
            print(f"[scan] cpu n_str={n_str:4d} dim={dim:7d} E={E:.10f} wall={t:.3f}s",
                  flush=True)

    # ---- GPU: 先 warm-up (进程级一次性, 不计入稳态) ----
    n_str0 = SCAN_NSTR[0]
    h1e, eri, norb, nelec, sa, sb = make_subspace(n_str0)
    sub_gpu0 = _Subspace(h1e, eri, norb, nelec, backend="gpu")
    t0 = time.perf_counter()
    E_warm, _, _, _ = sub_gpu0.diag(sa, sb)
    t_warm = time.perf_counter() - t0
    assert sub_gpu0.backend == "gpu", "有 GPU 时 backend='gpu' 应保持"
    if verbose:
        print(f"[scan] gpu warmup n_str={n_str0} dim={cpu[n_str0]['dim']} "
              f"E={E_warm:.10f} wall={t_warm:.3f}s (含 cupy/RawModule 初始化)", flush=True)

    # ---- GPU 全部 dim (稳态计时) ----
    gpu = {}
    for n_str in SCAN_NSTR:
        h1e, eri, norb, nelec, sa, sb = make_subspace(n_str)
        dim = len(sa) * len(sb)
        sub = _Subspace(h1e, eri, norb, nelec, backend="gpu")
        E, t = diag_timed(sub, sa, sb)
        gpu[n_str] = {"dim": dim, "E": E, "wall": t}
        if verbose:
            print(f"[scan] gpu n_str={n_str:4d} dim={dim:7d} E={E:.10f} wall={t:.3f}s",
                  flush=True)

    # ---- 汇总 + P0/P1 判定 ----
    rows = []
    p1_rows = []
    for n_str in SCAN_NSTR:
        c, g = cpu[n_str], gpu[n_str]
        assert c["dim"] == g["dim"]
        ratio = g["wall"] / c["wall"]
        speedup = c["wall"] / g["wall"]
        row = {
            "n_str": n_str, "dim": c["dim"],
            "E_cpu": c["E"], "E_gpu": g["E"],
            "E_diff": abs(c["E"] - g["E"]),
            "t_cpu": c["wall"], "t_gpu": g["wall"],
            "ratio": ratio, "speedup": speedup,
        }
        rows.append(row)
        if verbose:
            print(f"[scan] dim={c['dim']:7d} ratio={ratio:.3f} speedup={speedup:.2f}x "
                  f"Ediff={row['E_diff']:.2e}", flush=True)
        if c["dim"] > 1e5:
            p1_rows.append(row)

    p1_ok = all(r["E_diff"] <= 1e-10 for r in p1_rows)
    p0_dims = [r for r in rows if r["dim"] > 1e5]
    p0_ok = all(r["ratio"] <= 0.33 for r in p0_dims)
    if verbose:
        print(f"[scan] P0 dim>1e5 判定: {p0_ok}  (ratio≤0.33 逐点)",
              flush=True)
        print(f"[scan] P1 判定: {p1_ok}  (|E_gpu-E_cpu|≤1e-10 逐点)", flush=True)

    return {
        "scan": {
            "rows": rows,
            "gpu_warmup_wall": t_warm,
            "gpu_warmup_E": E_warm,
            "p0_ok": p0_ok, "p1_ok": p1_ok,
            "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
    }


def run_e2e(only=False, verbose=True):
    """端到端: 12,12 solve_sqd_improved backend='cpu' vs 'gpu' (C1-v2 配置)。"""
    if not have_gpu():
        print("[e2e] SKIP: 无 cupy/GPU", flush=True)
        return {"e2e": {"skipped": True}}

    h1e, eri, ecore, e_ref = load_1212()
    norb, nelec = 12, (6, 6)
    shots, seed = 500, 0
    bsm = np.random.default_rng(seed).random((shots, 2 * norb)) > 0.5
    probs = np.full(shots, 1.0 / shots)

    common = dict(
        bitstring_matrix=bsm, probabilities=probs,
        max_strings=100000, n_active_per_round=30,
        rand_seed=seed, ecore=ecore, return_details=True,
        tail_suppression=True, tail_max_draw_factor=10, tail_shots_ref=100,
    )

    results = {}
    for backend in ["cpu", "gpu"]:
        t0 = time.perf_counter()
        e, det = tc_sqd.solve_sqd_improved(
            h1e, eri, norb, nelec, backend=backend, **common)
        wall = time.perf_counter() - t0
        traj = det["trajectory"]
        pos = [t for t in traj if t["round"] >= 1]
        results[backend] = {
            "wall": wall,
            "E": float(e),
            "E_direct": float(det["E_direct"]),
            "E_PT2": float(det["E_PT2"]),
            "dim": int(det["dim"]),
            "err_vs_fci": float(abs(e - e_ref)),
            "n_rounds": len(pos),
            "per_round": [{"round": t["round"], "E": float(t["E"]),
                           "dim": int(t["dim"])} for t in pos],
        }
        if verbose:
            print(f"[e2e] {backend:3s} wall={wall:.1f}s E={e:.10f} "
                  f"dim={results[backend]['dim']} err={results[backend]['err_vs_fci']:.2e}",
                  flush=True)

    e_diff = abs(results["cpu"]["E"] - results["gpu"]["E"])
    wall_ratio = results["gpu"]["wall"] / results["cpu"]["wall"]
    results["E_diff_cpu_gpu"] = e_diff
    results["wall_ratio_gpu_cpu"] = wall_ratio
    results["speedup"] = results["cpu"]["wall"] / results["gpu"]["wall"]
    if verbose:
        print(f"[e2e] |E_cpu-E_gpu|={e_diff:.2e}  wall_ratio={wall_ratio:.3f} "
              f"speedup={results['speedup']:.2f}x", flush=True)

    return {"e2e": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "scan", "e2e"], default="all")
    ap.add_argument("--only", action="store_true", help="不落盘 (调试)")
    args = ap.parse_args()

    if args.phase in ("all", "scan"):
        res = run_scan(only=args.only)
        if not args.only:
            if os.path.exists(OUT):
                with open(OUT) as f:
                    data = json.load(f)
            else:
                data = {}
            data["scan"] = res["scan"]
            with open(OUT, "w") as f:
                json.dump(data, f, indent=2, default=float)
            print(f"[bench] saved scan -> {OUT}", flush=True)

    if args.phase in ("all", "e2e"):
        res = run_e2e(only=args.only)
        if not args.only:
            if os.path.exists(OUT):
                with open(OUT) as f:
                    data = json.load(f)
            else:
                data = {}
            data["e2e"] = res["e2e"]
            with open(OUT, "w") as f:
                json.dump(data, f, indent=2, default=float)
            print(f"[bench] saved e2e -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
