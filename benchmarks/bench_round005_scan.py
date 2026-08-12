"""R5 round_005: P0 hybrid dim 扫描 driver — **每点独立进程** (theory §3 声明 3)。

扫描 dim ∈ {1e4, 5e4, 1e5, 5e5} (n_str ∈ {100, 224, 317, 708}):
  - P0 : hybrid (backend="gpu", gpu_eigsh_mode="hybrid") vs cpu, 单次 diag wall。
         验收: dim 1e5 ratio ≤0.25 (≥4×) / dim 5e5 ratio ≤0.5 (≥2×) 证实。
  - 三模式对照 (dim 1e5/5e5): hybrid / cupyx / cpu_fallback / cpu 四模式 wall。
         预期: hybrid 最快 > cpu_fallback ≈ CPU > cupyx (cupyx 模式 maxiter 护栏 ->
         ArpackNoConvergence -> except 回退 CPU, shipped 行为)。
  - P1 : dim>1e5 的 hybrid vs cpu E diff ≤1e-10。

每个 (dim, mode) 用**独立 subprocess** 跑 `bench_round005_scan_point.py` (消除跨点
显存碎片 confound)。超时护栏按 (mode, dim) 设档 (cupyx 模式含 3000 matvec 失败尝试 +
CPU 回退, 给最宽)。

输出: 增量落盘 `benchmarks/_round005_results.json`
  {"scan": {"rows": [...], "p0_verdict":..., "p1_ok":..., "modes_3way": {...}}}
stdout 打印每点结果 + P0/P1 三态判定。
"""
import argparse
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "benchmarks", "_round005_results.json")
WORKER = os.path.join(BASE, "benchmarks", "bench_round005_scan_point.py")

# n_str -> dim (全 C(12,6)=924 字符串取前 n_str; round_003/004 scan 口径)
SCAN_NSTR = [100, 224, 317, 708]

# 每 (mode, n_str) 的 subprocess wall 超时 (s)。上界估算 (warm-up ~40s + 稳态 diag):
#   hybrid: dim 5e5 ~33s (R3 实测) -> 1500s 富余; cupyx 模式 = 3000×GPU matvec
#   (dim 5e5 ~40ms -> 120s) + CPU 回退 (~140s) -> 2400s; CPU/cpu_fallback ~140s。
TIMEOUT = {
    "cpu":          {100: 600, 224: 1200, 317: 1200, 708: 1200},
    "hybrid":       {100: 600, 224: 1200, 317: 1200, 708: 1500},
    "cpu_fallback": {100: 600, 224: 1200, 317: 1200, 708: 1200},
    "cupyx":        {100: 900, 224: 1500, 317: 1500, 708: 2400},
}


def run_point(n_str, mode):
    """spawn 独立进程跑单点, 返回 dict (含 status: ok / stall / exc / error)。"""
    timeout = TIMEOUT[mode][n_str]
    cmd = [sys.executable, WORKER, "--n_str", str(n_str), "--mode", mode]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"n_str": n_str, "dim": n_str * n_str, "mode": mode,
                "status": "stall", "timeout_s": timeout}
    if proc.returncode != 0:
        return {"n_str": n_str, "dim": n_str * n_str, "mode": mode,
                "status": "error", "returncode": proc.returncode,
                "stderr_tail": (proc.stderr or "")[-2000:]}
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        data["status"] = data.get("status", "ok")
        return data
    except Exception:
        return {"n_str": n_str, "dim": n_str * n_str, "mode": mode,
                "status": "error", "stdout_tail": (proc.stdout or "")[-2000:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nstrs", type=int, nargs="*", default=None,
                    help="扫描 n_str 子集 (默认全 4 点)")
    ap.add_argument("--only", action="store_true", help="不落盘 (调试)")
    args = ap.parse_args()

    scan_nstr = args.nstrs if args.nstrs else SCAN_NSTR

    rows = []          # P0: hybrid vs cpu
    modes_3way = {}    # 三模式对照: dim -> {mode: row}
    for n_str in scan_nstr:
        dim = n_str * n_str
        c = run_point(n_str, "cpu")
        h = run_point(n_str, "hybrid")
        row = {"n_str": n_str, "dim": dim,
               "t_cpu": c.get("wall"), "t_hybrid": h.get("wall"),
               "E_cpu": c.get("E"), "E_hybrid": h.get("E"),
               "cpu_status": c.get("status"), "hybrid_status": h.get("status"),
               "hybrid_warmup": h.get("warmup_wall")}
        if c.get("status") == "ok" and h.get("status") == "ok":
            row["ratio"] = h["wall"] / c["wall"]
            row["speedup"] = c["wall"] / h["wall"]
            row["E_diff"] = abs(c["E"] - h["E"])
            print(f"[scan] dim={dim:7d} cpu={c['wall']:9.2f}s "
                  f"hybrid={h['wall']:9.2f}s ratio={row['ratio']:.3f} "
                  f"speedup={row['speedup']:.2f}x "
                  f"Ediff={row['E_diff']:.2e}", flush=True)
        else:
            print(f"[scan] dim={dim:7d} status cpu={c.get('status')} "
                  f"hybrid={h.get('status')}", flush=True)
        rows.append(row)

        # 三模式对照 (dim 1e5 / 5e5): 补 cupyx + cpu_fallback
        if dim in (100489, 501264):
            cyx = run_point(n_str, "cupyx")
            cfb = run_point(n_str, "cpu_fallback")
            modes_3way[str(dim)] = {
                "hybrid": h, "cpu": c, "cupyx": cyx, "cpu_fallback": cfb}
            for mode in ("cupyx", "cpu_fallback"):
                r = {"hybrid": h, "cpu": c, "cupyx": cyx, "cpu_fallback": cfb}[mode]
                if r.get("status") == "ok":
                    print(f"[3way] dim={dim:7d} {mode:14s} "
                          f"wall={r['wall']:9.2f}s E={r['E']:.10f}", flush=True)
                else:
                    print(f"[3way] dim={dim:7d} {mode:14s} "
                          f"status={r.get('status')}", flush=True)

    # ---- P0 / P1 判定 (theory §3) ----
    # P0: dim 1e5 ratio≤0.25 (≥4×) 证实 / 0.25-0.5 部分 / >0.5 证伪;
    #     dim 5e5 ratio≤0.5 (≥2×) 证实 / 0.5-0.67 部分 / >0.67 证伪。
    p1_rows = [r for r in rows if r["dim"] > 1e5 and r.get("E_diff") is not None]
    p1_ok = all(r["E_diff"] <= 1e-10 for r in p1_rows) and len(p1_rows) > 0

    d1e5 = next((r for r in rows if r["dim"] == 100489 and r.get("ratio") is not None), None)
    d5e5 = next((r for r in rows if r["dim"] == 501264 and r.get("ratio") is not None), None)
    p0_verdicts = []
    if d1e5:
        p0_verdicts.append(f"dim1e5 ratio={d1e5['ratio']:.3f} "
                           f"({'证实' if d1e5['ratio'] <= 0.25 else '部分' if d1e5['ratio'] <= 0.5 else '证伪'})")
    if d5e5:
        p0_verdicts.append(f"dim5e5 ratio={d5e5['ratio']:.3f} "
                           f"({'证实' if d5e5['ratio'] <= 0.5 else '部分' if d5e5['ratio'] <= 0.67 else '证伪'})")
    p0_verdict = "; ".join(p0_verdicts) if p0_verdicts else "无有效点"
    print(f"[scan] P1 判定: {'证实' if p1_ok else '证伪'}  (|E_hybrid-E_cpu|≤1e-10, "
          f"n={len(p1_rows)})", flush=True)
    print(f"[scan] P0 判定: {p0_verdict}", flush=True)

    result = {"scan": {"rows": rows, "p0_verdict": p0_verdict, "p1_ok": p1_ok,
                       "modes_3way": modes_3way,
                       "timeouts": {}}}
    if not args.only:
        with open(OUT, "w") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"[bench] saved scan -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
