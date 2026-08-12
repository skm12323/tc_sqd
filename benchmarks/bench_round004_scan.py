"""R5 round_004: P0 dim 扫描 driver — **每点独立进程** (theory §3 声明 3)。

消除 round_003 单进程跨点 GPU 显存碎片 confound: 每个 (dim, backend) 用
**独立 subprocess** 跑 `bench_round004_scan_point.py`, 收集 wall/E。

扫描 dim ∈ {1e4, 5e4, 1e5, 5e5} (n_str ∈ {100, 224, 317, 708}), 每点:
  _Subspace(..., backend="gpu").diag vs backend="cpu").diag 单次 wall。

GPU worker wall 超时护栏 (round_003 同款 cupyx eigsh stall):
  超时 -> kill 子进程 + 记录 stall (不产出 ratio, 诚实标 "stall")。
  超时阈值按 dim 设档 (warm-up ~85s + 稳态 diag 上界)。

输出: 增量落盘 `benchmarks/_round004_results.json` {"scan": {rows, ...}}。
stdout 打印每点结果 + P0/P1 三态判定。
"""
import argparse
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "benchmarks", "_round004_results.json")
WORKER = os.path.join(BASE, "benchmarks", "bench_round004_scan_point.py")

# n_str -> dim² (全 C(12,6)=924 字符串取前 n_str; round_003 scan 口径)
SCAN_NSTR = [100, 224, 317, 708]

# GPU worker 总 wall 超时 (s): warm-up ~85s + 稳态 diag 上界 (round_003 实测:
# dim 5e4 433s, dim 5e5 371s 但 erratic 可 stall 至 500s+)。CPU worker 确定性,
# 给宽松 1200s (dim 5e5 CPU ~137s)。
GPU_TIMEOUT = {100: 600, 224: 1200, 317: 900, 708: 2400}
CPU_TIMEOUT = 1200


def run_point(n_str, backend):
    """spawn 独立进程跑单点, 返回 dict (含 status: ok / stall / exc / error)。"""
    timeout = GPU_TIMEOUT[n_str] if backend == "gpu" else CPU_TIMEOUT
    cmd = [sys.executable, WORKER, "--n_str", str(n_str), "--backend", backend]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # cupyx eigsh stall (round_003 同款): 记录 stall, 不产出 ratio
        return {"n_str": n_str, "dim": n_str * n_str, "backend": backend,
                "status": "stall", "timeout_s": timeout}
    if proc.returncode != 0:
        return {"n_str": n_str, "dim": n_str * n_str, "backend": backend,
                "status": "error", "returncode": proc.returncode,
                "stderr_tail": (proc.stderr or "")[-2000:]}
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        data["status"] = data.get("status", "ok")
        return data
    except Exception:
        return {"n_str": n_str, "dim": n_str * n_str, "backend": backend,
                "status": "error", "stdout_tail": (proc.stdout or "")[-2000:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nstrs", type=int, nargs="*", default=None,
                    help="扫描 n_str 子集 (默认全 4 点)")
    ap.add_argument("--only", action="store_true", help="不落盘 (调试)")
    args = ap.parse_args()

    scan_nstr = args.nstrs if args.nstrs else SCAN_NSTR

    cpu_rows, gpu_rows = {}, {}
    for n_str in scan_nstr:
        c = run_point(n_str, "cpu")
        g = run_point(n_str, "gpu")
        cpu_rows[n_str] = c
        gpu_rows[n_str] = g
        dim = n_str * n_str
        if c.get("status") == "ok" and g.get("status") == "ok":
            ratio = g["wall"] / c["wall"]
            print(f"[scan] dim={dim:7d} cpu={c['wall']:9.2f}s "
                  f"gpu={g['wall']:9.2f}s ratio={ratio:.3f} "
                  f"speedup={c['wall']/g['wall']:.2f}x "
                  f"Ediff={abs(c['E']-g['E']):.2e}", flush=True)
        elif g.get("status") == "stall":
            print(f"[scan] dim={dim:7d} cpu={c.get('wall', float('nan')):.2f}s "
                  f"gpu=STALL(t>{g['timeout_s']}s) — cupyx eigsh 停滞, 无有效 ratio",
                  flush=True)
        else:
            print(f"[scan] dim={dim:7d} status cpu={c.get('status')} "
                  f"gpu={g.get('status')}", flush=True)

    # ---- P0 / P1 判定 ----
    rows = []
    p1_rows = []
    for n_str in scan_nstr:
        c, g = cpu_rows[n_str], gpu_rows[n_str]
        row = {"n_str": n_str, "dim": n_str * n_str,
               "t_cpu": c.get("wall"), "t_gpu": g.get("wall"),
               "E_cpu": c.get("E"), "E_gpu": g.get("E"),
               "gpu_status": g.get("status"), "cpu_status": c.get("status")}
        if c.get("status") == "ok" and g.get("status") == "ok":
            row["ratio"] = g["wall"] / c["wall"]
            row["speedup"] = c["wall"] / g["wall"]
            row["E_diff"] = abs(c["E"] - g["E"])
            if row["dim"] > 1e5:
                p1_rows.append(row)
        rows.append(row)

    p1_ok = all(r.get("E_diff", 1.0) <= 1e-10 for r in p1_rows) and len(p1_rows) > 0
    # P0: dim>1e5 的 ok 点逐点 ratio≤0.33 (≥3×) 才证实; 三态按 theory §3
    p0_dims = [r for r in rows if r["dim"] > 1e5 and r.get("ratio") is not None]
    p0_confirmed = all(r["ratio"] <= 0.33 for r in p0_dims) and len(p0_dims) > 0
    p0_partial = any(r["ratio"] <= 0.5 for r in p0_dims)
    if p0_confirmed:
        p0_verdict = "证实"
    elif p0_partial:
        p0_verdict = "部分 (dim>1e5 有点 ratio≤0.5 但无点 ≤0.33)"
    else:
        p0_verdict = "证伪 (无任何 dim>1e5 点 ratio≤0.33)"
    print(f"[scan] P1 判定: {'证实' if p1_ok else '证伪'}  (|E_gpu-E_cpu|≤1e-10, "
          f"n={len(p1_rows)})", flush=True)
    print(f"[scan] P0 判定: {p0_verdict}  (ratio≤0.33 证实 / ≤0.5 部分 / >0.5 证伪)",
          flush=True)

    result = {"scan": {"rows": rows, "p0_verdict": p0_verdict,
                       "p1_ok": p1_ok,
                       "gpu_timeouts": {str(k): v.get("timeout_s")
                                        for k, v in gpu_rows.items()
                                        if v.get("status") == "stall"}}}
    if not args.only:
        with open(OUT, "w") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"[bench] saved scan -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
