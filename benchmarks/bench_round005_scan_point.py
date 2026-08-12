"""R5 round_005: P0 hybrid dim 扫描单点 worker — 每点独立进程 (theory §3 声明 3)。

用法:
  python bench_round005_scan_point.py --n_str 317 --mode hybrid
  python bench_round005_scan_point.py --n_str 317 --mode cpu

在**独立进程**中测 `_Subspace(backend=..., gpu_eigsh_mode=...).diag` 单次 wall
(round_005 hybrid 三模式 vs CPU, 消除跨点显存碎片 confound):
  - cpu          : backend="cpu"  (默认路径, scipy eigsh + contract_2e, 默认 tol)
  - hybrid       : backend="gpu", gpu_eigsh_mode="hybrid"  (scipy eigsh + GPU sigma
                   matvec, tol=1e-10; round_005 方向 A 主路径)
  - cupyx        : backend="gpu", gpu_eigsh_mode="cupyx"  (cupyx eigsh + GPU sigma
                   matvec, maxiter=3000 护栏; 预期 ArpackNoConvergence -> except 回退
                   CPU scipy = 本轮 shipped 行为)
  - cpu_fallback : backend="gpu", gpu_eigsh_mode="cpu_fallback"  (scipy eigsh +
                   contract_2e, tol=1e-10; GPU 不参与 matvec, 隔离 "慢在 eigsh 还是
                   GPU matvec")

GPU worker 先小维度 (n_str=100) warm-up (cupy 上下文 + RawModule 编译, 进程级一次性,
不计入稳态 wall) 再计时。stdout 单行 JSON (driver 解析):
  {"n_str":317,"dim":100489,"mode":"hybrid","wall":6.2,"E":-68.2240407675,
   "warmup_wall":40.0,"status":"ok","backend_effective":"gpu"}
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from pyscf.fci import cistring

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tc_sqd.cipsi import _Subspace  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTS = os.path.join(BASE, "_n2_1212_ints.npz")
WARMUP_NSTR = 100  # 小维度 warm-up, 触发 cupy 上下文 + RawModule 编译 (进程级一次性)

GPU_MODES = ("hybrid", "cupyx", "cpu_fallback")


def load_1212():
    d = np.load(INTS)
    return (d["h1e"], d["eri"], float(d["ecore"]), float(d["e_ref"]))


def make_subspace(n_str):
    """N2/cc-pVDZ 12-MO 窗口固定子空间: 前 n_str 个 alpha/beta 字符串 (sa==sb)。"""
    h1e, eri, _, _ = load_1212()
    norb, nelec = 12, (6, 6)
    full = np.array(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    sa = full[:n_str]
    sb = full[:n_str]
    return h1e, eri, norb, nelec, sa, sb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_str", type=int, required=True)
    ap.add_argument("--mode", choices=["cpu", "hybrid", "cupyx", "cpu_fallback"],
                    required=True)
    args = ap.parse_args()

    out = {"n_str": args.n_str, "mode": args.mode}
    backend = "gpu" if args.mode in GPU_MODES else "cpu"

    # warm-up (GPU backend 仅): 小维度 diag, 触发 cupy 上下文 + RawKernel 编译
    warmup_wall = 0.0
    if backend == "gpu":
        h1e, eri, norb, nelec, sa_w, sb_w = make_subspace(WARMUP_NSTR)
        sub_w = _Subspace(h1e, eri, norb, nelec, backend="gpu",
                          gpu_eigsh_mode=args.mode)
        out["backend_effective"] = sub_w.backend
        t0 = time.perf_counter()
        sub_w.diag(sa_w, sb_w)
        warmup_wall = time.perf_counter() - t0
        if sub_w.backend != "gpu":
            out["status"] = "degraded_cpu"
            out["warmup_wall"] = warmup_wall
            print(json.dumps(out), flush=True)
            return
    else:
        out["backend_effective"] = "cpu"

    h1e, eri, norb, nelec, sa, sb = make_subspace(args.n_str)
    dim = len(sa) * len(sb)
    out["dim"] = dim
    out["warmup_wall"] = warmup_wall

    sub = _Subspace(h1e, eri, norb, nelec, backend=backend,
                    gpu_eigsh_mode=args.mode)
    t0 = time.perf_counter()
    try:
        E, _, _, _ = sub.diag(sa, sb)
        wall = time.perf_counter() - t0
        out.update({"status": "ok", "E": float(E), "wall": wall,
                    "backend_effective": sub.backend})
    except Exception as ex:  # 不应发生 (diag 内部有回退护栏), 兜底记录
        wall = time.perf_counter() - t0
        out.update({"status": "exc", "exc": f"{type(ex).__name__}: {ex}",
                    "wall": wall})
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
