"""R5 round_004: P0 dim 扫描单点 worker — 每点独立进程 (theory §3 声明 3)。

用法:
  python bench_round004_scan_point.py --n_str 317 --backend gpu

在**独立进程**中测 `_Subspace(backend=...).diag` 单次 wall (消除 round_003
单进程跨点 GPU 显存碎片 confound):
  - CPU worker: 直接对目标 dim 计时 (scipy eigsh, 确定性 ~137s @ dim 5e5)。
  - GPU worker: 先小维度 (n_str=100) warm-up (cupy 上下文 + RawModule 编译,
    进程级一次性, 不计入稳态 wall), 再对目标 dim 计时 (内联 cupy LO + eri 缓存)。

stdout 单行 JSON (driver 解析):
  {"n_str":317,"dim":100489,"backend":"gpu","wall":43.2,"E":-68.2240407675,
   "warmup_wall":85.0,"status":"ok","backend_effective":"gpu"}

若 GPU 路径内部回退 CPU (OOM/不收敛 -> _Subspace.diag except), 以
backend_effective 标记并给出回退 wall (仍诚实记录)。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_str", type=int, required=True)
    ap.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    args = ap.parse_args()

    out = {"n_str": args.n_str, "backend": args.backend}

    # warm-up (GPU 仅): 小维度 diag, 触发 cupy 上下文 + RawKernel 编译
    warmup_wall = 0.0
    if args.backend == "gpu":
        h1e, eri, norb, nelec, sa_w, sb_w = make_subspace(WARMUP_NSTR)
        sub_w = _Subspace(h1e, eri, norb, nelec, backend="gpu")
        out["backend_effective"] = sub_w.backend
        t0 = time.perf_counter()
        sub_w.diag(sa_w, sb_w)
        warmup_wall = time.perf_counter() - t0
        if sub_w.backend != "gpu":
            # 无 GPU 降级 CPU (不应发生, 本机有 GPU; 诚实记录)
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

    sub = _Subspace(h1e, eri, norb, nelec, backend=args.backend)
    t0 = time.perf_counter()
    try:
        E, _, _, _ = sub.diag(sa, sb)
        wall = time.perf_counter() - t0
        out.update({"status": "ok", "E": float(E), "wall": wall,
                    "backend_effective": sub.backend})
    except Exception as ex:  # 不应发生 (diag 内部有回退护栏), 兜底记录
        wall = time.perf_counter() - t0
        out.update({"status": "exc", "exc": f"{type(ex).__name__}: {ex}", "wall": wall})
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
