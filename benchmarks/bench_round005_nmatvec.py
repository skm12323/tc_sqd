"""R5 round_005: P0' N_matvec probe — hybrid 继承 scipy 收敛的因果验证 (theory §3 P0')。

在 dim 1e5 (n_str=317) / 5e5 (n_str=708) 直接数 ARPACK matvec 次数, 三变体对照:
  - scipy_cpu : scipy eigsh (默认 tol=0) + CPU contract_2e matvec  — 镜像 CPU 分支
  - hybrid    : scipy eigsh (tol=1e-10) + GPU sigma matvec (.get() 回 numpy)
                — 镜像 round_005 hybrid 分支 (含 eri 缓存, 与 _Subspace 逐字相同)
  - cupyx     : cupyx eigsh (tol=1e-10, 无 maxiter, round_004 原始口径) + GPU sigma
                matvec (留 cupy) — 镜像 round_004 cupyx (未加护栏, 测原始 N_mv)

验收 (theory §3 P0'): hybrid N_mv ∈ [0.8×, 1.3×] scipy N_mv
  (dim 1e5 ~560-910 / dim 5e5 ~650-1050), 远小于 cupyx 的 ~7000。
若 cupyx stall (SIGALRM 超时), 记录已累积 matvec 数 + 标 "stall"。

每 dim 独立进程 (`--n_str` 一次 = 一个进程, 三变体按 scipy_cpu -> hybrid -> cupyx
顺序, hybrid 先跑触发 GPU RawKernel 编译 = cupyx 的 warm-up)。

用法:
  python bench_round005_nmatvec.py --n_str 317 [--cupyx_timeout 1500] [--only]

输出: 增量落盘 `benchmarks/_round005_results.json` {"nmatvec": {...}} + stdout。
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
from pyscf import ao2mo
from pyscf.fci import cistring, direct_spin1, selected_ci

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tc_sqd.selected_ci_gpu import (  # noqa: E402
    sigma_selected_ci_gpu, _selci_eri_aaaa, _selci_eri_bbaa, _get_kernels)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTS = os.path.join(BASE, "_n2_1212_ints.npz")
OUT = os.path.join(BASE, "benchmarks", "_round005_results.json")


class _Timeout:
    """cupyx eigsh stall 护栏: SIGALRM -> 置 flag, 由外层 loop 检查退出。"""

    def __init__(self):
        self.fired = False

    def _handler(self, signum, frame):
        self.fired = True

    def start(self, seconds):
        if seconds and seconds > 0:
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(int(seconds))


def load_1212():
    d = np.load(INTS)
    return d["h1e"], d["eri"]


def make_subspace(n_str):
    norb, nelec = 12, (6, 6)
    full = np.array(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    sa = full[:n_str]
    sb = full[:n_str]
    return sa, sb, norb, nelec


def run_scipy_cpu(sa, sb, norb, nelec, h1e, eri):
    """CPU 分支镜像: scipy eigsh 默认 tol + contract_2e (cipsi.py else 分支)。"""
    from scipy.sparse.linalg import LinearOperator as SciOp, eigsh
    dim = len(sa) * len(sb)
    link = selected_ci._all_linkstr_index((sa, sb), norb, nelec)
    h2e = ao2mo.restore(1, direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5), norb)
    myci = selected_ci.SCI()
    cnt = {"n": 0}

    def matvec(v):
        cnt["n"] += 1
        v = np.ascontiguousarray(v, dtype=np.float64)
        hv = myci.contract_2e(h2e, selected_ci._as_SCIvector(v, (sa, sb)),
                              norb, nelec, link).reshape(-1)
        return np.ascontiguousarray(hv, dtype=np.float64)
    A = SciOp((dim, dim), matvec=matvec, dtype=np.float64)
    t0 = time.perf_counter()
    try:
        e, c = eigsh(A, k=1, which="SA", maxiter=3000)
        return {"matvecs": cnt["n"], "wall": time.perf_counter() - t0,
                "E": float(e[0]), "status": "ok"}
    except Exception as ex:
        return {"matvecs": cnt["n"], "wall": time.perf_counter() - t0,
                "status": "exc", "exc": f"{type(ex).__name__}: {ex}"}


def _gpu_ctx(sa, sb, norb, nelec, h1e, eri):
    """GPU matvec 公共构件 (links + kernels + eri 缓存, 与 _Subspace GPU 路径逐字相同)。"""
    import cupy as cp
    from pyscf.fci import selected_ci as _sci
    links = [_sci.des_des_linkstr(sa, norb, nelec[0], True),
             _sci.des_des_linkstr(sb, norb, nelec[1], True),
             _sci.cre_des_linkstr(sa, norb, nelec[0], True),
             _sci.cre_des_linkstr(sb, norb, nelec[1], True)]
    kernels = _get_kernels()
    h2e = ao2mo.restore(1, direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5), norb)
    eri1_aaaa = cp.asarray(_selci_eri_aaaa(h2e, norb))
    eri1_bbaa = cp.asarray(_selci_eri_bbaa(h2e, norb, nelec))
    return links, kernels, eri1_aaaa, eri1_bbaa


def run_hybrid(sa, sb, norb, nelec, h1e, eri, tol=1e-10):
    """hybrid 分支镜像: scipy eigsh + GPU sigma matvec (.get() 回 numpy)。

    tol=1e-10  (默认) = shipped hybrid 分支 (cipsi.py:204);
    tol=0       = 与 CPU 分支同 tol 的引擎匹配测 (隔离变量: 仅 matvec 实现不同)。
    """
    from scipy.sparse.linalg import LinearOperator as SciOp, eigsh
    import cupy as cp
    dim = len(sa) * len(sb)
    nA = len(sa)
    links, kernels, eri1_aaaa, eri1_bbaa = _gpu_ctx(sa, sb, norb, nelec, h1e, eri)
    cnt = {"n": 0}

    def matvec(v):
        cnt["n"] += 1
        xv = cp.asarray(v).reshape(nA, nA)
        sigma = sigma_selected_ci_gpu(
            xv, sa, sb, norb, nelec, h1e, eri,
            links=links, kernels=kernels,
            eri1_aaaa=eri1_aaaa, eri1_bbaa=eri1_bbaa)
        return np.asarray(sigma.get()).ravel()
    A = SciOp((dim, dim), matvec=matvec, dtype=np.float64)
    t0 = time.perf_counter()
    try:
        e, c = eigsh(A, k=1, which="SA", tol=tol, maxiter=3000)
        return {"matvecs": cnt["n"], "wall": time.perf_counter() - t0,
                "E": float(e[0]), "status": "ok"}
    except Exception as ex:
        return {"matvecs": cnt["n"], "wall": time.perf_counter() - t0,
                "status": "exc", "exc": f"{type(ex).__name__}: {ex}"}


def run_cupyx(sa, sb, norb, nelec, h1e, eri, timeout_s):
    """cupyx 分支镜像 (round_004 原始口径, 无 maxiter): cupyx eigsh + GPU sigma 留 cupy。"""
    import cupy as cp
    from cupyx.scipy.sparse.linalg import eigsh, LinearOperator
    dim = len(sa) * len(sb)
    nA = len(sa)
    links, kernels, eri1_aaaa, eri1_bbaa = _gpu_ctx(sa, sb, norb, nelec, h1e, eri)
    cnt = {"n": 0}
    to = _Timeout()
    to.start(timeout_s)

    def matvec(x):
        cnt["n"] += 1
        if to.fired:
            raise RuntimeError("PROBE_TIMEOUT")
        xv = cp.asarray(x).reshape(nA, nA)
        return sigma_selected_ci_gpu(
            xv, sa, sb, norb, nelec, h1e, eri,
            links=links, kernels=kernels,
            eri1_aaaa=eri1_aaaa, eri1_bbaa=eri1_bbaa).reshape(-1)
    A = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    t0 = time.perf_counter()
    try:
        e, c = eigsh(A, k=1, which="SA", tol=1e-10)
        return {"matvecs": cnt["n"], "wall": time.perf_counter() - t0,
                "E": float(e[0]), "status": "ok"}
    except RuntimeError as ex:
        if "PROBE_TIMEOUT" in str(ex):
            return {"matvecs": cnt["n"], "wall": time.perf_counter() - t0,
                    "status": "stall", "timeout_s": timeout_s}
        return {"matvecs": cnt["n"], "wall": time.perf_counter() - t0,
                "status": "exc", "exc": f"{type(ex).__name__}: {ex}"}
    except Exception as ex:
        return {"matvecs": cnt["n"], "wall": time.perf_counter() - t0,
                "status": "exc", "exc": f"{type(ex).__name__}: {ex}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_str", type=int, required=True)
    ap.add_argument("--cupyx_timeout", type=int, default=1500,
                    help="cupyx eigsh wall 超时 (s), 0=不限")
    ap.add_argument("--only", action="store_true")
    args = ap.parse_args()
    n_str = args.n_str
    dim = n_str * n_str
    print(f"=== n_str={n_str} dim={dim} ===", flush=True)

    h1e, eri = load_1212()
    sa, sb, norb, nelec = make_subspace(n_str)

    scipy_res = run_scipy_cpu(sa, sb, norb, nelec, h1e, eri)
    print(f"[scipy_cpu] {scipy_res}", flush=True)

    hybrid_res = run_hybrid(sa, sb, norb, nelec, h1e, eri, tol=1e-10)
    print(f"[hybrid    ] {hybrid_res}", flush=True)

    hybrid_tol0 = run_hybrid(sa, sb, norb, nelec, h1e, eri, tol=0)
    print(f"[hybrid_t0 ] {hybrid_tol0}", flush=True)

    cupyx_res = run_cupyx(sa, sb, norb, nelec, h1e, eri, args.cupyx_timeout)
    print(f"[cupyx     ] {cupyx_res}", flush=True)

    out_entry = {"n_str": n_str, "dim": dim,
                 "scipy_cpu": scipy_res, "hybrid": hybrid_res,
                 "hybrid_tol0": hybrid_tol0, "cupyx": cupyx_res}
    if (scipy_res.get("status") == "ok" and hybrid_res.get("status") == "ok"):
        out_entry["hybrid_over_scipy"] = hybrid_res["matvecs"] / scipy_res["matvecs"]
    if (scipy_res.get("status") == "ok"
            and hybrid_tol0.get("status") == "ok"):
        out_entry["hybrid_tol0_over_scipy"] = (
            hybrid_tol0["matvecs"] / scipy_res["matvecs"])
    if (scipy_res.get("status") == "ok" and cupyx_res.get("status") == "ok"):
        out_entry["cupyx_over_scipy"] = cupyx_res["matvecs"] / scipy_res["matvecs"]

    if not args.only:
        if os.path.exists(OUT):
            with open(OUT) as f:
                data = json.load(f)
        else:
            data = {}
        data.setdefault("nmatvec", []).append(out_entry)
        with open(OUT, "w") as f:
            json.dump(data, f, indent=2, default=float)
        print(f"[probe] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
