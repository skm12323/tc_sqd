"""R5 round_004: cupyx vs scipy eigsh N_matvec 诊断 (theory §4 P1 #7, round_005 铺路)。

在 dim 1e5 (n_str=317) / 5e5 (n_str=708) 数 matvec 次数 (ARPACK 迭代):
  - scipy eigsh (which="SA", maxiter=3000, CPU contract_2e matvec) — 确定性
  - cupyx eigsh (which="SA", tol=1e-10, GPU sigma matvec) — 带 wall 超时
    (round_003 同款 cupyx ARPACK 收敛停滞: dim 5e4 33× / dim 1e5 9.2×)。

round_003 口径 (对齐可比较): scipy 默认 tol (机器精度), cupyx tol=1e-10。
cupyx 超时 -> 记录已累积 matvec 数 + wall, 标 "stall" (不产出收敛 E)。

用法:
  python _probe_eigsh_round004.py --n_str 317 [--only]
  python _probe_eigsh_round004.py --n_str 708 [--cupyx_timeout 1500]
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
from pyscf.fci import cistring, selected_ci

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tc_sqd.selected_ci_gpu import sigma_selected_ci_gpu, _get_kernels  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTS = os.path.join(BASE, "_n2_1212_ints.npz")
OUT = os.path.join(BASE, "benchmarks", "_round004_results.json")


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


def build_links(sa):
    norb, nelec = 12, (6, 6)
    return [selected_ci.des_des_linkstr(sa, norb, nelec[0], True),
            selected_ci.des_des_linkstr(sa, norb, nelec[1], True),
            selected_ci.cre_des_linkstr(sa, norb, nelec[0], True),
            selected_ci.cre_des_linkstr(sa, norb, nelec[1], True)]


def run_scipy(n_str):
    from scipy.sparse.linalg import LinearOperator as SciOp, eigsh
    from pyscf import ao2mo
    from pyscf.fci import direct_spin1
    d = np.load(INTS)
    h1e, eri = d["h1e"], d["eri"]
    norb, nelec = 12, (6, 6)
    full = np.array(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    sa = full[:n_str]
    na = dim = len(sa) * len(sa)
    link = selected_ci._all_linkstr_index((sa, sa), norb, nelec)
    h2e = ao2mo.restore(1, direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5), norb)
    myci = selected_ci.SCI()
    cnt = {"n": 0}

    def matvec(v):
        cnt["n"] += 1
        v = np.ascontiguousarray(v, dtype=np.float64)
        hv = myci.contract_2e(h2e, selected_ci._as_SCIvector(v, (sa, sa)),
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


def run_cupyx(n_str, timeout_s):
    import cupy as cp
    from cupyx.scipy.sparse.linalg import eigsh, LinearOperator
    d = np.load(INTS)
    h1e, eri = d["h1e"], d["eri"]
    norb, nelec = 12, (6, 6)
    full = np.array(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    sa = full[:n_str]
    na = len(sa)
    dim = na * na
    links = build_links(sa)
    kernels = _get_kernels()
    cnt = {"n": 0}
    to = _Timeout()
    to.start(timeout_s)

    def matvec(x):
        cnt["n"] += 1
        if to.fired:
            raise RuntimeError("PROBE_TIMEOUT")
        xv = cp.asarray(x).reshape(na, na)
        return sigma_selected_ci_gpu(xv, sa, sa, norb, nelec, h1e, eri,
                                     links, kernels).reshape(-1)
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

    scipy_res = run_scipy(n_str)
    print(f"[scipy ] {scipy_res}", flush=True)

    cupyx_res = run_cupyx(n_str, args.cupyx_timeout)
    print(f"[cupyx ] {cupyx_res}", flush=True)

    out = {"probe_eigsh": {
        "n_str": n_str, "dim": dim,
        "scipy": scipy_res, "cupyx": cupyx_res}}
    if scipy_res.get("status") == "ok" and cupyx_res.get("status") == "ok":
        out["probe_eigsh"]["ratio_matvecs"] = cupyx_res["matvecs"] / scipy_res["matvecs"]
    if not args.only:
        if os.path.exists(OUT):
            with open(OUT) as f:
                data = json.load(f)
        else:
            data = {}
        data.setdefault("probe_eigsh", []).append(out["probe_eigsh"])
        with open(OUT, "w") as f:
            json.dump(data, f, indent=2, default=float)
        print(f"[probe] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
