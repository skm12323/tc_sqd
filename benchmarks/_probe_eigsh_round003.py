"""Probe: cupyx eigsh vs scipy eigsh 迭代数 (matvec count) at dim=5e4 / 5e5。

假设: dim=5e4 GPU 433s / dim=5e5 GPU 371s 疑似 cupyx ARPACK 收敛停滞 (matvec 数爆炸),
非 OOM (t1 峰值 4.6GB << 17GB)。本 probe 直接数 matvec 次数验证。
"""
import os
import sys
import time

import numpy as np
from pyscf.fci import cistring, selected_ci

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tc_sqd.selected_ci_gpu import sigma_selected_ci_gpu, _get_kernels  # noqa: E402

d = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "_n2_1212_ints.npz"))
h1e, eri = d["h1e"], d["eri"]
norb, nelec = 12, (6, 6)
full = np.array(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)


def build_links(sa):
    return [selected_ci.des_des_linkstr(sa, norb, nelec[0], True),
            selected_ci.des_des_linkstr(sa, norb, nelec[1], True),
            selected_ci.cre_des_linkstr(sa, norb, nelec[0], True),
            selected_ci.cre_des_linkstr(sa, norb, nelec[1], True)]


def run_cupyx(n_str, tol=1e-10):
    import cupy as cp
    from cupyx.scipy.sparse.linalg import eigsh, LinearOperator
    sa = full[:n_str]
    na = nb = len(sa)
    dim = na * nb
    links = build_links(sa)
    kernels = _get_kernels()
    cnt = {"n": 0}

    def matvec(x):
        cnt["n"] += 1
        xv = cp.asarray(x).reshape(na, nb)
        return sigma_selected_ci_gpu(xv, sa, sa, norb, nelec, h1e, eri,
                                     links, kernels).reshape(-1)
    A = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    t0 = time.perf_counter()
    try:
        e, c = eigsh(A, k=1, which="SA", tol=tol)
        print(f"[cupyx] n_str={n_str} dim={dim} tol={tol:.0e} "
              f"matvecs={cnt['n']} wall={time.perf_counter()-t0:.1f}s E={float(e[0]):.10f}")
    except Exception as ex:
        print(f"[cupyx] n_str={n_str} dim={dim} EXC {type(ex).__name__}: {ex} "
              f"matvecs={cnt['n']} wall={time.perf_counter()-t0:.1f}s")


def run_scipy(n_str, maxiter=3000):
    from scipy.sparse.linalg import LinearOperator as SciOp, eigsh
    from pyscf import ao2mo
    from pyscf.fci import direct_spin1
    sa = full[:n_str]
    na = nb = len(sa)
    dim = na * nb
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
        e, c = eigsh(A, k=1, which="SA", maxiter=maxiter)
        print(f"[scipy ] n_str={n_str} dim={dim} maxiter={maxiter} "
              f"matvecs={cnt['n']} wall={time.perf_counter()-t0:.1f}s E={float(e[0]):.10f}")
    except Exception as ex:
        print(f"[scipy ] n_str={n_str} dim={dim} EXC {type(ex).__name__}: {ex} "
              f"matvecs={cnt['n']} wall={time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "224"
    n_str = int(which)
    print(f"=== n_str={n_str} dim={n_str*n_str} ===")
    run_scipy(n_str)
    run_cupyx(n_str)
