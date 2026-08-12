"""Probe: cupyx eigsh matvec count at n_str=708 (dim=5e5), flush 逐行输出。

验证 dim=5e5 GPU 371.5s 是 cupyx ARPACK 收敛停滞 (matvec 爆炸) 还是其他。
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

import cupy as cp  # noqa: E402
from cupyx.scipy.sparse.linalg import eigsh, LinearOperator  # noqa: E402

n_str = 708
sa = full[:n_str]
na = nb = len(sa)
dim = na * nb
links = [selected_ci.des_des_linkstr(sa, norb, nelec[0], True),
         selected_ci.des_des_linkstr(sa, norb, nelec[1], True),
         selected_ci.cre_des_linkstr(sa, norb, nelec[0], True),
         selected_ci.cre_des_linkstr(sa, norb, nelec[1], True)]
kernels = _get_kernels()
cnt = {"n": 0}
print(f"n_str={n_str} dim={dim} start", flush=True)


def matvec(x):
    cnt["n"] += 1
    xv = cp.asarray(x).reshape(na, nb)
    return sigma_selected_ci_gpu(xv, sa, sa, norb, nelec, h1e, eri,
                                 links, kernels).reshape(-1)


A = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
t0 = time.perf_counter()
try:
    e, c = eigsh(A, k=1, which="SA", tol=1e-10)
    print(f"RESULT cupyx n_str={n_str} dim={dim} tol=1e-10 "
          f"matvecs={cnt['n']} wall={time.perf_counter()-t0:.1f}s E={float(e[0]):.10f}",
          flush=True)
except Exception as ex:
    print(f"EXC {type(ex).__name__}: {ex} matvecs={cnt['n']} "
          f"wall={time.perf_counter()-t0:.1f}s", flush=True)
print("DONE", flush=True)
