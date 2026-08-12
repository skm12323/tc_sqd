"""tc_sqd.selected_ci_gpu —— selected-CI 子空间 GPU matrix-free matvec (3-contraction)。

**动机**：``solve_sci`` 用 PySCF ``selected_ci.contract_2e``（子空间专用，4-block
linkstr），而 ``matrixfree.linkstr_gpu``（direct_spin1 语义）仅全空间正确。本模块
移植 selected_ci 的 **3-contraction 算法**到 GPU（RawKernel scatter/gather +
batched matmul），**子空间正确** 且快于 pyscf C 核。

**算法**（pyscf/lib/mcscf/select_ci.c 的 SCIcontract_2e_aaaa / bbaa）：
- **aaaa_α/β**（同自旋双）：``des_des`` linkstr，intermediate = nelec-2 双消灭去重
  目标集（**含子空间外**，子空间正确的关键），antisym eri tril ⟨ij‖ab⟩×2；
- **bbaa**（αβ 交叉 + h1e）：``cre_des`` linkstr（子空间内单激发），eri×2+h_ps
  restore(4)。
每个 contraction = scatter (RawKernel atomicAdd) + eri batched matmul + gather。

**验证**（2026-08-11）：vs selected_ci.contract_2e —— H2 2.8e-17、Be 7.1e-15、
N2/STO-3G 全空间 2.3e-13；cupy 与 numpy 参考一致（≤3e-13）。N2 dim=14400 matvec
1.79ms（pyscf C 核 ~3ms）。

**关键调试记录**：numpy 参考初始 err 大，根因有二 —— ① ββ 的 fcivec 须传 ``v.T``
（_aaaa_np 内部按 β 索引取行）；② cupy ``v.T`` 是转置视图（非连续内存），kernel
线性索引错，须 ``np.ascontiguousarray(v.T)``。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["sigma_selected_ci_gpu", "eigsh_selected_ci_gpu"]

_SRC = r'''
extern "C" __global__
void scat_t1(const int* conn, const int* tril, const int* s0, const double* sg,
             const double* v, double* t1, int nconn, int nn, int nb) {
    int c = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= nconn || j >= nb) return;
    int inter = conn[c];
    int tr = tril[c], sv = s0[c];
    atomicAdd(&t1[(inter*nn + tr)*nb + j], sg[c] * v[sv*nb + j]);
}
extern "C" __global__
void gath_t1(const int* conn, const int* tril, const int* s0, const double* sg,
             const double* t1, double* ci1, int nconn, int nn, int nb) {
    int c = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= nconn || j >= nb) return;
    int inter = conn[c];
    int tr = tril[c], sv = s0[c];
    atomicAdd(&ci1[sv*nb + j], sg[c] * t1[(inter*nn + tr)*nb + j]);
}
extern "C" __global__
void scat_ba(const int* conn, const int* tril, const int* s0, const double* sg,
             const double* v, double* t1, int nconn, int npair, int nb) {
    int c = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= nconn || j >= nb) return;
    int stra = conn[c];
    atomicAdd(&t1[(stra*npair + tril[c])*nb + j], sg[c] * v[s0[c]*nb + j]);
}
extern "C" __global__
void gath_bb(const int* conn, const int* tril, const int* s0, const double* sg,
             const double* t1, double* ci1, int nconn, int npair, int na, int nb) {
    int c = blockIdx.x, A = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= nconn || A >= na) return;
    int strb = conn[c];
    atomicAdd(&ci1[A*nb + s0[c]], sg[c] * t1[(A*npair + tril[c])*nb + strb]);
}
'''


def _selci_eri_aaaa(eri, norb):
    eri1 = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)
    idx, idy = np.tril_indices(norb, -1)
    idx_flat = idx * norb + idy
    return np.take(np.take(eri1.reshape(norb * norb, -1), idx_flat, 0), idx_flat, 1) * 2


def _selci_eri_bbaa(eri, norb, nelec):
    h_ps = np.einsum('pqqs->ps', eri)
    eri1 = eri * 2
    for k in range(norb):
        eri1[:, :, k, k] += h_ps / nelec[0]
        eri1[k, k, :, :] += h_ps / nelec[1]
    from pyscf import ao2mo
    return ao2mo.restore(4, eri1, norb)


def _links_tril(link_index):
    """link_index (n,m,4) tril: (tril_idx,_,str0,sign) -> 扁平 GPU 连接数组 (conn,tril,s0,sg)。"""
    n, m = link_index.shape[:2]
    conn = np.repeat(np.arange(n), m)
    f = link_index.reshape(-1, 4)
    mask = f[:, 3] != 0
    import cupy as cp
    return (cp.asarray(conn[mask], np.int32), cp.asarray(f[mask, 0], np.int32),
            cp.asarray(f[mask, 2], np.int32), cp.asarray(f[mask, 3], np.float64), int(mask.sum()))


def _get_kernels():
    import cupy as cp
    if not hasattr(_get_kernels, "_cache"):
        mod = cp.RawModule(code=_SRC)
        _get_kernels._cache = (mod.get_function("scat_t1"), mod.get_function("gath_t1"),
                               mod.get_function("scat_ba"), mod.get_function("gath_bb"))
    return _get_kernels._cache


def sigma_selected_ci_gpu(v, ci_a, ci_b, norb, nelec, h1e, eri,
                          links=None, kernels=None,
                          eri1_aaaa=None, eri1_bbaa=None):
    """selected-CI 子空间 ``H·v`` (GPU, 3-contraction)。子空间正确。

    Parameters
    ----------
    v : (na, nb), numpy 或 cupy (C-contiguous)
    links : 4-tuple (dd_a, dd_b, cd_a, cd_b) tril-mode linkstr (可复用)
    kernels : 预编译 RawKernel (可复用)
    eri1_aaaa, eri1_bbaa : cupy ndarray, optional
        预算并缓存的第二电子积分重组 (round_004 方式 C)。``None`` 时内部从
        ``(h1e, eri, norb, nelec)`` 重算 (== round_003 现状, 向后兼容)。
        非 ``None`` 时直接复用, 跳过 ``absorb_h1e`` + ``ao2mo.restore`` +
        ``_selci_eri_*`` + 2× ``cp.asarray`` 的 per-matvec 重算。
        **必须是 cupy 数组** (本函数直接喂 ``cp.matmul``, 不再 ``cp.asarray``)。
        由调用方保证与 ``(h1e, eri, norb, nelec)`` 一致
        (``_Subspace.__init__`` 用同 ``self.h2e`` 预算, cipsi.py:99-100 与
        本函数重算分支逐字相同 -> 数值天然一致)。

    Returns
    -------
    sigma : cupy (na, nb)
    """
    import cupy as cp
    from pyscf import ao2mo
    from pyscf.fci import selected_ci, direct_spin1
    na, nb = len(ci_a), len(ci_b)
    if eri1_aaaa is None or eri1_bbaa is None:
        # 重算分支 (默认 == round_003 现状, 逐字一致)
        h2e = ao2mo.restore(1, direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5), norb)
        if eri1_aaaa is None:
            eri1_aaaa = cp.asarray(_selci_eri_aaaa(h2e, norb))
        if eri1_bbaa is None:
            eri1_bbaa = cp.asarray(_selci_eri_bbaa(h2e, norb, nelec))
    # else: 直接复用调用方传入的 cupy 缓存 (方式 C), 跳过 absorb_h1e+restore+
    # _selci_eri_*+2× cp.asarray 的 per-matvec 重算 (eri1_* 已是 cupy 数组)。
    if links is None:
        links = [selected_ci.des_des_linkstr(ci_a, norb, nelec[0], True),
                 selected_ci.des_des_linkstr(ci_b, norb, nelec[1], True),
                 selected_ci.cre_des_linkstr(ci_a, norb, nelec[0], True),
                 selected_ci.cre_des_linkstr(ci_b, norb, nelec[1], True)]
    if kernels is None:
        kernels = _get_kernels()
    scat_t1, gath_t1, scat_ba, gath_bb = kernels
    dd_a, dd_b, cd_a, cd_b = links
    vg = cp.ascontiguousarray(v, np.float64) if isinstance(v, cp.ndarray) \
        else cp.asarray(np.ascontiguousarray(v), np.float64)
    ci1 = cp.zeros((na, nb))
    TPB = 256
    nn = norb * (norb - 1) // 2
    npair = norb * (norb + 1) // 2
    # aaaa α
    if nelec[0] > 1:
        ca, tra, sa, sga, nca = _links_tril(dd_a)
        t1 = cp.zeros((dd_a.shape[0], nn, nb))
        scat_t1((nca, (nb + TPB - 1) // TPB, 1), (TPB, 1, 1), (ca, tra, sa, sga, vg, t1, nca, nn, nb))
        vt1 = cp.matmul(eri1_aaaa, t1)
        gath_t1((nca, (nb + TPB - 1) // TPB, 1), (TPB, 1, 1), (ca, tra, sa, sga, vt1, ci1, nca, nn, nb))
    # aaaa β (转置, v.T 须 C-contiguous)
    if nelec[1] > 1:
        cb, trb, sb, sgb, ncb = _links_tril(dd_b)
        t1b = cp.zeros((dd_b.shape[0], nn, na))
        vT = cp.ascontiguousarray(vg.T)
        scat_t1((ncb, (na + TPB - 1) // TPB, 1), (TPB, 1, 1), (cb, trb, sb, sgb, vT, t1b, ncb, nn, na))
        vt1b = cp.matmul(eri1_aaaa, t1b)
        ci1T = cp.zeros((nb, na))
        gath_t1((ncb, (na + TPB - 1) // TPB, 1), (TPB, 1, 1), (cb, trb, sb, sgb, vt1b, ci1T, ncb, nn, na))
        ci1 += ci1T.T
    # bbaa
    cba, trba, sba, sgba, ncba = _links_tril(cd_a)
    t1c = cp.zeros((na, npair, nb))
    scat_ba((ncba, (nb + TPB - 1) // TPB, 1), (TPB, 1, 1), (cba, trba, sba, sgba, vg, t1c, ncba, npair, nb))
    vt1c = cp.matmul(eri1_bbaa, t1c)
    cbb, trbb, sbb, sgbb, ncbb = _links_tril(cd_b)
    gath_bb((ncbb, (na + TPB - 1) // TPB, 1), (TPB, 1, 1), (cbb, trbb, sbb, sgbb, vt1c, ci1, ncbb, npair, na, nb))
    return ci1


def eigsh_selected_ci_gpu(ci_a, ci_b, norb, nelec, h1e, eri, *, k=1, which="SA", tol=1e-8,
                          eri1_aaaa=None, eri1_bbaa=None):
    """selected-CI 子空间 GPU 本征求解 (cupyx eigsh + 3-contraction matvec)。

    子空间正确 (与 selected_ci.contract_2e 一致, ≤1e-13), 用于 solve_sci(backend="gpu")。

    Parameters
    ----------
    eri1_aaaa, eri1_bbaa : cupy ndarray, optional
        预算缓存 (round_004 方式 C)。``None`` 时 matvec 内部每次重算 (== round_003
        现状, ``fermion.solve_sci`` 等既有调用者不传 -> 逐字零回归)。非 ``None``
        时透传给 :func:`sigma_selected_ci_gpu`, 跳过 per-matvec 重算。
    """
    import cupy as cp
    from pyscf.fci import selected_ci
    from cupyx.scipy.sparse.linalg import eigsh, LinearOperator
    na, nb = len(ci_a), len(ci_b)
    dim = na * nb
    links = [selected_ci.des_des_linkstr(ci_a, norb, nelec[0], True),
             selected_ci.des_des_linkstr(ci_b, norb, nelec[1], True),
             selected_ci.cre_des_linkstr(ci_a, norb, nelec[0], True),
             selected_ci.cre_des_linkstr(ci_b, norb, nelec[1], True)]
    kernels = _get_kernels()

    def matvec(x):
        xv = cp.asarray(x).reshape(na, nb)
        return sigma_selected_ci_gpu(xv, ci_a, ci_b, norb, nelec, h1e, eri, links, kernels,
                                     eri1_aaaa=eri1_aaaa, eri1_bbaa=eri1_bbaa).reshape(-1)
    A = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    e, c = eigsh(A, k=k, which=which, tol=tol)
    e = e.get() if hasattr(e, "get") else np.asarray(e)
    c = c.get() if hasattr(c, "get") else np.asarray(c)
    return np.asarray(e), np.asarray(c)
