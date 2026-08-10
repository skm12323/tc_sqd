"""tc_sqd.matrixfree —— 向量化 Slater-Condon σ-vector（matrix-free 对角化基础）。

**动机（GPU 后端第一步，arXiv:2601.16637 matrix-free 路线）**：
`solve_sci` 现有 CPU 路径用 PySCF ``selected_ci.contract_2e``；而"显式构建稀疏 H"
（O(dim) 次逐列 contract_2e）是 GPU 方案的瓶颈（实测占 95% 耗时）。本模块实现
**直接 Slater-Condon σ-vector**：枚举每个 CI 字符串的单/双激发连接，向量化
scatter-add 计算 ``H·v``，绕开逐列构建。

**算法**：子空间 H 在 α×β 字符串乘积基下作用在 ``v[ia,jb]`` 上，
``σ[i,j] = Σ_{i',j'} ⟨i'j'|H|ij⟩ v[i',j']``。Slater-Condon 非零耦合:
- 对角; α/β 单激发; α/β 双激发; **αβ 交叉双激发**。
矩阵元 (化学记号 (pq|rs)=eri[p,q,r,s], ⟨pq‖rs⟩=⟨pq|rs⟩-⟨pq|sr⟩):
- 对角 ``E[I,J]``: 闭合形式。
- α 单 (i' 由 i 做 p→a): ``sign·[h1e[p,a] + Σ_{k∈α(i)}(eri[a,p,k,k]-eri[a,k,p,k]) + Σ_{q∈β(j)} eri[a,p,q,q]]``
  (α 内部分依赖 i, αβ 部分依赖 j —— 预计算 Fock 型 Fa[i,a,p], Fb[j,a,p])
- α 双 (p,q→a,b): ``sign·(eri[a,p,b,q]-eri[a,q,b,p])``
- αβ 交叉 (p→a, q→b): ``sign·eri[a,p,b,q]``
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["sigma_vector", "prepare_sigma_tables", "prepare_sigma_operators",
           "sigma_vector_ops", "eigsh_gpu"]


# --------------------------------------------------------------------------- #
#  位串工具
# --------------------------------------------------------------------------- #
def _occ_bits(s: int, norb: int) -> np.ndarray:
    return np.array([(s >> i) & 1 for i in range(norb)], dtype=bool)


def _excite_single(s: int, norb: int, p: int, a: int) -> Tuple[int, int]:
    """单激发 p→a: 返回 (目标字符串, 符号)。

    符号 = (-1)^(p,a 间**严格** (lo,hi) 排除端点的被占位数)。
    注意: 源位 lo 本身被占据, 不计入 crossing (否则符号反)。
    """
    t = s ^ (1 << p) ^ (1 << a)
    lo, hi = sorted((p, a))
    ncross = bin((s >> (lo + 1)) & ((1 << (hi - lo - 1)) - 1)).count("1")
    sign = -1 if ncross % 2 else 1
    return t, sign


def _double_sign(s: int, norb: int, p: int, q: int, a: int, b: int) -> int:
    """双激发 p,q→a,b 符号 (按 p→a 再 q→b)。"""
    t1, sign1 = _excite_single(s, norb, p, a)
    t2, sign2 = _excite_single(t1, norb, q, b)
    return sign1 * sign2


def _enumerate_excitations(strs: np.ndarray, norb: int, nelec: int):
    """对每个字符串枚举到**目标集合内**的单/双激发。

    Returns
    -------
    singles : list[list[(target_idx, sign, p, a)]]
    doubles : list[list[(target_idx, sign, p, q, a, b)]]
    """
    pos = {int(s): k for k, s in enumerate(strs)}
    occ_lists = [_occ_bits(int(s), norb) for s in strs]
    singles = [[] for _ in strs]
    doubles = [[] for _ in strs]
    for k, s in enumerate(strs):
        occ = occ_lists[k]
        occ_idx = np.nonzero(occ)[0]
        vir_idx = np.nonzero(~occ)[0]
        np_ = [int(x) for x in occ_idx]
        nv_ = [int(x) for x in vir_idx]
        for p in np_:
            for a in nv_:
                t, sign = _excite_single(int(s), norb, p, a)
                if t in pos:
                    singles[k].append((pos[t], sign, p, a))
        for ii in range(len(np_)):
            p = np_[ii]
            for jj in range(ii + 1, len(np_)):
                q = np_[jj]
                for aa in range(len(nv_)):
                    a = nv_[aa]
                    for bb in range(aa + 1, len(nv_)):
                        b = nv_[bb]
                        t = int(s) ^ (1 << p) ^ (1 << q) ^ (1 << a) ^ (1 << b)
                        if t in pos:
                            sign = _double_sign(int(s), norb, p, q, a, b)
                            doubles[k].append((pos[t], sign, p, q, a, b))
    return singles, doubles


# --------------------------------------------------------------------------- #
#  σ-vector
# --------------------------------------------------------------------------- #
def prepare_sigma_tables(ci_strs_a: np.ndarray, ci_strs_b: np.ndarray,
                         norb: int, nelec: Tuple[int, int]):
    """预计算单/双激发连接表 (α 与 β)。"""
    na_elec, nb_elec = nelec
    sa, da = _enumerate_excitations(np.asarray(ci_strs_a), norb, na_elec)
    sb, db = _enumerate_excitations(np.asarray(ci_strs_b), norb, nb_elec)
    return (sa, da), (sb, db)


def sigma_vector(v: np.ndarray, ci_strs_a: np.ndarray, ci_strs_b: np.ndarray,
                 norb: int, nelec: Tuple[int, int],
                 h1e: np.ndarray, eri: np.ndarray,
                 tables=None) -> np.ndarray:
    """Matrix-free ``H·v`` (Slater-Condon σ-vector)。

    Parameters
    ----------
    v : ndarray (na, nb)
    ci_strs_a, ci_strs_b, norb, nelec
    h1e : (norb, norb); eri : (norb,)*4 化学记号
    tables : 预计算连接表 (可复用)

    Returns
    -------
    sigma : ndarray (na, nb)
    """
    v = np.asarray(v, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)
    na, nb = v.shape
    if tables is None:
        tables = prepare_sigma_tables(ci_strs_a, ci_strs_b, norb, nelec)
    (sa, da), (sb, db) = tables

    # 占据向量 (na,norb), (nb,norb)
    occ_a = np.array([[ (int(s)>>i)&1 for i in range(norb)] for s in ci_strs_a], dtype=float)
    occ_b = np.array([[ (int(s)>>i)&1 for i in range(norb)] for s in ci_strs_b], dtype=float)
    # Fock 型矩阵 (四块, 下标见各函数):
    #   Fa_ss[i,a,p] = Σ_k occ_a[i,k](eri[a,p,k,k]-eri[a,k,p,k])   α 单同自旋
    #   Fb_ab[j,a,p] = Σ_q occ_b[j,q] eri[a,p,q,q]                 α 单 α-β
    #   Fb_ss[j,b,q] = Σ_k occ_b[j,k](eri[b,q,k,k]-eri[b,k,q,k])   β 单同自旋
    #   Fa_ab[i,b,q] = Σ_p occ_a[i,p] eri[p,p,b,q]                 β 单 α-β
    h_diag = np.diag(h1e)
    Fa_ss = _fock_same_spin(occ_a, eri)
    Fb_ab = _fock_cross(occ_b, eri)
    Fb_ss = _fock_same_spin(occ_b, eri)
    Fa_ab = _fock_cross(occ_a, eri)

    sigma = np.zeros_like(v)
    # 对角
    sigma += _diag_energy(occ_a, occ_b, h_diag, eri) * v
    # α 单
    _alpha_singles(sigma, v, sa, Fa_ss, Fb_ab, h1e, norb)
    # β 单
    _beta_singles(sigma, v, sb, Fa_ab, Fb_ss, h1e, norb)
    # α 双
    _alpha_doubles(sigma, v, da, eri, norb)
    # β 双
    _beta_doubles(sigma, v, db, eri, norb)
    # αβ 交叉
    _cross_doubles(sigma, v, sa, sb, eri, norb)
    return sigma


def _fock_same_spin(occ, eri):
    """F_ss[i,a,p] = Σ_k occ[i,k] (eri[a,p,k,k] - eri[a,k,p,k]) (同自旋交换 Fock)。"""
    return np.einsum('ik,apkk->iap', occ, eri) - np.einsum('ik,akpk->iap', occ, eri)


def _fock_cross(occ, eri):
    """F_ab[j,a,p] = Σ_q occ[j,q] eri[a,p,q,q] (对侧自旋库仑 Fock)。"""
    return np.einsum('jq,apqq->jap', occ, eri)


def _diag_energy(occ_a, occ_b, h_diag, eri):
    """E[I,J] (na,nb)。"""
    e1 = np.einsum('p,Ip->I', h_diag, occ_a)[:, None] + np.einsum('q,Jq->J', h_diag, occ_b)[None, :]
    E_aa = 0.5 * (np.einsum('Ip,Iq,ppqq->I', occ_a, occ_a, eri)
                  - np.einsum('Ip,Iq,pqqp->I', occ_a, occ_a, eri))
    E_bb = 0.5 * (np.einsum('Jp,Jq,ppqq->J', occ_b, occ_b, eri)
                  - np.einsum('Jp,Jq,pqqp->J', occ_b, occ_b, eri))
    E_ab = np.einsum('Ip,Jq,ppqq->IJ', occ_a, occ_b, eri)
    return e1 + E_aa[:, None] + E_bb[None, :] + E_ab


def _T_single(links_list, norb, n):
    """单激发 T 表: T[m][i,i'] = Σ sign (i'→i via p→a), m 索引 a*norb+p。"""
    norb2 = norb * norb
    T = np.zeros((norb2, n, n))
    for i, links in enumerate(links_list):
        for (i_, sign, p, a) in links:
            T[a * norb + p, i, i_] += sign
    nz = np.nonzero(T.reshape(norb2, -1).any(axis=1))[0]
    return T[nz], nz


def _T_double(links_list, norb, n):
    """双激发 T 表: T[m][i,i'] = Σ sign (i'→i via p,q→a,b), m 按 (p,q,a,b) 类型。"""
    types = sorted({(p, q, a, b) for links in links_list for (_, _, p, q, a, b) in links})
    if not types:
        return np.zeros((0, n, n)), types
    idx = {t: k for k, t in enumerate(types)}
    T = np.zeros((len(types), n, n))
    for i, links in enumerate(links_list):
        for (i_, sign, p, q, a, b) in links:
            T[idx[(p, q, a, b)], i, i_] += sign
    return T, types


def _alpha_singles(sigma, v, sa, Fa_ss, Fb_ab, h1e, norb):
    """α 单激发 (向量化): σ[i,j] += Σ_m (h[p,a]+Fa_ss[i,a,p]+Fb_ab[j,a,p])·(T_m@v)[i,j]。"""
    T, nz = _T_single(sa, norb, v.shape[0])
    if not len(nz):
        return
    a_arr, p_arr = nz // norb, nz % norb
    w = np.matmul(T, v)
    c = h1e[p_arr, a_arr][None, :] + Fa_ss[:, a_arr, p_arr]      # (na, M) [i,m]
    sigma += np.einsum('im,mij->ij', c, w)
    sigma += np.einsum('jm,mij->ij', Fb_ab[:, a_arr, p_arr], w)  # Fb_ab[j,m]


def _beta_singles(sigma, v, sb, Fa_ab, Fb_ss, h1e, norb):
    """β 单激发 (向量化): σ[i,j] += Σ_m (h[q,b]+Fb_ss[j,b,q]+Fa_ab[i,b,q])·(v@T_mᵀ)[i,j]。"""
    T, nz = _T_single(sb, norb, v.shape[1])
    if not len(nz):
        return
    b_arr, q_arr = nz // norb, nz % norb
    w = np.matmul(v, T.transpose(0, 2, 1))
    c = h1e[q_arr, b_arr][None, :] + Fb_ss[:, b_arr, q_arr]      # (nb, M) [j,m]
    sigma += np.einsum('jm,mij->ij', c, w)
    sigma += np.einsum('im,mij->ij', Fa_ab[:, b_arr, q_arr], w)  # Fa_ab[i,m]


def _alpha_doubles(sigma, v, da, eri, norb):
    """α 双激发 (向量化): me_m = eri[a,p,b,q]-eri[a,q,b,p]。"""
    T, types = _T_double(da, norb, v.shape[0])
    if not len(types):
        return
    me = np.array([eri[a, p, b, q] - eri[a, q, b, p] for (p, q, a, b) in types])
    w = np.matmul(T, v)
    sigma += np.einsum('m,mij->ij', me, w)


def _beta_doubles(sigma, v, db, eri, norb):
    """β 双激发 (向量化): 作用在**列**索引, w[m,i,j] = (v @ T_mᵀ)[i,j]。"""
    T, types = _T_double(db, norb, v.shape[1])
    if not len(types):
        return
    me = np.array([eri[a, p, b, q] - eri[a, q, b, p] for (p, q, a, b) in types])
    w = np.matmul(v, T.transpose(0, 2, 1))
    sigma += np.einsum('m,mij->ij', me, w)


def _cross_doubles(sigma, v, sa, sb, eri, norb):
    """αβ 交叉双激发 (向量化)。

    ``σ_cross[i,j] = Σ_{a,p,b,q} eri[a,p,b,q] · (T_a[a,p] @ v @ T_b[q,b]ᵀ)[i,j]``
    其中 ``T_a[a,p][i,i']`` = α 单激发 (i'→i via p→a) 的符号, ``T_b`` 同理。
    """
    na, nb = v.shape
    norb2 = norb * norb
    T_a = np.zeros((norb2, na, na))
    T_b = np.zeros((norb2, nb, nb))
    for i, links in enumerate(sa):
        for (i_, sign, p, a) in links:
            T_a[a * norb + p, i, i_] += sign
    for j, links in enumerate(sb):
        for (j_, sign, q, b) in links:
            T_b[b * norb + q, j, j_] += sign
    nz_a = np.nonzero(T_a.reshape(norb2, -1).any(axis=1))[0]
    nz_b = np.nonzero(T_b.reshape(norb2, -1).any(axis=1))[0]
    T_a, T_b = T_a[nz_a], T_b[nz_b]
    # eri 子块 eri[a,p,b,q] for (a,p)∈nz_a, (b,q)∈nz_b
    aap = np.array([idx // norb for idx in nz_a]); pp = np.array([idx % norb for idx in nz_a])
    bbq = np.array([idx // norb for idx in nz_b]); qq = np.array([idx % norb for idx in nz_b])
    eri_apbq = eri[aap[:, None], pp[:, None], bbq[None, :], qq[None, :]]
    # W[m,i,j] = (v @ T_b[m]ᵀ); U[m,i,j] = Σ_n eri_apbq[m,n]·W[n,i,j]; σ += Σ_m T_a[m]@U[m]
    W = np.einsum('ik,mjk->mij', v, T_b)
    U = np.einsum('mn,nij->mij', eri_apbq, W)
    out = np.einsum('mik,mkj->ij', T_a, U)
    sigma += out


# --------------------------------------------------------------------------- #
#  算子预计算 + 后端无关 matvec (numpy / cupy)
# --------------------------------------------------------------------------- #
def prepare_sigma_operators(ci_strs_a, ci_strs_b, norb, nelec, h1e, eri):
    """预计算 σ-vector 的全部算子数据 (CPU, 一次), 供 numpy/cupy 复用。

    Returns
    -------
    dict with keys:
        diag (na,nb), Ta_s/Tb_s (M,na,nb 的 T 表), c_as (na,M), Fb_ab_s (nb,M),
        c_bs (nb,M), Fa_ab_s (na,M), Ta_d/Tb_d + me_ad/me_bd, Ta_c/Tb_c/eri_apbq
    """
    h1e = np.asarray(h1e, dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)
    ci_a = np.asarray(ci_strs_a); ci_b = np.asarray(ci_strs_b)
    na, nb = len(ci_a), len(ci_b)
    occ_a = np.array([[(int(s) >> i) & 1 for i in range(norb)] for s in ci_a], dtype=float)
    occ_b = np.array([[(int(s) >> i) & 1 for i in range(norb)] for s in ci_b], dtype=float)
    (sa, da), (sb, db) = prepare_sigma_tables(ci_a, ci_b, norb, nelec)

    Fa_ss = _fock_same_spin(occ_a, eri)
    Fb_ab = _fock_cross(occ_b, eri)
    Fb_ss = _fock_same_spin(occ_b, eri)
    Fa_ab = _fock_cross(occ_a, eri)

    # α 单
    Ta_s, nz_a = _T_single(sa, norb, na)
    a_a, p_a = nz_a // norb, nz_a % norb
    c_as = h1e[p_a, a_a][None, :] + Fa_ss[:, a_a, p_a]          # (na, M)
    Fb_ab_s = Fb_ab[:, a_a, p_a]                                # (nb, M)
    # β 单
    Tb_s, nz_b = _T_single(sb, norb, nb)
    b_b, q_b = nz_b // norb, nz_b % norb
    c_bs = h1e[q_b, b_b][None, :] + Fb_ss[:, b_b, q_b]          # (nb, M)
    Fa_ab_s = Fa_ab[:, b_b, q_b]                                # (na, M)
    # α/β 双
    Ta_d, ta = _T_double(da, norb, na)
    me_ad = np.array([eri[a, p, b, q] - eri[a, q, b, p] for (p, q, a, b) in ta])
    Tb_d, tb = _T_double(db, norb, nb)
    me_bd = np.array([eri[a, p, b, q] - eri[a, q, b, p] for (p, q, a, b) in tb])
    # 交叉
    norb2 = norb * norb
    Ta_c = np.zeros((norb2, na, na)); Tb_c = np.zeros((norb2, nb, nb))
    for i, links in enumerate(sa):
        for (i_, sign, p, a) in links:
            Ta_c[a * norb + p, i, i_] += sign
    for j, links in enumerate(sb):
        for (j_, sign, q, b) in links:
            Tb_c[b * norb + q, j, j_] += sign
    nz_ac = np.nonzero(Ta_c.reshape(norb2, -1).any(axis=1))[0]
    nz_bc = np.nonzero(Tb_c.reshape(norb2, -1).any(axis=1))[0]
    Ta_c, Tb_c = Ta_c[nz_ac], Tb_c[nz_bc]
    a_ac = np.array([idx // norb for idx in nz_ac]); p_ac = np.array([idx % norb for idx in nz_ac])
    b_bc = np.array([idx // norb for idx in nz_bc]); q_bc = np.array([idx % norb for idx in nz_bc])
    eri_apbq = eri[a_ac[:, None], p_ac[:, None], b_bc[None, :], q_bc[None, :]]

    diag = _diag_energy(occ_a, occ_b, np.diag(h1e), eri)
    return dict(na=na, nb=nb, diag=diag,
                Ta_s=Ta_s, c_as=c_as, Fb_ab_s=Fb_ab_s,
                Tb_s=Tb_s, c_bs=c_bs, Fa_ab_s=Fa_ab_s,
                Ta_d=Ta_d, me_ad=me_ad, Tb_d=Tb_d, me_bd=me_bd,
                Ta_c=Ta_c, Tb_c=Tb_c, eri_apbq=eri_apbq)


def sigma_vector_ops(v, ops, xp=np):
    """后端无关 ``H·v``: 用预计算算子 ``ops`` 与数组模块 ``xp`` (numpy 或 cupy)。

    ``v`` 与 ``ops`` 须在相同后端 (都 numpy 或都 cupy)。
    """
    sigma = ops["diag"] * v
    # α 单
    w = xp.matmul(ops["Ta_s"], v)
    sigma += xp.einsum("im,mij->ij", ops["c_as"], w)
    sigma += xp.einsum("jm,mij->ij", ops["Fb_ab_s"], w)
    # β 单
    w = xp.matmul(v, ops["Tb_s"].transpose(0, 2, 1))
    sigma += xp.einsum("jm,mij->ij", ops["c_bs"], w)
    sigma += xp.einsum("im,mij->ij", ops["Fa_ab_s"], w)
    # α 双
    if len(ops["me_ad"]):
        w = xp.matmul(ops["Ta_d"], v)
        sigma += xp.einsum("m,mij->ij", ops["me_ad"], w)
    # β 双
    if len(ops["me_bd"]):
        w = xp.matmul(v, ops["Tb_d"].transpose(0, 2, 1))
        sigma += xp.einsum("m,mij->ij", ops["me_bd"], w)
    # 交叉
    W = xp.einsum("ik,mjk->mij", v, ops["Tb_c"])
    U = xp.einsum("mn,nij->mij", ops["eri_apbq"], W)
    sigma += xp.einsum("mik,mkj->ij", ops["Ta_c"], U)
    return sigma


# --------------------------------------------------------------------------- #
#  GPU 求解器 (cupyx LinearOperator + eigsh)
# --------------------------------------------------------------------------- #
def eigsh_gpu(ops, dim, *, k=1, which="SA", tol=1e-8, maxiter=None, v0=None):
    """Matrix-free GPU 本征求解: cupyx LinearOperator + eigsh。

    ``ops`` 为 CPU 预计算算子 (``prepare_sigma_operators``), 内部移到 GPU。
    返回 ``(eigvals, eigvecs)`` (numpy, 已传回 CPU)。
    """
    import cupy as cp
    from cupyx.scipy.sparse.linalg import eigsh, LinearOperator
    na, nb = ops["na"], ops["nb"]
    ops_g = {k: cp.asarray(a) for k, a in ops.items() if hasattr(a, "shape")}

    def matvec(x):
        xv = cp.asarray(x).reshape(na, nb)
        return sigma_vector_ops(xv, ops_g, cp).reshape(-1)

    A = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    v0g = cp.asarray(v0) if v0 is not None else None
    e, c = eigsh(A, k=k, which=which, tol=tol, maxiter=maxiter, v0=v0g)
    e = e.get() if hasattr(e, "get") else np.asarray(e)
    c = c.get() if hasattr(c, "get") else np.asarray(c)
    return np.asarray(e), np.asarray(c)
