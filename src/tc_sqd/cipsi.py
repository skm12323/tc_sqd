"""CIPSI refinement --- Configuration Interaction by Perturbative Selection.

从用户提供的**种子 det 集合**（如 UCJ 辅助采样 ``ucj_assisted_configurations``
或 S+D 激发）出发，迭代做 PT2 筛选的生成集扩展，将子空间自动补全到近 FCI 精度。

定位（方向 B）：
    UCJ-SQD 用少量采样 shots 达化学精度；若需要更高精度（FCI 级），CIPSI 从
    UCJ 种子出发只需 1-2 轮即补全到全空间（种子字符串已覆盖全空间大部分）。
    代价是对角化维度 = 字符串乘积（可达全空间），与 HCI 近全空间相当——CIPSI
    是**高精度 refine 层**，不是"少量 det"路线。

算法（每轮）：
    1. 当前子空间对角化（复用 solve_sci 的稳健路径: dim≤1000 numpy eigh, 否则 eigsh）
    2. 取 |c| > ``dom_thresh`` 的主导 dets，枚举单/双激发连接 -> 候选 dets
    3. 扩展空间上 ``contract_2e`` 一次得 <a|H|Psi>（pyscf 矩阵元，免手写符号问题）
    4. PT2_a = <a|H|Psi>^2 / (E_gs - E_a)，按 |PT2| 加入 top 候选
    5. 重复至空间达全空间 / PT2 < ``pt2_floor`` / 无新候选 / ``max_iter``

空间表示与 det 计数口径：
    本实现沿用 SQD 库的字符串乘积表示（对角化维度 = n_str_a × n_str_b）。
    闭壳层 (n_a==n_b) 时 α/β 合并为同一字符串集合（与
    ``bitstring_matrix_to_ci_strs`` 默认一致）；开壳层用独立 α/β 集合。
    注意"采样 det 数"（bsm 行）远小于对角化维度——两者口径不同。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from pyscf.fci import cistring, selected_ci, direct_spin1
from pyscf import ao2mo
from scipy.sparse.linalg import eigsh, LinearOperator

from .configuration_recovery import recover_configurations
from .fermion import bitstring_matrix_to_ci_strs, SCIState
from .diagnostics import extrapolate_energy_variance, extrapolate_ev_pt2

__all__ = ["solve_cipsi", "solve_sqd_active", "solve_sqd_adaptive", "solve_hci",
           "solve_sqd_ev", "solve_sqd_distill", "eigenvector_importance_sample"]


# --------------------------------------------------------------------------- #
#  连接枚举 (Slater 行列式对之间的单/双激发目标)
# --------------------------------------------------------------------------- #
def _occ_bits(s: int, norb: int) -> list:
    return [i for i in range(norb) if (s >> i) & 1]


def _excited_dets(a: int, b: int, norb: int):
    """det (α=a, β=b) 的所有单/双激发目标 det 集合 {(α', β')}。"""
    oa, ob = _occ_bits(a, norb), _occ_bits(b, norb)
    va = [v for v in range(norb) if not (a >> v) & 1]
    vb = [v for v in range(norb) if not (b >> v) & 1]
    out = set()
    # 单激发 (α / β)
    for i in oa:
        for v in va:
            out.add((a ^ (1 << i) ^ (1 << v), b))
    for i in ob:
        for v in vb:
            out.add((a, b ^ (1 << i) ^ (1 << v)))
    # 双激发 αα / ββ
    for p in range(len(oa)):
        for q in range(p + 1, len(oa)):
            i, j = oa[p], oa[q]
            for r in range(len(va)):
                for s in range(r + 1, len(va)):
                    u, v = va[r], va[s]
                    out.add((a ^ (1 << i) ^ (1 << j) ^ (1 << u) ^ (1 << v), b))
    for p in range(len(ob)):
        for q in range(p + 1, len(ob)):
            i, j = ob[p], ob[q]
            for r in range(len(vb)):
                for s in range(r + 1, len(vb)):
                    u, v = vb[r], vb[s]
                    out.add((a, b ^ (1 << i) ^ (1 << j) ^ (1 << u) ^ (1 << v)))
    # 双激发 αβ
    for i in oa:
        for u in va:
            for j in ob:
                for v in vb:
                    out.add((a ^ (1 << i) ^ (1 << u), b ^ (1 << j) ^ (1 << v)))
    return out


# --------------------------------------------------------------------------- #
#  子空间对角化 (与 solve_sci 相同的稳健路径)
# --------------------------------------------------------------------------- #
class _Subspace:
    """字符串集合 (α, β) 的子空间对角化, 提供 <a|H|Psi> 的 PT2 矩阵元。"""

    def __init__(self, h1e, eri, norb, nelec):
        self.h1e = np.asarray(h1e)
        self.eri = np.asarray(eri)
        self.norb = norb
        self.nelec = nelec
        self.h2e = direct_spin1.absorb_h1e(self.h1e, self.eri, norb, nelec, 0.5)
        self.h2e = ao2mo.restore(1, self.h2e, norb)
        self.myci = selected_ci.SCI()

    def diag(self, str_a, str_b):
        """对角化 (str_a, str_b) 子空间, 返回 (E_gs, c2d, sa, sb)。"""
        sa = np.asarray(sorted(str_a), dtype=np.int64)
        sb = np.asarray(sorted(str_b), dtype=np.int64)
        nA, nB = len(sa), len(sb)
        dim = nA * nB
        link = selected_ci._all_linkstr_index((sa, sb), self.norb, self.nelec)

        def hop(v):
            v = np.ascontiguousarray(v, dtype=np.float64)
            hv = self.myci.contract_2e(
                self.h2e, selected_ci._as_SCIvector(v, (sa, sb)),
                self.norb, self.nelec, link).reshape(-1)
            return np.ascontiguousarray(hv, dtype=np.float64)

        if dim <= 1000:
            H = np.zeros((dim, dim))
            for col in range(dim):
                e = np.zeros(dim)
                e[col] = 1.0
                H[:, col] = hop(e)
            ev, cv = np.linalg.eigh(H)
            E, c1d = float(ev[0]), cv[:, 0]
        else:
            op = LinearOperator((dim, dim), matvec=hop, dtype=np.float64)
            ev, cv = eigsh(op, k=1, which="SA", maxiter=3000)
            E, c1d = float(ev[0]), np.asarray(cv).ravel()
        return E, c1d.reshape(nA, nB), sa, sb

    def pt2_matrix_elements(self, str_a, str_b, cand, c2d, sa, sb):
        """扩展空间 (str_a∪cand_α, str_b∪cand_β) 上算各候选的 <a|H|Psi> 与对角元。

        返回 dict {(ca, cb): (hpsi, Ea)}。
        """
        idx_b = {int(s): i for i, s in enumerate(sb)}
        set_a = set(str_a)
        set_b = set(str_b)
        for ca, cb in cand:
            set_a.add(ca)
            set_b.add(cb)
        sA = np.asarray(sorted(set_a), dtype=np.int64)
        sB = np.asarray(sorted(set_b), dtype=np.int64)
        nB = len(sB)
        idx_a = {int(s): i for i, s in enumerate(sA)}
        idx_b2 = {int(s): i for i, s in enumerate(sB)}
        dim2 = len(sA) * nB

        psi = np.zeros(dim2)
        for ia in range(len(sa)):
            for ib in range(len(sb)):
                psi[idx_a[int(sa[ia])] * nB + idx_b2[int(sb[ib])]] = c2d[ia, ib]
        link2 = selected_ci._all_linkstr_index((sA, sB), self.norb, self.nelec)
        hdiag = selected_ci.make_hdiag(self.h1e, self.eri, (sA, sB),
                                       self.norb, self.nelec)
        Hpsi = self.myci.contract_2e(
            self.h2e, selected_ci._as_SCIvector(psi, (sA, sB)),
            self.norb, self.nelec, link2).reshape(-1)

        out = {}
        for ca, cb in cand:
            k = idx_a[ca] * nB + idx_b2[cb]
            out[(ca, cb)] = (Hpsi[k], hdiag[k])
        return out


def solve_cipsi(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    seed_bitstring_matrix: np.ndarray,
    max_strings: Optional[int] = None,
    dom_thresh: float = 1e-3,
    pt2_floor: float = 1e-7,
    max_iter: int = 40,
    ecore: float = 0.0,
    verbose: bool = False,
) -> float:
    """CIPSI 迭代精化: 从种子 det 集合出发补全到近 FCI 精度。

    Parameters
    ----------
    one_body_tensor : ndarray, shape (norb, norb)
        单电子积分 (MO 基, 闭壳层单矩阵)。
    two_body_tensor : ndarray, shape (norb, norb, norb, norb)
        两电子积分 (chemist 记号)。
    norb : int
        空间轨道数。
    nelec : tuple(int, int)
        ``(n_alpha, n_beta)``。
    seed_bitstring_matrix : ndarray, shape (S, 2*norb)
        种子 det 集合 (位串矩阵, 如 ``ucj_assisted_configurations`` 输出或
        ``np.vstack([exc, ucj])``)。
    max_strings : int | None
        字符串集合上限 (对角化维度 ≈ n_str_a × n_str_b)。``None`` = 默认
        补全到全空间 ``C(norb, nelec[0])``。
    dom_thresh : float
        主导 det 的 |c| 阈值 (低于此的 det 不参与生成集扩展)。
    pt2_floor : float
        |PT2| 低于此的候选 det 不再加入。
    max_iter : int
        迭代轮数上限。
    ecore : float
        Core 能量偏移 (核排斥 + frozen-core), 计入返回值。
    verbose : bool
        打印每轮空间大小 / 能量 / PT2 信息。

    Returns
    -------
    energy : float
        基态能量 (含 ``ecore``)。

    Notes
    -----
    - 矩阵元全部走 PySCF ``contract_2e``, 避免手写 Slater-Condon 的相位/符号坑。
    - 子空间表示沿用库的字符串乘积: 闭壳层 α/β 合并, 开壳层独立。
    - 空间扩展按 PT2 排序; 由于种子 (UCJ 辅助) 已覆盖全空间大部分字符串,
      通常 1-2 轮即补全到全空间 = FCI。
    """
    from .fermion import bitstring_matrix_to_ci_strs

    if one_body_tensor.ndim == 3:
        if not np.allclose(one_body_tensor[0], one_body_tensor[1]):
            raise ValueError(
                "solve_cipsi 不支持自旋分辨 h1e (h_alpha != h_beta); "
                "请传单个 (norb, norb) 闭壳层 h1e。"
            )
        h1e = np.asarray(one_body_tensor[0])
    else:
        h1e = np.asarray(one_body_tensor)
    eri = np.asarray(two_body_tensor)

    # 种子 -> 字符串集合 (闭壳层合并, 开壳层独立)
    na, nb = nelec
    if na == nb:
        ci_a, ci_b = bitstring_matrix_to_ci_strs(seed_bitstring_matrix)
        str_a = sorted(set(int(x) for x in ci_a))
        str_b = str_a
    else:
        ci_a, ci_b = bitstring_matrix_to_ci_strs(seed_bitstring_matrix, open_shell=True)
        str_a = sorted(set(int(x) for x in ci_a))
        str_b = sorted(set(int(x) for x in ci_b))

    full_size = int(cistring.num_strings(norb, na))
    if max_strings is None:
        max_strings = full_size

    sub = _Subspace(h1e, eri, norb, nelec)
    for it in range(max_iter):
        if len(str_a) >= max_strings or len(str_b) >= max_strings:
            break
        E, c2d, sa, sb = sub.diag(str_a, str_b)
        idx_a = {int(s): i for i, s in enumerate(sa)}
        idx_b = {int(s): i for i, s in enumerate(sb)}

        # 主导 dets
        nA, nB = c2d.shape
        flat = np.abs(c2d).ravel()
        order = np.argsort(flat)[::-1]
        dom = []
        for k in order:
            if flat[k] > dom_thresh:
                ia, ib = divmod(int(k), nB)
                dom.append((int(sa[ia]), int(sb[ib])))
            else:
                break
        if not dom:
            break

        # 候选连接 (新 det: 笛卡尔积空间里 ca 或 cb 不在当前集合)
        cand = set()
        for a, b in dom:
            for ca, cb in _excited_dets(a, b, norb):
                if ca not in idx_a or cb not in idx_b:
                    cand.add((ca, cb))
        if not cand:
            break

        # PT2 筛选
        me = sub.pt2_matrix_elements(str_a, str_b, cand, c2d, sa, sb)
        pt2 = {det_: hpsi * hpsi / (E - Ea) for det_, (hpsi, Ea) in me.items()
               if abs(E - Ea) > 1e-12}
        ranked = sorted(pt2.items(), key=lambda kv: -abs(kv[1]))

        add = []
        pt2_sum = 0.0
        for det_, v in ranked:
            if abs(v) < pt2_floor:
                break
            if len(str_a) + len(add) >= max_strings:
                break
            add.append(det_)
            pt2_sum += v
        if not add:
            break
        for ca, cb in add:
            str_a.append(ca)
            if cb not in str_b:
                str_b.append(cb)
        str_a = sorted(set(str_a))
        str_b = sorted(set(str_b))
        if verbose:
            dim_now = len(str_a) * len(str_b)
            print(f"[CIPSI] it{it}: strings={len(str_a)}x{len(str_b)} "
                  f"diag_dim={dim_now} E={E + ecore:.6f} pt2_top={pt2_sum:.2e}")

    E, c2d, sa, sb = sub.diag(str_a, str_b)
    return float(E) + ecore


# --------------------------------------------------------------------------- #
#  真正的 HCI (heat-bath CI): |<j|H|i>| >= eps_hb 选态 (Holmes 2016 JCTC)
# --------------------------------------------------------------------------- #
def solve_hci(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    seed_bitstring_matrix: Optional[np.ndarray] = None,
    eps_hb: float = 1e-3,
    dom_thresh: float = 1e-3,
    pt2_floor: float = 1e-7,
    max_iter: int = 40,
    ecore: float = 0.0,
    verbose: bool = False,
    return_details: bool = False,
):
    """SHCI (heat-bath CI + PT2 修正, Holmes 2016 / Sharma 2017).

    **与 solve_cipsi 的区别 (heat-bath 选态 vs PT2 全排序)**:
      - :func:`solve_cipsi` (CIPSI): 候选加入用完整 Epstein-Nesbet PT2 得分
        ``⟨a|H|Ψ⟩²/(E−E_a)`` 排序选 top —— 每轮对全波函数求 H|Ψ⟩。
      - 本函数 (SHCI): **两阶段** —— ① heat-bath 选态 ``|⟨j|H|i⟩| ≥ eps_hb``
        构建变分空间 V (只用单参考 det 对矩阵元, 不求完整 ⟨a|H|Ψ⟩); ② 对角化 V
        得 ``E_V``, 对 V 外候选算 **PT2 能量修正** ``E_PT2 = Σ_a |⟨a|H|Ψ⟩|²/(E−E_a)``。
        返回标准 SHCI 报告的总能量 ``E_total = E_V + E_PT2``。

    **参数** (SHCI 双阈值): ``eps_hb`` = ε₁ (变分空间选态); ``pt2_floor`` = ε₂
    (PT2 修正精度参考, 用于判断变分空间是否足够; 本实现不做 semistochastic,
    ε₂ 仅标注)。

    **实现** (朴素 heat-bath, 同 pyscf/naive-hci 思路): 对每个主导 det |i⟩,
    枚举单/双激发候选 |j⟩, 用单位向量 ``e_i`` 经 PySCF ``contract_2e`` 一次算
    ``⟨j|H|i⟩`` (复用 :class:`_Subspace.pt2_matrix_elements`, 传 ``c2d=e_i``)。

    Parameters
    ----------
    one_body_tensor : ndarray (norb, norb)
        单电子积分 (闭壳层单矩阵)。
    two_body_tensor : ndarray (norb, norb, norb, norb)
        双电子积分 (chemist 记号)。
    norb : int
        空间轨道数。
    nelec : tuple(int, int)
        ``(n_alpha, n_beta)``。
    seed_bitstring_matrix : ndarray (S, 2*norb) | None
        种子 det 集合 (位串)。``None`` = 从 HF 出发 (标准 HCI)。
    eps_hb : float
        heat-bath 选态阈值 (ε₁): ``|⟨j|H|i⟩| ≥ eps_hb`` 的候选 det 加入变分空间。
        越小变分空间越大, E_PT2 越小, 越接近 FCI。
    dom_thresh : float
        主导 det 的 |c| 阈值 (低于此不参与生成集扩展)。
    pt2_floor : float
        PT2 修正阈值 (ε₂ 参考): 仅 verbose 标注变分空间是否足够, 不强制收敛。
    max_iter : int
        迭代轮数上限。
    ecore : float
        Core 能量偏移, 计入返回值。
    verbose : bool
        打印每轮变分空间/能量/PT2。
    return_details : bool
        ``True`` 返回 ``(E_total, E_PT2, dim)`` 元组 (含 ecore 的 E_total,
        不含 ecore 的 E_PT2, 变分空间维度) —— 供诊断/绘图。

    Returns
    -------
    float | tuple
        ``return_details=False``: SHCI 总能量 ``E_V + E_PT2`` (含 ``ecore``)。
        ``return_details=True``: ``(E_total, E_PT2, dim)``。
    """
    from .fermion import bitstring_matrix_to_ci_strs

    if one_body_tensor.ndim == 3:
        if not np.allclose(one_body_tensor[0], one_body_tensor[1]):
            raise ValueError(
                "solve_hci 不支持自旋分辨 h1e; 请传闭壳层 (norb, norb)。"
            )
        h1e = np.asarray(one_body_tensor[0])
    else:
        h1e = np.asarray(one_body_tensor)
    eri = np.asarray(two_body_tensor)
    na, nb = nelec
    open_shell = na != nb

    # 种子 -> 字符串集合 (默认 HF)
    if seed_bitstring_matrix is None:
        hf_a = (1 << na) - 1
        hf_b = (1 << nb) - 1
        str_a = [hf_a]
        str_b = [hf_b] if open_shell else [hf_a]
    else:
        ci_a, ci_b = bitstring_matrix_to_ci_strs(
            seed_bitstring_matrix, open_shell=open_shell)
        str_a = sorted(set(int(x) for x in ci_a))
        str_b = sorted(set(int(x) for x in ci_b))
        if not open_shell:
            str_b = str_a

    sub = _Subspace(h1e, eri, norb, nelec)

    def _dominant(c2d, sa, sb):
        nA, nB = c2d.shape
        flat = np.abs(c2d).ravel()
        order = np.argsort(flat)[::-1]
        dom = []
        for k in order:
            if flat[k] > dom_thresh:
                ia, ib = divmod(int(k), nB)
                dom.append((int(sa[ia]), int(sb[ib])))
            else:
                break
        return dom

    # ---- 阶段 1: heat-bath 选态构建变分空间 V (|⟨j|H|i⟩| ≥ eps_hb, 到无新增) ----
    for it in range(max_iter):
        E, c2d, sa, sb = sub.diag(str_a, str_b)
        idx_a = {int(s): i for i, s in enumerate(sa)}
        idx_b = {int(s): i for i, s in enumerate(sb)}
        dom = _dominant(c2d, sa, sb)
        if not dom:
            break

        hb_new = set()
        sa_list, sb_list = list(sa), list(sb)
        for a, b in dom:
            cand = _excited_dets(a, b, norb)
            cand = {(ca, cb) for (ca, cb) in cand
                    if ca not in idx_a or cb not in idx_b}
            if not cand:
                continue
            # 单位向量 e_i: 只在主导 det (a,b) 处为 1 -> H e_i 的第 j 分量 = <j|H|i>
            e_i = np.zeros((len(sa), len(sb)))
            e_i[sa_list.index(a), sb_list.index(b)] = 1.0
            me = sub.pt2_matrix_elements(str_a, str_b, cand, e_i, sa, sb)
            for (ca, cb), (hji, _) in me.items():
                if abs(hji) >= eps_hb:
                    hb_new.add((ca, cb))
        if not hb_new:
            break
        for ca, cb in hb_new:
            str_a.append(ca)
            if cb not in str_b:
                str_b.append(cb)
        str_a = sorted(set(str_a))
        str_b = str_a if not open_shell else sorted(set(str_b))
        if verbose:
            print(f"[HCI:hb] it{it+1}/{max_iter}: dim={len(str_a)*len(str_b)} "
                  f"E_V={E + ecore:.6f} new={len(hb_new)}")

    # ---- 阶段 2: 对角化 + PT2 能量修正 (E_PT2 = Σ |⟨a|H|Ψ⟩|²/(E−E_a)) ----
    E, c2d, sa, sb = sub.diag(str_a, str_b)
    idx_a = {int(s): i for i, s in enumerate(sa)}
    idx_b = {int(s): i for i, s in enumerate(sb)}
    dom = _dominant(c2d, sa, sb)

    cand_all = set()
    for a, b in dom:
        for ca, cb in _excited_dets(a, b, norb):
            if ca not in idx_a or cb not in idx_b:
                cand_all.add((ca, cb))
    if cand_all:
        me = sub.pt2_matrix_elements(str_a, str_b, cand_all, c2d, sa, sb)
        pt2 = {d: h * h / (E - Ea) for d, (h, Ea) in me.items()
               if abs(E - Ea) > 1e-12}
        e_pt2 = float(sum(pt2.values()))
    else:
        e_pt2 = 0.0
    dim = len(str_a) * len(str_b)
    e_total = float(E + e_pt2) + ecore

    if verbose:
        print(f"[HCI] dim={dim} E_V={E + ecore:.8f} E_PT2={e_pt2:.2e} "
              f"E_total={e_total:.8f} "
              f"{'PT2 OK' if abs(e_pt2) < pt2_floor else 'PT2 large (reduce eps_hb)'}")
    if return_details:
        return e_total, float(e_pt2), dim
    return e_total


# --------------------------------------------------------------------------- #
#  自适应 SQD (方向①②组合): 自洽换基表示层 + 受限 PT2 选态选择层
# --------------------------------------------------------------------------- #
def solve_sqd_adaptive(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    bitstring_matrix: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    avg_occupancies: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    max_strings: Optional[int] = None,
    n_active_per_round: int = 50,
    max_pt2_iters: int = 3,
    dom_thresh: float = 1e-3,
    pt2_floor: float = 1e-7,
    max_rounds: int = 10,
    energy_tol: float = 1e-9,
    ecore: float = 0.0,
    rand_seed: Optional[int] = 0,
    verbose: bool = False,
) -> float:
    """自适应 SQD: 自洽换基表示层 (方向①) + 受限 PT2 选态选择层 (方向②) 叠加。

    **统一视角** (REVIEW 方向③): 表示层 (自然轨道换基使展开系数集中) + 生成层
    (多样初猜) + 选择层 (PT2 确定性补足)。本函数组合**表示层与选择层**:

    每轮:
      1. 配置恢复 (当前基偏置平均占据) → 当前基 det 集合
      2. 受限 PT2 精化 (当轮/当前基): 主导 det 枚举单双激发 → PT2 top-K 注入
         (子空间受限, 不补全全空间)
      3. 子空间对角化 → E
      4. 解态 1-RDM → 自然轨道换基 → 更新 h1e/eri/平均占据 (下一轮采样更聚焦)
      5. 能量稳定则收敛

    **与单独方法的关系**:
      - ``solve_sqd_active`` 只有选择层 (基固定); 本函数换基使下一轮采样 det 更有效
      - ``solve_sqd_natural_orbitals`` 只有表示层 (无 PT2); 本函数当轮 PT2 补足
        采样缺口 → 更准的 1-RDM → 更准的换基 (正反馈)

    Parameters
    ----------
    同 :func:`solve_sqd_active`, 外加:
    max_pt2_iters : int
        每轮内受限 PT2 精化的迭代次数 (采样后确定性补足的程度)。
    energy_tol : float
        能量收敛阈值 (连续两轮变化小于它即停止换基)。

    Returns
    -------
    energy : float
        基态总能量 (含 ``ecore``)。
    """
    from .basis import rotate_to_natural_orbitals

    if one_body_tensor.ndim == 3:
        if not np.allclose(one_body_tensor[0], one_body_tensor[1]):
            raise ValueError(
                "solve_sqd_adaptive 不支持自旋分辨 h1e; 请传闭壳层 (norb, norb)。"
            )
        h1e = np.asarray(one_body_tensor[0])
    else:
        h1e = np.asarray(one_body_tensor)
    eri = np.asarray(two_body_tensor)
    na, nb = nelec
    open_shell = na != nb

    bsm = np.asarray(bitstring_matrix, dtype=bool)
    if bsm.ndim != 2 or bsm.shape[1] != 2 * norb:
        raise ValueError(
            f"bitstring_matrix must have shape (S, 2*norb={2*norb}), got {bsm.shape}."
        )
    probs = (np.full(bsm.shape[0], 1.0 / bsm.shape[0]) if probabilities is None
             else np.asarray(probabilities, dtype=np.float64))

    if avg_occupancies is not None:
        occ_a, occ_b = avg_occupancies
    else:
        occ_a = np.zeros(norb, dtype=np.float64)
        occ_a[:na] = 1.0
        occ_b = np.zeros(norb, dtype=np.float64)
        occ_b[:nb] = 1.0

    full_size = int(cistring.num_strings(norb, na))
    if max_strings is None:
        max_strings = full_size

    sub = _Subspace(h1e, eri, norb, nelec)
    e_prev = np.inf
    n_rounds_done = 0

    for r in range(max_rounds):
        # ① 配置恢复 (当前基偏置平均占据) → 当前基 det
        rec, _ = recover_configurations(
            bsm, probs, (occ_a, occ_b), na, nb, rand_seed=rand_seed
        )
        ci_a, ci_b = bitstring_matrix_to_ci_strs(rec, open_shell=open_shell)
        str_a = sorted(set(int(x) for x in ci_a))
        str_b = str_a if not open_shell else sorted(set(int(x) for x in ci_b))

        # ② 受限 PT2 精化 (当轮, 当前基)
        for _ in range(max_pt2_iters):
            E, c2d, sa, sb = sub.diag(str_a, str_b)
            idx_a = {int(s): i for i, s in enumerate(sa)}
            idx_b = {int(s): i for i, s in enumerate(sb)}
            nA, nB = c2d.shape
            flat = np.abs(c2d).ravel()
            order = np.argsort(flat)[::-1]
            dom = []
            for k in order:
                if flat[k] > dom_thresh:
                    ia, ib = divmod(int(k), nB)
                    dom.append((int(sa[ia]), int(sb[ib])))
                else:
                    break
            cand = set()
            if dom:
                for a, b in dom:
                    for ca, cb in _excited_dets(a, b, norb):
                        if ca not in idx_a or cb not in idx_b:
                            cand.add((ca, cb))
            if not cand:
                break
            me = sub.pt2_matrix_elements(str_a, str_b, cand, c2d, sa, sb)
            pt2 = {d: h * h / (E - Ea) for d, (h, Ea) in me.items()
                   if abs(E - Ea) > 1e-12}
            ranked = sorted(pt2.items(), key=lambda kv: -abs(kv[1]))
            add = []
            for d, v in ranked:
                if abs(v) < pt2_floor:
                    break
                if len(str_a) + len(add) >= max_strings:
                    break
                add.append(d)
            if len(add) > n_active_per_round:
                add = add[:n_active_per_round]
            if not add:
                break
            for ca, cb in add:
                str_a.append(ca)
                if cb not in str_b:
                    str_b.append(cb)
            str_a = sorted(set(str_a))
            str_b = str_a if not open_shell else sorted(set(str_b))

        # ③ 最终对角化 (当前基)
        E, c2d, sa, sb = sub.diag(str_a, str_b)
        dim_now = len(sa) * len(sb)

        # ④ 表示层: 解态 1-RDM → 自然轨道换基
        st = SCIState(amplitudes=c2d, ci_strs_a=np.asarray(sa),
                      ci_strs_b=np.asarray(sb), norb=norb, nelec=nelec)
        dm1 = st.rdm(rank=1, spin_summed=True)
        h1e, eri, U_step, occ_nat = rotate_to_natural_orbitals(h1e, eri, dm1)
        sub = _Subspace(h1e, eri, norb, nelec)  # 重建 (新基)
        occ_a = np.clip(occ_nat / 2.0, 0.0, 1.0)
        occ_b = occ_a.copy()

        if verbose:
            print(f"[adaptive r{r+1}/{max_rounds}] E={E + ecore:.6f} "
                  f"dim={dim_now} |c2|max={float(np.abs(c2d).max() ** 2):.4f}")

        n_rounds_done = r + 1
        if r > 0 and abs(E - e_prev) < energy_tol:
            break
        e_prev = E

    E, c2d, sa, sb = sub.diag(str_a, str_b)
    if verbose:
        print(f"[adaptive] 收敛 @ round {n_rounds_done}: E={E + ecore:.8f} "
              f"dim={len(sa) * len(sb)}")
    return float(E) + ecore


# --------------------------------------------------------------------------- #
#  主动采样 SQD (方向②): 受限 PT2 选态 + 采样聚焦 双闭环
# --------------------------------------------------------------------------- #
def solve_sqd_active(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    bitstring_matrix: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    avg_occupancies: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    max_strings: Optional[int] = None,
    n_active_per_round: int = 50,
    dom_thresh: float = 1e-3,
    pt2_floor: float = 1e-7,
    max_rounds: int = 10,
    ecore: float = 0.0,
    rand_seed: Optional[int] = 0,
    verbose: bool = False,
    # ---- B1 预算闭环 (增量采样 + 能量收敛停采) ----
    shots_budget: Optional[int] = None,
    shots_step: int = 0,
    energy_tol: Optional[float] = None,
    usage: Optional[list] = None,
    # ---- 能量-方差外推轨迹 (方向 D) ----
    trajectory: Optional[list] = None,
    # ---- 自蒸馏 (方向②): 取出最终本征矢供重采样 ----
    state_out: Optional[list] = None,
) -> float:
    """主动采样 SQD: 采样/配置恢复 ↔ 受限 PT2 选态 双闭环 (AS-SQD 思想, 方向②)。

    **动机**: 纯采样 SQD 的子空间只含"采到"的 det, 低采样/噪声下覆盖不全
    (C₂ 曾 3/8 失败)。AS-SQD (Miura, arXiv:2603.13536) 用 Epstein-Nesbet
    PT2 得分从外部候选**确定性补足**采样缺口 —— 无需额外量子测量, 且噪声
    bitstring 的 PT2 得分近零 (抗噪)。

    **与 solve_cipsi 的区别**: :func:`solve_cipsi` 是**纯经典 det 空间精化**
    (静态种子, 补全到全空间, 不碰采样); 本函数是**采样与选态双闭环** ——
    每轮先用**偏置的平均占据**做配置恢复 (采样聚焦), 再用受限 PT2 注入
    高价值 det (子空间不补全全空间), 两者交替直到收敛。

    Parameters
    ----------
    one_body_tensor : ndarray, shape (norb, norb)
        单电子积分 (闭壳层单矩阵)。
    two_body_tensor : ndarray, shape (norb, norb, norb, norb)
        双电子积分 (chemist 记号)。
    norb : int
        空间轨道数。
    nelec : tuple(int, int)
        ``(n_alpha, n_beta)``。
    bitstring_matrix : ndarray, shape (S, 2*norb)
        采样位串 (电路 shot 或经典随机种子)。配置恢复每轮按当前平均占据修正。
    probabilities : ndarray, shape (S,), optional
        对应概率; 省略时均匀。
    avg_occupancies : tuple(ndarray, ndarray), optional
        初始平均占据 (采样偏置)。省略时退化为 HF。
    max_strings : int | None
        字符串集合上限 (对角化维度 ≈ n_str_a × n_str_b)。``None`` = 默认
        全空间 ``C(norb, nelec[0])`` (受限时给较小值)。
    n_active_per_round : int
        每轮 PT2 选态注入的 top 候选 det 数上限 (受限核心参数)。
    dom_thresh : float
        主导 det 的 |c| 阈值 (低于此不参与生成集扩展)。
    pt2_floor : float
        |PT2| 低于此的候选 det 不再加入。
    max_rounds : int
        采样↔选态轮数上限。
    ecore : float
        Core 能量偏移, 计入返回值。
    rand_seed : int | None
        配置恢复 tie-breaking 种子。
    verbose : bool
        打印每轮空间/能量/PT2 信息。
    shots_budget : int | None
        B1 预算: 总采样预算。``bitstring_matrix`` 行数不足时预生成随机位串补足成
        采样池 (经典模拟)。``None`` = 用给定 ``bitstring_matrix`` 全量 (原行为)。
    shots_step : int
        B1 增量步长: 每轮用池的前 ``n_cur`` 行, ``n_cur`` 逐轮递增 (``>0`` 启用
        增量采样; ``0`` = 一次性全量, 原行为)。
    energy_tol : float | None
        B1 停采阈值: 连续两轮能量变化小于它即停止 (能量已收敛, 省 shots)。
        ``None`` = 不停采 (原行为)。
    usage : list | None
        B1 输出参数: 调用方传空 list, 结束后 ``usage[0]`` 为**实际使用的 shots 数**
        (预算闭环的量化指标; 不传则只返回能量)。
    trajectory : list | None
        方向 D 输出参数: 调用方传空 list, 每轮追加 ``dict`` 记录 ``{round, E,
        sigma2, e_pt2, dim, shots}``。``sigma2 = Σ_a |⟨a|H|Ψ⟩|²`` (子空间外
        PT2 分子平方和, 即对生成集的**精确方差**); ``e_pt2`` 为 Epstein-Nesbet
        PT2 全和; ``dim`` 为对角化维度。供 :func:`solve_sqd_ev` 做能量-方差
        外推 (E(σ²)→0)。不传则无额外开销。

    Returns
    -------
    energy : float
        基态能量 (含 ``ecore``)。
    """
    if one_body_tensor.ndim == 3:
        if not np.allclose(one_body_tensor[0], one_body_tensor[1]):
            raise ValueError(
                "solve_sqd_active 不支持自旋分辨 h1e; 请传闭壳层 (norb, norb)。"
            )
        h1e = np.asarray(one_body_tensor[0])
    else:
        h1e = np.asarray(one_body_tensor)
    eri = np.asarray(two_body_tensor)
    na, nb = nelec
    open_shell = na != nb

    bsm = np.asarray(bitstring_matrix, dtype=bool)
    if bsm.ndim != 2 or bsm.shape[1] != 2 * norb:
        raise ValueError(
            f"bitstring_matrix must have shape (S, 2*norb={2*norb}), got {bsm.shape}."
        )
    probs = (np.full(bsm.shape[0], 1.0 / bsm.shape[0]) if probabilities is None
             else np.asarray(probabilities, dtype=np.float64))

    # 初始平均占据 (采样偏置)
    if avg_occupancies is not None:
        occ_a, occ_b = avg_occupancies
    else:
        occ_a = np.zeros(norb, dtype=np.float64)
        occ_a[:na] = 1.0
        occ_b = np.zeros(norb, dtype=np.float64)
        occ_b[:nb] = 1.0

    full_size = int(cistring.num_strings(norb, na))
    if max_strings is None:
        max_strings = full_size

    sub = _Subspace(h1e, eri, norb, nelec)
    str_a: list = []
    str_b: list = []
    e_prev = np.inf

    # B1 预算闭环: 采样池 (预算 > 当前行数时补足随机位串) + 增量游标
    n_pool = bsm.shape[0]
    if shots_budget is not None and shots_budget > n_pool:
        rng = np.random.default_rng(rand_seed)
        extra = rng.random((shots_budget - n_pool, 2 * norb)) > 0.5
        bsm = np.vstack([bsm, extra])
        probs = np.concatenate(
            [probs, np.full(shots_budget - n_pool, 1.0 / n_pool)]
        )
        n_pool = shots_budget
    n_cur = n_pool if shots_step <= 0 else min(shots_step, n_pool)

    for r in range(max_rounds):
        # ① 采样聚焦: 配置恢复 (偏置平均占据) 生成当前基 det, 并入子空间。
        #    B1 增量采样: 每轮用池的前 n_cur 行 (shots 逐轮递增)。
        bsm_r = bsm[:n_cur] if shots_step > 0 else bsm
        probs_r = probs[:n_cur] if shots_step > 0 else probs
        rec, _ = recover_configurations(
            bsm_r, probs_r, (occ_a, occ_b), na, nb, rand_seed=rand_seed
        )
        ci_a, ci_b = bitstring_matrix_to_ci_strs(rec, open_shell=open_shell)
        n_before = len(str_a) + len(str_b)
        str_a = sorted(set(str_a) | set(int(x) for x in ci_a))
        str_b = sorted(set(str_a) if not open_shell else (set(str_b) | set(int(x) for x in ci_b)))
        n_sampled_new = len(str_a) + len(str_b) - n_before
        # 采样覆盖不受 max_strings 限制 (真实采样的 det 都应进子空间);
        # max_strings 只约束 PT2 扩展 (下方 ④), 与 solve_cipsi 语义一致。

        # ② 子空间对角化
        E, c2d, sa, sb = sub.diag(str_a, str_b)
        idx_a = {int(s): i for i, s in enumerate(sa)}
        idx_b = {int(s): i for i, s in enumerate(sb)}
        # 方向 D: 每轮默认 (无候选时方差/PT2 = 0)
        sigma2 = 0.0
        e_pt2_sum = 0.0

        # ③ 主导 dets
        nA, nB = c2d.shape
        flat = np.abs(c2d).ravel()
        order = np.argsort(flat)[::-1]
        dom = []
        for k in order:
            if flat[k] > dom_thresh:
                ia, ib = divmod(int(k), nB)
                dom.append((int(sa[ia]), int(sb[ib])))
            else:
                break

        # ④ 候选连接 → PT2 受限选态 (不补全全空间)
        cand = set()
        if dom:
            for a, b in dom:
                for ca, cb in _excited_dets(a, b, norb):
                    if ca not in idx_a or cb not in idx_b:
                        cand.add((ca, cb))
        if cand:
            me = sub.pt2_matrix_elements(str_a, str_b, cand, c2d, sa, sb)
            # 方向 D: σ² = Σ|⟨a|H|Ψ⟩|² (PT2 分子平方和, 对生成集的精确方差)
            sigma2 = sum(h * h for h, _ in me.values())
            pt2 = {d: h * h / (E - Ea) for d, (h, Ea) in me.items()
                   if abs(E - Ea) > 1e-12}
            e_pt2_sum = float(sum(pt2.values()))
            ranked = sorted(pt2.items(), key=lambda kv: -abs(kv[1]))
            add = []
            for d, v in ranked:
                if abs(v) < pt2_floor:
                    break
                if len(str_a) + len(add) >= max_strings:
                    break
                add.append(d)
            if len(add) > n_active_per_round:
                add = add[:n_active_per_round]
            for ca, cb in add:
                str_a.append(ca)
                if cb not in str_b:
                    str_b.append(cb)
            str_a = sorted(set(str_a))
            if open_shell:
                str_b = sorted(set(str_b))
            else:
                str_b = str_a
            n_pt2_new = len(add)
        else:
            n_pt2_new = 0

        # ⑤ 更新平均占据 (采样偏置): 解态 1-RDM 对角
        st = SCIState(amplitudes=c2d, ci_strs_a=np.asarray(sa),
                      ci_strs_b=np.asarray(sb), norb=norb, nelec=nelec)
        dm1 = st.rdm(rank=1, spin_summed=True)
        occ_a = np.clip(np.diag(dm1) / 2.0, 0.0, 1.0)
        occ_b = occ_a.copy()

        # 方向 D: 记录轨迹点 (E, σ², E_PT2, dim, shots) 供能量-方差外推
        if trajectory is not None:
            trajectory.append({
                "round": r + 1, "E": float(E), "sigma2": sigma2,
                "e_pt2": e_pt2_sum, "dim": len(str_a) * len(str_b),
                "shots": int(n_cur),
            })

        if verbose:
            print(f"[active r{r+1}/{max_rounds}] E={E + ecore:.6f} "
                  f"strings={len(str_a)}x{len(str_b)} "
                  f"sampled_new={n_sampled_new} pt2_new={n_pt2_new}")

        # 收敛: 无 PT2 新 det 且采样无新增 (子空间不再扩展) → 稳定
        if n_pt2_new == 0 and n_sampled_new == 0:
            break
        # PT2 贡献可忽略且能量稳定
        if n_pt2_new == 0 and abs(E - e_prev) < 1e-10:
            break
        # B1 预算闭环: 能量收敛停采 (ΔE < energy_tol → 已收敛, 省 shots)
        if energy_tol is not None and r > 0 and abs(E - e_prev) < energy_tol:
            break
        e_prev = E
        # B1 增量采样: 扩大下一轮使用的 shots
        if shots_step > 0:
            n_cur = min(n_cur + shots_step, n_pool)

    E, c2d, sa, sb = sub.diag(str_a, str_b)

    # 方向 D: 最终对角化点也进轨迹 (最大子空间 -> 方差最小, 外推最右端点)
    if trajectory is not None:
        idx_a = {int(s): i for i, s in enumerate(sa)}
        idx_b = {int(s): i for i, s in enumerate(sb)}
        nA, nB = c2d.shape
        flat = np.abs(c2d).ravel()
        order = np.argsort(flat)[::-1]
        dom = []
        for k in order:
            if flat[k] > dom_thresh:
                ia, ib = divmod(int(k), nB)
                dom.append((int(sa[ia]), int(sb[ib])))
            else:
                break
        cand_all = set()
        for a, b in dom:
            for ca, cb in _excited_dets(a, b, norb):
                if ca not in idx_a or cb not in idx_b:
                    cand_all.add((ca, cb))
        if cand_all:
            me = sub.pt2_matrix_elements(str_a, str_b, cand_all, c2d, sa, sb)
            sigma2 = sum(h * h for h, _ in me.values())
            e_pt2_sum = sum(h * h / (E - Ea) for h, Ea in me.values()
                            if abs(E - Ea) > 1e-12)
        else:
            sigma2, e_pt2_sum = 0.0, 0.0
        trajectory.append({
            "round": -1, "E": float(E), "sigma2": sigma2,
            "e_pt2": float(e_pt2_sum), "dim": len(str_a) * len(str_b),
            "shots": int(n_cur),
        })

    if state_out is not None:
        state_out.append((np.asarray(c2d), np.asarray(sa), np.asarray(sb)))
    if usage is not None:
        usage.append(int(n_cur))
    return float(E) + ecore


# --------------------------------------------------------------------------- #
#  方向 D: 能量-方差外推 (不增大维度降误差) + 本征矢重要性采样 (学习型采样先验)
# --------------------------------------------------------------------------- #
def solve_sqd_ev(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    bitstring_matrix: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    avg_occupancies: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    max_strings: Optional[int] = None,
    n_active_per_round: int = 50,
    dom_thresh: float = 1e-3,
    pt2_floor: float = 1e-7,
    max_rounds: int = 10,
    correction: str = "pt2",
    degree: int = 1,
    ecore: float = 0.0,
    rand_seed: Optional[int] = 0,
    verbose: bool = False,
    shots_budget: Optional[int] = None,
    shots_step: int = 0,
    energy_tol: Optional[float] = None,
    return_details: bool = False,
) -> float:
    """改进 SQD (方向 D/③): active 采样 + 基于方差的能量修正, 不增大维度降误差。

    **三种修正 (都用 PT2 分子 Σ|⟨a|H|Ψ⟩|² = 精确方差, 纯经典后处理)**:
      - ``correction="pt2"`` (**默认, 推荐**): ``E + E_PT2``, 其中
        ``E_PT2 = Σ_a |⟨a|H|Ψ⟩|²/(E−E_a)`` (Epstein-Nesbet)。SHCI/CIPSI 标准修正,
        **行为良好**——N₂/STO-3G 受限子空间直接 err 4.3e-4 → 6.2e-5; C₂ 直接
        err 7.9e-3 (超化学精度) → **5.0e-4** (达化学精度)。
      - ``correction="evpt2"`` (**方向③, 备选**): ``E_V`` vs ``E_PT2`` 两点外推
        (:func:`extrapolate_ev_pt2`, SHCI 社区 Holmes 2016/Sharma 2017 标准)。
        用轨迹各轮 ``(E_V, E_PT2)`` 拟合线性外推到 ``E_PT2→0``。x 轴是带能量分母
        加权的 ``E_PT2`` (物理上更接近漏掉的关联能), 经验**不过冲**——优于 σ² 线性。
        需轨迹 ≥2 个 ``E_PT2`` 非零点; 子空间饱和 (``E_PT2≈0``) 时退化为直接能量。
      - ``correction="ev"`` (**诊断用**): 用轨迹 ``(E, σ²)`` 线性外推到 σ²=0
        (:func:`extrapolate_energy_variance`)。**注意: 实测会过冲到 FCI 之下**
        (N₂ −5.8e-4, C₂ −1.7e-2), 不推荐作为默认——保留作方差标度诊断。

    **动机**: :func:`solve_sqd_active` 的最终子空间能量是 FCI 的变分上界
    (残余误差 ∝ 漏掉 det 的方差)。PT2/σ² 修正都用已算的候选矩阵元估计漏掉
    的关联, **不增大最终子空间维度**即降误差。饱和子空间 (全空间, σ²≈0)
    时修正自然趋零。

    Parameters
    ----------
    其余参数与 :func:`solve_sqd_active` 一致 (``ecore`` 在返回/诊断中计入;
    轨迹内部不含 ecore)。
    correction : {"pt2", "evpt2", "ev"}
        修正方式 (见上)。``"pt2"`` = E+E_PT2 (推荐); ``"evpt2"`` = E_V vs E_PT2
        两点外推 (方向③, 不过冲); ``"ev"`` = σ² 线性外推 (诊断, 可能过冲)。
    degree : int
        ``correction="evpt2"`` / ``"ev"`` 时外推多项式次数 (默认 1 = 线性)。
    return_details : bool
        ``True`` 返回 ``(能量, details_dict)``; ``details_dict`` 含
        ``E_direct`` (active 直接能量)、``correction``、``E_PT2`` (pt2 模式)、
        ``e_inf``/``alpha``(evpt2) 或 ``slope``(ev)/``r2``/``fit_std``、``trajectory``
        (每轮 E/σ²/PT2/dim/shots)。

    Returns
    -------
    float | tuple
        修正后能量 (含 ``ecore``); ``return_details=True`` 时返回 ``(能量, dict)``。
    """
    if correction not in ("pt2", "ev", "evpt2"):
        raise ValueError(f"correction 须为 'pt2' / 'ev' / 'evpt2', got {correction!r}.")
    trajectory: list = []
    solve_sqd_active(
        one_body_tensor, two_body_tensor, norb, nelec,
        bitstring_matrix=bitstring_matrix, probabilities=probabilities,
        avg_occupancies=avg_occupancies, max_strings=max_strings,
        n_active_per_round=n_active_per_round, dom_thresh=dom_thresh,
        pt2_floor=pt2_floor, max_rounds=max_rounds,
        ecore=0.0,                       # 轨迹 E 不含 ecore, 修正后统一加
        rand_seed=rand_seed, verbose=verbose,
        shots_budget=shots_budget, shots_step=shots_step,
        energy_tol=energy_tol, trajectory=trajectory,
    )
    if len(trajectory) < 2:
        raise ValueError(f"轨迹点不足 (<2), 无法修正: got {len(trajectory)}.")
    last = trajectory[-1]
    E = float(last["E"])
    e_direct = E + ecore                 # active 直接能量 (最终子空间)
    dim = int(last["dim"])

    if correction == "pt2":
        # E + E_PT2 (Epstein-Nesbet, 行为良好)
        e_pt2 = float(last["e_pt2"])
        e_corr = E + e_pt2 + ecore
        details = {
            "E_direct": e_direct, "correction": "pt2", "E_PT2": e_pt2,
            "dim": dim, "trajectory": trajectory,
        }
        if verbose:
            print(f"[EV:pt2] dim={dim} E_direct={e_direct:.8f} "
                  f"E_PT2={e_pt2:.2e} E_corr={e_corr:.8f}")
    elif correction == "ev":
        # σ² 线性外推 (诊断; 实测会过冲到 FCI 之下)
        es = np.asarray([t["E"] for t in trajectory], dtype=np.float64)
        vs = np.asarray([t["sigma2"] for t in trajectory], dtype=np.float64)
        if np.max(vs) < 1e-14:
            # 子空间饱和: 无残余可外推, 退化为直接能量
            e_corr = e_direct
            e_inf, slope, r2, fit_std = e_direct, 0.0, 1.0, 0.0
        else:
            e_inf, slope, r2, fit_std = extrapolate_energy_variance(
                es, vs, degree=degree)
            e_corr = float(e_inf) + ecore
        details = {
            "E_direct": e_direct, "correction": "ev", "e_inf": float(e_inf),
            "slope": slope, "r2": r2, "fit_std": fit_std, "dim": dim,
            "trajectory": trajectory,
        }
        if verbose:
            print(f"[EV:ev] dim={dim} r²={r2:.4f} E_direct={e_direct:.8f} "
                  f"E_ev={e_corr:.8f} (非变分, 可能过冲)")
    else:
        # E_V vs E_PT2 两点外推 (SHCI 标准, 方向③; 经验不过冲, 优于 σ² 线性)。
        # 用轨迹各轮 (E_V 变分能量, E_PT2 Epstein-Nesbet) 外推到 E_PT2→0。
        # **稳健性护栏**: solve_sqd_active 的 within-run 轨迹常退化 (受限时 round 间
        # E_PT2 重复, 或子空间饱和后 E_PT2≈0) —— 互异点 <2 时拟合病态 (alpha 爆炸),
        # 此时退化为 pt2 单点修正 (evpt2 永不劣于 pt2)。需稳健两点外推请用两次不同
        # max_strings 跑 solve_sqd_active, 再喂 :func:`extrapolate_ev_pt2`。
        es = np.asarray([t["E"] for t in trajectory], dtype=np.float64)
        pts = np.asarray([t["e_pt2"] for t in trajectory], dtype=np.float64)
        n_distinct = len(np.unique(np.round(pts, decimals=14)))
        if n_distinct < 2 or np.max(np.abs(pts)) < 1e-14:
            # 轨迹退化: 退化为 pt2 单点修正 (E + E_PT2)
            e_pt2_val = float(last["e_pt2"])
            e_corr = E + e_pt2_val + ecore
            details = {
                "E_direct": e_direct, "correction": "evpt2", "fallback": "pt2",
                "E_PT2": e_pt2_val, "dim": dim, "trajectory": trajectory,
                "note": "轨迹 E_PT2 互异点 <2 (受限/饱和), 外推病态, 退化为 pt2",
            }
            if verbose:
                print(f"[EV:evpt2→pt2 fallback] dim={dim} E_direct={e_direct:.8f} "
                      f"E_PT2={e_pt2_val:.2e} E_corr={e_corr:.8f}")
        else:
            e_inf, alpha, r2, fit_std = extrapolate_ev_pt2(es, pts, degree=degree)
            e_corr = float(e_inf) + ecore
            details = {
                "E_direct": e_direct, "correction": "evpt2", "fallback": None,
                "e_inf": float(e_inf), "alpha": alpha, "r2": r2,
                "fit_std": fit_std, "dim": dim, "trajectory": trajectory,
            }
            if verbose:
                print(f"[EV:evpt2] dim={dim} r²={r2:.4f} E_direct={e_direct:.8f} "
                      f"E_evpt2={e_corr:.8f}")
    if return_details:
        return e_corr, details
    return e_corr


def solve_sqd_distill(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    bitstring_matrix: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    n_rounds: int = 3,
    n_samples: Optional[int] = None,
    temperature_schedule: Optional[Sequence[float]] = None,
    max_strings: Optional[int] = None,
    n_active_per_round: int = 50,
    ecore: float = 0.0,
    rand_seed: Optional[int] = 0,
    keep_pool: bool = True,
    verbose: bool = False,
) -> float:
    """自蒸馏 SQD 闭环 (方向②): solve → 按 |c|^(2/T) 重采 → recover → solve。

    **思路 (库 TODO "solve_sqd_distill 蒸馏闭环" 落地)**: 子空间对角化的本征矢
    ``|Ψ⟩ = Σ_i c_i |i⟩`` 是体系当前最好的"波函数模型"。每轮用它驱动一次**重要性
    重采样** (:func:`eigenvector_importance_sample`, 按 ``p_i ∝ |c_i|^(2/T)`` 采 det),
    再喂回 :func:`solve_sqd_active`。这是 **EM 式量子-经典反馈**: E 步用当前波函数
    采, M 步重对角化。同 shots 下子空间对**主导 det 流形**覆盖更密 → 变分下界更低;
    或同精度省 shots。抗噪: 噪声 det 的 ``|c|²`` 自然小, 重采时淘汰 (自清洗)。

    **温度退火** ``temperature_schedule`` (高→低): 高温 ``T>1`` (``|c|^(2/T)`` 更平)
    保持探索, 低温 ``T<1`` (更锐) 聚焦主导 det。默认 ``[1.5]*(n_rounds-2) + [0.5]``
    (长度 ``n_rounds-1``, 最后一轮不重采; 前几轮探索, 倒数第二轮锐化)。

    Parameters
    ----------
    one_body_tensor, two_body_tensor, norb, nelec, ecore
        分子积分 (闭壳层单 h1e) + 电子数 + core 偏移。
    bitstring_matrix : ndarray (S, 2*norb)
        初始采样位串 (电路 shot 或随机种子)。第 0 轮的采样池。
    probabilities : ndarray | None
        初始概率 (第 0 轮); 省略均匀。后续轮重采位串用均匀 (来自 |c|²)。
    n_rounds : int
        solve→重采 循环次数 (≥1)。``n_rounds=1`` 退化为单次 :func:`solve_sqd_active`。
    n_samples : int | None
        每轮重采的 det 数; ``None`` = 用 ``bitstring_matrix`` 行数。
    temperature_schedule : Sequence[float] | None
        长度 ``n_rounds-1`` 的温度列表 (最后一轮不重采); ``None`` = 默认退火。
    max_strings, n_active_per_round
        透传 :func:`solve_sqd_active`。
    rand_seed : int | None
        第 0 轮配置恢复种子; 后续轮自动 +1 (避免重复采样序列)。
    keep_pool : bool
        ``True`` (默认): 每轮采样池 = ``vstack(初始 bsm, 重采 bsm)`` (不丢失原始
        电路覆盖, 仅聚焦增强); ``False``: 池 = 重采 bsm (纯蒸馏聚焦, 替换)。
    verbose : bool
        打印每轮能量 / 稀疏度。

    Returns
    -------
    float
        所有轮中**最低**的 active 能量 (含 ``ecore``)。变分保证 best_E 单调不增于
        各轮, 但因每轮采样池变化, 取 min 最稳。

    Notes
    -----
    - 依赖 :func:`eigenvector_importance_sample` (已修 F1 α/β 半区布局), 开壳层
      (na≠nb) 安全。
    - 与 :func:`solve_sqd_active` 的关系: 后者是单次"采样↔PT2 选态"闭环; 本函数
      在其外再套一层"解态驱动的采样分布更新"。可叠加 (内部仍跑 active)。
    - NQS 衔接 (research): 把"按 |c|² 采"升级为神经网络参数化 ``p_θ(det)`` 泛化到
      未采 det, 是本闭环的深度学习版 (见 REVIEW Part 2 B1)。
    """
    if one_body_tensor.ndim == 3:
        if not np.allclose(one_body_tensor[0], one_body_tensor[1]):
            raise ValueError(
                "solve_sqd_distill 不支持自旋分辨 h1e; 请传闭壳层 (norb, norb)。"
            )
        h1e = np.asarray(one_body_tensor[0])
    else:
        h1e = np.asarray(one_body_tensor)
    eri = np.asarray(two_body_tensor)

    bsm0 = np.asarray(bitstring_matrix, dtype=bool)
    if bsm0.ndim != 2 or bsm0.shape[1] != 2 * norb:
        raise ValueError(
            f"bitstring_matrix must have shape (S, 2*norb={2*norb}), got {bsm0.shape}."
        )
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}.")
    if n_samples is None:
        n_samples = bsm0.shape[0]
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    if temperature_schedule is None:
        # 长度 n_rounds-1 (最后一轮不重采); 前 n_rounds-2 轮高温探索, 倒数第二轮 0.5 锐化
        temperature_schedule = ([1.5] * max(n_rounds - 2, 0) + [0.5]) if n_rounds >= 2 else []
    # 最后一轮不重采 → schedule 长度应为 n_rounds-1
    if len(temperature_schedule) != max(n_rounds - 1, 0):
        raise ValueError(
            f"temperature_schedule 长度须为 n_rounds-1={max(n_rounds-1,0)}, "
            f"got {len(temperature_schedule)}。"
        )

    pool = bsm0
    pool_probs = (probabilities if probabilities is not None
                  else np.full(bsm0.shape[0], 1.0 / bsm0.shape[0]))
    best_E = np.inf
    cur_seed = rand_seed

    for r in range(n_rounds):
        state_out: list = []
        E = solve_sqd_active(
            h1e, eri, norb, nelec,
            bitstring_matrix=pool, probabilities=pool_probs,
            max_strings=max_strings, n_active_per_round=n_active_per_round,
            ecore=ecore, rand_seed=cur_seed, state_out=state_out,
        )
        c2d, sa, sb = state_out[0]
        if E < best_E:
            best_E = E
        if verbose:
            pmax = float(np.abs(np.asarray(c2d)).max() ** 2)
            print(f"[distill r{r+1}/{n_rounds}] E={E:.8f} pool={pool.shape[0]} "
                  f"|c|max²={pmax:.4f}")
        if r == n_rounds - 1:
            break
        # 解态驱动重要性重采 (温度退火)
        T = temperature_schedule[r]
        new_bsm = eigenvector_importance_sample(
            c2d, sa, sb, norb, n_samples, rand_seed=cur_seed, temperature=T)
        if keep_pool:
            pool = np.vstack([bsm0, new_bsm])
            pool_probs = np.full(pool.shape[0], 1.0 / pool.shape[0])
        else:
            pool = new_bsm
            pool_probs = np.full(n_samples, 1.0 / n_samples)
        cur_seed = (cur_seed or 0) + 1

    return float(best_E)


def eigenvector_importance_sample(
    c2d: np.ndarray,
    sa: np.ndarray,
    sb: np.ndarray,
    norb: int,
    n_shots: int,
    *,
    rand_seed: Optional[int] = 0,
    temperature: float = 1.0,
) -> np.ndarray:
    """本征矢重要性采样 (方向 D, 学习型采样先验): 按振幅平方 ∝c² 采样 det 位串。

    **思路**: 子空间对角化解出的本征矢 ``|Ψ⟩ = Σ_i c_i |i⟩`` 是体系当前最好
    的"波函数模型" (数据驱动先验)。按其振幅平方分布 ``p_i ∝ |c_i|²`` 重新
    采样 ``n_shots`` 个 det 位串 —— 高权重 det 被更多采样, 低权重 det 少量
    覆盖 —— 相比均匀/随机采样, 同 shots 下配置恢复更聚焦高价值 det, 子空间
    质量更高 (同维度误差更低)。

    **与 AI 方法衔接**: 这是"学习型采样分布"的最简实现 (从解态学分布)。更强
    版本可用神经网络/NQS 参数化 ``p_i`` 泛化到未采样 det (见 REVIEW 方向 D
    展望), 本函数是确定性、可验证的基线。

    Parameters
    ----------
    c2d : ndarray, shape (nA, nB)
        子空间对角化本征矢 (α × β 字符串网格振幅)。
    sa, sb : ndarray, shape (nA,) / (nB,)
        对应 α/β 字符串 (整数表示)。
    norb : int
        空间轨道数 (位串宽度)。
    n_shots : int
        采样 det 数。
    rand_seed : int | None
        随机种子。
    temperature : float
        分布锐度: ``p_i ∝ |c_i|^(2/temperature)``。``1.0`` = 原始振幅平方;
        ``<1`` 更锐 (只采主导 det), ``>1`` 更平 (更像均匀)。

    Returns
    -------
    ndarray, shape (n_shots, 2*norb)
        采样位串矩阵, 遵循库统一布局 ``[β_{n-1}..β_0 | α_{n-1}..α_0]``
        (左 norb 列 β = ``det_b``, 右 norb 列 α = ``det_a``)。开壳层消费者
        直接喂 ``bitstring_matrix_to_ci_strs(open_shell=True)`` 可还原 (α, β)。
    """
    from .counts import int_to_bitarray

    c2d = np.asarray(c2d)
    sa = np.asarray(sa)
    sb = np.asarray(sb)
    probs = np.abs(c2d) ** (2.0 / temperature)
    probs = probs.ravel()
    denom = probs.sum()
    if denom <= 0:
        raise ValueError("本征矢振幅全零, 无法采样。")
    probs = probs / denom
    rng = np.random.default_rng(rand_seed)
    idx = rng.choice(probs.size, size=n_shots, replace=True, p=probs)
    ia, ib = np.divmod(idx, c2d.shape[1])
    det_a = sa[ia]
    det_b = sb[ib]
    # 库比特串布局 [β_{n-1}..β_0 | α_{n-1}..α_0] (左 β 右 α, 见 counts.py /
    # fermion._det_to_bitstring / integrated carryover)。det_a=α → 右半,
    # det_b=β → 左半; int_to_bitarray 半内顺序 [orb_{n-1}..orb_0] 与约定一致。
    bsm = np.zeros((n_shots, 2 * norb), dtype=bool)
    bsm[:, :norb] = int_to_bitarray(det_b, norb)
    bsm[:, norb:] = int_to_bitarray(det_a, norb)
    return bsm
