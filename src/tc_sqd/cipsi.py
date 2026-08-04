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

__all__ = ["solve_cipsi", "solve_sqd_active"]


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

    for r in range(max_rounds):
        # ① 采样聚焦: 配置恢复 (偏置平均占据) 生成当前基 det, 并入子空间
        rec, _ = recover_configurations(
            bsm, probs, (occ_a, occ_b), na, nb, rand_seed=rand_seed
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
        e_prev = E

    E, c2d, sa, sb = sub.diag(str_a, str_b)
    return float(E) + ecore
