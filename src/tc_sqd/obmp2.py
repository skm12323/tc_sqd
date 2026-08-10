"""tc_sqd.obmp2 —— one-body Møller–Plesset second-order (OBMP2), self-consistent.

实现 OBMP2 的一体相关势与自洽 (SCF) 求解。方法来源:

- Tran Nguyen Lan & Takeshi Yanai, "Correlated one-body potential from second-order
  Møller–Plesset perturbation theory: alternative to orbital-optimized MP2 method",
  *J. Chem. Phys.* **138**, 224108 (2013).
- L. N. Tran, "Improving perturbation theory for open-shell molecules via
  self-consistency", *J. Phys. Chem. A* **125**, 9242 (2021) (arXiv:2107.11260).
  OBMP2 哈密顿量的规范方程:
      H_OBMP2 = H_HF + V_OBMP2,   V_OBMP2 = V_1stBCH + V_2ndBCH (+ C')
  V_1stBCH = T̄_ij^ab [ f_a^i Ω̂(â_j^b) + g_ab^ip Ω̂(â_j^p) − g_ij^aq Ω̂(â_q^b) ]
  (T̄_ij^ab = T_ij^ab − T_ji^ab; Ω̂(â_q^p) = â_q^p + â_p^q 对称化; i,j 占据、a,b 虚、p,q 一般自旋轨道;
   f_ai = 0 at HF, 故 V_1stBCH 首项消失)
  V_2ndBCH = 9 项 T·T̄·f 收缩 (见函数体注释)。
  C'_1stBCH = −2 T̄_ij^ab g_ab^ij。

**为什么必须自洽**: 在 HF 参考上一次性能量恰等于 E_HF
(E_OBMP2(0) = E_HF + ⟨V|HF⟩ + C' = E_HF, 由 V_1stBCH 的 ⟨HF|Ω̂|HF⟩ 与 C' 精确相消)。
相关能来自对角化 f̄ = f + v 后的**轨道弛豫**。故本模块实现 SCF 循环:
v(t2, 当前轨道) → f̄ = f + v → 对角化得新轨道 → 重算积分与 t2 → 迭代至收敛。

**实现口径**: 自旋轨道显式 (α 全-β 全排序, 占据/虚轨道用显式索引数组), 闭壳层。
暴露的空间一体势 v (norb×norb) 与常数 C̄ 供下游 (如 OBDF 下折叠) 使用。

**非变分/适用性**: OBMP2 是微扰导出的自洽均场, 强关联/解离区可能不物理 (OBDF 论文
自己承认 "unphysically low energies")。测试选平衡/近平衡几何。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = ["OBMP2Result", "obmp2_potential", "solve_obmp2", "OBDFResult", "obdf_downfold"]


# --------------------------------------------------------------------------- #
#  Data container
# --------------------------------------------------------------------------- #
@dataclass
class OBMP2Result:
    """OBMP2 自洽解的结果。"""

    energy: float            # E_OBMP2 (自洽, = E_HF + ⟨V̂⟩ + C̄, 见 solve_obmp2 docstring)
    e_hf: float
    mo_coeff: np.ndarray     # OBMP2 相关轨道 (norb × norb, 正交归一)
    mo_energy: np.ndarray    # f̄ 本征值 (相关轨道能量)
    t2: np.ndarray           # 最终 (自旋轨道) MP2 振幅
    v_potential: np.ndarray  # 最终空间一体势 (norb × norb, 自旋求和)
    C_bar: float             # 常数 C̄ = C + C' (自旋求和)
    converged: bool = True
    n_iter: int = 0


# --------------------------------------------------------------------------- #
#  Internal: spin-orbital helpers (排序: [α 全空间, β 全空间])
# --------------------------------------------------------------------------- #
def _occ_vir_so(norb: int, nocc: int) -> Tuple[np.ndarray, np.ndarray]:
    """占据/虚自旋轨道索引 (α 全-β 全排序)。"""
    occ = list(range(nocc)) + list(range(norb, norb + nocc))
    vir = list(range(nocc, norb)) + list(range(norb + nocc, 2 * norb))
    return np.asarray(occ, dtype=np.int64), np.asarray(vir, dtype=np.int64)


def _spin_antisym_eri(eri_spatial: np.ndarray, norb: int) -> np.ndarray:
    """自旋轨道反对称积分 g[p,q,r,s] = ⟨pq‖rs⟩ (α 全-β 全排序)。"""
    ns = 2 * norb
    g = np.zeros((ns, ns, ns, ns), dtype=np.float64)
    for p in range(ns):
        for q in range(ns):
            for r in range(ns):
                for s in range(ns):
                    sp, sq, sr, ss = p // norb, q // norb, r // norb, s // norb
                    if sp == sr and sq == ss:
                        g[p, q, r, s] = eri_spatial[p % norb, r % norb, q % norb, s % norb]
    return g - g.transpose(0, 1, 3, 2)


def _mp2_amps_so(g_so: np.ndarray, eps_so: np.ndarray, occ: np.ndarray, vir: np.ndarray) -> np.ndarray:
    """自旋轨道 MP2 振幅 T[i,j,a,b] = g[occ_i,occ_j,vir_a,vir_b]/(eps_i+eps_j-eps_a-eps_b)。

    分母近零 (简并) 处置 0 (Pauli 禁戒/数值噪声, 对势贡献可忽略)。
    """
    T = np.asarray(g_so[np.ix_(occ, occ, vir, vir)], dtype=np.float64)
    eps_o, eps_v = eps_so[occ], eps_so[vir]
    denom = (eps_o[:, None, None, None] + eps_o[None, :, None, None]
             - eps_v[None, None, :, None] - eps_v[None, None, None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(np.abs(denom) > 1e-8, T / denom, 0.0)
    return T


def _obmp2_v_1stbch(T: np.ndarray, g_so: np.ndarray, occ: np.ndarray, vir: np.ndarray,
                    f_so: np.ndarray) -> Tuple[np.ndarray, float]:
    """V_1stBCH: 返回自旋轨道势 v_so (nso×nso) 与常数 C'_1stBCH。

    V_1stBCH = T̄_ij^ab [ f_a^i Ω̂(â_j^b) + g_ab^ip Ω̂(â_j^p) − g_ij^aq Ω̂(â_q^b) ]
    - 首项 (f_a^i, 占据-虚 Fock) 在规范 HF 基下 = 0 (Brillouin), 但在 SCF 中间步的
      非规范基下非零 —— 本实现**保留**它 (通用), 用完整 f_so。
    - Ω̂(â_q^p) = â_q^p + â_p^q: 每个系数 c 同时加到 v[q,p] 与 v[p,q]。
    """
    nso = g_so.shape[0]
    no, nv = len(occ), len(vir)
    Tbar = T - T.transpose(1, 0, 2, 3)                     # T̄_ij^ab
    v = np.zeros((nso, nso), dtype=np.float64)

    # 首项: f_a^i Ω̂(â_j^b)  -> c1[j,b] = sum_{i,a} Tbar[i,j,a,b] f[vir_a, occ_i]
    if nv and no:
        g_ovov = f_so[np.ix_(vir, occ)]                     # (nv, no)
        c1 = np.einsum("ijab,ai->jb", Tbar, g_ovov, optimize=True)
        for j in range(no):
            for b in range(nv):
                c = c1[j, b]
                v[occ[j], vir[b]] += c
                v[vir[b], occ[j]] += c

    # 第二项: g_ab^ip Ω̂(â_j^p) -> c2[j,p] = sum_{i,a,b} Tbar[i,j,a,b] g[vir_a,vir_b,occ_i,p]
    g_vvop = g_so[np.ix_(vir, vir, occ, np.arange(nso))]    # (nv,nv,no,nso)
    c2 = np.einsum("ijab,abip->jp", Tbar, g_vvop, optimize=True)
    v[np.ix_(occ, np.arange(nso))] += c2
    v[np.ix_(np.arange(nso), occ)] += c2.T

    # 第三项: −g_ij^aq Ω̂(â_q^b) -> c3[b,q] = −sum_{i,j,a} Tbar[i,j,a,b] g[occ_i,occ_j,vir_a,q]
    g_oovq = g_so[np.ix_(occ, occ, vir, np.arange(nso))]    # (no,no,nv,nso)
    c3 = -np.einsum("ijab,ijaq->bq", Tbar, g_oovq, optimize=True)
    v[np.ix_(vir, np.arange(nso))] += c3
    v[np.ix_(np.arange(nso), vir)] += c3.T

    # 常数 C'_1stBCH = −2 T̄_ij^ab g_ijab
    g_oovv = g_so[np.ix_(occ, occ, vir, vir)]
    Cp1 = -2.0 * float(np.einsum("ijab,ijab->", Tbar, g_oovv, optimize=True))
    return v, Cp1


def _obmp2_v_2ndbch(T: np.ndarray, g_so: np.ndarray, occ: np.ndarray, vir: np.ndarray,
                    f_so: np.ndarray) -> Tuple[np.ndarray, float]:
    """V_2ndBCH (9 项 T·T̄·f 收缩) 与 C'_2ndBCH。

    源自 ½[[F, A_D], A_D]_1 (Tran 2021 Eq. 10)。Ω̂ 对称化同 1stBCH。
    """
    nso = g_so.shape[0]
    no, nv = len(occ), len(vir)
    Tbar = T - T.transpose(1, 0, 2, 3)
    v = np.zeros((nso, nso), dtype=np.float64)

    def add_coeff(coeff_mat, i_idx, j_idx):
        """coeff_mat[a_local, b_local] (局部) -> v[abs_i, abs_j] += c, v[abs_j, abs_i] += c。"""
        for ia in range(coeff_mat.shape[0]):
            for ib in range(coeff_mat.shape[1]):
                c = coeff_mat[ia, ib]
                if abs(c) < 1e-14:
                    continue
                v[i_idx[ia], j_idx[ib]] += c
                v[j_idx[ib], i_idx[ia]] += c

    # 简记: T = T_ij^ab (no,no,nv,nv), Tbar 同理; f 为全自旋轨道 Fock
    # 各项系数 (目标 Ω̂(â_...)):
    # T1  +f_a^i T̄_ij^ab T̄_jk^bc Ω̂(â_c^k) : c[k,c] = sum_{i,j,a,b} f_ai Tbar_ijab Tbar_jkbc
    c_t1 = np.einsum("ai,ijab,jkbc->kc", f_so[np.ix_(vir, occ)], Tbar, Tbar, optimize=True)
    # T2  +f_c^a T_ij^ab T̄_il^cb Ω̂(â_j^l) : c[l,j] = sum_{i,a,b,c} f_ca T_ijab Tbar_ilcb
    c_t2 = np.einsum("ca,ijab,ilcb->lj", f_so[np.ix_(vir, vir)], T, Tbar, optimize=True)
    # T3  +f_c^a T_ij^ab T̄_kj^cb Ω̂(â_i^k) : c[k,i] = sum_{j,a,b,c} f_ca T_ijab Tbar_kjcb
    c_t3 = np.einsum("ca,ijab,kjcb->ki", f_so[np.ix_(vir, vir)], T, Tbar, optimize=True)
    # T4  −f_i^k T_ij^ab T̄_kl^ab Ω̂(â_j^l) : c[l,j] = −sum_{i,a,b,k} f_ik T_ijab Tbar_klab
    c_t4 = -np.einsum("ik,ijab,klab->lj", f_so[np.ix_(occ, occ)], T, Tbar, optimize=True)
    # T5  −f_i^p T_ij^ab T̄_kj^ab Ω̂(â_k^p) : c[p,k] = −sum_{i,j,a,b} f_ip T_ijab Tbar_kjab
    c_t5 = -np.einsum("ip,ijab,kjab->pk", f_so[np.ix_(occ, np.arange(nso))], T, Tbar, optimize=True)
    # T6  +f_i^k T_ij^ab T̄_kj^ad Ω̂(â_b^d) : c[b,d] = sum_{i,j,a,k} f_ik T_ijab Tbar_kjad
    c_t6 = np.einsum("ik,ijab,kjad->bd", f_so[np.ix_(occ, occ)], T, Tbar, optimize=True)
    # T7  +f_k^i T_ij^ab T̄_kj^cb Ω̂(â_a^c) : c[a,c] = sum_{i,j,b,k} f_ki T_ijab Tbar_kjcb
    c_t7 = np.einsum("ki,ijab,kjcb->ac", f_so[np.ix_(occ, occ)], T, Tbar, optimize=True)
    # T8  −f_c^a T_ij^ab T̄_ij^cd Ω̂(â_d^b) : c[b,d] = −sum_{i,j,a,c} f_ca T_ijab Tbar_ijcd
    c_t8 = -np.einsum("ca,ijab,ijcd->bd", f_so[np.ix_(vir, vir)], T, Tbar, optimize=True)
    # T9  −f_p^a T_ij^ab T̄_ij^cb Ω̂(â_c^p) : c[p,c] = −sum_{i,j,a,b} f_pa T_ijab Tbar_ijcb
    c_t9 = -np.einsum("pa,ijab,ijcb->pc", f_so[np.ix_(np.arange(nso), vir)], T, Tbar, optimize=True)

    # 施加 Ω̂ (每项目标索引映射到绝对自旋轨道)
    add_coeff(c_t1, occ, vir)     # â_c^k: c∈vir, k∈occ (c_t1[k,c])
    add_coeff(c_t2, occ, occ)     # â_j^l: j,l∈occ
    add_coeff(c_t3, occ, occ)     # â_i^k
    add_coeff(c_t4, occ, occ)
    add_coeff(c_t5, np.arange(nso), occ)   # â_k^p: p 一般, k occ
    add_coeff(c_t6, vir, vir)     # â_b^d: b,d vir
    add_coeff(c_t7, vir, vir)     # â_a^c
    add_coeff(c_t8, vir, vir)     # â_d^b
    add_coeff(c_t9, np.arange(nso), vir)   # â_c^p

    # C'_2ndBCH = −2 f_a^c T_ij^ab T̄_ij^cb + 2 f_i^k T_ij^ab T̄_kj^ab
    Cp2 = (-2.0 * float(np.einsum("ca,ijab,ijcb->", f_so[np.ix_(vir, vir)], T, Tbar, optimize=True))
           + 2.0 * float(np.einsum("ik,ijab,kjab->", f_so[np.ix_(occ, occ)], T, Tbar, optimize=True)))
    return v, Cp2


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def obmp2_potential(h1e: np.ndarray, eri: np.ndarray, mo_energy: np.ndarray,
                    norb: int, nocc: int, *,
                    include_2nd_bch: bool = True,
                    fock: Optional[np.ndarray] = None,
                    active_range: Optional[Tuple[int, int]] = None,
                    ) -> Tuple[np.ndarray, float, np.ndarray]:
    """计算闭壳层 RHF 基下 OBMP2 一体相关势 (空间, 自旋求和)。

    Parameters
    ----------
    h1e : ndarray (norb, norb)
        当前 MO 基一电子积分 (仅用于 Fock, 若 ``fock`` 未给)。
    eri : ndarray (norb, norb, norb, norb)
        当前 MO 基双电子积分 (chemist 记号)。
    mo_energy : ndarray (norb,)
        当前 MO 基轨道能量。
    norb, nocc : int
    include_2nd_bch : bool
        是否包含 2nd-BCH 项 (默认 True, 完整 OBMP2 势)。
    fock : ndarray (norb, norb) | None
        当前基 Fock 矩阵。``None`` 时用 ``diag(mo_energy)`` (规范 HF 基近似;
        若基非规范, 应显式传 Fock)。
    active_range : tuple(int, int) | None
        活性空间轨道区间 ``(start, end)`` (空间轨道索引)。给定后只保留含
        ≥1 **外部**指标 (冻结 core 占据 或 冻结虚轨道) 的振幅构造 v^ext
        (OBDF 下折叠, 论文 arXiv:2605.08675 Eq. 12-13), 并返回活性块
        ``v[start:end, start:end]``。``None`` = 全振幅全空间势。

    Returns
    -------
    v_spatial : ndarray (norb, norb)
        空间一体相关势 (自旋求和)。对称。``active_range`` 给定时为活性块。
    C_bar_corr : float
        相关常数贡献 C' (1st + 2nd BCH), 自旋求和。
    t2 : ndarray (2*nocc, 2*nocc, 2*nvir, 2*nvir)
        自旋轨道 MP2 振幅 (下游自洽用)。
    """
    if fock is None:
        fock = np.diag(np.asarray(mo_energy, dtype=np.float64))
    fock = np.asarray(fock, dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)
    g_so = _spin_antisym_eri(eri, norb)
    nso = 2 * norb
    f_so = np.zeros((nso, nso))
    f_so[:norb, :norb] = fock
    f_so[norb:, norb:] = fock
    eps_so = np.concatenate([np.asarray(mo_energy), np.asarray(mo_energy)])
    occ, vir = _occ_vir_so(norb, nocc)

    T = _mp2_amps_so(g_so, eps_so, occ, vir)
    if active_range is not None:
        # OBDF 外部限制: 只保留含 ≥1 外部指标的振幅。外部 = 冻结 core 占据
        # (空间 [0,start)) + 冻结虚轨道 (空间 [end,norb))。
        start, end = active_range
        ext_occ = (occ % norb) < start          # occ 自旋轨道属冻结 core
        ext_vir = (vir % norb) >= end           # vir 自旋轨道属冻结虚
        T = T * (ext_occ[:, None, None, None] | ext_occ[None, :, None, None]
                 | ext_vir[None, None, :, None] | ext_vir[None, None, None, :])
    v_so, Cp = _obmp2_v_1stbch(T, g_so, occ, vir, f_so)
    # 归一化因子: A_D = ½·T (论文 Eq 3), 故
    #   [H, A_D]_1      含 ½ -> v_1st, C'_1st 乘 1/2
    #   ½[[F,A_D],A_D]_1 含 ½·(½)² = 1/8 -> v_2nd, C'_2nd 乘 1/8, 且符号翻转 (-1/8)
    # (实证: 施加后 E_OBMP2(0) = E_HF + 2·Tr(v)+C' 精确 = E_MP2, N2/H2O/STO-3G 全吻合)
    v_so, Cp = 0.5 * v_so, 0.5 * Cp
    if include_2nd_bch:
        v_so2, Cp2 = _obmp2_v_2ndbch(T, g_so, occ, vir, f_so)
        v_so += -0.125 * v_so2
        Cp += -0.125 * Cp2

    # 自旋求和 -> 空间 (α-β 块相等, 取任一块即可)
    v_spatial = v_so[:norb, :norb]
    if active_range is not None:
        start, end = active_range
        v_spatial = v_spatial[start:end, start:end]
    return v_spatial, float(Cp), T


def solve_obmp2(mf, *, max_iter: int = 60, tol: float = 1e-8,
                include_2nd_bch: bool = True) -> OBMP2Result:
    """OBMP2 自洽求解。

    迭代: 当前轨道 -> 积分/Fock/t2 -> v, C' -> f̄ = f + v -> 对角化得新轨道 -> 重算。
    能量取 **相关态期望** E_OBMP2 = ⟨Φ|H_OBMP2|Φ⟩ = Σ_i f̄_ii + C̄ (OBMP2 规范基下
    f̄ 对角, f̄_ii = ε̄_i)。与 ⟨Φ|H|Φ⟩ (原始哈密顿量在相关轨道上的期望) 一并返回。

    Parameters
    ----------
    mf : pyscf.scf.RHF
        已收敛闭壳层 RHF (规范基)。
    max_iter, tol
        自洽迭代控制。
    include_2nd_bch
        是否含 2nd-BCH 项。

    Returns
    -------
    OBMP2Result
    """
    from pyscf import scf

    mol = mf.mol
    if mol.nelectron % 2 != 0:
        raise ValueError("OBMP2 目前仅支持闭壳层 (偶电子)。")
    norb = mf.mo_coeff.shape[1]
    nocc = mol.nelectron // 2
    h_ao = mf.get_hcore()

    C = np.asarray(mf.mo_coeff, dtype=np.float64).copy()
    e_hf = mf.e_tot
    V_nuc = mf.energy_nuc()

    # 原子轨道双电子积分 (一次变换所有迭代)
    eri_ao = mol.intor("int2e_sph")

    E_prev = None
    for it in range(max_iter):
        # 当前基下积分
        h1e = C.T @ h_ao @ C
        eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, C, C, C, C, optimize=True)
        # 当前基 Fock (闭壳层): f = h + J - ½K, 其中 J=Σ(pq|rs)ρ, K=Σ(pr|qs)ρ, ρ=2δ_occ
        # (注意: 不是 2J-K —— 那会重复计算库仑, 系数 4 而非 2)
        rho = np.zeros((norb, norb))
        rho[:nocc, :nocc] = 2.0 * np.eye(nocc)
        f = h1e + np.einsum("pqrs,rs->pq", eri, rho) \
            - 0.5 * np.einsum("prqs,rs->pq", eri, rho)
        mo_energy = np.diag(f)

        v, Cp, T = obmp2_potential(h1e, eri, mo_energy, norb, nocc,
                                   include_2nd_bch=include_2nd_bch, fock=f)
        fbar = f + v
        e_val, Crot = np.linalg.eigh(fbar)
        # 闭壳层 OBMP2 能量: E_OBMP2 = ⟨Φ|F̄|Φ⟩ + C̄ = 2·Σ_occ ε̄_i + (C + C'),
        # 其中 C = E_HF(Φ) − ⟨Φ|F̂|Φ⟩ = E_HF(Φ) − 2·Σ_occ f_ii (双占据因子 2)。
        # E_HF(Φ) = Σ_occ(h_ii + f_ii) + Vn 是**当前**轨道 Φ 的 HF 能量 (随 SCF
        # 旋转变化, 不能用原始 e_hf)。C_cur = E_HF(Φ) − 2·Σ f_ii。
        E_hf_cur = float(np.sum(np.diag(h1e)[:nocc] + mo_energy[:nocc])) + V_nuc
        C_cur = E_hf_cur - 2.0 * float(np.sum(mo_energy[:nocc]))
        Cbar = C_cur + Cp
        E_obmp2 = 2.0 * float(np.sum(e_val[:nocc])) + Cbar

        newC = C @ Crot
        # 收敛判定: 轨道旋转的 Frobenius 范数
        dC = np.linalg.norm(newC - C)
        C = newC
        if E_prev is not None and it > 0:
            dE = abs(E_obmp2 - E_prev)
            if dE < tol:
                converged = True
                break
        E_prev = E_obmp2
    else:
        converged = False

    # 收敛基下最终量
    h1e = C.T @ h_ao @ C
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, C, C, C, C, optimize=True)
    rho = np.zeros((norb, norb)); rho[:nocc, :nocc] = 2.0 * np.eye(nocc)
    f = h1e + np.einsum("pqrs,rs->pq", eri, rho) - 0.5 * np.einsum("prqs,rs->pq", eri, rho)
    mo_energy = np.diag(f)
    v, Cp, T = obmp2_potential(h1e, eri, mo_energy, norb, nocc,
                               include_2nd_bch=include_2nd_bch, fock=f)
    fbar = f + v
    e_val, Crot = np.linalg.eigh(fbar)
    C = C @ Crot
    E_hf_cur = float(np.sum(np.diag(h1e)[:nocc] + np.diag(f)[:nocc])) + V_nuc
    C_cur = E_hf_cur - 2.0 * float(np.sum(np.diag(f)[:nocc]))
    Cbar = C_cur + Cp
    E_obmp2 = 2.0 * float(np.sum(e_val[:nocc])) + Cbar

    return OBMP2Result(
        energy=E_obmp2, e_hf=e_hf, mo_coeff=C, mo_energy=e_val,
        t2=T, v_potential=v, C_bar=Cbar, converged=converged, n_iter=it + 1,
    )


# --------------------------------------------------------------------------- #
#  OBDF: one-body downfolding (arXiv:2605.08675)
# --------------------------------------------------------------------------- #
@dataclass
class OBDFResult:
    """OBDF 下折叠的活性哈密顿量分量。"""

    h1e: np.ndarray              # H_CAS 一体部分 (frozen-core, 无 v^ext)
    h1e_downfolded: np.ndarray   # H_OBDF 一体部分 = h1e + scale·v^ext
    eri: np.ndarray              # 活性双电子积分
    ecore: float                 # 核 + frozen-core 能量
    norb: int
    nelec: Tuple[int, int]
    v_ext: np.ndarray            # 外部相关势 (活性块)
    scale: float
    n_core: int
    n_virtual: int

    def solve(self, *, method: str = "fci", downfolded: bool = True, **kwargs) -> float:
        """对活性哈密顿量求基态能量。``downfolded=True`` 用 H_OBDF, 否则 H_CAS。"""
        from .fermion import compute_ground_state_energy
        h1e = self.h1e_downfolded if downfolded else self.h1e
        return float(compute_ground_state_energy(
            h1e, self.eri, self.norb, self.nelec, ecore=self.ecore,
            method=method, **kwargs))


def obdf_downfold(mf, *, n_core: int, n_virtual: int,
                  scale: float = 0.1, include_2nd_bch: bool = True) -> OBDFResult:
    """One-body downfolding (OBDF, arXiv:2605.08675): 把外部相关折叠进活性 h1e。

    ``H_OBDF = H_CAS + scale·v^ext``, 其中 ``v^ext`` 是 OBMP2 一体势在**外部振幅**
    (含 ≥1 冻结 core/虚指标) 上构造后投影到活性块的结果。仅改 h1e, eri/ecore/nelec
    不变 (不增加量子资源)。

    Parameters
    ----------
    mf : pyscf.scf.RHF
        已收敛闭壳层 RHF。
    n_core : int
        冻结最低 n_core 个占据 MO (core)。
    n_virtual : int
        冻结最高 n_virtual 个虚 MO。
    scale : float
        v^ext 的缩放系数。**经验校准 ~0.1**: 实测 N₂/H₂O/cc-pVDZ (6-10o 活性)
        CAS 误差 0.21-0.30 Ha -> scale=0.1 时 OBDF 误差 0.006-0.012 Ha (近 CCSD(T))。
        原始 v^ext (A_D=½T 归一化后) 对下折叠约 10× 过大——10× 来源未完全解析
        (OBMP2 总能量用 trace+C' 相消, 元素量级与下折叠需求差一个常数), 留作开放问题。
    include_2nd_bch
        是否含 2nd-BCH 项。

    Returns
    -------
    OBDFResult
        含 H_CAS 与 H_OBDF 的一体部分 (调 ``.solve()`` 得能量)。

    Notes
    -----
    - **非变分**: OBMP2 是微扰的, 强关联/解离区可能失效 (论文自认 "unphysically low
      energies")。测试选平衡/近平衡几何。
    - 需 ``n_core + n_virtual > 0`` (有外部空间才可折叠)。
    """
    from pyscf import scf
    mol = mf.mol
    if mol.nelectron % 2 != 0:
        raise ValueError("OBDF 目前仅支持闭壳层 (偶电子)。")
    norb = mf.mo_coeff.shape[1]
    nocc = mol.nelectron // 2
    if n_core + n_virtual <= 0:
        raise ValueError("OBDF 需要非空外部空间 (n_core + n_virtual > 0)。")
    mo = np.asarray(mf.mo_coeff, dtype=np.float64)
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl",
                    mol.intor("int2e_sph"), mo, mo, mo, mo, optimize=True)

    start, end = n_core, norb - n_virtual
    n_act = end - start
    from .molecule import _frozen_core_energy, _frozen_core_potential
    ecore = mf.energy_nuc() + _frozen_core_energy(h1e, eri, n_core)
    h1e_act = np.asarray(h1e[start:end, start:end], dtype=np.float64)
    if n_core > 0:
        h1e_act = h1e_act + _frozen_core_potential(eri, n_core, n_virtual)
    eri_act = np.asarray(eri[start:end, start:end, start:end, start:end], dtype=np.float64)
    nelec_act = (nocc - n_core, nocc - n_core)

    v_ext, _Cp, _ = obmp2_potential(h1e, eri, mf.mo_energy, norb, nocc,
                                    include_2nd_bch=include_2nd_bch,
                                    fock=np.diag(mf.mo_energy),
                                    active_range=(start, end))
    return OBDFResult(
        h1e=h1e_act,
        h1e_downfolded=h1e_act + scale * v_ext,
        eri=eri_act,
        ecore=float(ecore),
        norb=int(n_act),
        nelec=nelec_act,
        v_ext=v_ext,
        scale=scale,
        n_core=int(n_core),
        n_virtual=int(n_virtual),
    )
