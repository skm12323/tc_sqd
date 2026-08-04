"""tc_sqd.basis —— 基设计: 自然轨道换基 (方向①).

思路 (详见 SURVEY §7 基设计方向 / REVIEW 方向①):
SQD 子空间 = 采样 det 张的空间, 其效率取决于基态波函数在计算基下的**稀疏度**.
把积分旋转到自然轨道基 (1-RDM 对角化) 可大幅压缩波函数长尾系数 —— N2/STO-3G
拉伸实测: 99.9% 覆盖所需 det 数 189→62, 达到化学精度所需子空间维度 2116→676.

本模块提供**非侵入**的换基工具: 输入某基 (通常 MO 基) 的 ``h1e``/``eri`` +
一个 1-RDM (来自 SQD 解 ``SCIResult.sci_state.rdm``、FCI 解或 CCSD), 输出
自然轨道基的积分与变换矩阵. 换基后把新的 ``h1e``/``eri`` 喂回 ``solve_sqd`` /
``solve_sci`` 即可 —— 采样 det 的轨道占据定义随之切换 (经典模拟层无需重编译电路).

与 :func:`tc_sqd.fermion.rotate_integrals` 的区别: 后者由反厄米参数 K 生成
``U=exp(K)``; 本模块直接接受/生成酉矩阵 U (自然轨道), 面向"已知 1-RDM 的对角化换基".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "natural_orbitals_from_rdm",
    "rotate_to_natural_orbitals",
    "ccsd_natural_orbitals",
    "rdm1_from_sci_result",
    "natural_orbital_occupancies",
    "NaturalOrbitalResult",
    "solve_sqd_natural_orbitals",
]


def natural_orbitals_from_rdm(
    rdm1: np.ndarray,
    *,
    descend: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """对角化空间求和 1-RDM 得自然轨道。

    Parameters
    ----------
    rdm1 : ndarray, shape (norb, norb)
        空间求和 (spin-summed) 一电子约化密度矩阵。闭壳层 ``FCI`` / ``SQD``
        解可用 ``direct_spin1.make_rdm1`` 或 ``SCIResult.sci_state.rdm`` 取得。
    descend : bool
        是否按占据数**降序**排列自然轨道 (标准惯例)。``False`` 则升序。

    Returns
    -------
    U : ndarray, shape (norb, norb)
        酉矩阵, **列**为自然轨道在新基旧轨道下的展开系数
        (``φ'_p = Σ_i φ_i U_{ip}``)。
    occ : ndarray, shape (norb,)
        自然轨道占据数 (对角化后的 RDM 本征值), 与 ``U`` 列一一对应。
    """
    rdm1 = np.asarray(rdm1, dtype=np.float64)
    if rdm1.ndim != 2 or rdm1.shape[0] != rdm1.shape[1]:
        raise ValueError(
            f"rdm1 must be a square matrix, got shape {rdm1.shape}."
        )
    if not np.allclose(rdm1, rdm1.T, atol=1e-8):
        raise ValueError(
            "rdm1 must be symmetric (spin-summed closed-shell 1-RDM). "
            "For spin-resolved RDM use (rdm_a + rdm_b)."
        )
    # eigvalsh 保证对称阵数值上稳妥 (Hermitian eig)。
    occ, U = np.linalg.eigh(rdm1)
    if descend:
        # 降序: 占据数最大的轨道排最前, 使 HF 型 det 对应前 na 个轨道。
        U = U[:, ::-1]
        occ = occ[::-1]
    return U, occ


def rotate_to_natural_orbitals(
    h1e: np.ndarray,
    eri: np.ndarray,
    rdm1: np.ndarray,
    *,
    descend: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把 ``h1e``/``eri`` 旋转到自然轨道基。

    Parameters
    ----------
    h1e : ndarray, shape (norb, norb)
        当前基 (通常 MO 基) 一电子积分。
    eri : ndarray, shape (norb, norb, norb, norb)
        当前基双电子积分 (chemist's notation, 空间)。
    rdm1 : ndarray, shape (norb, norb)
        与 ``h1e``/``eri`` **同基**的空间求和 1-RDM。
    descend : bool
        传给 :func:`natural_orbitals_from_rdm`。

    Returns
    -------
    h1e_nat : ndarray, shape (norb, norb)
        自然轨道基一电子积分。
    eri_nat : ndarray, shape (norb, norb, norb, norb)
        自然轨道基双电子积分。
    U : ndarray, shape (norb, norb)
        变换矩阵, 列 = 自然轨道在旧基下的展开 (见 :func:`natural_orbitals_from_rdm`)。
    occ : ndarray, shape (norb,)
        自然轨道占据数。

    Notes
    -----
    变换约定与 :func:`tc_sqd.fermion.rotate_integrals` 一致 (U 列 = 新基):
    ``h'_pq = (U^T h U)_pq``, ``g'_pqrs = Σ_{ijkl} U_ip U_jq U_kr U_ls g_ijkl``。
    """
    U, occ = natural_orbitals_from_rdm(rdm1, descend=descend)
    h1e = np.asarray(h1e, dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)
    norb = h1e.shape[0]
    if h1e.shape != (norb, norb) or eri.shape != (norb, norb, norb, norb):
        raise ValueError(
            f"h1e/eri shape mismatch: h1e={h1e.shape}, eri={eri.shape}, "
            f"expected {(norb, norb)} / {(norb, norb, norb, norb)}."
        )
    if U.shape[0] != norb:
        raise ValueError(
            f"rdm1 dimension {U.shape[0]} != h1e dimension {norb}."
        )
    h1e_nat = U.T @ h1e @ U
    eri_nat = np.einsum(
        "pqrs,pi,qj,rk,sl->ijkl", eri, U, U, U, U, optimize=True,
    )
    return (
        np.ascontiguousarray(h1e_nat),
        np.ascontiguousarray(eri_nat),
        U,
        occ,
    )


def ccsd_natural_orbitals(
    mf,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 CCSD 的 MO 基 1-RDM 求自然轨道 (真实可自举先验)。

    不需要 FCI 解 —— 经典 CCSD 即可给出自然轨道先验, 适合作为换基的第一轮
    猜测。注意: 强关联区 CCSD 自身多参考缺失, 其自然轨道劣于 FCI-NO
    (N2 拉伸实测达化学精度维度 1849 vs 676); 更好的是用 SQD 解自洽迭代
    (见 :func:`rdm1_from_sci_result` 与 REVIEW 方向①自洽方案)。

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        已收敛的 RHF/ROHF 对象 (``tc_sqd.from_pyscf(mol).mf``)。

    Returns
    -------
    U : ndarray, shape (norb, norb)
        自然轨道 (列) 变换矩阵。
    occ : ndarray, shape (norb,)
        自然轨道占据数。
    dm1_cc : ndarray, shape (norb, norb)
        CCSD 的空间求和 MO 基 1-RDM (诊断用)。
    """
    if mf is None or not hasattr(mf, "mo_coeff") or getattr(mf, "e_tot", None) is None:
        raise ValueError(
            "mf must be a converged PySCF SCF object (e.g. from_pyscf(mol).mf)."
        )
    from pyscf import cc as _cc

    mycc = _cc.CCSD(mf).run(verbose=0)
    # ao_repr=False -> MO 基自旋求和 1-RDM (与 h1e/eri 同基)
    dm1_cc = np.asarray(mycc.make_rdm1(ao_repr=False), dtype=np.float64)
    U, occ = natural_orbitals_from_rdm(dm1_cc)
    return U, occ, dm1_cc


def rdm1_from_sci_result(result) -> np.ndarray:
    """从 SQD 对角化解 ``SCIResult`` 提取空间求和 1-RDM。

    这是"自洽换基"的核心接口: 解出 SQD 基态后取 1-RDM, 换基到自然轨道,
    下一轮用新基积分 + 新平均占据重解, 迭代至收敛 (见 REVIEW 方向①自洽方案)。
    """
    return np.asarray(
        result.sci_state.rdm(rank=1, spin_summed=True), dtype=np.float64
    )


def natural_orbital_occupancies(
    rdm1: np.ndarray,
    *,
    spin: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """从空间求和 1-RDM 求平均轨道占据数。

    Parameters
    ----------
    rdm1 : ndarray, shape (norb, norb)
        空间求和 1-RDM (对角即轨道占据, 0~2)。
    spin : bool
        ``False`` 返回空间占据 ``(occ,)`` (每轨道 0~2); ``True`` 返回闭壳层
        自旋分辨平均占据 ``(occ/2, occ/2)`` (每自旋 0~1, 可直接作为
        :func:`tc_sqd.configuration_recovery.recover_configurations` 的
        ``avg_occupancies``)。

    Returns
    -------
    tuple
        ``(occ,)`` (``spin=False``) 或 ``(occ_a, occ_b)`` (``spin=True``)。
    """
    d = np.clip(np.diag(np.asarray(rdm1, dtype=np.float64)), 0.0, 2.0)
    if spin:
        return d / 2.0, d / 2.0
    return (d,)


# --------------------------------------------------------------------------- #
# 便捷封装: 从 pyscf FCI 解直接换基 (验证/测试用)
# --------------------------------------------------------------------------- #
def natural_orbital_basis_from_fci(
    h1e: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    civec: Optional[np.ndarray] = None,
    *,
    conv_tol: float = 1e-12,
    max_cycle: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(验证工具) 用 FCI 1-RDM 换基 —— 理想极限基准, 用于衡量换基上限。

    Parameters
    ----------
    h1e, eri : ndarray
        MO 基积分。
    norb : int
        轨道数。
    nelec : tuple(int, int)
        电子数。
    civec : ndarray | None
        可选; 给出则直接用它算 1-RDM (省一次 FCI)。否则先解 FCI。
    conv_tol, max_cycle : float, int
        FCI 收敛参数 (与 ``fermion.py`` FCI 分支一致, 避免假收敛陷阱)。

    Returns
    -------
    h1e_nat, eri_nat, U, occ
        同 :func:`rotate_to_natural_orbitals`。
    """
    from pyscf.fci import direct_spin1

    if civec is None:
        _, civec = direct_spin1.kernel(
            h1e, eri, norb, nelec, conv_tol=conv_tol, max_cycle=max_cycle
        )
    dm1 = direct_spin1.make_rdm1(civec, norb, nelec)
    return rotate_to_natural_orbitals(h1e, eri, dm1)


@dataclass
class NaturalOrbitalResult:
    """:func:`solve_sqd_natural_orbitals` 的结果。

    Attributes
    ----------
    energy : float
        最终基下 SQD 电子能量 (总能量 = ``energy + ecore``)。
    h1e, eri : ndarray
        最终自然轨道基的积分 (可直接再喂 ``solve_sqd``/``solve_sci``)。
    orbitals : ndarray, shape (norb, norb)
        **累计**旋转矩阵: 原始 MO 基轨道经各轮自然轨道变换后的最终轨道
        (``φ_final = φ_MO @ orbitals``)。仅旋转回原始轨道时才需要; 若不关心
        轨道显式形式, 直接使用 ``h1e``/``eri`` 即可。
    occ : ndarray, shape (norb,)
        最终自然轨道占据数。
    history : list[dict]
        每轮迭代记录: ``energy`` / ``dim`` (字符串乘积维度) / ``ndet`` /
        ``maxc2`` / ``pr`` / ``k999`` (解态系数稀疏度, 供监控换基收敛)。
    """
    energy: float
    h1e: np.ndarray
    eri: np.ndarray
    orbitals: np.ndarray
    occ: np.ndarray
    history: List[Dict] = field(default_factory=list)

    @property
    def total_energy(self) -> float:
        return self.energy


def solve_sqd_natural_orbitals(
    h1e: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    ecore: float = 0.0,
    bitstring_matrix: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    n_samples: int = 500,
    avg_occupancies: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    rand_seed: Optional[int] = 0,
    max_basis_iters: int = 4,
    energy_tol: float = 1e-9,
    verbose: bool = False,
    **solve_sci_kwargs,
) -> NaturalOrbitalResult:
    """自洽换基 SQD: 解 SQD → 1-RDM → 自然轨道换基 → 重解, 迭代至收敛。

    **动机** (方向①验证, 见 REVIEW): FCI-NO 换基使 N₂/STO-3G 拉伸达到化学精度
    所需子空间维度 2116→676; 低采样自洽换基突破配置恢复在 MO 基的覆盖瓶颈,
    能量误差最多改善 263×。此函数把该闭环固化为库功能, 是方向② (自适应采样)
    的表示层地基。

    **流程** (经典模拟, 配置恢复替代电路采样): 每轮 ① 配置恢复 (按当前基平均
    占据生成 det, ``recover_configurations``——只依赖平均占据, 故换基无缝) →
    ② ``solve_sci`` 子空间对角化 → ③ 解态 1-RDM (``rdm1_from_sci_result``)
    → ④ 对角化 1-RDM 得自然轨道, 换基 ``h1e``/``eri`` → ⑤ 更新平均占据
    (闭壳层: 每自旋 = 占据数/2)。迭代直到能量变化 < ``energy_tol`` 或达到
    ``max_basis_iters``。

    Parameters
    ----------
    h1e, eri : ndarray
        初始基 (通常 MO 基) 积分。
    norb : int
        空间轨道数。
    nelec : tuple(int, int)
        电子数。**当前实现要求闭壳层** (α=β 占据), 与 ``natural_orbital_occupancies``
        的自旋分辨假设一致。
    ecore : float
        核排斥能; 总能量 = ``energy + ecore`` (结果属性 ``total_energy``)。
    bitstring_matrix : ndarray, shape (S, 2*norb), optional
        配置恢复的种子位串。省略时用 ``n_samples`` 个均匀随机位串 (经典初猜)。
        注意: 换基后种子位串的轨道占据语义随基变化, 但配置恢复只依赖平均占据,
        恢复出的 det 始终在当前基下 —— 这是自洽换基能在经典层无缝运行的原因。
    probabilities : ndarray, shape (S,), optional
        对应概率; 省略时均匀。
    n_samples : int
        ``bitstring_matrix`` 为 ``None`` 时的随机种子数。
    avg_occupancies : tuple(ndarray, ndarray), optional
        初始平均占据 (闭壳层)。省略时退化为 HF 占据。
    rand_seed : int | None
        随机种子 (配置恢复 tie-breaking + 随机种子生成)。
    max_basis_iters : int
        最大换基轮数。
    energy_tol : float
        能量收敛阈值: 连续两轮能量变化小于它即停止换基。
    verbose : bool
        打印每轮能量/维度/稀疏度。
    **solve_sci_kwargs
        透传给 ``solve_sci`` (如 ``spin_sq``)。

    Returns
    -------
    NaturalOrbitalResult
        ``energy`` 为最终电子能量; ``h1e``/``eri`` 为最终自然基积分。

    Notes
    -----
    * **换基后的量子电路对接** (真机): 若电路在原始 MO 基下采样, 换基后需把
      电路参数重编译到自然基 (单粒子变换合成) 才能保持物理一致。本函数面向
      配置恢复路径 (经典/模拟), 该路径换基无缝。
    * 收敛时能量可能在小数后几位波动 (数值噪声), ``energy_tol`` 不宜过严;
      观测建议 1e-9 ~ 1e-10 对 N₂/STO-3G 量级体系已足够。
    """
    from .configuration_recovery import recover_configurations
    from .fermion import bitstring_matrix_to_ci_strs, solve_sci

    h1e = np.asarray(h1e, dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)
    na, nb = nelec
    if na != nb:
        raise ValueError(
            "solve_sqd_natural_orbitals 当前仅支持闭壳层 (nelec[0]==nelec[1]): "
            f"got {(na, nb)}。开壳层需自旋分辨 1-RDM 换基, 见 TODO。"
        )

    # 采样种子 (省略时随机)
    if bitstring_matrix is None:
        rng = np.random.default_rng(rand_seed)
        bsm = (rng.random((n_samples, 2 * norb)) > 0.5)
        probs = np.full(n_samples, 1.0 / n_samples)
    else:
        bsm = np.asarray(bitstring_matrix, dtype=bool)
        if bsm.ndim != 2 or bsm.shape[1] != 2 * norb:
            raise ValueError(
                f"bitstring_matrix must have shape (S, 2*norb={2*norb}), got {bsm.shape}."
            )
        probs = (np.full(bsm.shape[0], 1.0 / bsm.shape[0]) if probabilities is None
                 else np.asarray(probabilities, dtype=np.float64))

    # 初始平均占据
    if avg_occupancies is not None:
        occ_a, occ_b = avg_occupancies
    else:
        occ_a = np.zeros(norb, dtype=np.float64)
        occ_a[:na] = 1.0
        occ_b = np.zeros(norb, dtype=np.float64)
        occ_b[:nb] = 1.0

    h1e_cur, eri_cur = h1e, eri
    U_total = np.eye(norb, dtype=np.float64)
    history: List[Dict] = []
    energy_prev = np.inf

    for it in range(max_basis_iters):
        # ① 配置恢复 (当前基平均占据) ② 子空间对角化
        rec, _ = recover_configurations(
            bsm, probs, (occ_a, occ_b), na, nb, rand_seed=rand_seed
        )
        ci_a, ci_b = bitstring_matrix_to_ci_strs(rec)
        result = solve_sci(
            (ci_a, ci_b), h1e_cur, eri_cur, norb, nelec, **solve_sci_kwargs
        )
        dim = ci_a.shape[0] * ci_b.shape[0]

        # ③ 解态 1-RDM → ④ 自然轨道换基
        dm1 = rdm1_from_sci_result(result)
        h1e_cur, eri_cur, U_step, occ_nat = rotate_to_natural_orbitals(
            h1e_cur, eri_cur, dm1
        )
        U_total = U_total @ U_step

        # ⑤ 更新平均占据 (闭壳层)
        occ_a = np.clip(occ_nat / 2.0, 0.0, 1.0)
        occ_b = occ_a.copy()

        # 稀疏度监控 (当前基下解态系数的长尾指标; 各轮口径一致可作相对比较)
        c2 = np.abs(np.asarray(result.sci_state.amplitudes).ravel()) ** 2
        c2 = c2[c2 > 1e-15]
        p = c2 / c2.sum()
        history.append({
            "energy": float(result.energy),
            "dim": dim,
            "ndet": len(rec),
            "maxc2": float(p.max()),
            "pr": float(1.0 / (p**2).sum()),
            "k999": int(np.sort(p)[::-1].cumsum().searchsorted(0.999) + 1),
        })
        if verbose:
            print(f"[basis iter {it+1}/{max_basis_iters}] E(elec)={result.energy:.8f} "
                  f"dim={dim} ndet={len(rec)} k999={history[-1]['k999']}")

        if abs(result.energy - energy_prev) < energy_tol:
            if verbose:
                print(f"  能量收敛: ΔE={abs(result.energy - energy_prev):.2e} < {energy_tol}")
            break
        energy_prev = result.energy

    return NaturalOrbitalResult(
        energy=float(result.energy),
        h1e=h1e_cur,
        eri=eri_cur,
        orbitals=U_total,
        occ=occ_nat,
        history=history,
    )
