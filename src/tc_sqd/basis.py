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

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "natural_orbitals_from_rdm",
    "rotate_to_natural_orbitals",
    "ccsd_natural_orbitals",
    "rdm1_from_sci_result",
    "natural_orbital_occupancies",
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
