"""tc_sqd.molecule —— 从 PySCF 一键构建 SQD 输入 (分子数据)。

把"PySCF 分子 → MO 基 h1e/eri/ecore/norb/nelec"这段最容易出错的手写转换
(MO 变换、核排斥、冻结核修正、电子数) 封装成单个调用 :func:`from_pyscf`,
消灭整类积分/布局错误。

用法
----
>>> data = tc_sqd.from_pyscf(mol)            # mol (gto.Mole) 或已收敛的 mf (scf.RHF)
>>> data.norb, data.nelec, data.ecore        # 直接拿到 SQD 输入
>>> e = data.solve(method="fci")             # 一键求基态能量
>>> e = data.solve(method="sqd", bitstring_matrix=bsm)   # 或给采样喂 SQD
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["MolecularData", "from_pyscf"]


@dataclass
class MolecularData:
    """从 PySCF 构建的 SQD 输入 (MO 基积分 + 核/冻结能量 + 电子数)。

    Attributes
    ----------
    h1e : ndarray, shape (norb, norb)
        MO 基一电子积分 (空间轨道)。
    eri : ndarray, shape (norb, norb, norb, norb)
        MO 基双电子积分 (chemist's notation, 空间)。
    ecore : float
        核排斥能 + 冻结核修正 (直接用作 SQD 的 ecore)。
    norb : int
        活性空间轨道数。
    nelec : tuple(int, int)
        活性空间电子数 (n_alpha, n_beta)。
    n_core : int
        冻结轨道数 (0 = 全轨道)。
    mo_coeff : ndarray | None
        活性 MO 系数 (norb_full × norb)。
    nuc_energy : float
        纯核排斥能 (不含冻结核修正)。
    mf : object
        底层 PySCF SCF 对象 (RHF)。

    Methods
    -------
    solve(method=..., **kwargs) -> float
        一键求基态能量, 内部转 ``compute_ground_state_energy``。
    """

    h1e: np.ndarray
    eri: np.ndarray
    ecore: float
    norb: int
    nelec: Tuple[int, int]
    n_core: int = 0
    mo_coeff: Optional[np.ndarray] = None
    nuc_energy: float = 0.0
    mf: object = None

    def solve(self, *, method: str = "sqd", **kwargs) -> float:
        """一键求基态能量 (内部转 ``compute_ground_state_energy``)。

        ``method`` = ``"fci"`` / ``"sqd"`` / ``"direct"``; ``method="sqd"``
        时需传 ``bitstring_matrix`` (采样比特串)。
        """
        from .fermion import compute_ground_state_energy

        return float(
            compute_ground_state_energy(
                self.h1e, self.eri, self.norb, self.nelec,
                ecore=self.ecore, method=method, **kwargs,
            )
        )


def _frozen_core_energy(h1e: np.ndarray, eri: np.ndarray, n_core: int) -> float:
    """闭壳层冻结核能量 (core-core):

        E_frozen = 2 Σ_c h_cc + Σ_{c,d} [2 (cc|dd) - (cd|dc)]

    其中 ``h1e``/``eri`` 为**全 MO 基**积分 (含 core 块), ``n_core`` 为冻结
    轨道数 (占据最低的 n_core 个 MO)。core 对活性电子的库仑/交换势修正
    单独在 :func:`from_pyscf` 里加进活性 ``h1e``。
    """
    e = 0.0
    for c in range(n_core):
        e += 2.0 * h1e[c, c]
        for d in range(n_core):
            e += 2.0 * eri[c, c, d, d] - eri[c, d, d, c]
    return float(e)


def _frozen_core_potential(eri_full: np.ndarray, n_core: int) -> np.ndarray:
    """core 平均场对活性一电子积分的修正 (frozen-core 近似):

        Δh_ij = Σ_c [2 (cc|ij) - (ci|jc)]          (i, j 为活性轨道)

    活性空间 FCI 应使用 ``h1e_eff = h1e_act + Δh``; 这是 McWeeny 标准
    frozen-core 公式中缺失就错的部分 (只做 core-core 能量不闭合)。
    """
    n_act = eri_full.shape[0] - n_core
    pot = np.zeros((n_act, n_act), dtype=np.float64)
    for c in range(n_core):
        pot += 2.0 * eri_full[c, c, n_core:, n_core:]       # 库仑 (cc|ij)
        pot -= eri_full[c, n_core:, n_core:, c]             # 交换 (ci|jc)
    return pot


def from_pyscf(mf_or_mol, *, n_active: Optional[int] = None) -> MolecularData:
    """从 PySCF 分子 / SCF 对象一键构建 SQD 输入。

    Parameters
    ----------
    mf_or_mol : pyscf.gto.Mole | pyscf.scf.RHF
        已收敛的 RHF 对象, 或分子对象 (自动跑 RHF)。
    n_active : int | None
        活性空间轨道数 (冻结内层 ``norb_full - n_active`` 个 MO)。
        ``None`` = 全轨道 (n_core=0)。冻结时 ``ecore``/``nelec`` 已按闭壳层
        修正, 且 MO 假设按能量升序排列 (RHF 默认)。

    Returns
    -------
    MolecularData
        可直接传给 ``compute_ground_state_energy`` 或调 ``.solve()``。

    Raises
    ------
    ValueError
        非闭壳层 (奇电子)、``n_active`` 超出轨道数、输入既非 Mole 亦非 SCF。

    Notes
    -----
    **开壳层 (n_α≠n_β)**: 原生支持 (P2-1b)。闭壳层自动 RHF, 开壳层 (``mol.spin!=0``)
    自动 ROHF (单 ``mo_coeff``, 空间共用轨道, ROHF 式单 ``h1e``); ``nelec`` 由
    ``mf.nelec`` / ``mol.spin`` 推断 ``(na,nb)``。frozen-core 下 core 双占,
    活性 ``nelec = (na-n_core, nb-n_core)``。

    **不支持 UHF** (自旋分辨轨道, ``mo_coeff`` 形如 ``(2,norb,norb)``): SQD 后端
    拒 ``h_alpha != h_beta``, 请用 ROHF。UHF mf 会显式 raise。
    """
    from pyscf import gto, scf

    if isinstance(mf_or_mol, gto.Mole):
        mol = mf_or_mol
        # 开壳层 (mol.spin != 0) 自动 ROHF, 闭壳层 RHF (单 mo_coeff, 空间共用)
        if int(getattr(mol, "spin", 0)) != 0:
            mf = scf.ROHF(mol).run()
        else:
            mf = scf.RHF(mol).run()
    else:
        mf = mf_or_mol
        mol = getattr(mf, "mol", None)
        if mol is None or not isinstance(mf, scf.hf.SCF):
            raise ValueError(
                "mf_or_mol 必须是 pyscf gto.Mole 或已构建的 scf 对象 "
                "(RHF/ROHF); got {type(mf_or_mol).__name__}."
            )
        # UHF 显式拒绝: 自旋分辨轨道 (α/β 不同 h1e), SQD 后端不支持
        if isinstance(mf, scf.uhf.UHF):
            raise ValueError(
                "from_pyscf 不接受 UHF (自旋分辨轨道, mo_coeff 形如 (2,norb,norb)): "
                "SQD 后端 (fermion.solve_sci) 拒 h_alpha != h_beta, 请用 ROHF "
                "(单 mo_coeff, 空间共用轨道)。"
            )

    mo = np.asarray(mf.mo_coeff, dtype=np.float64)
    if mo.ndim != 2:
        raise ValueError(
            "mo_coeff 必须是单空间基 (RHF/ROHF), got ndim={mo.ndim} "
            "(UHF 自旋分辨轨道不支持, 请用 ROHF)。"
        )
    norb_full = mo.shape[1]

    # nelec 推断: 优先 mf.nelec (ROHF/RHF 直接给 (na,nb)), 回退 mol.spin
    mf_nelec = getattr(mf, "nelec", None)
    if mf_nelec is not None:
        na, nb = int(mf_nelec[0]), int(mf_nelec[1])
    else:
        spin = int(getattr(mol, "spin", 0))
        na = (mol.nelectron + spin) // 2
        nb = (mol.nelectron - spin) // 2
    if na < 0 or nb < 0 or na + nb != mol.nelectron:
        raise ValueError(
            f"nelec 推断不一致: na={na}, nb={nb}, nelectron={mol.nelectron}."
        )

    # 全 MO 基积分 (含 core 块, 供冻结修正 / 切片)
    h1e_ao = mf.get_hcore()
    h1e_full = mo.T @ h1e_ao @ mo
    eri_full = np.einsum(
        "pqrs,pi,qj,rk,sl->ijkl",
        mol.intor("int2e_sph"), mo, mo, mo, mo, optimize=True,
    )

    if n_active is not None:
        if not (0 < n_active <= norb_full):
            raise ValueError(
                f"n_active 必须在 (0, norb_full={norb_full}] 内, got {n_active}."
            )
        n_core = norb_full - n_active
    else:
        n_core = 0

    # 活性空间积分 (MO 按能量升序, 内层为 core)
    h1e = h1e_full[n_core:, n_core:]
    eri = eri_full[n_core:, n_core:, n_core:, n_core:]

    # frozen-core 修正: core-core 能量 + core 平均场打进活性 h1e (缺后者不闭合)
    ecore = mf.energy_nuc() + _frozen_core_energy(h1e_full, eri_full, n_core)
    if n_core > 0:
        h1e = h1e + _frozen_core_potential(eri_full, n_core)
    # frozen-core: core 双占 (ROHF 式, core 能量/平均场公式不变), 仅 nelec 减 core
    n_act_a = na - n_core
    n_act_b = nb - n_core
    if n_act_a < 0 or n_act_b < 0:
        raise ValueError(
            f"n_core={n_core} 超过电子数: na={na}, nb={nb}."
        )
    nelec = (n_act_a, n_act_b)

    return MolecularData(
        h1e=np.asarray(h1e, dtype=np.float64),
        eri=np.asarray(eri, dtype=np.float64),
        ecore=float(ecore),
        norb=int(norb_full - n_core),
        nelec=nelec,
        n_core=int(n_core),
        mo_coeff=mo[:, n_core:],
        nuc_energy=float(mf.energy_nuc()),
        mf=mf,
    )
