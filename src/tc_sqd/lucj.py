"""LUCJ (Local Unitary Cluster Jastrow) ansatz for TensorCircuit.

本模块从 PySCF CCSD 振幅构造一个简化的 LUCJ 电路, 作为 tc_sqd 的"量子态制备"
侧: HF 初态 + 由 CCSD 双激发振幅 t2 参数化的占据-空轨道纠缠门。

算法背景
--------
完整 UCJ ansatz 为 |Ψ_UCJ> = ∏_ℓ (Û_ℓ e^{iĴ_ℓ} Û_ℓ†) |HF>, 其中
Û = e^κ (轨道旋转, κ anti-Hermitian)、Ĵ = Σ_{pq} J_{pq} n_p n_q (对角 Coulomb)。
权威实现 ffsim.UCJOpSpinBalanced.from_t_amplitudes 从 t2 的 SVD 同时构造 Û 和 Ĵ。
"Local" UCJ (LUCJ) 把 e^{iĴ} 的 R_ZZ 项限制在相邻 qubit 以降低深度。

本模块的实现 (简化)
-------------------
- 对每个占据-空轨道对 (i, a), 用涉及该对的 CCSD 双激发振幅块 t2[i, :, a, :]
  的 Frobenius 范数作为 Givens 旋转角度, 在 alpha / beta 自旋上各施加一个
  Givens-like 门 (ry + cnot + ry)。两个自旋的单激发组合即产生双激发 determinant
  (α_i β_i → α_a β_a), 使 SQD 子空间含相关能。
- 关键: 必须由 t2 (而非 t1) 驱动 —— H2/STO-3G 的 t1≈0 (Brillouin 定理),
  相关能几乎全部来自 t2 双激发; t1 在许多闭壳层小体系都接近零。
- 尚未实现 t2 → Û/Ĵ 的精确 SVD 分解 (ffsim 流程); 当前简化已足以让 LUCJ-SQD
  在 H2 上精确复现 FCI、在 LiH 上捕获大部分相关能。

布局约定 (与 tc_sqd 其余模块一致)
---------------------------------
比特串: [β_{n-1}..β0 | α_{n-1}..α0]。TensorCircuit 的 q0 是采样整数最高位
(MSB), 即 bsm col_i == qubit i, 故 α_i → qubit(2norb-1-i), β_i → qubit(norb-1-i)。
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import tensorcircuit as tc

__all__ = ["get_ccsd_amplitudes", "build_lucj_circuit"]


def get_ccsd_amplitudes(mf):
    """跑 RHF-CCSD 并返回 (t1, t2, mycc)。

    Parameters
    ----------
    mf : pyscf scf.RHF 对象 (已完成 kernel)

    Returns
    -------
    t1 : ndarray, shape (nocc, nvir)
    t2 : ndarray, shape (nocc, nocc, nvir, nvir)
    mycc : pyscf CC 对象

    Warns
    -----
    RuntimeWarning
        若 CCSD 未收敛。
    """
    from pyscf import cc as _cc

    mycc = _cc.CCSD(mf)
    mycc.kernel()
    if not mycc.converged:
        warnings.warn(
            "CCSD did not converge; LUCJ amplitudes may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )
    return mycc.t1, mycc.t2, mycc


def _qubit(spin: str, orb: int, norb: int) -> int:
    """自旋轨道 → TensorCircuit qubit 编号。

    α_i → 2norb-1-i ; β_i → norb-1-i  (q0=MSB, [β|α] 布局)。
    """
    if spin == "a":
        return 2 * norb - 1 - orb
    return norb - 1 - orb


def build_lucj_circuit(
    mf,
    norb: int,
    nelec: Tuple[int, int],
    *,
    ccsd_scale: float = 1.0,
    max_excitations: Optional[int] = None,
    angle_multiplier: float = 2.0,
):
    """构造简化 LUCJ 电路: HF 初态 + t2 指导的占据-空 Givens-like 门。

    Parameters
    ----------
    mf : pyscf SCF 对象
    norb : 空间轨道数
    nelec : (n_alpha, n_beta)  (闭壳层: n_alpha == n_beta)
    ccsd_scale : float
        振幅整体缩放 (tex 中的 λ)。=0 退化为纯 HF。
    max_excitations : int | None
        只取强度 (t2 范数) 最大的前 K 个 occ-vir 对 (None=全部)。
    angle_multiplier : float
        t2 范数 → 旋转角的额外倍数 (CCSD 振幅通常很小, 需放大才能产生足够覆盖)。

    Returns
    -------
    tensorcircuit.Circuit
        可直接交给 ``tc_sqd.sample_from_circuit`` 采样。

    Raises
    ------
    ValueError
        若 nelec 非闭壳层 (当前仅支持 n_alpha == n_beta)。
    """
    if nelec[0] != nelec[1]:
        raise ValueError(
            "build_lucj_circuit 当前仅支持闭壳层 (n_alpha == n_beta); "
            f"got nelec={nelec}."
        )
    _t1, t2, _mycc = get_ccsd_amplitudes(mf)

    nq = 2 * norb
    c = tc.Circuit(nq)

    # 1) HF 初态: 占据最低的 nalpha / nbeta 个轨道
    for i in range(nelec[0]):
        c.x(_qubit("a", i, norb))
    for i in range(nelec[1]):
        c.x(_qubit("b", i, norb))

    # 2) 从 t2 提取每个 occ-vir 对 (i,a) 的耦合强度 (Frobenius 范数)。
    #    H2 的 t1≈0 (Brillouin), 相关能几乎全来自 t2, 故由 t2 驱动。
    #    SQD 只依赖采样的 |振幅|², 相位/符号不影响子空间构成。
    nocc = nelec[0]
    nvir = norb - nocc
    pairs = []  # (strength, spin, occ_orb, vir_orb, theta)
    for i in range(nocc):
        for a in range(nvir):
            strength = float(np.linalg.norm(t2[i, :, a, :])) if t2 is not None else 0.0
            theta = ccsd_scale * angle_multiplier * strength
            pairs.append((strength, "a", i, nocc + a, theta))
            pairs.append((strength, "b", i, nocc + a, theta))

    if max_excitations is not None:
        pairs.sort(key=lambda x: -x[0])
        pairs = pairs[:max_excitations]

    # 3) 对每个 occ-vir 对施加 Givens-like 门 (ry + cnot + ry)。
    #    粒子数违例由后续 tc_sqd.recover_configurations 修正。
    for _strength, spin, occ_orb, vir_orb, theta in pairs:
        if abs(theta) < 1e-12:
            continue
        q_occ = _qubit(spin, occ_orb, norb)
        q_vir = _qubit(spin, vir_orb, norb)
        c.ry(q_occ, theta=theta)
        c.cnot(q_occ, q_vir)
        c.ry(q_occ, theta=-theta)

    return c
