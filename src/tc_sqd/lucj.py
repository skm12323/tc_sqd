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

__all__ = [
    "get_ccsd_amplitudes",
    "build_lucj_circuit",
    "optimize_ansatz_parameters",
    "circuit_stats",
    "lucj_report",
    "ucj_decomposition",
    "ucj_matrix_energy",
    "ucj_subspace_energy",
]

# 门集合 (TC gate_summary 的键; 按作用 qubit 数分类)。
# 真机深度预算以 2Q 门数为主 (2Q 层主导 depth, 腾讯 qcloud 有 depth 上限)。
_1Q_GATES = {
    "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz",
    "u", "u1", "u2", "u3", "p", "r", "sx", "sxdg", "id", "i",
}
_2Q_GATES = {
    "cx", "cnot", "cz", "swap", "iswap", "ch", "csx", "crx", "cry",
    "crz", "cu", "cu1", "cu3", "cp", "ecr", "rxx", "ryy", "rzz",
    "xx_plus_yy",
}
_MULTI_GATES = {"ccx", "toffoli", "cswap", "ccz", "fredkin"}


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

    Notes
    -----
    结果缓存在 ``mf._tc_sqd_ccsd``, 供多次调用的变分流程复用 (如
    :func:`optimize_ansatz_parameters` 每次评估都要建电路, 但 CCSD 只需一次)。
    **mf 的轨道/积分变化后缓存不会自动失效**, 需要重建 mf。
    """
    cached = getattr(mf, "_tc_sqd_ccsd", None)
    if cached is not None:
        return cached

    from pyscf import cc as _cc

    mycc = _cc.CCSD(mf)
    mycc.kernel()
    if not mycc.converged:
        warnings.warn(
            "CCSD did not converge; LUCJ amplitudes may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )
    result = (mycc.t1, mycc.t2, mycc)
    try:
        mf._tc_sqd_ccsd = result
    except (AttributeError, TypeError):
        pass  # 只读/特殊 mf 不可缓存
    return result


def _qubit(spin: str, orb: int, norb: int) -> int:
    """自旋轨道 → TensorCircuit qubit 编号。

    α_i → 2norb-1-i ; β_i → norb-1-i  (q0=MSB, [β|α] 布局)。
    """
    # int() 防御: pyscf 某些版本 (如 2.7) mol.nao_nr() 返回 numpy int,
    # 传给 tensorcircuit 的 c.x(index) 会触发 "Illegal index specification"
    if spin == "a":
        return int(2 * norb - 1 - orb)
    return int(norb - 1 - orb)


def _lucj_pairs_from_t2(t2, norb, nelec, *, ccsd_scale, max_excitations,
                        angle_multiplier, theta_list=None):
    """构造 (strength, spin, occ_orb, vir_orb, theta) 列表。

    - 每个 occ-vir 对 (i,a) 产生 alpha + beta 两个 entry, 角度由 t2 范数推导
      (H2 的 t1≈0 / Brillouin, 相关能几乎全来自 t2);
    - ``max_excitations`` 裁剪: 按强度降序取前 K 个 entry;
    - ``theta_list`` 变分覆盖: 长度必须等于激活 entry 数, 覆盖推导角度
      (ccsd_scale/angle_multiplier 此时被忽略)。
    """
    nocc = nelec[0]
    nvir = norb - nocc
    pairs = []
    for i in range(nocc):
        for a in range(nvir):
            strength = float(np.linalg.norm(t2[i, :, a, :])) if t2 is not None else 0.0
            theta = ccsd_scale * angle_multiplier * strength
            pairs.append((strength, "a", i, nocc + a, theta))
            pairs.append((strength, "b", i, nocc + a, theta))

    if max_excitations is not None:
        pairs.sort(key=lambda x: -x[0])
        pairs = pairs[:max_excitations]

    if theta_list is not None:
        theta_list = list(theta_list)
        if len(theta_list) != len(pairs):
            raise ValueError(
                f"theta_list 长度 {len(theta_list)} != 激活 entry 数 {len(pairs)} "
                f"(max_excitations={max_excitations}, nocc={nocc}, nvir={nvir})。"
            )
        pairs = [
            (s, sp, o, v, float(theta_list[i]))
            for i, (s, sp, o, v, _) in enumerate(pairs)
        ]
    return pairs


def build_lucj_circuit(
    mf,
    norb: int,
    nelec: Tuple[int, int],
    *,
    ccsd_scale: float = 1.0,
    max_excitations: Optional[int] = None,
    angle_multiplier: float = 2.0,
    theta_list: Optional[list] = None,
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
    theta_list : list[float] | None
        **变分入口**: 覆盖 t2 推导的角度, 长度必须等于激活 entry 数
        (= min(max_excitations, 2·nocc·nvir) 或全量)。给出后
        ``ccsd_scale`` / ``angle_multiplier`` 被忽略。配合
        :func:`optimize_ansatz_parameters` 做 SQD+VQE 优化。

    Returns
    -------
    tensorcircuit.Circuit
        可直接交给 ``tc_sqd.sample_from_circuit`` 采样。

    Raises
    ------
    ValueError
        若 nelec 非闭壳层 (当前仅支持 n_alpha == n_beta);
        或 theta_list 长度与激活 entry 数不符。

    Notes
    -----
    **开壳层 (n_α≠n_β)**: LUCJ 暂保持闭壳层 (见 P2-1 决议)。开壳层请用 HF 电路
    (``tc.Circuit`` 手动设 α/β 占据) 或用户自带电路采样, 交给 SQD 核心
    (``solve_sci`` 原生支持 (na,nb) 不等); 开壳层 LUCJ (α/β 不同占据的 Givens)
    列为后续工作。
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

    # 2) occ-vir 对 + 角度 (t2 推导或 theta_list 变分覆盖)
    pairs = _lucj_pairs_from_t2(
        t2, norb, nelec, ccsd_scale=ccsd_scale,
        max_excitations=max_excitations, angle_multiplier=angle_multiplier,
        theta_list=theta_list,
    )

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


# --------------------------------------------------------------------------- #
#  SQD + VQE 混合优化: LUCJ 角度变分, SQD 能量作损失
# --------------------------------------------------------------------------- #
def optimize_ansatz_parameters(
    mf,
    h1e,
    eri,
    norb: int,
    nelec: Tuple[int, int],
    *,
    ecore: float = 0.0,
    n_samples: int = 2000,
    max_excitations: Optional[int] = None,
    num_restarts: int = 10,
    maxiter: int = 100,
    seed: int = 42,
    n_seeds: int = 1,
    max_iterations: int = 3,
    verbose: bool = False,
) -> dict:
    """SQD+VQE 混合优化: Nelder-Mead 优化 LUCJ Givens 角度, 目标 = SQD 能量。

    VQE 的变分原理 (参数优化) 与 SQD 的子空间对角化 (误差吸收) 互补: 电路角度
    作为变分参数, 以**真实采样后的 SQD 总能量**为损失 (而非期望值), 优化器
    搜索使 SQD 能量最小的角度。这是 tc_sqd 对"结合 qiskit + tc 优势"的落地之一。

    **确定性**: 固定 ``seed`` 使每次目标评估 (采样 + recover + 对角化) 完全
    可复现, Nelder-Mead 可稳定收敛。代价: 结果可能过拟合该 seed 的采样;
    建议 ``num_restarts`` 多次重启并取最优。

    Parameters
    ----------
    mf : pyscf scf.RHF 对象
    h1e, eri, norb, nelec, ecore
        分子积分与电子数 (SQD 输入)。
    n_samples : int
        每次评估的采样数 (shots)。更大更稳但更慢。
    max_excitations : int | None
        激活的 occ-vir entry 数 (None = 全部 2·nocc·nvir)。
    num_restarts : int
        Nelder-Mead 重启次数 (从 CCSD 初始角度 + 扰动出发)。
    maxiter : int
        每次重启的 Nelder-Mead 最大迭代。
    seed : int
        随机种子 (采样 + recover + 重启扰动)。
    n_seeds : int
        **每次目标评估的采样种子数**。=1 (默认) 时目标 = 单 seed 的 SQD 能量
        (确定但可能**过拟合该 seed 的统计涨落**); >1 时目标 = ``n_seeds`` 个
        不同 seed 的 SQD 能量平均 (跨 seed 稳健, 消除过拟合, 但每次评估成本
        ×n_seeds)。推荐 3-5。
    max_iterations : int
        SQD 迭代轮数 (传给 diagonalize)。

    Returns
    -------
    dict
        ``theta`` 最优角度 (与 build_lucj_circuit(theta_list=...) 兼容);
        ``energy`` 最优 SQD 总能量 (含 ecore, n_seeds>1 时为多 seed 均值);
        ``energies`` 每重启的最优能量历史;
        ``n_params`` 参数数; ``method="sqd+vqe"``。
    """
    from .counts import sample_from_circuit
    from .fermion import compute_ground_state_energy

    _t1, t2, _mycc = get_ccsd_amplitudes(mf)
    pairs0 = _lucj_pairs_from_t2(
        t2, norb, nelec, ccsd_scale=1.0,
        max_excitations=max_excitations, angle_multiplier=2.0,
    )
    theta0 = np.array([p[4] for p in pairs0], dtype=np.float64)
    K = theta0.size
    if K <= 0:
        raise ValueError(
            f"无可优化参数 (K={K}): 检查 nelec/norb/max_excitations。"
        )

    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}.")

    def _objective(theta: np.ndarray) -> float:
        # 多 seed 平均: 消除固定 seed 的过拟合 (见 docstring n_seeds)
        es = []
        for s in range(n_seeds):
            seed_s = int(seed) + s
            c = build_lucj_circuit(
                mf, norb, nelec, ccsd_scale=1.0,
                max_excitations=max_excitations, theta_list=list(theta),
            )
            bsm, probs = sample_from_circuit(
                c, n_samples=n_samples,
                random_generator=np.random.default_rng(seed_s),
            )
            es.append(compute_ground_state_energy(
                h1e, eri, norb, nelec, ecore=ecore, method="sqd",
                bitstring_matrix=bsm, probabilities=probs,
                max_iterations=max_iterations, seed=seed_s,
            ))
        return float(np.mean(es))

    from scipy.optimize import minimize

    best_energy = np.inf
    best_theta = theta0.copy()
    energies = []
    rng = np.random.default_rng(seed)
    for restart in range(num_restarts):
        if restart == 0:
            x0 = theta0
        else:
            x0 = theta0 + 0.1 * rng.standard_normal(K)
        res = minimize(
            _objective, x0, method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-7},
        )
        x = np.asarray(res.x, dtype=np.float64)
        e = _objective(x)
        energies.append(float(e))
        if e < best_energy:
            best_energy, best_theta = e, x.copy()
        if verbose:
            print(f"[sqd+vqe] restart {restart + 1}/{num_restarts}: "
                  f"E = {e:.6f}")

    return {
        "theta": best_theta,
        "energy": float(best_energy),
        "energies": energies,
        "n_params": int(K),
        "method": "sqd+vqe",
    }


# --------------------------------------------------------------------------- #
#  真机深度预算 (腾讯 qcloud 有 depth 上限, 如 1500; 2Q 层主导 depth)
# --------------------------------------------------------------------------- #
def circuit_stats(circuit) -> dict:
    """TC 电路门统计: 1Q / 2Q / 多体门数与总量。

    返回 ``{"n_qubits", "n_1q", "n_2q", "n_multi", "n_gates", "gate_summary"}``。
    真机深度预算以 ``n_2q`` 为主: 编译映射后 depth 主要由 2Q 层决定, 故
    ``n_2q`` 是保守深度代理 (串行下界; 实际可并行时会更小)。
    """
    gs = dict(circuit.gate_summary())
    n_1q = n_2q = n_multi = 0
    for name, cnt in gs.items():
        base = name.lower()
        if base in _1Q_GATES:
            n_1q += cnt
        elif base in _2Q_GATES:
            n_2q += cnt
        else:
            n_multi += cnt  # 未知门按多体兜底 (保守)
    return {
        "n_qubits": int(circuit._nqubits),
        "n_1q": int(n_1q),
        "n_2q": int(n_2q),
        "n_multi": int(n_multi),
        "n_gates": int(circuit.gate_count()),
        "gate_summary": gs,
    }


def lucj_report(mf, norb: int, nelec: Tuple[int, int], *,
                ccsd_scale: float = 1.0,
                max_excitations: Optional[int] = None,
                max_depth: Optional[int] = None) -> dict:
    """构建 LUCJ 电路并返回真机深度预算报告 (对接 qcloud 深度上限)。

    Parameters
    ----------
    mf, norb, nelec, ccsd_scale, max_excitations
        与 :func:`build_lucj_circuit` 相同。
    max_depth : int | None
        真机深度预算 (如 1500)。None = 不检查。

    Returns
    -------
    dict
        ``circuit_stats`` 门统计 (含 ``n_2q`` 保守深度代理);
        ``depth_proxy``  = ``n_2q`` (保守 2Q 深度);
        ``within_budget``: bool | None —— ``max_depth`` 给定时 ``n_2q``
        是否在预算内 (None = 未给预算);
        ``max_entries_by_2q_budget``: int | None —— 2Q 门数预算下最多能放
        几个 occ-vir entry (每个 entry 贡献 1 个 cnot; None = 未给预算)。
    """
    c = build_lucj_circuit(mf, norb, nelec, ccsd_scale=ccsd_scale,
                           max_excitations=max_excitations)
    stats = circuit_stats(c)
    depth_proxy = int(stats["n_2q"])
    if max_depth is None:
        within = None
        max_entries = None
    else:
        within = bool(depth_proxy <= max_depth)
        max_entries = max(0, int(max_depth))   # 每 entry 1 cnot (2Q), 预算即上限
    return {
        "circuit_stats": stats,
        "depth_proxy": depth_proxy,
        "within_budget": within,
        "max_entries_by_2q_budget": max_entries,
    }


# --------------------------------------------------------------------------- #
#  P2-2a: UCJ 精确分解 (对标 ffsim UCJOpSpinBalanced.from_t_amplitudes)
#  |Ψ_UCJ> = Π_ℓ (Û_ℓ e^{iĴ_ℓ} Û_ℓ†) |HF>,  Û=e^κ,  Ĵ=Σ J_pq n_p n_q
# --------------------------------------------------------------------------- #
def ucj_decomposition(t2, norb: int, nocc: int, nlayers: int = 1,
                      scale: float = 1.0):
    """t2 (nocc,nocc,nvir,nvir) -> 多层 (kappa, J) UCJ 参数 (闭壳层)。

    对标 ffsim ``UCJOpSpinBalanced.from_t_amplitudes``: 每层对 t2 的
    occ-vir 对矩阵 (``mat[(i,a),(j,b)] = t2[i,j,a,b]``) 做 SVD, 取 top-k
    (k = min(nocc,nvir)) 奇异分量, 构造:
      - ``kappa`` (norb,norb) 实 anti-Hermitian 轨道旋转生成元 (Û = expm(kappa));
      - ``J`` (norb,norb) 对角 Coulomb (e^{iĴ} = ∏ exp(i J_pq n_p n_q))。
    每层剥离已分解的 top-k 分量到残差, 供下一层继续提取 (nlayers 收敛)。

    Parameters
    ----------
    t2 : ndarray (nocc,nocc,nvir,nvir)
        CCSD 双激发振幅 (空间轨道, 如 ``get_ccsd_amplitudes`` 的 t2)。
    norb : int
        空间轨道数 (norb = nocc + nvir)。
    nocc : int
        占据轨道数 (闭壳层)。
    nlayers : int
        UCJ 层数。每层提取一组 (kappa, J)。默认 1。
    scale : float
        kappa 整体缩放 (CCSD t2 振幅通常 ~1e-2, 直接作轨道旋转角太小;
        放大到 O(1) 才产生有效双激发混合, 类似 build_lucj_circuit 的
        angle_multiplier)。J 不缩放 (对角相位对 |ψ|² 无影响, 但影响干涉)。

    Returns
    -------
    list[(kappa, J)]
        每项 ``(kappa, J)`` 均为 (norb, norb); ``layers`` 顺序从内到外作用
        (层 0 先作用)。

    Notes
    -----
    **诚实标注**: 本实现是 **UCJ-inspired 简化 SVD** —— kappa 从 occ-vir 左奇异
    向量 × 奇异值构造、J 对角为启发式分配, **非 ffsim ``UCJOpSpinBalanced``
    的精确实现**。子空间路径 (:func:`ucj_subspace_energy`) 已验证有效 (H₂ = FCI、
    LiH 趋近 FCI); 单态期望 (:func:`ucj_matrix_energy`) 仅参考 (见其 docstring
    局限)。
    """
    nvir = norb - nocc
    t2 = np.asarray(t2, dtype=np.float64)
    if t2.shape != (nocc, nocc, nvir, nvir):
        raise ValueError(
            f"t2 shape {t2.shape} != (nocc,nocc,nvir,nvir)="
            f"{(nocc, nocc, nvir, nvir)}"
        )
    if nlayers < 1:
        raise ValueError(f"nlayers must be >= 1, got {nlayers}.")

    layers = []
    residual = t2.copy()
    for _ in range(nlayers):
        # occ-vir 对索引: mat[(i,a),(j,b)] = residual[i,j,a,b]
        mat = residual.transpose(0, 2, 1, 3).reshape(nocc * nvir, nocc * nvir)
        U, S, Vt = np.linalg.svd(mat)
        k = min(nocc, nvir)
        kappa = np.zeros((norb, norb))
        J = np.zeros((norb, norb))
        for r in range(k):
            u = U[:, r].reshape(nocc, nvir)        # 左奇异向量: occ-vir 耦合权重
            for i in range(nocc):
                for a in range(nvir):
                    kappa[i, nocc + a] += u[i, a] * S[r]
            if r < nvir:
                J[nocc + r, nocc + r] = S[r]       # 对角 Coulomb 分配到 vir 轨道
        kappa = kappa - kappa.T                    # 实 anti-Hermitian
        kappa = kappa * scale                      # 放大 (CCSD t2 振幅太小)
        layers.append((kappa, J))
        # 剥离 top-k 分量到残差
        recon = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        residual = (mat - recon).reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3)
    return layers


def ucj_matrix_energy(layers, h1e, eri, norb: int, nelec):
    """矩阵层验证: UCJ|HF⟩ 的 ⟨Ψ|H|Ψ⟩ (纯矩阵, 不含 SQD/电路)。

    用 PySCF ``transform_ci_for_orbital_rotation`` 应用 Û = expm(kappa),
    对角相位 ``e^{iĴ}`` (逐 determinant 按占据数), 返回 UCJ 态期望能量。
    验证 UCJ 分解对相关空间的覆盖: 层数/参数有效时 ⟨Ψ|H|Ψ⟩ 应显著低于 HF
    且随 nlayers 趋近 FCI。

    Parameters
    ----------
    layers : list[(kappa, J)]
        :func:`ucj_decomposition` 输出 (从内到外作用)。
    h1e, eri, norb, nelec
        分子积分 (闭壳层 nelec=(nocc,nocc))。

    Returns
    -------
    float
        ⟨Ψ|H|Ψ⟩ (含核排斥需调用方加 ecore; 本函数为电子能量)。
    """
    from scipy.linalg import expm
    from pyscf import fci as _fci
    from pyscf.fci import cistring, addons
    from .fermion import build_ci_matrix

    na, nb = nelec
    if na != nb:
        raise ValueError("ucj_matrix_energy 当前仅闭壳层 (na==nb)。")

    ci_strs_a = cistring.make_strings(range(norb), na)
    ci_strs_b = cistring.make_strings(range(norb), nb)
    dim_a, dim_b = len(ci_strs_a), len(ci_strs_b)

    # HF 态在完整 CI 空间
    hf_a, hf_b = (1 << na) - 1, (1 << nb) - 1
    ci = np.zeros((dim_a, dim_b), dtype=complex)
    ia = int(np.where(ci_strs_a == hf_a)[0][0])
    ib = int(np.where(ci_strs_b == hf_b)[0][0])
    ci[ia, ib] = 1.0

    # |Ψ> = Π_ℓ Û_ℓ e^{iĴ_ℓ} Û_ℓ† |HF>   (层 0 先作用)
    for kappa, J in layers:
        U = expm(kappa)                              # Û
        # 闭壳层: α/β 相同轨道旋转 (PySCF 签名 (ci, norb, nelec, u))
        ci = addons.transform_ci_for_orbital_rotation(
            ci, norb, nelec, (U.T.conj(), U.T.conj()))  # Û†
        # e^{iĴ}: 对角相位 per determinant
        for a in range(dim_a):
            for b in range(dim_b):
                sa, sb = int(ci_strs_a[a]), int(ci_strs_b[b])
                occ = [((sa >> p) & 1) + ((sb >> p) & 1) for p in range(norb)]
                phase = 0.0
                for p in range(norb):
                    for q in range(norb):
                        phase += J[p, q] * occ[p] * occ[q]
                ci[a, b] *= np.exp(1j * phase)
        ci = addons.transform_ci_for_orbital_rotation(ci, norb, nelec, (U, U))  # Û

    H = build_ci_matrix(ci_strs_a, ci_strs_b, h1e, eri, norb, nelec)
    cvec = ci.ravel()
    return float((cvec.conj() @ H @ cvec).real)


def ucj_subspace_energy(layers, h1e, eri, norb: int, nelec):
    """矩阵层验证 (确定性 SQD): UCJ|HF⟩ 支持的 det 子空间对角化能量。

    **UCJ 单态期望 ⟨Ψ|H|Ψ⟩ 对单层对角 J 无法低于 HF** (UCJ|HF⟩ 的 Û 旋转会
    引入单激发污染)。UCJ 的真正价值在**子空间对角化**: UCJ 态非零 CI 系数的
    determinant 作子空间, 对角化 H —— 参数有效时覆盖相关 det, 能量接近 FCI
    (H₂ 全空间 → = FCI)。这是 SQD 的确定性 (无采样) 矩阵层版本。

    Parameters
    ----------
    layers : list[(kappa, J)]
        :func:`ucj_decomposition` 输出。
    h1e, eri, norb, nelec
        分子积分 (闭壳层 nelec=(nocc,nocc))。

    Returns
    -------
    float
        子空间对角化基态能量 (电子能量, 不含核排斥)。
    """
    from scipy.linalg import expm
    from pyscf import fci as _fci
    from pyscf.fci import cistring, addons
    from .fermion import build_ci_matrix

    na, nb = nelec
    if na != nb:
        raise ValueError("ucj_subspace_energy 当前仅闭壳层 (na==nb)。")

    ci_strs_a = cistring.make_strings(range(norb), na)
    ci_strs_b = cistring.make_strings(range(norb), nb)
    dim_a, dim_b = len(ci_strs_a), len(ci_strs_b)

    hf_a, hf_b = (1 << na) - 1, (1 << nb) - 1
    ci = np.zeros((dim_a, dim_b), dtype=complex)
    ia = int(np.where(ci_strs_a == hf_a)[0][0])
    ib = int(np.where(ci_strs_b == hf_b)[0][0])
    ci[ia, ib] = 1.0

    for kappa, J in layers:
        U = expm(kappa)
        ci = addons.transform_ci_for_orbital_rotation(
            ci, norb, nelec, (U.T.conj(), U.T.conj()))
        for a in range(dim_a):
            for b in range(dim_b):
                sa, sb = int(ci_strs_a[a]), int(ci_strs_b[b])
                occ = [((sa >> p) & 1) + ((sb >> p) & 1) for p in range(norb)]
                phase = 0.0
                for p in range(norb):
                    for q in range(norb):
                        phase += J[p, q] * occ[p] * occ[q]
                ci[a, b] *= np.exp(1j * phase)
        ci = addons.transform_ci_for_orbital_rotation(ci, norb, nelec, (U, U))

    # 子空间: CI 系数非零的 det (确定性 SQD)。build_ci_matrix 用 α/β 集合的
    # 笛卡尔积, 故对 sub_a/sub_b 取唯一 (重复字符串会让 H 维度虚增、eig 出错)。
    mask = np.abs(ci) > 1e-10
    idx_a, idx_b = np.nonzero(mask)
    if len(idx_a) == 0:
        return float("inf")
    sub_a = np.unique(ci_strs_a[idx_a])
    sub_b = np.unique(ci_strs_b[idx_b])
    H = build_ci_matrix(sub_a, sub_b, h1e, eri, norb, nelec)
    return float(np.linalg.eigvalsh(H)[0])
