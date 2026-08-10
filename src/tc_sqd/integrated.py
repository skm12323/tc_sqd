"""Integrated SQD: one-call entry point for the quantum SQD algorithm.

本模块把 tc_sqd 中分散的若干步骤（采样 → 配置恢复 → CI 字符串 → 子空间对角化
→ 多次迭代更新占据数）封装进**单个函数** ``solve_sqd``，并通过 ``mode`` 参数在
「单次运算」(single) 与「多次迭代」(iterative) 两种模式之间切换：

* ``single``     : 配置恢复 → CI 字符串 → 子空间对角化（执行一次）
* ``iterative``  : 在单次运算基础上，用解出的占据数更新平均占据、按概率批量子采样，
                   反复迭代直到 ``max_iterations``（等价于原 ``diagonalize_fermionic_hamiltonian``）

两种模式均直接复用底层已验证的构建块（``recover_configurations``、
``bitstring_matrix_to_ci_strs``、``solve_sci``、``subsample`` 等），返回与
``solve_sci`` / ``diagonalize_fermionic_hamiltonian`` 一致的 ``SCIResult``，因此
总能量为 ``result.energy + ecore``，与全代码库约定保持一致。
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from .counts import sample_from_circuit
from .configuration_recovery import (
    recover_configurations,
    recover_configurations_clustered,
    postselect_by_hamming_weight,
)
from .subsampling import subsample
from .fermion import (
    bitstring_matrix_to_ci_strs,
    solve_sci,
    SCIResult,
    _int_to_bits,
)

__all__ = ["solve_sqd", "solve_sqd_auto", "solve_sqd_best", "solve_sqd_improved"]


def solve_sqd(
    h1e: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    ecore: float = 0.0,
    bitstring_matrix: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    circuit=None,
    n_samples: int = 2000,
    mode: str = "iterative",
    samples_per_batch: Optional[int] = None,
    num_batches: int = 1,
    max_iterations: int = 5,
    seed: Optional[int] = None,
    rand_seed: Optional[int] = None,
    include_configurations: Optional[np.ndarray] = None,
    carryover_threshold: float = 0.0,
    avg_occupancy: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    spin_sq: Optional[float] = None,
    recovery: str = "global",
    n_clusters: int = 4,
    verbose: bool = False,
    **solver_kwargs,
) -> SCIResult:
    """Run the full Sample-based Quantum Diagonalization (SQD) algorithm.

    把之前分散在 ``counts`` / ``configuration_recovery`` / ``fermion`` 等模块里的
    步骤打包成一个调用。

    Parameters
    ----------
    h1e, eri : ndarray
        MO 基（或旋转后）单 / 双电子积分。``h1e`` 形状 ``(norb, norb)`` 或
        ``(2, norb, norb)``（自旋分辨，但要求 alpha == beta）。
    norb : int
        空间轨道数。
    nelec : (n_alpha, n_beta)
        电子数。
    ecore : float
        核排斥能；总能量 = ``result.energy + ecore``。
    bitstring_matrix : ndarray, shape (S, 2*norb), optional
        已采样的比特串矩阵（bool）。与 ``circuit`` 二选一；若都给则优先用
        ``bitstring_matrix``。
    probabilities : ndarray, shape (S,), optional
        对应概率（可非归一化）。省略时按均匀分布处理。
    circuit : tensorcircuit.Circuit, optional
        若提供且未给 ``bitstring_matrix``，则先 ``sample_from_circuit`` 采样。
    n_samples : int
        ``circuit`` 模式下的采样数。
    mode : {"single", "iterative"}
        ``single``      —— 单次运算（恢复一次 + 对角化一次）。
        ``iterative``   —— 多次迭代（更新占据数 + 可选批量子采样）。
    samples_per_batch : int, optional
        ``num_batches > 1`` 时每批子采样的比特串数。
    num_batches : int
        子采样批数（``iterative`` 模式，``>1`` 时启用批量子采样）。
    max_iterations : int
        迭代次数下限（``iterative`` 模式）。``single`` 模式忽略。
    seed, rand_seed : int, optional
        ``seed`` 用于配置恢复的随机性；``rand_seed`` 用于批量子采样。
    include_configurations : ndarray, optional
        强制纳入子空间的确定性比特串（如 HF determinant）。
    carryover_threshold : float, in [0, 1]
        振幅阈值 carryover（与 ``diagonalize_fermionic_hamiltonian`` 语义一致, B4 统一）:
        保留上一轮解态 ``|c| >= carryover_threshold·max|c|`` 的 det, 确定性注入下一轮
        子空间 (0 = 不启用 carryover)。
    avg_occupancy : (ndarray, ndarray), optional
        初始平均占据数 ``(occ_a, occ_b)``；省略时退化为 HF 占据。
    spin_sq :
        透传给 ``solve_sci`` 的目标 S^2 约束（``None`` = 无约束）。
    recovery : {"global", "clustered"}
        配置恢复策略。``"global"`` (默认) 用单一平均占据向量；``"clustered"``
        (CSQD, arXiv:2603.09346) 把每自旋位串池按 weighted k-modes 分成
        ``n_clusters`` 簇、每簇各自参考占据做恢复——保留多占据模式、利好
        强关联体系。``"clustered"`` 时 ``avg_occupancy`` 被忽略（参考占据由
        聚类统计产生）。

        **推荐用法**：``recovery="clustered"`` 配合 ``mode="single"`` 使用。
        在 N₂/cc-pVDZ 强关联基准上，clustered single 误差降低 1.3–2.8×。
        ``mode="iterative"`` 下 occupancy refinement 与聚类语义存在张力
        (迭代循环的单模式收敛会抵消聚类的多模式保留)，此时 clustered 不
        必优于 global；迭代精化场景建议用 ``recovery="global"``。
    n_clusters : int
        ``recovery="clustered"`` 时每自旋的簇数（默认 4）。
    verbose : bool
        打印每次迭代的能量与子空间维度。

    Returns
    -------
    SCIResult
        与 ``solve_sci`` 相同；总能量用 ``result.energy + ecore`` 取得。

    Raises
    ------
    ValueError
        ``mode`` 非法、比特串宽度与 ``norb`` 不符、``max_iterations <= 0``、
        ``carryover_threshold`` 越界等。

    See Also
    --------
    tc_sqd.fermion.compute_ground_state_energy :
        积分→能量 的快速单入口 (采样外部提供, 返回 float)。
        **分工**: 本函数 = 端到端 (含电路采样、single/iterative、返回 SCIResult);
        只拿能量数字用 ``compute_ground_state_energy``。
    """
    if mode not in ("single", "iterative"):
        raise ValueError(
            f"mode must be 'single' or 'iterative', got {mode!r}."
        )
    if recovery not in ("global", "clustered"):
        raise ValueError(
            f"recovery must be 'global' or 'clustered', got {recovery!r}."
        )

    # ---- 1) 取得比特串矩阵（采样或直接传入） ----
    if bitstring_matrix is None:
        if circuit is not None:
            bitstring_matrix, probabilities = sample_from_circuit(
                circuit, n_samples=n_samples
            )
        else:
            raise ValueError(
                "solve_sqd requires either 'bitstring_matrix' or 'circuit'."
            )

    bsm_all = np.asarray(bitstring_matrix, dtype=bool)
    if bsm_all.ndim != 2 or bsm_all.shape[1] != 2 * norb:
        raise ValueError(
            f"bitstring_matrix must have shape (S, 2*norb={2 * norb}), "
            f"got {bsm_all.shape}."
        )

    if probabilities is None:
        probs_all = np.full(bsm_all.shape[0], 1.0 / bsm_all.shape[0])
    else:
        probs_all = np.asarray(probabilities, dtype=np.float64)
        if probs_all.shape[0] != bsm_all.shape[0]:
            raise ValueError(
                "probabilities length must match bitstring_matrix row count."
            )

    na, nb = nelec

    # ---- 2) 初始平均占据数 ----
    if avg_occupancy is not None:
        occ_a, occ_b = avg_occupancy
        occ_a = np.asarray(occ_a, dtype=np.float64)
        occ_b = np.asarray(occ_b, dtype=np.float64)
    else:
        occ_a = np.zeros(norb, dtype=np.float64)
        occ_a[:na] = 1.0
        occ_b = np.zeros(norb, dtype=np.float64)
        occ_b[:nb] = 1.0

    # ---- 3) 共用的一次"恢复 + 对角化"步骤 ----
    def _recover_and_solve(bsm, probs):
        if recovery == "clustered":
            recovered, _ = recover_configurations_clustered(
                bsm, probs, na, nb,
                n_clusters=n_clusters, rand_seed=seed,
            )
        else:
            recovered, _ = recover_configurations(
                bsm, probs, (occ_a, occ_b), na, nb, rand_seed=seed
            )
        if include_configurations is not None:
            inc = np.asarray(include_configurations, dtype=bool)
            recovered = np.vstack([recovered, inc])
        ci_a, ci_b = bitstring_matrix_to_ci_strs(recovered)
        return solve_sci(
            (ci_a, ci_b),
            h1e,
            eri,
            norb,
            nelec,
            spin_sq=spin_sq,
            **solver_kwargs,
        )

    # ================= 单次运算 =================
    if mode == "single":
        result = _recover_and_solve(bsm_all, probs_all)
        if verbose:
            print(f"[SQD:single] E(elec) = {result.energy:.8f} "
                  f"(total = {result.energy + ecore:.8f})")
        return result

    # ================= 多次迭代 =================
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive for iterative mode.")
    if carryover_threshold < 0.0 or carryover_threshold > 1.0:
        raise ValueError("carryover_threshold must lie in [0, 1].")
    if num_batches > 1:
        if samples_per_batch is None or samples_per_batch <= 0:
            raise ValueError(
                "samples_per_batch must be positive when num_batches > 1."
            )

    # 预生成各批比特串（若启用批量子采样）。批内恢复使用真实采样概率
    # (subsample return_probs=True), 不再丢弃原始 probs (B4 修复)。
    bsm_list = []
    if num_batches > 1:
        bsm_list = subsample(
            bsm_all, probs_all, samples_per_batch, num_batches,
            rand_seed=rand_seed, return_probs=True,
        )

    result: Optional[SCIResult] = None
    # 振幅阈值 carryover (与 diagonalize_fermionic_hamiltonian 语义一致, B4 统一):
    # 保留上一轮解态 |c| >= carryover_threshold·max|c| 的 det, 确定性注入下一轮
    # 子空间 —— 保留高置信配置, 替代原 Hamming-weight postselect (采样层语义)。
    carryover_bsm: Optional[np.ndarray] = None
    for iteration in range(max_iterations):
        # CSQD ("clustered"): every iteration re-clusters from the bitstring
        # pool (which grows via carryover).  occ refinement is used for
        # convergence checking only, not fed to recovery.
        use_clustered = (recovery == "clustered")
        if num_batches > 1:
            recovered_blocks = []
            for bsm, p in bsm_list:
                if use_clustered:
                    rec, _ = recover_configurations_clustered(
                        bsm, p, na, nb,
                        n_clusters=n_clusters, rand_seed=seed,
                    )
                else:
                    rec, _ = recover_configurations(
                        bsm, p, (occ_a, occ_b), na, nb, rand_seed=seed
                    )
                recovered_blocks.append(rec)
            all_recovered = (
                np.vstack(recovered_blocks) if recovered_blocks else bsm_all
            )
        else:
            if use_clustered:
                all_recovered, _ = recover_configurations_clustered(
                    bsm_all, probs_all, na, nb,
                    n_clusters=n_clusters, rand_seed=seed,
                )
            else:
                all_recovered, _ = recover_configurations(
                    bsm_all, probs_all, (occ_a, occ_b), na, nb, rand_seed=seed
                )

        rec_for_solve = all_recovered
        if include_configurations is not None:
            inc = np.asarray(include_configurations, dtype=bool)
            rec_for_solve = np.vstack([all_recovered, inc])
        if carryover_bsm is not None:
            rec_for_solve = np.vstack([rec_for_solve, carryover_bsm])

        ci_a, ci_b = bitstring_matrix_to_ci_strs(rec_for_solve)
        result = solve_sci(
            (ci_a, ci_b),
            h1e,
            eri,
            norb,
            nelec,
            spin_sq=spin_sq,
            **solver_kwargs,
        )

        if verbose:
            print(f"[SQD:iter {iteration + 1}/{max_iterations}] "
                  f"dim={ci_a.shape[0] * ci_b.shape[0]} "
                  f"E(elec) = {result.energy:.8f} "
                  f"(total = {result.energy + ecore:.8f})")

        # 用解出的占据数更新平均占据，进入下一次迭代
        occ_a, occ_b = result.sci_state.orbital_occupancies()
        # 防护: 退化子空间 (如恢复后仅 1 个行列式) 可能产出 NaN/越界占据,
        # clip 回 [0,1] 避免下一轮 recover_configurations 校验失败。
        occ_a = np.clip(np.nan_to_num(occ_a, nan=0.0), 0.0, 1.0)
        occ_b = np.clip(np.nan_to_num(occ_b, nan=0.0), 0.0, 1.0)

        # 振幅阈值 carryover: 提取解态大振幅 det (与 fermion.diagonalize 一致)
        if carryover_threshold > 0.0:
            amps = np.abs(np.asarray(result.sci_state.amplitudes))
            keep = amps >= carryover_threshold * amps.max()
            ia, ib = np.nonzero(keep)
            st = result.sci_state
            carry_rows = []
            for a_i, b_i in zip(ia, ib):
                bits_a = _int_to_bits(int(st.ci_strs_a[a_i]), norb)[::-1]
                bits_b = _int_to_bits(int(st.ci_strs_b[b_i]), norb)[::-1]
                carry_rows.append(np.concatenate([bits_b, bits_a]))
            carryover_bsm = (
                np.array(carry_rows, dtype=bool) if carry_rows else None
            )
        else:
            carryover_bsm = None

    assert result is not None
    return result


# --------------------------------------------------------------------------- #
#  自动 SQD 流程 (方向 E): 超参推荐 -> 采样 -> 自适应收敛 -> PT2 能量修正
# --------------------------------------------------------------------------- #
def solve_sqd_auto(
    h1e: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    ecore: float = 0.0,
    circuit=None,
    bitstring_matrix: Optional[np.ndarray] = None,
    T1_us: Optional[float] = None,
    t_gate_ns: Optional[float] = None,
    target: Optional[float] = None,
    shots_budget: Optional[int] = None,
    shots_step: Optional[int] = None,
    energy_tol: Optional[float] = None,
    max_strings: Optional[int] = None,
    n_active_per_round: Optional[int] = None,
    extrapolate_ev: bool = True,
    correction: Optional[str] = None,
    seed: Optional[int] = 0,
    verbose: bool = False,
    return_details: bool = False,
) -> Union[float, dict]:
    """自动 SQD 流程: 超参推荐 → 采样 → 自适应收敛 → (可选) PT2 能量修正。

    把"推荐 + 执行 + 收敛判断 + 精度提升"串成一条流水线 (工程自动化入口):
      1. **超参推荐**: 给真机参数 (``T1_us``/``t_gate_ns``) 时用
         :func:`recommend_sqd_params` 自动取 ``shots``/``depth``/``max_strings``
         /``n_active_per_round``; 不给则用调用方传入值或库默认。
      2. **采样**: 给 ``circuit`` 用 ``sample_from_circuit``; 否则随机位串
         (经典自举)。
      3. **自适应收敛**: :func:`solve_sqd_active` 的 B1 预算闭环 (增量采样 +
         ``energy_tol`` 停采), 自动判断收敛并记录轨迹。
      4. **PT2 能量修正**: 默认用轨迹末点的 Epstein-Nesbet 修正 ``E+E_PT2``
         (方向 D, SHCI 式, 行为良好; σ² 线性外推实测过冲, 不启用)。修正残余
         截断误差 (不增大维度)。

    Parameters
    ----------
    h1e, eri, norb, nelec, ecore
        分子积分与电子数 (SQD 输入)。
    circuit : tensorcircuit.Circuit | None
        采样电路。``None`` = 随机位串自举 (经典模拟; 真机用 ``sample_on_hw``
        先采样再传 ``bitstring_matrix``)。
    bitstring_matrix : ndarray (S, 2*norb) | None
        预采样位串 (优先级最高; 与 ``circuit`` 二选一)。
    T1_us, t_gate_ns : float | None
        真机校准 (启用超参推荐; 否则用传入的 shots 预算)。
    target : float | None
        推荐目标精度 (默认化学精度)。
    shots_budget, shots_step : int | None
        B1 预算与增量步长。``None`` = 由推荐 (有 T1) 或默认 2000/300 给出。
    energy_tol : float | None
        能量收敛停采阈值。``None`` = 默认 1e-5 (自选停止)。
    max_strings : int | None
        子空间维度上限 (覆盖推荐)。
    n_active_per_round : int | None
        每轮 PT2 选态注入数 (覆盖推荐)。
    extrapolate_ev : bool
        ``True`` (默认) 用轨迹末点做 PT2 能量修正 ``E+E_PT2`` (方向 D, 行为
        良好); ``False`` 返回 active 直接能量。
    seed : int | None
        随机种子。
    verbose : bool
        打印每轮进度。
    return_details : bool
        ``True`` 返回 dict (见下)。

    Returns
    -------
    float | dict
        ``return_details=False``: 能量 (PT2 修正版若启用, 否则 active 直接能量;
        含 ``ecore``)。``return_details=True``: ``{"energy", "E_direct", "E_ev",
        "shots_used", "recommendation" (SqdParams | None), "trajectory",
        "converged", "n_rounds"}``。
    """
    from .cipsi import solve_sqd_active
    from .predict import recommend_sqd_params, CHEMICAL_ACCURACY

    if target is None:
        target = CHEMICAL_ACCURACY

    # 1) 超参推荐 (有硬件参数时)
    recommendation = None
    shots = shots_budget
    if T1_us is not None and t_gate_ns is not None:
        recommendation = recommend_sqd_params(
            norb, nelec, T1_us=T1_us, t_gate_ns=t_gate_ns, target=target)
        if shots is None:
            shots = recommendation.shots
        if max_strings is None:
            max_strings = recommendation.max_strings
        if n_active_per_round is None:
            n_active_per_round = recommendation.n_active_per_round
    if shots is None:
        shots = 2000
    shots_step = shots_step if shots_step is not None else 300
    energy_tol = energy_tol if energy_tol is not None else 1e-5
    if n_active_per_round is None:
        n_active_per_round = 50

    # 2) 采样
    if bitstring_matrix is None:
        if circuit is not None:
            from .counts import sample_from_circuit
            bsm = np.asarray(sample_from_circuit(circuit, int(shots)), dtype=bool)
            probs = np.full(bsm.shape[0], 1.0 / bsm.shape[0])
        else:
            rng = np.random.default_rng(seed)
            bsm = rng.random((int(shots), 2 * norb)) > 0.5
            probs = np.full(int(shots), 1.0 / int(shots))
    else:
        bsm = np.asarray(bitstring_matrix, dtype=bool)
        probs = None

    # 3) 自适应收敛 (B1 预算闭环 + 轨迹)
    traj: list = []
    usage: list = []
    e_direct = solve_sqd_active(
        h1e, eri, norb, nelec,
        bitstring_matrix=bsm, probabilities=probs,
        max_strings=max_strings, n_active_per_round=n_active_per_round,
        ecore=ecore, rand_seed=seed, verbose=verbose,
        shots_budget=int(shots), shots_step=int(shots_step),
        energy_tol=energy_tol, usage=usage, trajectory=traj,
    )
    shots_used = int(usage[0]) if usage else int(shots)

    # 4) 能量修正 (方向 D/③, 用已有轨迹, 不重跑): correction 选 pt2 / evpt2 / none。
    #    extrapolate_ev (旧 bool, 向后兼容): True→"pt2", False→"none"。
    #    evpt2 = within-run trajectory 多点外推 (互异点<2 退化 pt2; 稳健正道用
    #    :func:`solve_sqd_best` 多 shots 外推, 近收敛体系实测 30×)。
    from .diagnostics import extrapolate_ev_pt2
    if correction is None:
        correction = "pt2" if extrapolate_ev else "none"
    if correction not in ("pt2", "evpt2", "none"):
        raise ValueError(f"correction 须为 'pt2'/'evpt2'/'none', got {correction!r}.")
    e_ev = None
    corr_used = correction
    if len(traj) >= 1:
        last = traj[-1]
        if correction == "pt2":
            e_ev = float(last["E"] + last["e_pt2"]) + ecore
        elif correction == "evpt2":
            es = np.asarray([t["E"] for t in traj], dtype=np.float64)
            pts = np.asarray([t["e_pt2"] for t in traj], dtype=np.float64)
            n_distinct = len(np.unique(np.round(pts, decimals=14)))
            if n_distinct < 2 or np.max(np.abs(pts)) < 1e-14:
                e_ev = float(last["E"] + last["e_pt2"]) + ecore
                corr_used = "evpt2→pt2(fallback)"
            else:
                e_inf, _alpha, _r2, _std = extrapolate_ev_pt2(es, pts, degree=1)
                e_ev = float(e_inf) + ecore
                corr_used = "evpt2"
    energy = e_ev if e_ev is not None else float(e_direct)

    # 收敛判定: 末两轮能量变化 < energy_tol 且无 PT2 新 det
    converged = (len(traj) >= 2
                 and abs(traj[-1]["E"] - traj[-2]["E"]) < energy_tol
                 and traj[-1]["e_pt2"] is not None
                 and abs(traj[-1]["e_pt2"]) < 1e-6)
    if verbose:
        print(f"[auto] shots_used={shots_used}/{shots} "
              f"E_direct={e_direct:.8f} "
              f"E_ev={e_ev if e_ev is not None else e_direct:.8f} "
              f"converged={converged} rounds={len(traj)}")

    if not return_details:
        return energy
    return {
        "energy": float(energy),
        "correction": corr_used,
        "E_direct": float(e_direct),
        "E_ev": e_ev,
        "shots_used": shots_used,
        "shots_budget": int(shots),
        "recommendation": recommendation,
        "trajectory": traj,
        "converged": bool(converged),
        "n_rounds": len(traj),
    }


# --------------------------------------------------------------------------- #
#  当前最优 SQD (2026-08-10 跨体系实测最优配置; benchmark/测试用)
# --------------------------------------------------------------------------- #
def solve_sqd_best(
    h1e: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    ecore: float = 0.0,
    bitstring_matrix: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    n_shots: Optional[int] = None,
    max_strings: Optional[int] = None,
    n_active_per_round: int = 30,
    evpt2: bool = True,
    evpt2_scales: Tuple[float, ...] = (0.5, 1.0, 2.0),
    degree: int = 1,
    rand_seed: Optional[int] = 0,
    return_details: bool = False,
    verbose: bool = False,
) -> Union[float, dict]:
    """当前最优 SQD 配置 (2026-08-10 跨体系实测最优; benchmark/测试用)。

    基于 L1/L2 在 n2_ccpvdz + 12,12 的实测, 封装**当前已知最优** SQD 组合:

    - **active** (采样↔PT2 选态双闭环, AS-SQD) + **PT2 修正** (普适基线, 所有体系行为良好);
    - **evpt2 多 shots 外推** (近收敛精修; N₂/cc-pVDZ 10o R=3.0 实测改进 30×): 用
      ``evpt2_scales`` 个不同 shots 各跑一次 active, 取各 trajectory 末轮
      ``(E_V, e_pt2)``, :func:`extrapolate_ev_pt2` 外推 ``E_PT2→0``;
    - **不用** distill / adaptive / UCJ (L1/L2 实测在所测体系无增益或有害)。

    与 :func:`solve_sqd_auto` 的区别: ``auto`` 用 within-run trajectory (B1 预算闭环
    各轮) 做 evpt2, 受限/饱和时常退化; 本函数用**多次独立 active** (不同 shots) 喂外推,
    是 REVIEW「L1 改进方法」实证的稳健多点外推正道 (近收敛体系收益最大)。

    Parameters
    ----------
    n_shots : int | None
        基准 shots (baseline active 的采样数); ``None`` = ``bitstring_matrix`` 行数或 60。
    evpt2_scales : tuple[float]
        evpt2 外推的 shots 缩放 (各 shots = n_shots × scale); 默认 (0.5, 1.0, 2.0) 三点。
        互异 ``e_pt2`` 点 <2 时退化为 baseline PT2 (evpt2 永不劣于 pt2)。
    degree : int
        evpt2 外推多项式次数 (默认 1 = 线性)。

    Returns
    -------
    float | dict
        ``return_details=False``: 最优能量 (evpt2 外推版若启用且非退化, 否则 PT2; 含 ``ecore``)。
        ``return_details=True``: ``{"energy", "E_direct", "E_pt2", "E_evpt2", "dim",
        "evpt2"(alpha/r2/fit_std/n_pts/dims) | None}``。
    """
    from .cipsi import solve_sqd_active
    from .diagnostics import extrapolate_ev_pt2

    if n_shots is None:
        n_shots = bitstring_matrix.shape[0] if bitstring_matrix is not None else 60

    if bitstring_matrix is None:
        bsm0 = np.random.default_rng(rand_seed).random((int(n_shots), 2 * norb)) > 0.5
        probs0 = np.full(int(n_shots), 1.0 / int(n_shots))
    else:
        bsm0 = np.asarray(bitstring_matrix, dtype=bool)
        probs0 = (probabilities if probabilities is not None
                  else np.full(bsm0.shape[0], 1.0 / bsm0.shape[0]))

    def _run(bsm, probs):
        traj: list = []
        E = solve_sqd_active(
            h1e, eri, norb, nelec, ecore=ecore, bitstring_matrix=bsm,
            probabilities=probs, max_strings=max_strings,
            n_active_per_round=n_active_per_round, rand_seed=rand_seed,
            trajectory=traj, verbose=verbose)
        return E, (traj[-1] if traj else None)

    # baseline (n_shots): active 变分 + PT2
    e_direct, last = _run(bsm0, probs0)
    e_pt2 = (float(last["E"] + last["e_pt2"]) + ecore) if last is not None else float(e_direct)
    out: dict = {"E_direct": float(e_direct), "E_pt2": e_pt2,
                 "dim": (last["dim"] if last is not None else None)}

    energy = e_pt2
    if evpt2:
        Es, pts, dims = [], [], []
        for sc in evpt2_scales:
            s = max(int(round(n_shots * float(sc))), 4)
            bsm = np.random.default_rng(rand_seed).random((s, 2 * norb)) > 0.5
            _, lr = _run(bsm, np.full(s, 1.0 / s))
            if lr is None:
                continue
            Es.append(lr["E"]); pts.append(lr["e_pt2"]); dims.append(lr["dim"])
        n_distinct = len(set(round(float(p), 14) for p in pts))
        if len(Es) >= 2 and n_distinct >= 2:
            e_inf, alpha, r2, fit_std = extrapolate_ev_pt2(
                np.asarray(Es), np.asarray(pts), degree=degree)
            e_evpt2 = float(e_inf) + ecore
            out["E_evpt2"] = e_evpt2
            out["evpt2"] = {"alpha": alpha, "r2": r2, "fit_std": fit_std,
                            "n_pts": len(Es), "dims": dims}
            energy = e_evpt2
        else:
            out["E_evpt2"] = None  # 退化: 用 baseline PT2
            out["evpt2"] = None
    else:
        out["E_evpt2"] = None
        out["evpt2"] = None

    if verbose:
        print(f"[best] E_direct={e_direct:.8f} E_pt2={e_pt2:.8f} "
              f"E_evpt2={out.get('E_evpt2')} dim={out['dim']}")
    out["energy"] = energy
    return energy if not return_details else out


def solve_sqd_improved(
    h1e: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    ecore: float = 0.0,
    bitstring_matrix: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    max_strings: Optional[int] = None,
    n_active_per_round: int = 30,
    rand_seed: Optional[int] = 0,
    return_details: bool = False,
    verbose: bool = False,
    **kwargs,
) -> Union[float, dict]:
    """Improved SQD = active (采样↔PT2 选态, AS-SQD) + PT2 修正 (E+E_PT2)。

    图 / SURVEY / REVIEW 中 **"improved SQD"** 的显式入口 —— 此前散在
    :func:`solve_sqd_ev` 的 ``correction="pt2"`` 选项, 无独立函数。本函数固定 PT2 修正
    (Epstein-Nesbet, SHCI 式, **普适行为良好**, 所有体系实测有效)。

    与相关入口的区别:
    - :func:`solve_sqd_active`: 仅 active 变分 (无 PT2), 是本函数的"基";
    - 本函数: active + PT2 修正 (improved SQD);
    - :func:`solve_sqd_best`: 在本函数基础上加 **evpt2 多 shots 外推** (近收敛精修,
      N₂/cc-pVDZ 10o 实测 30×), 是当前最优配置。

    Parameters 与 :func:`solve_sqd_active` 一致 (``ecore``/``bitstring_matrix``/
    ``max_strings``/``n_active_per_round``/``rand_seed`` 等); ``return_details=True``
    返回 ``(energy, details)`` (含 ``E_direct``/``E_PT2``/``dim``/``trajectory``)。
    """
    from .cipsi import solve_sqd_ev
    kwargs.pop("correction", None)  # improved SQD 固定 PT2; 要 evpt2 用 solve_sqd_best
    return solve_sqd_ev(
        h1e, eri, norb, nelec, ecore=ecore, bitstring_matrix=bitstring_matrix,
        probabilities=probabilities, max_strings=max_strings,
        n_active_per_round=n_active_per_round, rand_seed=rand_seed,
        correction="pt2", return_details=return_details, verbose=verbose, **kwargs)
