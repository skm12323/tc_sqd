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
    postselect_by_hamming_weight,
)
from .subsampling import subsample
from .fermion import (
    bitstring_matrix_to_ci_strs,
    solve_sci,
    SCIResult,
    _int_to_bits,
)

__all__ = ["solve_sqd"]


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
        if num_batches > 1:
            recovered_blocks = []
            for bsm, p in bsm_list:
                rec, _ = recover_configurations(
                    bsm, p, (occ_a, occ_b), na, nb, rand_seed=seed
                )
                recovered_blocks.append(rec)
            all_recovered = (
                np.vstack(recovered_blocks) if recovered_blocks else bsm_all
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
