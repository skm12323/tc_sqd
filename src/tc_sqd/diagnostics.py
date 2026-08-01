"""tc_sqd.diagnostics —— 采样质量诊断报告。

对 SQD 采样 bitstring 生成一份"采样质量报告": 子空间维度、采样熵、配置分布、
以及能量随 shots 的收敛曲线。帮助判断电路质量与噪声影响:
- 子空间维度太小     -> 电路纠缠不够 (只采到少量 determinant);
- 采样熵异常低       -> 采样坍缩在少数配置 (电路太平凡或过拟合);
- 能量随 shots 不收敛 -> 采样数不够 / 有噪声漏采。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["shannon_entropy", "subspace_dimension", "energy_convergence",
           "sampling_report"]


def shannon_entropy(probs: np.ndarray) -> float:
    """采样概率的香农熵 (nat)。均匀分布最大, 确定性分布为 0。"""
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def subspace_dimension(bsm) -> Tuple[int, int, int]:
    """唯一 alpha/beta 字符串数与行列式对数 (子空间维度)。"""
    from .fermion import bitstring_matrix_to_ci_strs

    bsm = np.asarray(bsm, dtype=bool)
    ci_a, ci_b = bitstring_matrix_to_ci_strs(bsm)
    return len(ci_a), len(ci_b), len(ci_a) * len(ci_b)


def _default_shots_grid(n: int, max_points: int = 6):
    """均匀分布的 shots 子集 (从 1..n, 约 max_points 个点)。"""
    n = int(n)
    if n <= 1:
        return [n]
    step = max(1, n // max_points)
    return sorted(set(range(step, n + 1, step)) | {n})


def energy_convergence(h1e, eri, norb, nelec, bsm, *, probs=None, ecore=0.0,
                       shots_grid=None, seed=42, method="sqd", **kwargs) -> dict:
    """能量随 shots 收敛: 在 shots 子集上重算 SQD 能量。

    对每个 shots 值, 从 ``bsm`` 中按概率无放回抽子集, 跑
    ``compute_ground_state_energy``, 返回收敛曲线。

    Returns
    -------
    dict
        ``{"shots": list, "energies": list, "converged_energy": float}``
    """
    from .fermion import compute_ground_state_energy

    bsm = np.asarray(bsm, dtype=bool)
    n = bsm.shape[0]
    if n == 0:
        raise ValueError("bsm 为空, 无法计算收敛曲线。")
    if shots_grid is None:
        shots_grid = _default_shots_grid(n)
    shots_grid = [min(int(s), n) for s in shots_grid]
    shots_grid = sorted(set(shots_grid) - {0}) or [n]

    rng = np.random.default_rng(seed)
    energies = []
    for s in shots_grid:
        idx = rng.choice(n, size=s, replace=False)
        sub_probs = None
        if probs is not None:
            p = np.asarray(probs, dtype=np.float64)[idx]
            sub_probs = p / p.sum()
        e = compute_ground_state_energy(
            h1e, eri, norb, nelec, ecore=ecore, method=method,
            bitstring_matrix=bsm[idx], probabilities=sub_probs, **kwargs)
        energies.append(float(e))
    return {"shots": list(shots_grid), "energies": energies,
            "converged_energy": float(energies[-1])}


def sampling_report(h1e, eri, norb, nelec, bsm, *, probs=None, ecore=0.0,
                    shots_grid=None, seed=42, **kwargs) -> dict:
    """综合采样诊断报告 (去重合并 + 统计 + 能量收敛曲线)。

    Parameters
    ----------
    h1e, eri, norb, nelec, ecore
        分子积分与电子数 (SQD 输入)。
    bsm : ndarray (S, 2*norb)
        采样 bitstring 矩阵。
    probs : ndarray (S,) | None
        采样概率 (None = 均匀)。
    shots_grid : iterable | None
        收敛曲线的 shots 子集 (None = 自动, 约 6 个点)。
    **kwargs
        透传给 ``compute_ground_state_energy`` (如 ``max_iterations``)。

    Returns
    -------
    dict
        ``n_samples`` / ``n_unique`` 采样数与去重数;
        ``n_alpha_strs`` / ``n_beta_strs`` / ``subspace_dim`` 子空间维度;
        ``entropy_nat`` 采样熵 (nat);
        ``top_configs`` 概率最高的 5 个配置 (bitstring 整数 + 概率);
        ``energy_convergence`` = ``{shots, energies, converged_energy}``。
    """
    from .counts import bitarray_to_int, int_to_bitarray

    bsm = np.asarray(bsm, dtype=bool)
    n = bsm.shape[0]
    if n == 0:
        raise ValueError("bsm 为空。")

    # 去重合并概率
    ints = bitarray_to_int(bsm)
    uniq_ints, inverse = np.unique(ints, return_inverse=True)
    w = (np.ones(n) / n) if probs is None else (np.asarray(probs, dtype=np.float64))
    w = w / w.sum()
    merged = np.zeros(len(uniq_ints))
    np.add.at(merged, inverse, w)
    probs_uniq = merged / merged.sum()
    uniq_bsm = int_to_bitarray(uniq_ints, bsm.shape[1])

    n_alpha, n_beta, dim = subspace_dimension(uniq_bsm)
    entropy = shannon_entropy(probs_uniq)

    order = np.argsort(probs_uniq)[::-1][:5]
    top_configs = [
        {"bitstring": int(uniq_ints[i]), "probability": float(probs_uniq[i])}
        for i in order
    ]

    conv = energy_convergence(
        h1e, eri, norb, nelec, uniq_bsm, probs=probs_uniq, ecore=ecore,
        shots_grid=shots_grid, seed=seed, **kwargs,
    )

    return {
        "n_samples": n,
        "n_unique": int(len(uniq_ints)),
        "n_alpha_strs": int(n_alpha),
        "n_beta_strs": int(n_beta),
        "subspace_dim": int(dim),
        "entropy_nat": entropy,
        "top_configs": top_configs,
        "energy_convergence": conv,
    }
