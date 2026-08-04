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
           "sampling_report", "extrapolate_infinite_samples"]


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


def extrapolate_infinite_samples(
    energies,
    shots,
) -> Tuple[float, float, float, float]:
    """无限采样外推 (A1): 拟合 ``E(S) = E∞ + a/√S``, 取 ``E∞``。

    采样能量随 shots S 单调收敛到子空间极限, 且统计收敛主导项 ~ 1/√S
    (采样 det 覆盖随 √S 增长)。对 :func:`energy_convergence` 的 S→E 曲线做
    ``E vs 1/√S`` 线性最小二乘拟合, 外推到 S→∞ 的 ``E∞`` —— 比最大 shots
    点的能量更接近真值 (纯经典后处理, 零额外量子资源)。

    Parameters
    ----------
    energies : array-like, shape (K,)
        ``energy_convergence`` 输出的能量序列 (含 ecore 或电子能量均可, 口径不变)。
    shots : array-like, shape (K,)
        对应采样数序列 (需与 ``energies`` 同序)。

    Returns
    -------
    (e_inf, a, r2, fit_std) : (float, float, float, float)
        ``e_inf`` 外推无限采样能量; ``a`` 斜率 (a/√S 修正); ``r2`` 拟合优度;
        ``fit_std`` 拟合残差标准差 (误差带参考)。

    Notes
    -----
    - 至少 2 个点; 建议 shots 跨度 ≥ 一个数量级 (覆盖充分, 外推更稳)。
    - **对 SQD 子空间能量不适用 (A1 验证证伪)**: SQD 能量是采样 det 覆盖决定的
      变分下界, 非统计量, ``E(S)`` 随覆盖阶梯式收敛而非 ``1/√S`` 平滑收敛
      (N₂ 拉伸实测: 外推反而不如最大 shots 点, 差 ~1000×)。本函数面向**统计量**
      (如期望值测量、噪声外推 ``E(γ)`` 等) 的无限采样/参数外推。
    """
    y = np.asarray(energies, dtype=np.float64)
    s = np.asarray(shots, dtype=np.float64)
    if y.ndim != 1 or len(y) < 2:
        raise ValueError(
            f"至少需要 2 个 (shots, energy) 点, got {len(y)}."
        )
    if len(y) != len(s):
        raise ValueError(
            f"energies 与 shots 长度不一致: {len(y)} vs {len(s)}."
        )
    if np.any(s <= 0):
        raise ValueError("shots 必须为正。")
    # E vs 1/√S 线性拟合: E = e_inf + a·(1/√S)
    x = 1.0 / np.sqrt(s)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    e_inf, a = float(coef[0]), float(coef[1])
    resid = y - (e_inf + a * x)
    fit_std = float(np.sqrt(np.mean(resid**2)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return e_inf, a, r2, fit_std
