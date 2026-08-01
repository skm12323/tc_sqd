"""tc_sqd.predict —— SQD 噪声容限预测器 (基于已验证的噪声鲁棒性机制)。

给定真机校准 (T1) + 电路 (depth/单门时间) + shots, 解析预测 SQD 基态/激发态可达精度。
这是 tc_sqd 的独有工具 (qiskit-addon-sqd 没有), 来自 D:\\explore 方向1/2 的机制发现。

模型 (误差 vs FCI, Ha):
  ε_SQD = ε_sample + ε_T1 + ε_T2(=0) + ε_readout(=0)
    ε_sample    = KS / √shots                 (采样统计涨落)
    ε_T1(基态)  = KT1 × γ_T1                  (振幅阻尼漏 determinant)
    ε_T1(激发)  = EXCITED_FACTOR × KT1 × γ_T1 (激发态 ~3× 基态, 方向2)
    γ_T1        = 1 - exp(-depth × t_gate / T1)
    T2 (退相干) = 免疫 (方向1: 计算基 diag 不变); 读出 = recover 纠正 (CAR)

校准: KS, KT1 从 H4 (norb=4) 数据最小二乘。**注意**: KT1 是 shots 依赖的 (D:\\explore
calibrate_kt1 发现高 shots 下 KT1->0, recover 吸收 T1), 本模型适用于子空间未饱和的
中低 shots 区间; 高 shots 时实际 T1 误差会小于预测 (更乐观)。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

# ---- 校准常数 (H4 / STO-3G, 方向1/2 数据最小二乘) ----
# 注意: KS/KT1 是 H4/STO-3G 小体系的拟合值, **跨体系/基组只应作数量级参考**,
# 不建议当作精确误差界使用。对具体分子建议先用 predict 模块做经典标定
# (在模拟器上跑若干 T1/shots 组合, 拟合自己的 KS/KT1)。
KS = 0.0175          # 采样: ε_sample = KS/√shots
KT1 = 4.7e-3         # 振幅阻尼基态: ε_T1 = KT1 × γ_T1  (Ha per γ, @ 中低 shots)
EXCITED_FACTOR = 3.0  # 激发态 T1 误差 ~3× 基态 (方向2, 低 shots)
CHEMICAL_ACCURACY = 1.6e-3   # 化学精度 (Ha, ≈ 1 kcal/mol)


def gamma_T1(depth: int, t_gate_ns: float, T1_us: float) -> float:
    """真机单 qubit 振幅阻尼率: γ = 1 - exp(-depth × t_gate / T1)。

    depth = 电路深度 (门数), t_gate_ns = 单门时长 (ns), T1_us = T1 (µs)。
    """
    t_circuit_us = depth * t_gate_ns / 1000.0
    return 1.0 - exp(-t_circuit_us / T1_us)


def predict_sqd_error(T1_us: float, depth: int, t_gate_ns: float, shots: int,
                      n_excited: int = 1) -> dict:
    """预测 SQD 基态 + 前 n_excited 激发态的误差 (vs FCI, Ha)。

    Returns
    -------
    dict
        gamma_T1, eps_sample, eps_T1_ground, eps_T2 (0), eps_readout (0),
        ground (基态误差), excited (各激发态误差列表), dominant (主因 "T1"/"sampling"),
        ground_chemical (bool, 基态是否达化学精度), excited_chemical (list[bool])。
    """
    g = gamma_T1(depth, t_gate_ns, T1_us)
    eps_sample = KS / sqrt(shots)
    eps_T1_g = KT1 * g
    eps_T2 = 0.0           # 退相干免疫 (方向1)
    eps_readout = 0.0      # recover 纠正 (CAR)
    ground = eps_sample + eps_T1_g
    excited = [eps_sample + EXCITED_FACTOR * eps_T1_g] * n_excited
    return {
        "gamma_T1": g,
        "eps_sample": eps_sample,
        "eps_T1_ground": eps_T1_g,
        "eps_T2": eps_T2,
        "eps_readout": eps_readout,
        "ground": ground,
        "excited": excited,
        "dominant": "T1" if eps_T1_g > eps_sample else "sampling",
        "ground_chemical": ground < CHEMICAL_ACCURACY,
        "excited_chemical": [e < CHEMICAL_ACCURACY for e in excited],
    }


@dataclass
class DepthBudget:
    """深度预算求解结果 (结构化, 推荐用 :func:`depth_budget` 获取)。

    Attributes
    ----------
    max_depth : int | None
        允许的最大电路深度。``None`` 表示 T1 不构成限制
        (``status`` = ``"sampling_limited"`` 或 ``"t1_unlimited"``)。
    status : str
        ``"ok"``                —— T1 主导, ``max_depth`` 有效;
        ``"sampling_limited"``  —— 采样误差已 ≥ target, 提 shots 比降 depth 更有效;
        ``"t1_unlimited"``      —— T1 误差预算充裕, depth 不受 T1 限制。
    reason : str
        人类可读的判定说明。
    """

    max_depth: "int | None" = None
    status: str = "ok"
    reason: str = ""


def depth_budget(T1_us: float, t_gate_ns: float, shots: int,
                 target: float = CHEMICAL_ACCURACY,
                 excited: bool = False) -> DepthBudget:
    """反向预测: 给定 T1/shots/目标精度, 推最大电路深度预算 (T1 主导时)。

    excited=True 时按激发态 (3×) 算。返回结构化 :class:`DepthBudget`:
    - ``status="ok"``                 -> ``max_depth`` 有效 (超过则 T1 误差超标);
    - ``status="sampling_limited"``   -> 采样误差已 ≥ target, 加 shots 比压 depth 有效;
    - ``status="t1_unlimited"``       -> T1 任意深度都不超 (极乐观, 主要受限采样)。
    """
    eps_sample = KS / sqrt(shots)
    if eps_sample >= target:
        return DepthBudget(
            max_depth=None, status="sampling_limited",
            reason=f"采样误差 ε_sample={eps_sample:.2e} 已 ≥ target={target:.2e}; "
                   f"先加 shots 而不是压 depth。",
        )
    budget = target - eps_sample                     # 留给 T1 的预算
    factor = EXCITED_FACTOR if excited else 1.0
    gamma_budget = budget / (factor * KT1)           # ε_T1 = factor*KT1*γ <= budget
    if gamma_budget >= 1:
        return DepthBudget(
            max_depth=None, status="t1_unlimited",
            reason=f"T1 误差预算充裕 (γ 上限 {gamma_budget:.3f} ≥ 1), "
                   f"depth 不受 T1 限制, 主要受限采样 ε_sample={eps_sample:.2e}。",
        )
    # γ = 1-exp(-depth*t_gate/T1) <= gamma_budget  ->  depth <= -ln(1-γ_budget) * T1 / t_gate
    from math import log
    max_depth = int(-log(1 - gamma_budget) * T1_us / (t_gate_ns / 1000.0))
    return DepthBudget(
        max_depth=max_depth, status="ok",
        reason=f"T1 主导下最大 depth = {max_depth} (γ 上限 {gamma_budget:.4f})。",
    )


def max_depth_for_accuracy(T1_us: float, t_gate_ns: float, shots: int,
                           target: float = CHEMICAL_ACCURACY,
                           excited: bool = False) -> int:
    """反向预测 depth 上限 (向后兼容的 int 薄封装)。

    excited=True 时按激发态 (3×) 算。返回 depth 上限 (超过则 T1 误差超标)。
    哨兵: ``-1`` = 采样误差已 ≥ target (先加 shots); ``-2`` = T1 任意深度都不超。
    需要区分两种哨兵原因时, 请用 :func:`depth_budget` 的结构化返回。
    """
    b = depth_budget(T1_us, t_gate_ns, shots, target, excited)
    if b.status == "sampling_limited":
        return -1
    if b.status == "t1_unlimited":
        return -2
    assert b.max_depth is not None
    return b.max_depth


@dataclass
class SamplingPlan:
    """一个 (shots, depth) 采样方案及其预测误差/成本。

    Attributes
    ----------
    shots, depth : int
        采样数与电路深度。
    error : float
        预测 SQD 总误差 (vs FCI, Ha)。
    eps_sample, eps_T1 : float
        采样误差分量与 T1 误差分量。
    dominant : str
        主误差来源 (``"T1"`` / ``"sampling"``)。
    chemical : bool
        是否达到目标精度 (化学精度)。
    cost : float
        方案成本 = ``shots_cost·shots + depth_cost·depth``。
    """

    shots: int
    depth: int
    error: float
    eps_sample: float
    eps_T1: float
    dominant: str
    chemical: bool
    cost: float


def _geom_grid(lo: float, hi: float, n: int = 9):
    """几何序列网格 (整型, 去重)。"""
    import numpy as np
    vals = set(int(x) for x in np.geomspace(lo, hi, n))
    return sorted(vals)


def plan_sampling(T1_us: float, t_gate_ns: float, *,
                  target: float = CHEMICAL_ACCURACY,
                  excited: bool = False,
                  shots_grid=None, depth_grid=None,
                  shots_cost: float = 1.0, depth_cost: float = 1e-4) -> dict:
    """采样预算分配: 枚举 (shots, depth) 网格, 返回可行方案并按成本排序。

    这是把 :func:`predict_sqd_error` 从"预报"升级为"决策"的入口: 给定真机
    T₁ / 单门时长 / 目标精度, 自动找"哪些 (shots, depth) 组合能达到化学精度,
    哪个最便宜"。

    Parameters
    ----------
    T1_us, t_gate_ns : float
        真机校准 (T₁ in µs, 单门时长 in ns)。
    target : float
        目标误差 (默认化学精度 1.6e-3 Ha)。
    excited : bool
        True 时按激发态误差 (3×) 评估。
    shots_grid, depth_grid : iterable | None
        候选 shots / depth。None = 几何序列默认网格
        (shots: 500..1e6; depth: 20..2000)。
    shots_cost, depth_cost : float
        成本权重 (真机场景 shots 通常主导成本, 默认 depth 权重很小)。

    Returns
    -------
    dict
        ``{"all": list[SamplingPlan],          # 全部组合 (未过滤)
            "feasible": list[SamplingPlan],     # 达到 target 的方案, 按 cost 升序
            "best": SamplingPlan | None,        # 最便宜的可行方案
            "target": target}``
    """
    import numpy as np

    if shots_grid is None:
        shots_grid = _geom_grid(500, 1_000_000)
    if depth_grid is None:
        depth_grid = _geom_grid(20, 2000)
    shots_grid = [int(s) for s in shots_grid]
    depth_grid = [int(d) for d in depth_grid]

    plans = []
    for s in shots_grid:
        for d in depth_grid:
            r = predict_sqd_error(T1_us, d, t_gate_ns, s, n_excited=1)
            eps_T1 = r["eps_T1_ground"] * (EXCITED_FACTOR if excited else 1.0)
            err = r["excited"][0] if excited else r["ground"]
            plans.append(SamplingPlan(
                shots=int(s), depth=int(d), error=err,
                eps_sample=r["eps_sample"], eps_T1=eps_T1,
                dominant=r["dominant"], chemical=bool(err < target),
                cost=shots_cost * int(s) + depth_cost * int(d),
            ))

    feasible = [p for p in plans if p.chemical]
    feasible.sort(key=lambda p: p.cost)
    return {
        "all": plans,
        "feasible": feasible,
        "best": feasible[0] if feasible else None,
        "target": target,
    }
