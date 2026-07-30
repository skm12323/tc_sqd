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

from math import exp, sqrt

# ---- 校准常数 (H4 / STO-3G, 方向1/2 数据最小二乘) ----
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


def max_depth_for_accuracy(T1_us: float, t_gate_ns: float, shots: int,
                           target: float = CHEMICAL_ACCURACY,
                           excited: bool = False) -> int:
    """反向预测: 给定 T1/shots/目标精度, 推最大电路深度 (T1 主导时)。

    excited=True 时按激发态 (3×) 算。返回 depth 上限 (超过则 T1 误差超标)。
    若采样已超 target (KS/√shots > target), 返回 -1 (提 shots 无效, 需先加 shots)。
    """
    eps_sample = KS / sqrt(shots)
    if eps_sample >= target:
        return -1
    budget = target - eps_sample                     # 留给 T1 的预算
    factor = EXCITED_FACTOR if excited else 1.0
    gamma_budget = budget / (factor * KT1)           # ε_T1 = factor*KT1*γ <= budget
    if gamma_budget >= 1:
        return -2                                    # T1 任意都不超 (极乐观)
    # γ = 1-exp(-depth*t_gate/T1) <= gamma_budget  ->  depth <= -ln(1-γ_budget) * T1 / t_gate
    from math import log
    return int(-log(1 - gamma_budget) * T1_us / (t_gate_ns / 1000.0))
