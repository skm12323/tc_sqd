"""tc_sqd.predict —— SQD 噪声容限预测器 (基于已验证的噪声鲁棒性机制)。

给定真机校准 (T1) + 电路 (depth/单门时间) + shots, 解析预测 SQD 基态/激发态可达精度。
这是 tc_sqd 的独有工具 (qiskit-addon-sqd 没有), 来自早期方向 1/2 的机制发现。

模型 (误差 vs FCI, Ha):
  ε_SQD = ε_sample + ε_T1 + ε_T2(=0) + ε_readout(=0)
    ε_sample    = KS / √shots                 (采样统计涨落)
    ε_T1(基态)  = KT1 × γ_T1                  (振幅阻尼漏 determinant)
    ε_T1(激发)  = EXCITED_FACTOR × KT1 × γ_T1 (激发态 ~3× 基态, 方向2)
    γ_T1        = 1 - exp(-depth × t_gate / T1)
    T2 (退相干) = 免疫 (方向1: 计算基 diag 不变); 读出 = recover 纠正 (CAR)

校准: KS, KT1 从 H4 (norb=4) 数据最小二乘。**注意**: KT1 是 shots 依赖的 (早期实验
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


@dataclass
class SqdParams:
    """SQD 超参数推荐结果 (结构化, 推荐用 :func:`recommend_sqd_params` 获取)。

    Attributes
    ----------
    shots, depth : int
        硬件预算: 采样次数与电路深度 (来自 ``plan_sampling`` 最便宜可行方案)。
    max_strings : int
        子空间维度上限 (对角化维度 ≈ n_str_a × n_str_b 的上界启发式)。
    n_active_per_round : int
        每轮 PT2 选态注入的 top 候选 det 数。
    dom_thresh, pt2_floor : float
        主动采样选态阈值 (主导 det 阈值 / PT2 贡献阈值)。
    predicted_error : float
        模型预测的 SQD 总误差 (vs FCI, Ha)。
    dominant : str
        主误差来源 (``"T1"`` / ``"sampling"``)。
    feasible : bool
        是否有 (shots, depth) 组合能达到目标精度。
    reason : str
        人类可读的推荐依据 (含各参数来源)。
    """

    shots: int
    depth: int
    max_strings: int
    n_active_per_round: int
    dom_thresh: float
    pt2_floor: float
    predicted_error: float
    dominant: str
    feasible: bool
    reason: str


def recommend_sqd_params(norb: int, nelec, *,
                         T1_us: float, t_gate_ns: float,
                         target: float = CHEMICAL_ACCURACY,
                         excited: bool = False,
                         shots_max: int = 1_000_000,
                         max_strings_override: "int | None" = None) -> SqdParams:
    """根据分子体系 + 硬件参数自动推荐 SQD 超参数 (工程自动化入口)。

    **组装已有的决策零件**:
      - 硬件/采样预算: :func:`plan_sampling` 枚举 (shots, depth) 网格, 取**最便宜**
        的可行方案 (达到 ``target`` 精度) -> ``shots`` / ``depth``;
      - 子空间规模: 对角化维度 ≈ ``n_str_a × n_str_b``, 用分子尺寸 (norb, nelec)
        的启发式给 ``max_strings`` 上限 (保持对角化可解);
      - 选态阈值: ``dom_thresh`` / ``pt2_floor`` 用库默认值, ``n_active_per_round``
        按子空间规模缩放。

    **精度模型注意**: ``plan_sampling`` 用 H₄/STO-3G 拟合的 KS/KT1 (见模块头
    校准说明), 跨体系只作**数量级起点**。对具体分子建议用 ``calibrate`` 在
    模拟器上重新标定 KS/KT1, 再喂回本函数 (传 ``target`` 不变即可)。

    Parameters
    ----------
    norb : int
        空间轨道数。
    nelec : tuple(int, int)
        ``(n_alpha, n_beta)``。
    T1_us, t_gate_ns : float
        真机校准 (T₁ in µs, 单门时长 in ns)。
    target : float
        目标误差 (默认化学精度 1.6e-3 Ha)。
    excited : bool
        True 时按激发态误差 (3×) 评估。
    shots_max : int
        shots 上限 (防止推荐超预算)。
    max_strings_override : int | None
        手动指定子空间上限 (覆盖启发式)。

    Returns
    -------
    SqdParams
        结构化推荐 (见类文档)。``feasible=False`` 时 ``shots/depth`` 取上限、
        ``reason`` 说明为何不可行 (目标过紧或噪声预算不足)。
    """
    import numpy as np
    from pyscf.fci import cistring as _cistr

    na, nb = nelec
    full = int(_cistr.num_strings(norb, na))

    # 1) 硬件预算: 最便宜可行 (shots, depth)
    plan = plan_sampling(T1_us, t_gate_ns, target=target, excited=excited)
    best = plan["best"]
    if best is not None:
        shots, depth = int(best.shots), int(best.depth)
        pred_err, dominant = float(best.error), best.dominant
        feasible = True
    else:
        # 无可行方案: 给上限 + 明确警告
        feasible = False
        shots = int(min(shots_max, 1_000_000))
        # 深度: 无 T1 限制时给一个保守中值 (20..2000 网格中位)
        depth = 200
        r_hi = predict_sqd_error(T1_us, depth, t_gate_ns, shots, n_excited=1)
        pred_err = r_hi["excited"][0] if excited else r_hi["ground"]
        dominant = r_hi["dominant"]

    # 2) 子空间规模启发式 (对角化维度 ≈ n_str_a×n_str_b 上界)
    if max_strings_override is not None:
        max_strings = int(max_strings_override)
    else:
        # 经验: dim ≤ ~(25·norb)² 可快速对角化; 大体系需更大 (受限精度优先)
        cap = max(50, min(250, 25 * norb))
        max_strings = min(full, cap)

    # 3) 选态阈值按子空间规模缩放
    n_active = int(max(10, min(50, max_strings // 3)))
    dom_thresh = 1e-3
    pt2_floor = 1e-7

    # 4) 推荐理由
    if feasible:
        reason = (
            f"硬件预算来自 plan_sampling: 最便宜可行方案 shots={shots} "
            f"depth={depth} (预测误差 {pred_err:.2e}, {dominant} 主导)。"
            f"子空间 max_strings={max_strings} (全空间 {full}); "
            f"主动采样每轮注入 n_active={n_active} 个 PT2 候选。"
            f"注: 精度模型基于 H₄/STO-3G, 跨体系建议先 calibrate 再精调。"
        )
    else:
        reason = (
            f"目标 {target:.2e} 在硬件参数 (T1={T1_us}µs, "
            f"t_gate={t_gate_ns}ns) 下无可行 (shots, depth) 组合: "
            f"即使上限 shots={shots} 预测误差仍 {pred_err:.2e} ({dominant} 主导)。"
            f"建议: 增大 shots 上限 / 用 ZNE (solve_sqd_robust) 抑 T1 / "
            f"换基 (solve_sqd_natural_orbitals) 减截断。"
        )
    return SqdParams(
        shots=shots, depth=depth, max_strings=max_strings,
        n_active_per_round=n_active, dom_thresh=dom_thresh,
        pt2_floor=pt2_floor, predicted_error=pred_err,
        dominant=dominant, feasible=feasible, reason=reason,
    )


def calibrate(h1e, eri, norb, nelec, *, ecore: float = 0.0, circuit=None,
              shots_grid=None, gamma_grid=None, n_avg: int = 1,
              seed: int = 42, max_iterations: int = 3) -> dict:
    """跨体系校准 SQD 噪声模型的 KS / KT1 (二元线性最小二乘)。

    **模型**: ``ε = KS/√shots + KT1·γ``, 其中 γ 为振幅阻尼率, ε 为 SQD 总能量
    对 FCI 的误差。默认的 ``KS=0.0175 / KT1=4.7e-3`` 只来自 H₄/STO-3G, 跨体系
    仅数量级; 用本函数在**模拟器上跑你自己的 (shots, γ) 网格**重新拟合。

    T1 注入走现成 noise 链: 态 → ``statevector_to_density`` →
    ``apply_amp_damping`` → ``density_to_bitstring_matrix`` → SQD (模拟器),
    不新造噪声路径。本函数**纯解析 → 需要模拟**, 故对 noise/fermion/pyscf 用
    **函数内 lazy import**, 保持 ``predict`` 模块顶层零重依赖、防循环导入。

    Parameters
    ----------
    h1e, eri, norb, nelec, ecore
        分子积分 (SQD 输入; 与本库其余模块一致)。
    circuit : tensorcircuit.Circuit | None
        **采样电路 (双模式)**:
        - 给定时 —— **实际电路采样** (``sample_from_circuit``) + **位串级 T1**
          (``noise.apply_t1_bitstrings``), 反映真机/带噪 LUCJ 的真实采样 regime,
          与 :func:`predict_sqd_error` 预报的 regime 一致; 测得的 KS/KT1 可直接
          喂回 predict。H₄ LUCJ ≥2000 shots 下测得非零 KS (~1e-2~1e-1 量级)。
        - ``None`` (默认) —— 用 FCI 态密度采样 (**coverage/saturation benchmark**,
          非真机预报用; 高 shots 下子空间饱和, KS→0)。docstring 已标注, 勿把
          此模式的 KS/KT1 直接喂 predict (会系统性低估真机 LUCJ 采样的 KS)。
    shots_grid : list[int] | None
        候选 shots (默认 ``[2000, 4000, 8000]``)。
    gamma_grid : list[float] | None
        候选振幅阻尼率 (默认 ``[0.05, 0.2, 0.4]``)。
    n_avg : int
        每 (shots, γ) 点重复采样次数取平均 (降统计噪声)。
    seed : int
        随机种子 (采样 + recover 确定性)。
    max_iterations : int
        SQD 迭代轮数。

    Returns
    -------
    dict
        ``{"KS", "KT1", "grid_points": [(shots, γ), ...], "errors": ndarray,
        "rmse", "mode": "circuit" | "fci_density"}``。

    Notes
    -----
    - 体系相关: 对给定分子 (h1e/eri/norb/nelec) 校准出的 KS/KT1 是**该体系专属**
      的, 换体系应得到不同值 (即本函数非空操作)。
    - **KS 依赖电路覆盖质量**: LUCJ/真机电路 vs 覆盖充分态, 拟合出的 KS 量级不同,
      **勿锚定历史值 0.0175** (那是方向 1 特定电路的拟合值); 同电路跨体系才有
      可比性。
    - 代价: circuit 模式无需密度矩阵 (位串级), 可跑大体系; fci_density 模式做
      density 矩阵模拟 (2^nq), 只适合小体系 (nq ≲ 12)。
    """
    import numpy as _np
    from pyscf import fci as _fci
    from pyscf.fci import cistring as _cistr
    from . import noise as _noise
    from .fermion import compute_ground_state_energy as _cgse

    if shots_grid is None:
        shots_grid = [2000, 4000, 8000]
    if gamma_grid is None:
        gamma_grid = [0.05, 0.2, 0.4]
    if n_avg < 1:
        raise ValueError(f"n_avg must be >= 1, got {n_avg}.")

    # FCI 参考 (两种模式共用)
    e_fci, civec = _fci.direct_spin1.kernel(h1e, eri, norb, nelec)
    nq = 2 * norb
    X, y, grid = [], [], []

    if circuit is not None:
        # ---- 电路模式: 实际电路采样 + 位串级 T1 (真机/带噪 LUCJ regime) ----
        from .counts import sample_from_circuit as _samp
        mode = "circuit"
        for gamma in gamma_grid:
            for shots in shots_grid:
                es = []
                for k in range(n_avg):
                    sk = int(seed) + k
                    bsm, probs = _samp(
                        circuit, n_samples=int(shots),
                        random_generator=_np.random.default_rng(sk))
                    bsm_t = _noise.apply_t1_bitstrings(bsm, float(gamma), seed=sk)
                    es.append(_cgse(
                        h1e, eri, norb, nelec, ecore=ecore, method="sqd",
                        bitstring_matrix=bsm_t, probabilities=probs,
                        max_iterations=max_iterations, seed=sk))
                eps = float(_np.mean(es)) - (e_fci + ecore)
                X.append([1.0 / _np.sqrt(int(shots)), float(gamma)])
                y.append(eps)
                grid.append((int(shots), float(gamma)))
    else:
        # ---- FCI 密度模式: FCI 态 -> density 计算基 (bit p = α_p, bit norb+p = β_p) ----
        ci_a = _cistr.make_strings(range(norb), nelec[0])
        ci_b = _cistr.make_strings(range(norb), nelec[1])
        psi = _np.zeros(2 ** nq, dtype=complex)
        civec = _np.asarray(civec).reshape(len(ci_a), len(ci_b))
        for ia, sa in enumerate(ci_a):
            for ib, sb in enumerate(ci_b):
                amp = civec[ia, ib]
                if abs(amp) < 1e-15:
                    continue
                idx = 0
                for p in range(norb):
                    if (int(sa) >> p) & 1:
                        idx |= (1 << p)
                for p in range(norb):
                    if (int(sb) >> p) & 1:
                        idx |= (1 << (norb + p))
                psi[idx] += amp
        rho = _noise.statevector_to_density(psi)
        mode = "fci_density"
        for gamma in gamma_grid:
            rho_g = _noise.apply_amp_damping(rho, float(gamma), nq)
            diag = _np.diag(rho_g).real
            for shots in shots_grid:
                es = []
                for k in range(n_avg):
                    bsm = _noise.density_to_bitstring_matrix(
                        diag, norb, int(shots), seed=int(seed) + k)
                    e = _cgse(h1e, eri, norb, nelec, ecore=ecore, method="sqd",
                              bitstring_matrix=bsm,
                              max_iterations=max_iterations, seed=int(seed) + k)
                    es.append(e)
                eps = float(_np.mean(es)) - (e_fci + ecore)
                X.append([1.0 / _np.sqrt(int(shots)), float(gamma)])
                y.append(eps)
                grid.append((int(shots), float(gamma)))

    X = _np.asarray(X)
    y = _np.asarray(y)
    coef, *_ = _np.linalg.lstsq(X, y, rcond=None)
    KS, KT1 = float(coef[0]), float(coef[1])
    rmse = float(_np.sqrt(_np.mean((X @ coef - y) ** 2)))
    return {"KS": KS, "KT1": KT1, "grid_points": grid,
            "errors": y, "rmse": rmse, "mode": mode}
