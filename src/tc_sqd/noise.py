"""tc_sqd.noise —— 密度矩阵 Kraus 噪声模拟 (qiskit-Aer 风格 + tc GPU 优势)。

提供 SQD 噪声鲁棒性研究/预测所需的密度矩阵噪声通道:
  - apply_dephasing       退相干 (T2, 相位阻尼) —— SQD 免疫 (diag 不变)
  - apply_amp_damping     振幅阻尼 (T1, |1>->|0>) —— SQD 主导误差源
  - apply_depolarizing    去极化 (通用 bit 翻转)
  - density_to_bitstring_matrix  密度矩阵 diag -> 采样 bitstring matrix (接 recover_configurations)

GPU: cupy 可选。gpu=True 且装了 cupy 时, 密度矩阵运算走 GPU (大矩阵 2^nq 加速);
没装 cupy 则 numpy。这样 tc_sqd.noise 同时具备:
  - qiskit Aer 的噪声模拟能力 (密度矩阵 Kraus 通道)
  - tc 的 GPU 优势 (cupy 后端)

参考: D:\\explore 的方向1/2 (SQD 噪声鲁棒性) 验证了这些通道对 SQD 的影响。

**内存边界**: 密度矩阵是 2^nq × 2^nq 的复数矩阵 (complex128 约 16·4^nq 字节):
nq=10 约 16 MiB, nq=12 约 0.25 GiB, nq=14 约 4 GiB, nq=16 约 68 GiB。
因此本模块**只适合小体系 (nq ≲ 12, 对应 ~6 空间轨道)**; 59-qubit 真机无法做
密度矩阵模拟。大体系 / 真机规模的噪声评估请用 ``predict`` 预测器 (解析模型,
无内存开销)。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# cupy 可选 (GPU)。没装则 numpy。
try:
    import cupy as _cp
    _HAS_CUPY = True
except Exception:
    _cp = None
    _HAS_CUPY = False


def has_gpu() -> bool:
    """是否可用 cupy GPU 后端。"""
    return _HAS_CUPY


def _xp(gpu: bool):
    """返回数组后端: cupy (gpu=True 且可用) 或 numpy。"""
    if gpu and _HAS_CUPY:
        return _cp
    return np


def statevector_to_density(psi, gpu: bool = False):
    """纯态向量 |ψ> -> 密度矩阵 ρ = |ψ><ψ| (2^nq × 2^nq)。"""
    xp = _xp(gpu)
    psi = xp.asarray(psi)
    return xp.outer(psi, psi.conj())


def _kron_at(op, q: int, nq: int, xp):
    """单 qubit 算符 op 嵌入 qubit q 的 2^nq 算符 (np/cp.kron, qubit 0 = 最低 bit)。"""
    eye2 = xp.eye(2, dtype=op.dtype)
    full = xp.eye(1, dtype=op.dtype)
    for i in range(nq):
        full = xp.kron(full, op if i == q else eye2)
    return full


def apply_dephasing(rho, p: float, nq: int, gpu: bool = False):
    """逐 qubit 相位阻尼 (退相干 T2): ρ -> (1-p)ρ + p Z_q ρ Z_q。

    物理注意: 相位阻尼的 Kraus (I, Z) 都对角, **不改 diag** -> SQD 计算基采样免疫。
    主要用于验证/教学, 以及对非 SQD 量 (如期望值) 的退相干影响。
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0,1], got {p}")
    xp = _xp(gpu)
    rho = xp.asarray(rho)
    Z = xp.array([[1, 0], [0, -1]], dtype=rho.dtype)
    for q in range(nq):
        Zq = _kron_at(Z, q, nq, xp)
        rho = (1 - p) * rho + p * (Zq @ rho @ Zq.conj().T)
    return rho


def apply_amp_damping(rho, gamma: float, nq: int, gpu: bool = False):
    """逐 qubit 振幅阻尼 (T1): K0=diag(1,√(1-γ)), K1=√γ|0><1|。|1> -> |0> 衰减。

    这是 SQD 的主导误差源 (方向1): 漏低概率 determinant; 高 shots 下被 recover 吸收。
    γ 对应真机单 qubit 振幅阻尼率 = 1 - exp(-depth × t_gate / T1) (见 predict 模块)。
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"gamma must be in [0,1], got {gamma}")
    xp = _xp(gpu)
    rho = xp.asarray(rho)
    K0 = xp.array([[1, 0], [0, xp.sqrt(1 - gamma)]], dtype=rho.dtype)
    K1 = xp.array([[0, xp.sqrt(gamma)], [0, 0]], dtype=rho.dtype)
    for q in range(nq):
        K0q, K1q = _kron_at(K0, q, nq, xp), _kron_at(K1, q, nq, xp)
        rho = K0q @ rho @ K0q.conj().T + K1q @ rho @ K1q.conj().T
    return rho


def apply_depolarizing(rho, p: float, nq: int, gpu: bool = False):
    """逐 qubit 去极化 (标准 4-Kraus 通道, 迹保持):

        ρ -> (1-p) ρ + (p/3)(X ρ X + Y ρ Y + Z ρ Z)

    Kraus 算符 ``{√(1-p)·I, √(p/3)·X, √(p/3)·Y, √(p/3)·Z}``; ``p ∈ [0,1]`` 为
    非恒等 Kraus 的总概率 (X/Y/Z 三等分)。对 nq 个 qubit 逐个施加该通道。
    通用 bit 翻转 + 相位噪声, 区别于纯振幅阻尼 (T1) / 纯退相干 (T2)。
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0,1], got {p}")
    xp = _xp(gpu)
    rho = xp.asarray(rho)
    # 单 qubit 去极化 Kraus: √(1-p) I, √(p/3) X, √(p/3) Y, √(p/3) Z
    I = xp.array([[1, 0], [0, 1]], dtype=rho.dtype)
    X = xp.array([[0, 1], [1, 0]], dtype=rho.dtype)
    Y = xp.array([[0, -1j], [1j, 0]], dtype=rho.dtype)
    Z = xp.array([[1, 0], [0, -1]], dtype=rho.dtype)
    c0, c1 = xp.sqrt(1 - p), xp.sqrt(p / 3)
    kraus = [(c0, I), (c1, X), (c1, Y), (c1, Z)]
    for q in range(nq):
        acc = None
        for c, K in kraus:
            Kc = c * K
            Kq = _kron_at(Kc, q, nq, xp)
            term = Kq @ rho @ Kq.conj().T
            acc = term if acc is None else acc + term
        rho = acc
    return rho


def density_to_bitstring_matrix(diag, norb: int, n_samples: int,
                                 seed: Optional[int] = None, gpu: bool = False):
    """密度矩阵 diag (计算基概率) -> 采样 bitstring matrix [β0..|α0..] (col=轨道, 升序)。

    返回的 bsm 可直接喂 ``recover_configurations`` / ``build_ci_matrix``。

    约定
    ----
    密度矩阵计算基 (本模块 ``statevector_to_density`` / Kraus 通道一致): 整数 ``i``
    的 bit ``orb``        (0..norb-1) = α 轨道 ``orb``
                   ``norb+orb``       = β 轨道 ``orb``
    输出 bsm 遵循 tc_sqd 全库降序约定 ``[β_{norb-1}..β0 | α_{norb-1}..α0]`` (列内
    降序, 与 ``counts.py`` / ``bitstring_matrix_to_ci_strs`` 一致), 故
    β 轨道 ``orb`` -> 列 ``norb-1-orb`` ;  α 轨道 ``orb`` -> 列 ``2*norb-1-orb``。
    """
    xp = _xp(gpu)
    diag = xp.asarray(diag)
    if gpu and _HAS_CUPY:
        diag = _cp.asnumpy(diag)   # 采样回 CPU (np.random.choice)
    prob = diag.real.clip(0)
    s = prob.sum()
    if s <= 0:
        raise ValueError("diag has no positive probability mass.")
    prob = prob / s
    rng = np.random.default_rng(seed)
    nq = 2 * norb
    bs_ints = rng.choice(2 ** nq, size=n_samples, p=prob)
    bsm = np.zeros((n_samples, nq), dtype=bool)
    for orb in range(norb):
        # density bit (norb+orb) = β 轨道 orb  ->  β 块内降序列 norb-1-orb
        bsm[:, norb - 1 - orb] = (bs_ints >> (norb + orb)) & 1
        # density bit orb        = α 轨道 orb  ->  α 块内降序列 2*norb-1-orb
        bsm[:, 2 * norb - 1 - orb] = (bs_ints >> orb) & 1
    return bsm


def apply_t1_bitstrings(bitstring_matrix, gamma: float, *, seed=None):
    """位串级振幅阻尼 (T1): 每个 |1⟩ 以概率 ``gamma`` 独立翻 0。

    对纯态/计算基, 与密度矩阵 ``apply_amp_damping`` 在 diag 上等价, 但**无需
    构造 2^nq 密度矩阵**, 支持大体系 (nq 不限)。用于 :func:`predict.calibrate`
    的电路采样模式 (对实际电路采样的位串施加 T1)。

    Parameters
    ----------
    bitstring_matrix : ndarray (S, N), bool
        无噪声位串 (如 ``sample_from_circuit`` 输出)。
    gamma : float in [0, 1]
        振幅阻尼率 (1→0 翻转概率)。
    seed : int | None

    Returns
    -------
    ndarray (S, N), bool
        T1 后的位串 (含粒子数违例, 交由配置恢复修正)。
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
    bsm = np.asarray(bitstring_matrix, dtype=bool).copy()
    rng = np.random.default_rng(seed)
    flip = rng.random(bsm.shape) < gamma
    bsm &= ~flip
    return bsm


def zero_noise_extrapolate_t1(
    h1e,
    eri,
    norb: int,
    nelec,
    *,
    bitstring_matrix,
    probabilities=None,
    gammas=(0.05, 0.1, 0.2, 0.3),
    extrapolation_order: int = 1,
    ecore: float = 0.0,
    seed: Optional[int] = 0,
    **sqd_kwargs,
):
    """T1 零噪声外推 (ZNE, A3): 位串级 T1 噪声能量多项式外推到 γ→0。

    T1 (振幅阻尼) 是 SQD 主导误差源 (漏低概率 determinant)。对一组噪声强度
    γ 用 :func:`apply_t1_bitstrings` 破坏无噪声位串, 跑 SQD 得 ``E(γ)``,
    最小二乘多项式外推到 γ=0 (零噪声极限) —— 提高受 T1 影响时的能量精度。

    外推模型: 低阶多项式 ``E(γ) ≈ Σ c_k γ^k`` (k=0..order), ``c_0 = E(0)``。
    推荐线性 (order=1); 高阶可吸收曲率但放大测量噪声 (arXiv:2502.20673
    建议最小二乘 + 低阶避免过拟合)。验证用模拟器: 对比外推 E0 与无噪声
    SQD 参考 (γ=0 直接跑)。

    Parameters
    ----------
    h1e, eri, norb, nelec, ecore
        分子积分 (SQD 输入)。
    bitstring_matrix : ndarray (S, 2*norb), bool
        无噪声采样位串 (如 ``sample_from_circuit`` 输出)。
    probabilities : ndarray (S,) | None
        位串权重 (T1 翻转后近似沿用, 仅配置恢复去重合并用)。
    gammas : tuple[float]
        T1 率网格 (外推点, 建议覆盖 γ∈[0.02, 0.4])。
    extrapolation_order : int
        外推多项式阶数 (1=线性)。
    ecore : float
        Core 能量偏移 (计入返回值与 E(γ) 序列)。
    seed : int | None
        T1 翻转随机种子。
    **sqd_kwargs
        透传给 ``compute_ground_state_energy`` (如 ``max_iterations``)。

    Returns
    -------
    (e_zero, energies_by_gamma) : (float, list[float])
        ``e_zero`` 外推零噪声能量 (含 ecore); ``energies_by_gamma`` 各 γ 的
        噪声能量 (含 ecore, 与 ``gammas`` 同序)。
    """
    from .fermion import compute_ground_state_energy

    if extrapolation_order < 1:
        raise ValueError(f"extrapolation_order must be >= 1, got {extrapolation_order}.")
    if len(gammas) < extrapolation_order + 1:
        raise ValueError(
            f"gammas 点数 ({len(gammas)}) 不足以拟合 order={extrapolation_order} "
            f"多项式 (至少需 {extrapolation_order + 1})。"
        )
    if any(g < 0 or g > 1 for g in gammas):
        raise ValueError("gammas 必须在 [0, 1]。")

    bsm0 = np.asarray(bitstring_matrix, dtype=bool)
    # 固定 SQD 链路的随机性 (compute_ground_state_energy -> diagonalize 的
    # 配置恢复 tie-breaking 需固定 seed, 否则 E(γ) 受全局随机状态影响)。
    sqd_kwargs.setdefault("seed", seed)
    energies = []
    for g in gammas:
        bsm_noisy = apply_t1_bitstrings(bsm0, float(g), seed=seed)
        e = compute_ground_state_energy(
            h1e, eri, norb, nelec, ecore=ecore, method="sqd",
            bitstring_matrix=bsm_noisy, probabilities=probabilities,
            **sqd_kwargs,
        )
        energies.append(float(e))
    e_arr = np.asarray(energies, dtype=np.float64)
    g_arr = np.asarray(gammas, dtype=np.float64)
    coef = np.polyfit(g_arr, e_arr, deg=extrapolation_order)
    e_zero = float(coef[-1])  # 常数项 = E(γ=0)
    return e_zero, energies


def solve_sqd_robust(
    h1e,
    eri,
    norb: int,
    nelec,
    *,
    bitstring_matrix,
    probabilities=None,
    gammas=(0.05, 0.1, 0.2, 0.3),
    shots_budget: Optional[int] = None,
    shots_step: int = 0,
    energy_tol: Optional[float] = None,
    extrapolation_order: int = 1,
    ecore: float = 0.0,
    seed: Optional[int] = 0,
    verbose: bool = False,
    **active_kwargs,
):
    """鲁棒自适应采样: A3 T1-ZNE × B1 预算闭环 统一 API。

    组合两个已验证的方向, 同时获得**噪声鲁棒**(ZNE 外推 γ→0)与**预算高效**
    (能量收敛停采省 shots):

      B1 (预算闭环)  ×  A3 (T1 零噪声外推)
      ──────────────────────────────────────
      每个 γ 噪声水平下, 用自适应预算的 :func:`solve_sqd_active` (增量采样 +
      energy_tol 收敛停采) 求收敛能量 E(γ) 与实际 shots 用量;
      对 E(γ) 做低阶多项式外推得零噪声能量 E(0)。

    Parameters
    ----------
    h1e, eri, norb, nelec, ecore
        分子积分 (SQD 输入)。
    bitstring_matrix : ndarray (S, 2*norb), bool
        无噪声采样位串 (采样池; ``shots_budget`` 大于池行数时预生成随机补足)。
    probabilities : ndarray (S,) | None
        位串权重。
    gammas : tuple[float]
        T1 率网格 (A3 外推点)。
    shots_budget : int | None
        B1 总采样预算 (每个 γ 的 solve_sqd_active 采样池上限)。
    shots_step : int
        B1 增量采样步长 (0 = 一次性全量)。
    energy_tol : float | None
        B1 能量收敛停采阈值。
    extrapolation_order : int
        ZNE 外推多项式阶数 (1 = 线性)。
    seed : int | None
        T1 翻转 + SQD 链路随机种子 (确定性)。
    verbose : bool
        打印每 γ 能量与 shots。
    **active_kwargs
        透传给 ``solve_sqd_active`` (如 ``n_active_per_round``, ``max_rounds``)。

    Returns
    -------
    dict
        ``energy``          —— ZNE 外推零噪声能量 (含 ecore);
        ``energies_by_gamma`` —— 各 γ 收敛能量 (含 ecore);
        ``shots_by_gamma``   —— 各 γ 实际使用的 shots (B1 停采后);
        ``total_shots``      —— Σ shots (预算对比指标);
        ``gammas``           —— γ 网格。
    """
    from .cipsi import solve_sqd_active

    if extrapolation_order < 1:
        raise ValueError(f"extrapolation_order must be >= 1, got {extrapolation_order}.")
    if len(gammas) < extrapolation_order + 1:
        raise ValueError(
            f"gammas 点数 ({len(gammas)}) 不足以拟合 order={extrapolation_order} "
            f"多项式 (至少需 {extrapolation_order + 1})。"
        )
    if any(g < 0 or g > 1 for g in gammas):
        raise ValueError("gammas 必须在 [0, 1]。")

    # 采样池: 预算 > 池行数时预生成随机补足 (每个 γ 共用同一池, 只做 T1 翻转)
    bsm0 = np.asarray(bitstring_matrix, dtype=bool)
    n_pool = bsm0.shape[0]
    probs0 = probabilities
    if shots_budget is not None and shots_budget > n_pool:
        rng = np.random.default_rng(seed)
        extra = rng.random((shots_budget - n_pool, 2 * norb)) > 0.5
        bsm0 = np.vstack([bsm0, extra])
        if probs0 is None:
            probs0 = np.full(shots_budget, 1.0 / shots_budget)
        else:
            probs0 = np.concatenate(
                [np.asarray(probs0, dtype=np.float64),
                 np.full(shots_budget - n_pool, 1.0 / n_pool)]
            )
        n_pool = shots_budget

    energies = []
    shots_used = []
    # 确定性: rand_seed 统一用 seed (active_kwargs 里若传了则弹出, 避免冲突)
    active_kwargs.pop("rand_seed", None)
    for g in gammas:
        bsm_noisy = apply_t1_bitstrings(bsm0, float(g), seed=seed)
        usage: list = []
        e = solve_sqd_active(
            h1e, eri, norb, nelec,
            bitstring_matrix=bsm_noisy, probabilities=probs0,
            ecore=ecore, rand_seed=seed,
            shots_budget=shots_budget, shots_step=shots_step,
            energy_tol=energy_tol, usage=usage,
            verbose=verbose, **active_kwargs,
        )
        energies.append(float(e))
        shots_used.append(int(usage[0]) if usage else int(n_pool))
        if verbose:
            print(f"[robust] γ={g}: E={e:.6f} shots={shots_used[-1]}")

    e_arr = np.asarray(energies, dtype=np.float64)
    g_arr = np.asarray(gammas, dtype=np.float64)
    coef = np.polyfit(g_arr, e_arr, deg=extrapolation_order)
    e_zero = float(coef[-1])
    return {
        "energy": e_zero,
        "energies_by_gamma": energies,
        "shots_by_gamma": shots_used,
        "total_shots": int(sum(shots_used)),
        "gammas": list(gammas),
    }
