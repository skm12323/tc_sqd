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
