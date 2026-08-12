"""tc_sqd.cipsi._Subspace GPU backend 测试 (round_003 R3)。

三锚点 (theory.md §3):
- **P2 零回归**: backend="cpu" (默认) 路径逐位不变; dim≤1000 始终 CPU (不读 backend)。
- **P1 正确性**: 固定子空间 dim>1e5, backend="gpu" vs "cpu" E diff ≤1e-10。
- **P0 性能**: 固定子空间 dim>1e5, 单次 diag wall GPU/CPU ≤0.5 (≥2× 稳健阈值;
  理论目标 ≤0.33 即 ≥3×)。
- **回退**: mock has_gpu()=False → backend="gpu" 静默降级 "cpu", 绝不 raise。

无 cupy/GPU 环境: P1/P0 测试 skip, 其余 (P2/回退) 必通过。
"""
import time

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.fci import cistring

import tc_sqd
from tc_sqd.cipsi import _Subspace


def _have_gpu():
    """cupy + 真实 GPU 设备可用?"""
    try:
        import cupy  # noqa: F401
        if not cupy.cuda.runtime.getDeviceCount():
            return False
        return True
    except Exception:
        return False


def _large_subspace_ints(n_str=400):
    """N2/cc-pVDZ 前 12 个 MO 的积分窗口 (norb=12, nelec=(6,6))。

    全 C(12,6)=924 字符串; 取前 ``n_str`` → dim=n_str²>1e5 (n_str=400 → 160000)。
    dim>1e5 锁定 GPU crossover 右侧 (REVIEW: 6× @9e4, 2× @4.9e5), 防 P0 假证伪。

    积分取自真实分子 MO 窗口 (对称有效); nelec=(6,6) 对 norb=12 合法。测试关心的是
    CPU/GPU **代数等价**与 wall-time 对照 (同一 h1e/eri/sa/sb), 非物理精度。
    """
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="cc-pvdz", verbose=0)
    mf = scf.RHF(mol).run()
    mo = np.asarray(mf.mo_coeff)
    mo12 = mo[:, :12]
    h1e = mo12.T @ mf.get_hcore() @ mo12
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl",
                    mol.intor("int2e_sph"), mo12, mo12, mo12, mo12, optimize=True)
    norb = 12
    nelec = (6, 6)
    full = np.array(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    sa = full[:n_str]
    sb = full[:n_str]
    return h1e, eri, norb, nelec, sa, sb


# --------------------------------------------------------------------------- #
#  P2 零回归: 默认 backend="cpu", dim≤1000 始终 CPU
# --------------------------------------------------------------------------- #
def test_subspace_default_backend_cpu():
    """_Subspace 默认 backend="cpu" (keyword-only, 不破坏位置参数)。"""
    h1e = np.zeros((4, 4))
    eri = np.zeros((4, 4, 4, 4))
    sub = _Subspace(h1e, eri, 4, (2, 2))
    assert sub.backend == "cpu"
    # 位置参数不受影响 (backend 是 keyword-only)
    sub2 = _Subspace(h1e, eri, 4, (2, 2), backend="cpu")
    assert sub2.backend == "cpu"


def test_subspace_small_dim_always_cpu():
    """dim≤1000 分支不读 backend: backend="gpu" 时小子空间仍走 CPU numpy eigh。

    构造 6×6=36 dim 子空间 (远 <1000), 无论是否有 GPU, diag 都走分支 ① (CPU),
    逐位一致。验证 theory §1.3 / 叮嘱 #3: 小 dim 不接 GPU。
    """
    h1e = np.diag([-2.0, -1.0, -0.5, 0.0]).astype(float)
    eri = np.zeros((4, 4, 4, 4))
    strs = np.array(cistring.make_strings(range(4), 2), dtype=np.int64)  # 6 strings
    sub_cpu = _Subspace(h1e, eri, 4, (2, 2), backend="cpu")
    sub_gpu = _Subspace(h1e, eri, 4, (2, 2), backend="gpu")
    E_cpu, c_cpu, _, _ = sub_cpu.diag(strs, strs)
    E_gpu, c_gpu, _, _ = sub_gpu.diag(strs, strs)
    assert abs(E_cpu - E_gpu) < 1e-12, "小 dim 应始终 CPU numpy eigh, 逐位一致"
    assert np.allclose(c_cpu, c_gpu, atol=1e-12)


def test_solve_sqd_active_default_cpu_smoke():
    """solve_sqd_active 默认 backend (端到端零回归冒烟): 默认 == 显式 cpu。

    小体系 N2/STO-3G, 确认透传链默认 "cpu" 与改造前行为一致。
    """
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((40, 2 * norb)) > 0.5
    e_default = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, max_rounds=2, rand_seed=0)
    e_explicit = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, max_rounds=2, rand_seed=0,
        backend="cpu")
    assert abs(e_default - e_explicit) < 1e-12


# --------------------------------------------------------------------------- #
#  回退: mock has_gpu()=False → backend="gpu" 静默降级, 绝不 raise
# --------------------------------------------------------------------------- #
def test_gpu_graceful_fallback_no_cupy(monkeypatch):
    """has_gpu()=False 时 backend="gpu" 不 raise, 降级 self.backend="cpu"。

    theory §6.4 硬性要求: 无 cupy 环境 backend="gpu" 必须 pass/skip 而非 fail。
    本测试模拟无 GPU 环境 (mock), 验证静默降级路径。
    """
    monkeypatch.setattr("tc_sqd.noise.has_gpu", lambda: False)
    h1e = np.diag([-2.0, -1.0, -0.5, 0.0]).astype(float)
    eri = np.zeros((4, 4, 4, 4))
    sub = _Subspace(h1e, eri, 4, (2, 2), backend="gpu")
    assert sub.backend == "cpu", "无 GPU 时 backend='gpu' 应静默降级为 'cpu'"
    # diag 不应 raise (走 CPU 路径)
    strs = np.array(cistring.make_strings(range(4), 2), dtype=np.int64)
    E, _, _, _ = sub.diag(strs, strs)
    assert np.isfinite(E)


# --------------------------------------------------------------------------- #
#  P1 正确性 + P0 性能: dim>1e5 (需 cupy+GPU, 否则 skip)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have_gpu(), reason="cupy / GPU 不可用")
def test_gpu_correctness_large_subspace():
    """P1: dim>1e5 子空间, backend="gpu" (tol=1e-10) vs "cpu" E diff ≤1e-10。

    theory §3 P1: REVIEW 基线 sigma vs contract_2e ≤2e-13; _Subspace 包装后
    E diff 应 ≤1e-10 (tol 收紧对齐 CPU scipy 默认机器精度)。
    """
    h1e, eri, norb, nelec, sa, sb = _large_subspace_ints(n_str=400)
    dim = len(sa) * len(sb)
    assert dim > 1e5, f"测试子空间 dim={dim} 须 >1e5 (P0/P1 前置)"

    sub_cpu = _Subspace(h1e, eri, norb, nelec, backend="cpu")
    sub_gpu = _Subspace(h1e, eri, norb, nelec, backend="gpu")
    assert sub_gpu.backend == "gpu", "有 GPU 时 backend='gpu' 保持"

    E_cpu, _, _, _ = sub_cpu.diag(sa, sb)
    E_gpu, _, _, _ = sub_gpu.diag(sa, sb)
    diff = abs(E_cpu - E_gpu)
    assert diff <= 1e-10, f"GPU/CPU E diff {diff:.2e} >1e-10 (E_cpu={E_cpu:.10f}, E_gpu={E_gpu:.10f})"


@pytest.mark.skipif(not _have_gpu(), reason="cupy / GPU 不可用")
def test_gpu_performance_large_subspace():
    """P0: dim>1e5 子空间, 单次 diag wall GPU/CPU ≤0.5 (≥2× 稳健; 目标 ≤0.33 即 ≥3×)。

    GPU 先 warm-up (cupy 上下文初始化 + RawModule 编译一次性开销, 不计入稳态计时),
    再计时。断言 dim>1e5 防 crossover 假证伪 (theory §3 风险声明 2)。
    """
    h1e, eri, norb, nelec, sa, sb = _large_subspace_ints(n_str=400)
    dim = len(sa) * len(sb)
    assert dim > 1e5, f"测试子空间 dim={dim} 须 >1e5 (P0 前置, 防 crossover 假证伪)"

    sub_cpu = _Subspace(h1e, eri, norb, nelec, backend="cpu")
    sub_gpu = _Subspace(h1e, eri, norb, nelec, backend="gpu")

    # GPU warm-up: 一次性 cupy/RawModule 初始化, 不计入稳态计时
    sub_gpu.diag(sa, sb)

    t0 = time.perf_counter()
    sub_gpu.diag(sa, sb)
    t_gpu = time.perf_counter() - t0

    t0 = time.perf_counter()
    sub_cpu.diag(sa, sb)
    t_cpu = time.perf_counter() - t0

    ratio = t_gpu / t_cpu
    # 理论目标 ≤0.33 (≥3×); 用 ≤0.5 (≥2×) 作稳健硬阈值避免 CI 单次抖动。
    # 实测 ratio 记录在 implementation.md (R4/R5 据此判三态)。
    assert ratio <= 0.5, (
        f"GPU diag 未加速: t_gpu={t_gpu:.3f}s t_cpu={t_cpu:.3f}s ratio={ratio:.3f} "
        f"(理论目标 ratio≤0.33 即 ≥3×, 稳健阈值 ≤0.5)"
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print(f"--- {name} ---")
            fn()
            print("PASS")
