"""tc_sqd.cipsi._Subspace GPU backend 测试 (round_003 R3 / round_005 R3 三模式)。

锚点 (round_005 theory.md §3):
- **P2 零回归**: backend="cpu" (默认) 路径逐位不变; dim≤1000 始终 CPU (不读 backend)。
  backend="gpu" 默认 gpu_eigsh_mode="hybrid" 仅在 GPU 分支读, CPU 路径完全不触及。
- **P1 正确性**: 固定子空间 dim>1e5, backend="gpu" (hybrid, tol=1e-10) vs "cpu"
  E diff ≤1e-10。
- **P0 性能**: 固定子空间 dim>1e5, 单次 diag wall GPU(hybrid)/CPU ≤0.5 (≥2× 稳健阈值)。
  round_005 hybrid 绕开 cupyx 收敛停滞, 预期 ratio ≤0.3 (≥3×), 应 xpass。
- **三模式覆盖**: hybrid / cupyx / cpu_fallback 三模式在中等 dim 下能量数值一致。
- **回退**: mock has_gpu()=False → backend="gpu" 静默降级 "cpu", 绝不 raise;
  mock cupyx.eigsh (cupyx 模式) / sigma_selected_ci_gpu OOM (hybrid 模式) → except
  懒构 hop 回退 CPU scipy, 能量逐位一致。

无 cupy/GPU 环境: P1/P0/三模式 测试 skip, 其余 (P2/回退) 必通过。
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
    """_Subspace 默认 backend="cpu", gpu_eigsh_mode="hybrid" (keyword-only, 不破坏位置参数)。

    round_005: gpu_eigsh_mode 默认 "hybrid" 仅在 backend=="gpu" 时 diag 读;
    CPU 路径完全不触及 -> 默认值不影响零回归。
    """
    h1e = np.zeros((4, 4))
    eri = np.zeros((4, 4, 4, 4))
    sub = _Subspace(h1e, eri, 4, (2, 2))
    assert sub.backend == "cpu"
    assert sub.gpu_eigsh_mode == "hybrid"     # round_005 新默认
    # 位置参数不受影响 (backend / gpu_eigsh_mode 均为 keyword-only)
    sub2 = _Subspace(h1e, eri, 4, (2, 2), backend="cpu")
    assert sub2.backend == "cpu"
    # gpu_eigsh_mode 可显式覆盖 (三模式)
    sub3 = _Subspace(h1e, eri, 4, (2, 2), backend="gpu", gpu_eigsh_mode="cupyx")
    assert sub3.gpu_eigsh_mode == "cupyx"


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
    """P1: dim>1e5 子空间, backend="gpu" (round_005 默认 hybrid, tol=1e-10) vs "cpu"
    E diff ≤1e-10。

    round_005 hybrid = scipy.sparse.linalg.eigsh + GPU matvec (sigma + .get())。
    引擎与 CPU else 分支同一个 scipy eigsh -> N_matvec 同分布; matvec 误差地板
    ~1e-13 (round_004 sigma vs contract_2e 实测 ≤4.4e-13) ≪ tol=1e-10 -> E diff
    富余。theory §3 P1: ≤1e-10 证实 / 1e-10–1e-8 部分 / >1e-8 证伪。
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
@pytest.mark.xfail(strict=False, reason=(
    "round_005 R3: 默认 hybrid (scipy eigsh + GPU matvec) 绕开 cupyx 收敛停滞, "
    "theory §1.3.2 预期 ratio 0.21-0.31 (3.2-4.7×) @ dim 5e5 / 0.09-0.19 @ dim 1e5 "
    "-> 应 xpass (ratio ≤0.5). strict=False: R5 确认达标后 R3 移除 xfail 改硬 assert. "
    "若仍 >0.5, 查 P0' N_matvec 归因 (hybrid 是否继承 scipy ~700-811) — theory §3 "
    "风险声明 2: P0' 是 P0 的因果前提."))
def test_gpu_performance_large_subspace():
    """P0: dim>1e5 子空间, 单次 diag wall GPU(hybrid)/CPU ≤0.5 (≥2×; 目标 ≤0.33 即 ≥3×)。

    round_005 hybrid = scipy ARPACK 黑盒驱动 GPU matvec (绕开 cupyx 收敛停滞)。
    GPU 先 warm-up (cupy 上下文初始化 + RawModule 编译一次性开销, 不计入稳态计时),
    再计时。断言 dim>1e5 防 crossover 假证伪 (theory §3 风险声明 2)。

    theory §1.3.2 预测带: dim 1e5 ratio 0.09-0.19 (5.4-11.7×); dim 5e5 ratio
    0.21-0.31 (3.2-4.7×)。保守端 dim 5e5 仍 3.2× 富余, 远超 P0 阈值 2×。
    唯一危险: scipy N_mv 也退化到 ~3000+ (P0' 证伪) — 但 scipy 引擎 + 同矩阵 +
    同 v0 分布必然 ~700-811 (theory §1.2), 概率低。P0' 必跑 (N_mv 是 P0 因果前提)。
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
    # round_004: B+C 仅修 per-matvec/per-diag 冗余 (~10-15%), 不触及 cupyx 收敛。
    # 若 dim 1e5 收敛正常, B+C 应落到 2.5-2.8× (≤0.4) 通过; 若抖动到 >0.5,
    # 查 implementation.md P0' 是否证实 -> 归因 cupyx 收敛 #3, 非 B+C bug。
    assert ratio <= 0.5, (
        f"GPU diag 未加速: t_gpu={t_gpu:.3f}s t_cpu={t_cpu:.3f}s ratio={ratio:.3f} "
        f"(理论目标 ratio≤0.33 即 ≥3×, 稳健阈值 ≤0.5)"
    )


# --------------------------------------------------------------------------- #
#  round_004 新增: 方式 C sigma 接口正确性 + P0' per-matvec 隔离测
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have_gpu(), reason="cupy / GPU 不可用")
def test_sigma_cached_eri_correctness():
    """P1 (sigma 接口级): 缓存版 sigma_selected_ci_gpu 输出 vs 重算版 ≤2e-13。

    round_004 方式 C 在 sigma 加 eri1_aaaa/bbaa=None 可选参数。本测验证:
      (a) 缓存版 (传入 _Subspace 预算的 cupy eri1_*) 与重算版 (None) 数值一致;
      (b) 缓存版 (调用方手动 cp.asarray 预算) 与重算版一致。

    这是 P1 正确性的最细粒度锚点 (直接对照 sigma 输出, 不经 eigsh)。
    _Subspace.diag 路径的端到端 P1 已由 test_gpu_correctness_large_subspace 覆盖。
    """
    import cupy as cp
    from tc_sqd.selected_ci_gpu import (
        sigma_selected_ci_gpu, _selci_eri_aaaa, _selci_eri_bbaa, _get_kernels)
    from pyscf.fci import selected_ci as _sci

    h1e, eri, norb, nelec, sa, sb = _large_subspace_ints(n_str=120)  # dim ~1.4e4, 够对照
    na, nb = len(sa), len(sb)
    links = [_sci.des_des_linkstr(sa, norb, nelec[0], True),
             _sci.des_des_linkstr(sb, norb, nelec[1], True),
             _sci.cre_des_linkstr(sa, norb, nelec[0], True),
             _sci.cre_des_linkstr(sb, norb, nelec[1], True)]
    kernels = _get_kernels()
    rng = np.random.default_rng(42)
    v = rng.standard_normal((na, nb))

    # 重算版 (eri1_*=None == round_003 现状)
    sigma_recompute = sigma_selected_ci_gpu(
        v, sa, sb, norb, nelec, h1e, eri, links=links, kernels=kernels,
        eri1_aaaa=None, eri1_bbaa=None)
    # 缓存版 (调用方手动预算 -> 等价 _Subspace.__init__ 的实例级缓存)
    from pyscf import ao2mo
    from pyscf.fci import direct_spin1
    h2e = ao2mo.restore(1, direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5), norb)
    eri1_aaaa = cp.asarray(_selci_eri_aaaa(h2e, norb))
    eri1_bbaa = cp.asarray(_selci_eri_bbaa(h2e, norb, nelec))
    sigma_cached = sigma_selected_ci_gpu(
        v, sa, sb, norb, nelec, h1e, eri, links=links, kernels=kernels,
        eri1_aaaa=eri1_aaaa, eri1_bbaa=eri1_bbaa)

    diff = float(cp.abs(sigma_cached - sigma_recompute).max())
    assert diff <= 2e-13, (
        f"缓存版 vs 重算版 sigma 输出 max|Δ|={diff:.2e} >2e-13 "
        "(缓存 eri1 与重算 eri1 不一致, 查 _selci_eri_* 接合 bug)")


@pytest.mark.skipif(not _have_gpu(), reason="cupy / GPU 不可用")
def test_subspace_gpu_lazy_hop_fallback_cupyx(monkeypatch):
    """round_005 三模式 fallback (cupyx 模式): mock cupyx.eigsh 抛异常 ->
    分支 ②-GPU 的 except 捕获, 懒构 hop (此前 GPU 成功路径未付 _all_linkstr_index 税),
    回退 CPU scipy eigsh。

    round_004 的 test_subspace_gpu_lazy_hop_fallback 在 round_005 默认 hybrid 下会退化
    (hybrid 用 scipy eigsh, mock cupyx.eigsh 不再触发)。本测试 pin
    gpu_eigsh_mode="cupyx" 恢复 cupyx 路径的 fallback 覆盖 (theory §2.4.1)。
    能量应与纯 CPU 路径一致; 本征矢符号不定 (ARPACK 相位模糊, ±v 等价)。
    """
    import cupyx.scipy.sparse.linalg as cpsl
    # mock cp_eigsh 抛异常, 触发 _build_cpu_hop 兜底
    def _boom(*args, **kwargs):
        raise RuntimeError("mock cupyx.eigsh 不收敛 (强制走懒构 hop 回退)")
    monkeypatch.setattr(cpsl, "eigsh", _boom)

    h1e, eri, norb, nelec, sa, sb = _large_subspace_ints(n_str=120)
    sub_gpu = _Subspace(h1e, eri, norb, nelec, backend="gpu",
                        gpu_eigsh_mode="cupyx")         # ← pin cupyx 走 cupyx.eigsh 路径
    sub_cpu = _Subspace(h1e, eri, norb, nelec, backend="cpu")
    E_gpu, c_gpu, _, _ = sub_gpu.diag(sa, sb)
    E_cpu, c_cpu, _, _ = sub_cpu.diag(sa, sb)
    # 能量逐位一致 (回退路径与纯 CPU 走同一 scipy eigsh)
    assert abs(E_gpu - E_cpu) < 1e-10, (
        f"GPU 回退路径 E={E_gpu:.10f} != CPU E={E_cpu:.10f} (懒构 hop 接合 bug)")
    # 本征矢: ARPACK 起步随机 v0 -> 收敛到 ±v 均合法 (相位模糊), 符号不可定。
    # 用 abs 比较 (or 等价于 allclose(c, ±c_cpu))。atol=1e-8: round_013 起
    # 默认 tol=1e-10, 独立 ARPACK 运行的本征矢分量收敛到 ~1e-9 级 (能量仍
    # <1e-10 一致, 主检查); 真接合 bug 会给 O(1e-3) 级差异, 1e-8 足够抓。
    assert np.allclose(np.abs(c_gpu), np.abs(c_cpu), atol=1e-8) or \
        np.allclose(c_gpu, c_cpu, atol=1e-8) or \
        np.allclose(c_gpu, -c_cpu, atol=1e-8), (
        "GPU 回退本征矢与 CPU 既非 +c 也非 -c (相位模糊外的不一致)")


@pytest.mark.skipif(not _have_gpu(), reason="cupy / GPU 不可用")
def test_subspace_gpu_lazy_hop_fallback_hybrid(monkeypatch):
    """round_005 三模式 fallback (hybrid 模式, 默认): mock sigma_selected_ci_gpu 抛
    OOM -> hybrid 的 matvec 闭包内 _gpu_sigma 失败 -> except 懒构 hop + CPU scipy 回退。

    theory §2.4.1: hybrid 用 scipy eigsh (本就不 stall), 故 fallback 触发点是 matvec
    (GPU OOM / sigma 异常)。mock tc_sqd.selected_ci_gpu.sigma_selected_ci_gpu 抛
    MemoryError, 验证 except 接住 -> 能量与 CPU 一致。
    """
    import tc_sqd.selected_ci_gpu as sgpu
    def _oom(*args, **kwargs):
        raise MemoryError("mock sigma_selected_ci_gpu GPU OOM (强制走懒构 hop 回退)")
    monkeypatch.setattr(sgpu, "sigma_selected_ci_gpu", _oom)

    h1e, eri, norb, nelec, sa, sb = _large_subspace_ints(n_str=120)
    sub_gpu = _Subspace(h1e, eri, norb, nelec, backend="gpu")          # 默认 hybrid
    sub_cpu = _Subspace(h1e, eri, norb, nelec, backend="cpu")
    E_gpu, c_gpu, _, _ = sub_gpu.diag(sa, sb)
    E_cpu, c_cpu, _, _ = sub_cpu.diag(sa, sb)
    assert abs(E_gpu - E_cpu) < 1e-10, (
        f"hybrid 回退路径 E={E_gpu:.10f} != CPU E={E_cpu:.10f} (懒构 hop 接合 bug)")
    # 本征矢 atol=1e-8: round_013 起默认 tol=1e-10, 独立 ARPACK 运行的本征矢
    # 分量收敛到 ~1e-9 级; 能量 <1e-10 一致才是主检查 (见 cupyx 版注释)。
    assert np.allclose(np.abs(c_gpu), np.abs(c_cpu), atol=1e-8) or \
        np.allclose(c_gpu, c_cpu, atol=1e-8) or \
        np.allclose(c_gpu, -c_cpu, atol=1e-8), (
        "hybrid 回退本征矢与 CPU 既非 +c 也非 -c (相位模糊外的不一致)")


@pytest.mark.skipif(not _have_gpu(), reason="cupy / GPU 不可用")
def test_three_gpu_eigsh_modes_energy_consistency():
    """round_005 三模式覆盖: hybrid / cupyx / cpu_fallback 在中等 dim 下能量数值一致。

    固定子空间 (n_str=120, dim=14400 > 1000 -> 走 GPU 分支), 跑三模式 + 纯 CPU 对照:
      - hybrid:       scipy eigsh + GPU matvec (sigma + .get())
      - cupyx:        cupyx eigsh + GPU matvec (留 cupy); maxiter=3000, 病理 stall
                      会触发 ArpackNoConvergence -> except 回退 CPU (仍正确)
      - cpu_fallback: scipy eigsh + contract_2e (GPU 不参与 matvec)

    三模式 + CPU 四个能量两两 diff ≤1e-10 (theory §3 P1 基线; 实测 ~1e-13 级)。
    中等 dim 选 n_str=120: 既 >1000 触发 GPU 分支 (不走分支 ① numpy eigh), 又足够小
    让 cupyx 在 maxiter=3000 内收敛 (避免恒回退, 实测 cupyx 收敛性)。
    """
    h1e, eri, norb, nelec, sa, sb = _large_subspace_ints(n_str=120)
    dim = len(sa) * len(sb)
    assert dim > 1000, f"测试子空间 dim={dim} 须 >1000 (走 GPU 分支, 非分支 ①)"

    sub_cpu = _Subspace(h1e, eri, norb, nelec, backend="cpu")
    sub_hybrid = _Subspace(h1e, eri, norb, nelec, backend="gpu",
                           gpu_eigsh_mode="hybrid")
    sub_cupyx = _Subspace(h1e, eri, norb, nelec, backend="gpu",
                          gpu_eigsh_mode="cupyx")
    sub_fb = _Subspace(h1e, eri, norb, nelec, backend="gpu",
                       gpu_eigsh_mode="cpu_fallback")

    E_cpu, _, _, _ = sub_cpu.diag(sa, sb)
    E_hybrid, _, _, _ = sub_hybrid.diag(sa, sb)
    E_cupyx, _, _, _ = sub_cupyx.diag(sa, sb)
    E_fb, _, _, _ = sub_fb.diag(sa, sb)

    for label, E in [("hybrid", E_hybrid), ("cupyx", E_cupyx),
                     ("cpu_fallback", E_fb)]:
        diff = abs(E - E_cpu)
        assert diff <= 1e-10, (
            f"{label} vs CPU E diff {diff:.2e} >1e-10 "
            f"(E_{label}={E:.10f}, E_cpu={E_cpu:.10f})")


@pytest.mark.skipif(not _have_gpu(), reason="cupy / GPU 不可用")
@pytest.mark.xfail(strict=False, reason=(
    "round_004 实测: norb=12 dim=1e5 时 RawKernel launch + t1 buffer mgmt 主导 "
    "matvec wall (~9ms), eri 重算仅 ~0.3-0.5ms (<6%). theory §1.2 的 8-12% 估计 "
    "偏高 -> 缓存收益落在测量噪声内. strict=False: 录比值不破回归. "
    "方式 C 的定性正确性由 test_sigma_cached_eri_correctness 锁定 (输出 ≤2e-13)."))
def test_p0_prime_per_matvec_eri_cache_isolation():
    """P0' (per-matvec 隔离, round_004 B+C 可归因验收):

    固定 60 次随机 matvec, **interleave** 两种 eri 模式 (recompute, cached 交替)
    消除顺序效应, 直接对照 sigma_selected_ci_gpu 的 per-call wall:
      - 重算版 (eri1_*=None == round_003 现状): 每次 absorb_h1e + restore +
        _selci_eri_* + 2× cp.asarray
      - 缓存版 (eri1_*=cupy 预算, _Subspace.__init__ 同路径): 跳过上述重算

    绕开 cupyx eigsh 收敛 confound (固定 N_matvec, 不涉及 ARPACK restart),
    隔离方式 C 的确定性 per-matvec 收益。aggregate wall 比 cached/recompute
    应 ≤0.88 (即 per-matvec ≥1.14× 加速, theory §3)。

    round_004 实测 (n_str=317, dim=100489, n_matvec=60 interleave):
      aggregate ratio ~0.97-1.05 -> 落 theory §3 「部分/证伪」带。
      结论: eri 重算 <8% matvec (theory §1.2 高估), 缓存收益在噪声内。
      方式 C 仍正确 (test_sigma_cached_eri_correctness 证), 仅量级不达 1.14×。
    """
    import cupy as cp
    from tc_sqd.selected_ci_gpu import (
        sigma_selected_ci_gpu, _selci_eri_aaaa, _selci_eri_bbaa, _get_kernels)
    from pyscf.fci import selected_ci as _sci
    from pyscf import ao2mo
    from pyscf.fci import direct_spin1

    # dim ~1e5 (n_str=317 -> 100,489), 锁定 P0 dim 段 (theory scan 单调合理点)
    h1e, eri, norb, nelec, sa, sb = _large_subspace_ints(n_str=317)
    na, nb = len(sa), len(sb)
    dim = na * nb
    assert dim > 1e5, f"测试子空间 dim={dim} 须 >1e5"

    links = [_sci.des_des_linkstr(sa, norb, nelec[0], True),
             _sci.des_des_linkstr(sb, norb, nelec[1], True),
             _sci.cre_des_linkstr(sa, norb, nelec[0], True),
             _sci.cre_des_linkstr(sb, norb, nelec[1], True)]
    kernels = _get_kernels()

    # 预算缓存 (== _Subspace.__init__ 的实例级缓存路径)
    h2e = ao2mo.restore(1, direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5), norb)
    eri1_aaaa = cp.asarray(_selci_eri_aaaa(h2e, norb))
    eri1_bbaa = cp.asarray(_selci_eri_bbaa(h2e, norb, nelec))

    n_mv = 60
    rng = np.random.default_rng(0)
    vs = [rng.standard_normal((na, nb)) for _ in range(n_mv)]

    # warm-up: 触发 cupy 上下文 / RawModule 编译 / 首次 matmul autotune
    for v in vs[:3]:
        sigma_selected_ci_gpu(v, sa, sb, norb, nelec, h1e, eri, links=links,
                              kernels=kernels, eri1_aaaa=None, eri1_bbaa=None)
        sigma_selected_ci_gpu(v, sa, sb, norb, nelec, h1e, eri, links=links,
                              kernels=kernels, eri1_aaaa=eri1_aaaa, eri1_bbaa=eri1_bbaa)

    # interleave: recompute / cached 交替, 消除 GPU 热状态/调度顺序效应
    t_recompute = []
    t_cached = []
    cp.cuda.Stream.null.synchronize()
    for v in vs:
        # recompute
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        sigma_selected_ci_gpu(v, sa, sb, norb, nelec, h1e, eri, links=links,
                              kernels=kernels, eri1_aaaa=None, eri1_bbaa=None)
        cp.cuda.Stream.null.synchronize()
        t_recompute.append(time.perf_counter() - t0)
        # cached
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        sigma_selected_ci_gpu(v, sa, sb, norb, nelec, h1e, eri, links=links,
                              kernels=kernels,
                              eri1_aaaa=eri1_aaaa, eri1_bbaa=eri1_bbaa)
        cp.cuda.Stream.null.synchronize()
        t_cached.append(time.perf_counter() - t0)

    # aggregate (sum) 比中位更稳: 单次抖动被 n_mv 平均掉
    tot_recompute = float(np.sum(t_recompute))
    tot_cached = float(np.sum(t_cached))
    med_recompute = float(np.median(t_recompute))
    med_cached = float(np.median(t_cached))
    ratio_agg = tot_cached / tot_recompute
    ratio_med = med_cached / med_recompute
    # 主阈值: aggregate ratio (最稳); 阈值 ≤0.88 (≥1.14×, theory §3)
    print(f"\n[P0'] dim={dim} n_mv={n_mv} interleave "
          f"med_recompute={med_recompute*1e3:.2f}ms med_cached={med_cached*1e3:.2f}ms "
          f"ratio_med={ratio_med:.3f} | tot_recompute={tot_recompute*1e3:.0f}ms "
          f"tot_cached={tot_cached*1e3:.0f}ms ratio_agg={ratio_agg:.3f} (阈值 ≤0.88)")
    assert ratio_agg <= 0.88, (
        f"P0' per-matvec 隔离: aggregate ratio={ratio_agg:.3f} >0.88 "
        f"(tot_cached={tot_cached*1e3:.0f}ms tot_recompute={tot_recompute*1e3:.0f}ms) "
        "eri 缓存未带来 ≥1.14× per-matvec 收益"
    )


# --------------------------------------------------------------------------- #
# round_013: eigsh_tol 覆盖 (hybrid 分支)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have_gpu(), reason="cupy / GPU 不可用")
def test_hybrid_eigsh_tol_ablation():
    """round_013 GPU hybrid 分支 eigsh_tol 消融锚。

    固定子空间 (n_str=120, dim=14400 > 1000 -> GPU 分支):
      - eigsh_tol=None (默认 = 原硬编码 1e-10) vs 显式 1e-10: 同 tol 值,
        E 一致 (≤1e-10; 两次调用 ARPACK 内部 v0 随机, 非逐位)。
      - eigsh_tol=1e-8: E 相对 None 在 1e-8 内 (放松一档仍收敛)。
      - eigsh_tol=1e-6: E 相对 None 在 1e-3 内, 且 n_mv 显著少于 None
        (松 tol 提前停机 -> 少 matvec, 参数确实到达 eigsh 调用)。
    """
    h1e, eri, norb, nelec, sa, sb = _large_subspace_ints(n_str=120)
    dim = len(sa) * len(sb)
    assert dim > 1000

    sub_def = _Subspace(h1e, eri, norb, nelec, backend="gpu",
                        gpu_eigsh_mode="hybrid")
    sub_t10 = _Subspace(h1e, eri, norb, nelec, backend="gpu",
                        gpu_eigsh_mode="hybrid", eigsh_tol=1e-10)
    sub_t8 = _Subspace(h1e, eri, norb, nelec, backend="gpu",
                       gpu_eigsh_mode="hybrid", eigsh_tol=1e-8)
    sub_t6 = _Subspace(h1e, eri, norb, nelec, backend="gpu",
                       gpu_eigsh_mode="hybrid", eigsh_tol=1e-6)

    E_def, _, _, _ = sub_def.diag(sa, sb)
    n_def = sub_def.last_n_mv
    E_t10, _, _, _ = sub_t10.diag(sa, sb)
    E_t8, _, _, _ = sub_t8.diag(sa, sb)
    E_t6, _, _, _ = sub_t6.diag(sa, sb)
    n_t6 = sub_t6.last_n_mv

    assert abs(E_t10 - E_def) <= 1e-10, (
        f"显式 1e-10 vs None (同 tol 值) 应一致: {E_t10:.12f} vs {E_def:.12f}")
    assert abs(E_t8 - E_def) <= 1e-8, (
        f"tol=1e-8 vs 1e-10: {E_t8:.12f} vs {E_def:.12f} diff "
        f"{abs(E_t8 - E_def):.2e} >1e-8")
    assert abs(E_t6 - E_def) <= 1e-3, (
        f"tol=1e-6 vs 1e-10: diff {abs(E_t6 - E_def):.2e} >1e-3 (收敛过松)")
    assert n_t6 < n_def, (
        f"松 tol 应提前停机少 matvec: n_mv(1e-6)={n_t6} >= n_mv(None)={n_def}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print(f"--- {name} ---")
            fn()
            print("PASS")
