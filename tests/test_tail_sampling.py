"""tc_sqd.tail_sampling 测试 —— C1 尾部发现采样 (round_001, SQD-AA 经典模拟版)
  + round_002 C1-v2 预算随 shots 缩放 (n_tgt ∝ shots, @100 零回归)。

覆盖 (theory.md §4 P0 第 4 条):
  - 正确性: suppress_seen_bitstrings 每条保留位串至少引入一个新 α/β 字符串;
    discover_tail_pool 收集到的全是新贡献者。
  - 边界: seen=全空间→空池; 空 seen→全保留; 空 bsm→不崩; 预算上界。
  - distill 隔离 (锁认知): suppress_seen_bitstrings 的 API 不接受 c2d 参数仍正常工作。
  - 默认关回归: solve_sqd_active(tail_suppression=False) 与不带 tail 参数逐位等价。
  - 集成: tail_suppression=True 不饿死 (能量有限); solve_sqd_improved 透传可达。

round_002 增量 (theory.md round_002 §1.2/§1.6 + 任务单 P0):
  - L2 命门: @100 shots + tail_shots_ref=100 与 round_001 C1-v1 (tail_shots_ref=0)
    逐位一致 (n_tgt=30, discover_tail_pool 调用签名完全一致)。
  - n_tgt(S) 表驱动: @100→30, @200→60, @500→90, @1000→90 (cap=3 封顶)。
  - 边界: n_cur < tail_shots_ref → clip 下限 n_active_per_round。
  - n_drawn 接住 (round_001 用 _ 丢弃): >0 且 ≤ max_draw_factor × n_tgt。
  - L1 回归: tail_suppression=False 时 tail_shots_ref 取值不影响结果 (hook 跳过)。
"""
import inspect
import itertools
from unittest.mock import patch

import numpy as np
import pytest

import tc_sqd
import tc_sqd.cipsi as _cipsi_mod
from tc_sqd import suppress_seen_bitstrings, discover_tail_pool
from tc_sqd.fermion import bitstring_matrix_to_ci_strs


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _per_row_strs(bsm, norb):
    """逐行 (α_str, β_str) 整数 (与 bitstring_matrix_to_ci_strs 同位 packing)。

    测试专用: 保留行对应关系, 用于逐行断言抑制判据 (模块内部 _bsm_to_ci_strs_per_row
    的独立复现, 交叉验证 packing 一致)。
    """
    bsm = np.asarray(bsm, dtype=bool)
    alpha_bits = bsm[:, norb:]
    beta_bits = bsm[:, :norb]
    powers = (1 << np.arange(norb, dtype=np.uint64)).astype(np.int64)
    a = (alpha_bits[:, ::-1].astype(np.int64) @ powers).ravel()
    b = (beta_bits[:, ::-1].astype(np.int64) @ powers).ravel()
    return a, b


def _all_ci_strs(norb, n):
    """枚举 norb 轨道、n 电子的全部 CI 字符串整数 (轨道 0 = LSB)。"""
    out = set()
    for occ in itertools.combinations(range(norb), n):
        s = 0
        for p in occ:
            s |= 1 << p
        out.add(s)
    return out


def _n2_stretch_data():
    from pyscf import gto
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    return tc_sqd.from_pyscf(mol)


NORB4, NA4, NB4 = 4, 2, 2


def _make_bsm_probs(n_samples=80, norb=NORB4, seed=0):
    rng = np.random.default_rng(seed)
    bsm = rng.random((n_samples, 2 * norb)) > 0.5
    probs = np.full(n_samples, 1.0 / n_samples)
    return bsm, probs


def _moderate_occ(norb=NORB4):
    # 不锐的占据 → 恢复映像不退化到单一 Det, 给抑制留多样本
    return (np.array([0.7, 0.6, 0.4, 0.3])[:norb] if norb == 4
            else np.full(norb, 0.5))


def _spy_discover_tail_pool(capture_list):
    """构造一个 side_effect 包裹真实 discover_tail_pool, 把每次调用的
    ``n_target_new`` / ``max_draw_factor`` 入参 + 返回的 ``n_drawn`` 记入 capture_list。

    round_002 测试用: 直接断言 hook 传给 discover_tail_pool 的 n_tgt (= n_target_new)
    是否符合 §1.2 shots 缩放公式, 以及 n_drawn 是否被 hook 接住 (>0 且 ≤ 预算)。

    引用源模块 ``tc_sqd.tail_sampling.discover_tail_pool`` (不受 cipsi 命名空间 patch
    影响), 保证 spy 调用的是真实函数。
    """
    from tc_sqd.tail_sampling import discover_tail_pool as _real

    def _wrapper(*args, **kwargs):
        n_tgt_call = int(kwargs.get("n_target_new", 0))
        mdf_call = int(kwargs.get("max_draw_factor", 0))
        bsm, probs, n_drawn = _real(*args, **kwargs)
        capture_list.append({
            "n_target_new": n_tgt_call,
            "max_draw_factor": mdf_call,
            "n_drawn": int(n_drawn),
        })
        return bsm, probs, n_drawn

    return _wrapper


# --------------------------------------------------------------------------- #
# suppress_seen_bitstrings: 正确性 + 边界 + distill 隔离
# --------------------------------------------------------------------------- #
def test_suppress_seen_keeps_only_new_contributors():
    """正确性: 每条保留位串至少引入一个新字符串 (α∉seen_a 或 β∉seen_b)。

    构造 seen_a/seen_b 含恢复后字符串的**一部分**; 断言输出中不存在
    (α∈seen_a ∧ β∈seen_b) 的行 (即对乘积子空间零新增者全被抑制)。
    """
    bsm, probs = _make_bsm_probs()
    occ = _moderate_occ()
    rec, _ = tc_sqd.recover_configurations(bsm, probs, (occ, occ), NA4, NB4,
                                           rand_seed=0)
    a_unique, _ = bitstring_matrix_to_ci_strs(rec)        # 闭壳层: α=β 合并集
    all_strs = sorted(int(x) for x in a_unique)
    assert len(all_strs) >= 2, "恢复映像过窄, 测试无意义"
    half = len(all_strs) // 2
    seen_a = set(all_strs[:half])                          # 含一部分
    seen_b = set(all_strs[:half])

    out_bsm, out_probs = suppress_seen_bitstrings(
        bsm, probs, (occ, occ), NA4, NB4, seen_a=seen_a, seen_b=seen_b,
        rand_seed=0,
    )
    # 概率重归一化
    assert out_probs.shape[0] == out_bsm.shape[0]
    if out_probs.shape[0] > 0:
        assert np.isclose(out_probs.sum(), 1.0), "概率应重归一化到 1"
    # 每条保留行: α∉seen_a 或 β∉seen_b (至少引入一个新字符串)
    a_rows, b_rows = _per_row_strs(out_bsm, NORB4)
    for a, b in zip(a_rows, b_rows):
        ai, bi = int(a), int(b)
        assert (ai not in seen_a) or (bi not in seen_b), (
            f"保留行 ({ai:#x},{bi:#x}) 对子空间零新增, 不应被保留")


def test_suppress_seen_full_space_empty_pool():
    """边界: seen_a/seen_b 覆盖全空间 → 所有恢复 det 被抑制 → 返回空池。"""
    bsm, probs = _make_bsm_probs()
    occ = _moderate_occ()
    full = _all_ci_strs(NORB4, NA4)                        # 闭壳层: α=β 同集
    out_bsm, out_probs = suppress_seen_bitstrings(
        bsm, probs, (occ, occ), NA4, NB4,
        seen_a=full, seen_b=full, rand_seed=0,
    )
    assert out_bsm.shape == (0, 2 * NORB4), f"应空池, got {out_bsm.shape}"
    assert out_probs.shape == (0,)


def test_suppress_seen_empty_seen_keeps_all():
    """边界: seen 为空 → 不抑制, 保留全部恢复后 det (重归一化)。"""
    bsm, probs = _make_bsm_probs()
    occ = _moderate_occ()
    rec, _ = tc_sqd.recover_configurations(bsm, probs, (occ, occ), NA4, NB4,
                                           rand_seed=0)
    out_bsm, out_probs = suppress_seen_bitstrings(
        bsm, probs, (occ, occ), NA4, NB4,
        seen_a=set(), seen_b=set(), rand_seed=0,
    )
    # 空 seen → 无抑制; recover 内部去重后行数应与直接 recover 一致
    assert out_bsm.shape[0] == rec.shape[0]
    assert np.isclose(out_probs.sum(), 1.0)


def test_suppress_seen_empty_bsm_no_crash():
    """边界: 空 bsm → 返回空池, 不崩 (不调用 recover_configurations 的空校验)。"""
    occ = _moderate_occ()
    empty = np.empty((0, 2 * NORB4), dtype=bool)
    out_bsm, out_probs = suppress_seen_bitstrings(
        empty, np.empty(0), (occ, occ), NA4, NB4,
        seen_a=set(), seen_b=set(), rand_seed=0,
    )
    assert out_bsm.shape == (0, 2 * NORB4)
    assert out_probs.shape == (0,)


def test_suppress_seen_no_c2d_param_distill_isolation():
    """distill 隔离 (锁认知): API 不接受 c2d 参数, 无 c2d 仍正常工作。

    参照 test_include_excitations_not_fci_strong_correlation 风格: 用反例/边界
    锁住 "C1 ≠ distill" 的认知 —— C1 代码路径绝不读 c2d, 只读 (seen_a, seen_b)。
    """
    sig = inspect.signature(suppress_seen_bitstrings)
    assert "c2d" not in sig.parameters, (
        "suppress_seen_bitstrings 绝不接受 c2d (distill 边界: C1 ≠ distill)")
    assert "seen_a" in sig.parameters and "seen_b" in sig.parameters
    # 无 c2d 输入下正常工作 (功能正确性不依赖任何解态振幅)
    bsm, probs = _make_bsm_probs()
    occ = _moderate_occ()
    out_bsm, _ = suppress_seen_bitstrings(
        bsm, probs, (occ, occ), NA4, NB4,
        seen_a={0}, seen_b={0}, rand_seed=0,
    )
    assert out_bsm.shape[1] == 2 * NORB4


def test_suppress_seen_rejects_bad_seen_type():
    """输入校验: seen_a/seen_b 须为 int 的 set/frozenset。"""
    bsm, probs = _make_bsm_probs()
    occ = _moderate_occ()
    with pytest.raises(TypeError):
        suppress_seen_bitstrings(bsm, probs, (occ, occ), NA4, NB4,
                                 seen_a=[1, 2], seen_b=set())   # list 非法
    with pytest.raises(TypeError):
        suppress_seen_bitstrings(bsm, probs, (occ, occ), NA4, NB4,
                                 seen_a={"x"}, seen_b=set())    # 非 int 元素


# --------------------------------------------------------------------------- #
# discover_tail_pool: 正确性 + 边界 + 预算
# --------------------------------------------------------------------------- #
def test_discover_tail_pool_finds_new_contributors():
    """正确性: discover 收集到的全是新贡献者 (α∉seen_a 或 β∉seen_b)。

    bootstrap 模式过抽: seen 含一部分, discover 返回的每条均引入新字符串。
    """
    occ = _moderate_occ()
    half = set(list(_all_ci_strs(NORB4, NA4))[:2])         # seen 含 2 个
    bsm, probs, n_drawn = discover_tail_pool(
        (occ, occ), NA4, NB4, NORB4,
        seen_a=half, seen_b=half, n_target_new=5,
        base_distribution="bootstrap", max_draw_factor=10, rand_seed=0,
    )
    assert bsm.shape[1] == 2 * NORB4
    assert bsm.shape[0] >= 1, "应能发现新贡献者 (全空间 6, seen 仅 2)"
    # 每条: α∉seen_a 或 β∉seen_b; 且池内去重
    a_rows, b_rows = _per_row_strs(bsm, NORB4)
    pairs = set()
    for a, b in zip(a_rows, b_rows):
        ai, bi = int(a), int(b)
        assert (ai not in half) or (bi not in half), (
            f"发现的 ({ai:#x},{bi:#x}) 对 seen 零新增")
        assert (ai, bi) not in pairs, "池内应去重"
        pairs.add((ai, bi))
    assert n_drawn > 0
    assert np.isclose(probs.sum(), 1.0)


def test_discover_tail_pool_seen_full_returns_empty():
    """边界: seen 覆盖全空间 → discover 无新可发现 → 空池 (调用方回退原池)。"""
    occ = _moderate_occ()
    full = _all_ci_strs(NORB4, NA4)
    bsm, probs, n_drawn = discover_tail_pool(
        (occ, occ), NA4, NB4, NORB4,
        seen_a=full, seen_b=full, n_target_new=5,
        base_distribution="bootstrap", max_draw_factor=4, rand_seed=0,
    )
    assert bsm.shape == (0, 2 * NORB4), "全空间 seen → 应空池"
    assert probs.shape == (0,)
    assert n_drawn == 4 * 5                            # 耗尽预算


def test_discover_tail_pool_budget_respected():
    """预算上界: n_drawn <= max_draw_factor * n_target_new。"""
    occ = _moderate_occ()
    seen = set(list(_all_ci_strs(NORB4, NA4))[:1])
    for mdf in (1, 3, 10):
        _, _, n_drawn = discover_tail_pool(
            (occ, occ), NA4, NB4, NORB4,
            seen_a=seen, seen_b=seen, n_target_new=8,
            max_draw_factor=mdf, rand_seed=0,
        )
        assert n_drawn <= mdf * 8, f"n_drawn={n_drawn} 超预算 {mdf*8}"


def test_discover_tail_pool_empty_seen_collects_target():
    """边界: 空 seen → 全是新的 → 至少收集到 n_target_new 个 (除非预算/全空间不够)。"""
    occ = _moderate_occ()
    n_tgt = 4
    bsm, probs, n_drawn = discover_tail_pool(
        (occ, occ), NA4, NB4, NORB4,
        seen_a=set(), seen_b=set(), n_target_new=n_tgt,
        max_draw_factor=10, rand_seed=0,
    )
    # 空 seen + 过抽: 应能收齐 n_target_new (norb=4 闭壳层恢复映像足够多样)
    assert bsm.shape[0] >= n_tgt, (
        f"空 seen 应收齐 {n_tgt}, got {bsm.shape[0]}")
    assert np.isclose(probs.sum(), 1.0)


def test_discover_tail_pool_circuit_not_implemented():
    """P1 占位: base_distribution='circuit' 在 round_001 P0 抛 NotImplementedError。"""
    occ = _moderate_occ()
    with pytest.raises(NotImplementedError):
        discover_tail_pool((occ, occ), NA4, NB4, NORB4,
                           seen_a=set(), seen_b=set(), n_target_new=4,
                           base_distribution="circuit", circuit=object())


def test_discover_tail_pool_validates_inputs():
    """输入校验: base_distribution 非法 / n_target_new<=0 / max_draw_factor<1 / occ 越界。"""
    occ = _moderate_occ()
    with pytest.raises(ValueError):
        discover_tail_pool((occ, occ), NA4, NB4, NORB4, seen_a=set(),
                           seen_b=set(), n_target_new=4,
                           base_distribution="bogus")
    with pytest.raises(ValueError):
        discover_tail_pool((occ, occ), NA4, NB4, NORB4, seen_a=set(),
                           seen_b=set(), n_target_new=0)
    with pytest.raises(ValueError):
        discover_tail_pool((occ, occ), NA4, NB4, NORB4, seen_a=set(),
                           seen_b=set(), n_target_new=4, max_draw_factor=0)
    with pytest.raises(ValueError):
        discover_tail_pool((occ, occ), 99, NB4, NORB4, seen_a=set(),
                           seen_b=set(), n_target_new=4)


# --------------------------------------------------------------------------- #
# solve_sqd_active: 默认关回归 (逐位等价) + 启用不饿死
# --------------------------------------------------------------------------- #
def test_solve_sqd_active_tail_off_is_bit_identical():
    """默认关回归: tail_suppression=False (显式) 与不带 tail 参数**代码路径等价**。

    hook 在 tail_suppression=False 时整块跳过 → 计算路径与原 solve_sqd_active 完全
    一致 → 同 rand_seed 下结果应一致到 BLAS 线程约整数级别。真正的逻辑分歧会表现为
    ~1e-4 量级; 此处用 rtol=1e-10 锁 "默认关零行为变化" (远严于任何真实分歧, 远松于
    多线程 eigh 的 ULP 噪声 ~1e-13)。R4 全库回归 0 失败的根因。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((100, 2 * norb)) > 0.5
    probs = np.full(100, 1.0 / 100)
    common = dict(max_strings=None, n_active_per_round=50, max_rounds=8,
                  rand_seed=0)
    e_default = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs, **common)
    e_off = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        tail_suppression=False, tail_max_draw_factor=10,
        tail_n_target_per_round=0, **common)
    # rtol=1e-10: 吸收多线程 BLAS eigh 的 ULP 级噪声 (~1e-13), 仍远严于真实分歧 (~1e-4)
    assert np.isclose(e_default, e_off, rtol=1e-10, atol=1e-10), (
        f"默认关应代码路径等价: default={e_default!r} off={e_off!r} "
        f"diff={abs(e_default - e_off):.2e}")


def test_solve_sqd_active_tail_on_runs_without_starving():
    """集成: tail_suppression=True 不饿死 (能量有限, 子空间非空)。

    C1 hook 每轮过抽 + 抑制; seen 覆盖恢复映像时回退原池 → ② 仍有子空间可对角化。
    不断言 "优于 baseline" (那是 R5 的 A/B 对照 + P0 阈值职责), 只锁 "机制可运行"。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((100, 2 * norb)) > 0.5
    probs = np.full(100, 1.0 / 100)
    e_c1 = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        max_strings=None, n_active_per_round=50, max_rounds=8, rand_seed=0,
        tail_suppression=True, tail_max_draw_factor=10,
        tail_n_target_per_round=50,
    )
    assert np.isfinite(e_c1), "C1 启用应返回有限能量 (不饿死)"


def test_solve_sqd_improved_tail_passthrough():
    """透传: solve_sqd_improved(tail_suppression=True) 经 solve_sqd_ev 到达
    solve_sqd_active (A/B 对照入口); 返回有限能量。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((100, 2 * norb)) > 0.5
    e_c1 = tc_sqd.solve_sqd_improved(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        max_strings=None, n_active_per_round=30, rand_seed=0,
        tail_suppression=True, tail_max_draw_factor=10,
        tail_n_target_per_round=30,
    )
    assert np.isfinite(e_c1), "solve_sqd_improved 透传 tail_suppression 应达"


def test_solve_sqd_improved_tail_off_bit_identical():
    """默认关回归 (透传层): solve_sqd_improved 不带 tail 参数 vs tail_suppression=False
    经 solve_sqd_ev 透传后代码路径等价 (rtol=1e-10, 同 active 测试口径, 吸收 BLAS 噪声)。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((100, 2 * norb)) > 0.5
    common = dict(max_strings=None, n_active_per_round=30, rand_seed=0)
    e_default = tc_sqd.solve_sqd_improved(h1e, eri, norb, nelec,
                                          bitstring_matrix=bsm, **common)
    e_off = tc_sqd.solve_sqd_improved(h1e, eri, norb, nelec,
                                      bitstring_matrix=bsm,
                                      tail_suppression=False, **common)
    assert np.isclose(e_default, e_off, rtol=1e-10, atol=1e-10), (
        f"默认关 (透传) 应等价: default={e_default!r} off={e_off!r} "
        f"diff={abs(e_default - e_off):.2e}")


# --------------------------------------------------------------------------- #
# round_002: C1-v2 预算随 shots 缩放 (n_tgt ∝ shots, @100 零回归)
# ---------------------------------------------------------------------------
# 命门 (theory.md round_002 §1.6 + 任务单硬要求): tail_shots_ref=100 时 @100 shots
# 必须给 n_tgt=30 (= round_001 C1-v1), 且 discover_tail_pool 其余调用参数完全一致 →
# RNG 序列逐位一致 → 输出与 round_001 逐位相同。本节测试锁该保证 + 缩放生效。
def test_solve_sqd_active_c1v2_at100_zero_regression_vs_round001():
    """L2 命门: @100 shots + tail_shots_ref=100 与 round_001 C1-v1 (tail_shots_ref=0)
    逐位一致 (np.isclose rtol=1e-10)。

    n_tgt(100) = clip(ceil(30*100/100), 30, 90) = 30 = round_001 (tail_n_target=0 →
    n_active=30)。两者 discover_tail_pool 调用签名完全一致 (n_target_new=30,
    max_draw_factor=10, rand_seed=rand_seed+r+1, seen_a/seen_b 同源) → RNG 推进序列
    一致 → bsm_new/probs_new 逐位一致 → recover/diag 逐位一致 → 能量 rtol=1e-10。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((100, 2 * norb)) > 0.5
    probs = np.full(100, 1.0 / 100)
    common = dict(max_strings=None, n_active_per_round=30, max_rounds=5, rand_seed=0)
    e_v1 = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        tail_suppression=True, tail_max_draw_factor=10,
        tail_n_target_per_round=0, tail_shots_ref=0, **common)        # round_001 路径
    e_v2 = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        tail_suppression=True, tail_max_draw_factor=10,
        tail_n_target_per_round=0, tail_shots_ref=100, **common)       # C1-v2 @100
    assert np.isclose(e_v1, e_v2, rtol=1e-10, atol=1e-10), (
        f"@100 L2 零回归失败: v1(tail_shots_ref=0)={e_v1!r} "
        f"v2(tail_shots_ref=100)={e_v2!r} diff={abs(e_v1 - e_v2):.2e}")


def test_solve_sqd_active_c1v2_n_tgt_scales_with_shots():
    """n_tgt(S) 表驱动 (theory.md round_002 §1.2 + 任务单 P0 #4):
    @100→30, @200→60, @500→90, @1000→90 (cap=3 封顶)。

    用 spy 包裹 cipsi 命名空间的 discover_tail_pool, 断言 hook 每轮传的 n_target_new
    == 预期。max_rounds=1 保持每例只 1 次调用, 快速且确定性。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    common = dict(max_strings=None, n_active_per_round=30, max_rounds=1,
                  rand_seed=0, tail_suppression=True, tail_shots_ref=100)
    # (n_shots, expected_n_tgt)
    table = [(100, 30), (200, 60), (500, 90), (1000, 90)]
    for n_shots, expected_n_tgt in table:
        bsm = np.random.default_rng(0).random((n_shots, 2 * norb)) > 0.5
        probs = np.full(n_shots, 1.0 / n_shots)
        calls = []
        with patch.object(_cipsi_mod, "discover_tail_pool",
                          side_effect=_spy_discover_tail_pool(calls)):
            tc_sqd.solve_sqd_active(
                h1e, eri, norb, nelec, bitstring_matrix=bsm,
                probabilities=probs, **common)
        assert len(calls) >= 1, f"@{n_shots}: hook 应至少调用 1 次 discover_tail_pool"
        for c in calls:
            assert c["n_target_new"] == expected_n_tgt, (
                f"@{n_shots} shots: n_tgt 应为 {expected_n_tgt} (cap=3, n_active=30), "
                f"got {c['n_target_new']}")


def test_solve_sqd_active_c1v2_boundary_n_cur_below_ref_floors_to_n_active():
    """边界 (任务单 P0 #3 边界): n_cur < tail_shots_ref 时 n_tgt 不低于 n_active_per_round
    (clip 下限)。

    构造 @50 shots + tail_shots_ref=100: ceil(30*50/100)=15 → clip(15, 30, 90) = 30
    (下限兜底)。用 spy 断言 n_target_new=30 (= n_active_per_round)。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    n_shots = 50
    bsm = np.random.default_rng(0).random((n_shots, 2 * norb)) > 0.5
    probs = np.full(n_shots, 1.0 / n_shots)
    calls = []
    with patch.object(_cipsi_mod, "discover_tail_pool",
                      side_effect=_spy_discover_tail_pool(calls)):
        tc_sqd.solve_sqd_active(
            h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
            max_strings=None, n_active_per_round=30, max_rounds=1, rand_seed=0,
            tail_suppression=True, tail_shots_ref=100)
    assert len(calls) >= 1
    for c in calls:
        assert c["n_target_new"] == 30, (
            f"@{n_shots} (<ref=100): n_tgt 应被下限 clip 到 30 (n_active), "
            f"got {c['n_target_new']}")


def test_solve_sqd_active_c1v2_n_drawn_captured_and_bounded():
    """n_drawn 接住 (theory.md round_002 §4 P0 #5 + 任务单 P0 #2):
    hook 把 discover_tail_pool 返回的 n_drawn 接住 (round_001 用 _ 丢弃), 且
    满足契约 0 < n_drawn ≤ max_draw_factor × n_target_new。

    用 spy 同时捕获入参 (n_target_new/max_draw_factor) 与返回 (n_drawn), 锁该契约。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    n_shots = 200
    bsm = np.random.default_rng(0).random((n_shots, 2 * norb)) > 0.5
    probs = np.full(n_shots, 1.0 / n_shots)
    calls = []
    with patch.object(_cipsi_mod, "discover_tail_pool",
                      side_effect=_spy_discover_tail_pool(calls)):
        tc_sqd.solve_sqd_active(
            h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
            max_strings=None, n_active_per_round=30, max_rounds=3, rand_seed=0,
            tail_suppression=True, tail_shots_ref=100, tail_max_draw_factor=10)
    assert len(calls) >= 1, "应至少 1 次 discover_tail_pool 调用"
    for c in calls:
        assert c["n_drawn"] > 0, f"n_drawn 应 >0, got {c['n_drawn']}"
        budget = c["max_draw_factor"] * c["n_target_new"]
        assert c["n_drawn"] <= budget, (
            f"n_drawn={c['n_drawn']} 超 预算 {budget} "
            f"(= max_draw_factor {c['max_draw_factor']} × n_tgt {c['n_target_new']})")


def test_solve_sqd_active_tail_shots_ref_inert_when_suppression_off():
    """L1 回归 (新默认值): tail_suppression=False 时 tail_shots_ref 取值不影响结果
    (hook 整块跳过, tail_shots_ref 不被读取)。

    锁住 "默认 tail_shots_ref=100 不会污染 tail_suppression=False 的代码路径" ——
    R4 全库 167 测试 0 失败的根因。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((100, 2 * norb)) > 0.5
    probs = np.full(100, 1.0 / 100)
    common = dict(max_strings=None, n_active_per_round=30, max_rounds=5, rand_seed=0)
    e_ref = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        tail_suppression=False, **common)                              # 默认 tail_shots_ref=100
    for sref in (0, 100, 999):
        e = tc_sqd.solve_sqd_active(
            h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
            tail_suppression=False, tail_shots_ref=sref, **common)
        assert np.isclose(e, e_ref, rtol=1e-10, atol=1e-10), (
            f"tail_suppression=False 时 tail_shots_ref={sref} 应无影响: "
            f"diff={abs(e - e_ref):.2e}")


def test_solve_sqd_improved_c1v2_passthrough_n_tgt_scales():
    """透传 + 缩放 (theory.md round_002 §2.1): solve_sqd_improved(tail_suppression=True,
    tail_shots_ref=100) 经 solve_sqd_ev 到达 solve_sqd_active hook, @500 shots 给
    n_tgt=90 (C1-v2 缩放生效, R5 A/B 对照入口)。返回有限能量 + spy 锁 n_tgt=90。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    n_shots = 500
    bsm = np.random.default_rng(0).random((n_shots, 2 * norb)) > 0.5
    calls = []
    with patch.object(_cipsi_mod, "discover_tail_pool",
                      side_effect=_spy_discover_tail_pool(calls)):
        e = tc_sqd.solve_sqd_improved(
            h1e, eri, norb, nelec, bitstring_matrix=bsm,
            max_strings=None, n_active_per_round=30, rand_seed=0,
            tail_suppression=True, tail_shots_ref=100)
    assert np.isfinite(e), "solve_sqd_improved 透传 C1-v2 应返回有限能量"
    assert len(calls) >= 1, "hook 应被透传触发"
    for c in calls:
        assert c["n_target_new"] == 90, (
            f"@500 shots 经 solve_sqd_improved 透传: n_tgt 应为 90 (C1-v2 缩放), "
            f"got {c['n_target_new']}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_tail_sampling: all PASS")
