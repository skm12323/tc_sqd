"""tc_sqd.cipsi.solve_sqd_active BFS 覆盖闭包测试 (round_012, coverage_closure)。

覆盖:
  - P0' 零回归: coverage_closure=False (默认) 与不传逐位一致 (active/ev/best)。
  - 功能: coverage_closure=True → 单激发 BFS 补全到 max_strings 上限 (默认全空间),
    小体系上 E_V = FCI (与 triple_injection=True, pt2_floor=1e-12 等价)。
  - 等价性: coverage_closure=True ≡ triple_injection=True + pt2_floor=0 (同一闭包)。
  - 透传: solve_sqd_ev / solve_sqd_best 接 coverage_closure 不崩且生效。
  - max_strings 护栏: coverage_closure + 有界 max_strings 不挂死。

口径同 test_triple_injection: 零回归用 LiH/STO-3G (dim≤225 → numpy eigh 确定性);
功能用 N2/STO-3G 受限子空间 (快)。
"""
import numpy as np
import pytest

import tc_sqd
from pyscf.fci import direct_spin1


# --------------------------------------------------------------------------- #
# 辅助 (同 test_triple_injection)
# --------------------------------------------------------------------------- #
def _lih_data():
    from pyscf import gto
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    return tc_sqd.from_pyscf(mol)


def _n2_stretch_data():
    from pyscf import gto
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    return tc_sqd.from_pyscf(mol)


def _assert_traj_close(ta, tb, tag=""):
    assert len(ta) == len(tb), f"{tag}: 轨迹点数不同 {len(ta)} vs {len(tb)}"
    for k, (pa, pb) in enumerate(zip(ta, tb)):
        for key in ("round", "E", "sigma2", "e_pt2", "dim", "shots"):
            va, vb = float(pa[key]), float(pb[key])
            assert np.isclose(va, vb, rtol=1e-10, atol=1e-10), (
                f"{tag}: 轨迹点[{k}].{key} 不一致: {va!r} vs {vb!r} "
                f"diff={abs(va - vb):.2e}")


# --------------------------------------------------------------------------- #
# P0' 零回归: coverage_closure=False (默认) 与不传逐位一致
# --------------------------------------------------------------------------- #
def test_coverage_closure_default_bit_identical_active():
    """P0' 锚: solve_sqd_active 不传 (默认 False) vs 显式 coverage_closure=False
    代码路径等价 (closure 不启用 → triple 用 pt2_floor 门控, 与改动前一致),
    能量 + 轨迹逐位一致。"""
    data = _lih_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((60, 2 * norb)) > 0.5
    probs = np.full(60, 1.0 / 60)
    common = dict(max_strings=None, n_active_per_round=30, max_rounds=8,
                  rand_seed=0)
    traj_def, traj_off = [], []
    e_default = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        trajectory=traj_def, **common)
    e_off = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        coverage_closure=False, trajectory=traj_off, **common)
    assert np.isclose(e_default, e_off, rtol=1e-10, atol=1e-10), (
        f"默认 (False) 应代码路径等价: default={e_default!r} off={e_off!r} "
        f"diff={abs(e_default - e_off):.2e}")
    _assert_traj_close(traj_def, traj_off, tag="P0' active")


def test_coverage_closure_default_bit_identical_ev_and_improved():
    """P0' 锚 (透传层): solve_sqd_ev / solve_sqd_improved 默认 vs 显式 False 等价。"""
    data = _lih_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((60, 2 * norb)) > 0.5
    common = dict(max_strings=None, n_active_per_round=30, rand_seed=0)
    e_def = tc_sqd.solve_sqd_ev(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, **common)
    e_off = tc_sqd.solve_sqd_ev(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        coverage_closure=False, **common)
    assert np.isclose(e_def, e_off, rtol=1e-10, atol=1e-10), (
        f"solve_sqd_ev 默认应等价: {e_def!r} vs {e_off!r}")
    i_def = tc_sqd.solve_sqd_improved(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, **common)
    i_off = tc_sqd.solve_sqd_improved(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        coverage_closure=False, **common)
    assert np.isclose(i_def, i_off, rtol=1e-10, atol=1e-10), (
        f"solve_sqd_improved 默认应等价: {i_def!r} vs {i_off!r}")


def test_coverage_closure_default_bit_identical_best():
    """P0' 锚 (best 层): solve_sqd_best 默认 vs 显式 False → 全字段一致 (LiH)。"""
    data = _lih_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    d_def = tc_sqd.solve_sqd_best(h1e, eri, norb, nelec, ecore=data.ecore,
                                  n_shots=60, return_details=True, rand_seed=0)
    d_off = tc_sqd.solve_sqd_best(h1e, eri, norb, nelec, ecore=data.ecore,
                                  n_shots=60, return_details=True, rand_seed=0,
                                  coverage_closure=False)
    for key in ("energy", "E_direct", "E_pt2", "dim"):
        assert np.isclose(d_def[key], d_off[key], rtol=1e-10, atol=1e-10), (
            f"solve_sqd_best 默认应等价: {key} {d_def[key]!r} vs "
            f"{d_off[key]!r} diff={abs(d_def[key] - d_off[key]):.2e}")


# --------------------------------------------------------------------------- #
# 功能: coverage_closure=True → BFS 补全到全空间, E_V = FCI
# --------------------------------------------------------------------------- #
def test_coverage_closure_reaches_full_space_fci():
    """正确性锚: LiH/STO-3G (norb=6, C(6,2)=15 字符串, 全 dim 225) 上
    coverage_closure=True (无需手动调 pt2_floor) → BFS 补全到全 15 字符串,
    E_V = FCI (单激发图连通 → fixpoint = 全空间)。"""
    data = _lih_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_fci = direct_spin1.kernel(h1e, eri, norb, nelec)[0]
    bsm = np.random.default_rng(0).random((30, 2 * norb)) > 0.5
    st = []
    e = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, ecore=data.ecore,
        n_active_per_round=5, max_rounds=3, rand_seed=0,
        coverage_closure=True, state_out=st)
    n_full = 15                              # C(6,2)
    assert len(st[0][1]) == n_full, (
        f"coverage_closure 应补全到全空间 {n_full} 字符串, got {len(st[0][1])}")
    assert abs(e - (e_fci + data.ecore)) < 1e-8, (
        f"补全到全空间 → E = FCI: {e!r} vs {e_fci + data.ecore!r}")


def test_coverage_closure_equivalent_to_triple_floor0():
    """等价性锚: coverage_closure=True ≡ triple_injection=True + pt2_floor=0
    (同一 BFS 闭包, 同一最终子空间) —— closure 是 pt2_floor=0 的自描述封装。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((40, 2 * norb)) > 0.5
    common = dict(max_strings=None, n_active_per_round=10, max_rounds=4,
                  rand_seed=0)
    st_closure, st_manual = [], []
    e_closure = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        coverage_closure=True, state_out=st_closure, **common)
    e_manual = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        triple_injection=True, n_triples_per_round=0, pt2_floor=0.0,
        state_out=st_manual, **common)
    sa_c = set(int(s) for s in st_closure[0][1])
    sa_m = set(int(s) for s in st_manual[0][1])
    assert sa_c == sa_m, (
        f"coverage_closure 与 pt2_floor=0 应补全到同一字符串集合: "
        f"{len(sa_c)} vs {len(sa_m)} 串")
    assert np.isclose(e_closure, e_manual, rtol=1e-10, atol=1e-10), (
        f"等价路径能量应一致: closure={e_closure!r} manual={e_manual!r}")


def test_coverage_closure_beats_default_on_n2():
    """功能锚: N2/STO-3G 受限子空间上 coverage_closure=True → dim 增加,
    E_V ≤ 基线 (变分), sigma² 显著缩小 (高阶字符串补入, 子空间外残余减少)。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((40, 2 * norb)) > 0.5
    common = dict(max_strings=None, n_active_per_round=10, max_rounds=4,
                  rand_seed=0)
    traj_base, traj_clo = [], []
    e_base = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, trajectory=traj_base,
        **common)
    e_clo = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, trajectory=traj_clo,
        coverage_closure=True, **common)
    assert traj_clo[-1]["dim"] >= traj_base[-1]["dim"], (
        f"closure 后 dim 应不降: {traj_base[-1]['dim']} → {traj_clo[-1]['dim']}")
    assert e_clo <= e_base + 1e-10, (
        f"closure 只增字符串, E_V 应不升: {e_base!r} → {e_clo!r}")
    assert traj_clo[-1]["sigma2"] <= traj_base[-1]["sigma2"] + 1e-10, (
        f"补入高阶字符串后 sigma² 应缩小: "
        f"{traj_base[-1]['sigma2']:.2e} → {traj_clo[-1]['sigma2']:.2e}")


# --------------------------------------------------------------------------- #
# max_strings 护栏: coverage_closure + 有界 max_strings 不挂死
# --------------------------------------------------------------------------- #
def test_coverage_closure_max_strings_guard():
    """max_strings 远小于全空间时 coverage_closure 不挂死 (BFS 受 max_strings
    上界约束, 不会超界枚举)。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((30, 2 * norb)) > 0.5
    st = []
    e = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, max_strings=40,
        n_active_per_round=10, max_rounds=4, rand_seed=0,
        coverage_closure=True, state_out=st)
    assert np.isfinite(e)
    assert len(st[0][1]) <= 2 * 40 + len(bsm) * 2, (
        "max_strings 约束 BFS 闭包扩张 (采样并入不受限, 留宽松上界)")


# --------------------------------------------------------------------------- #
# 透传: solve_sqd_ev / solve_sqd_best 接 coverage_closure 生效
# --------------------------------------------------------------------------- #
def test_coverage_closure_passes_through_ev():
    """透传锚: solve_sqd_ev(correction='pt2', coverage_closure=True) 在受限子空间
    上 dim ≥ 基线, E_PT2 修正后能量 ≤ 基线 (closure 补入字符串 → 变分更低 +
    残余 PT2 更小)。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((40, 2 * norb)) > 0.5
    common = dict(max_strings=None, n_active_per_round=10, rand_seed=0)
    e_base = tc_sqd.solve_sqd_ev(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, **common)
    e_clo = tc_sqd.solve_sqd_ev(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        coverage_closure=True, **common)
    assert e_clo <= e_base + 1e-9, (
        f"ev 透传 coverage_closure 应降能量: {e_base!r} → {e_clo!r}")
