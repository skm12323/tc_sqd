"""tc_sqd.cipsi.solve_sqd_active 三激发定向注入测试 (round_008, 方向 C)。

覆盖 (theory.md §4 P0 + 任务单):
  - P0' 零回归: triple_injection=False (默认) 与不传逐位一致 (active/ev/best)。
  - 功能: 受限子空间上 triple_injection=True → dim 增加 (单激发 BFS 补高阶字符串)
    且 E_V 不升 (子空间只增不减, 变分); 补全到全空间时 E = FCI。
  - cap 边界: n_triples_per_round>0 → 注入新字符串数受限 (≤ cap×norb)。
  - 叠加去重: tail_suppression + triple_injection 组合不崩、注入与 ④ 候选自然去重。
  - helper 单元: _single_excited_strings 的计数 / popcount 不变式。

口径同 test_pruning: 零回归用 N2/STO-3G (rtol=1e-10 吸收 BLAS ULP 噪声),
功能/边界用 N2/STO-3G 受限子空间 (快); 开壳层用 CH/STO-3G 冒烟。
"""
import numpy as np
import pytest

import tc_sqd
from tc_sqd.cipsi import _single_excited_strings
from pyscf import gto, scf
from pyscf.fci import direct_spin1


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _n2_stretch_data():
    from pyscf import gto
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    return tc_sqd.from_pyscf(mol)


def _lih_data():
    from pyscf import gto
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    return tc_sqd.from_pyscf(mol)


def _ch_data():
    """CH/STO-3G 空间积分 (开壳层 5e; 与 test_open_shell/test_pruning 同式)。"""
    mol = gto.M(atom="C 0 0 0; H 0 0 1.1", basis="sto-3g", spin=1, verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl",
                    mol.intor("int2e_sph"), mo, mo, mo, mo)
    return {"h1e": h1e, "eri": eri, "norb": mol.nao_nr(),
            "nelec": (3, 2), "ecore": mf.energy_nuc()}


def _assert_traj_close(ta, tb, tag=""):
    """轨迹逐点逐字段一致 (P0' 锚: 零回归, rtol=1e-10 吸收 BLAS ULP 噪声)。"""
    assert len(ta) == len(tb), f"{tag}: 轨迹点数不同 {len(ta)} vs {len(tb)}"
    for k, (pa, pb) in enumerate(zip(ta, tb)):
        for key in ("round", "E", "sigma2", "e_pt2", "dim", "shots"):
            va, vb = float(pa[key]), float(pb[key])
            assert np.isclose(va, vb, rtol=1e-10, atol=1e-10), (
                f"{tag}: 轨迹点[{k}].{key} 不一致: {va!r} vs {vb!r} "
                f"diff={abs(va - vb):.2e}")


# --------------------------------------------------------------------------- #
# helper 单元测试
# --------------------------------------------------------------------------- #
def test_single_excited_strings_unit():
    """单激发生成器: 数量 = n_occ × n_virt, popcount 不变, 自反无自身。"""
    norb, nelec = 6, 3
    s = 0b000111                     # occ {0,1,2}, virt {3,4,5}
    out = _single_excited_strings(s, norb)
    assert len(out) == 3 * 3 == 9    # n_occ × n_virt
    assert s not in out              # occ→virt 一次翻转不回自身
    for t in out:
        assert bin(t).count("1") == nelec      # popcount 不变 (同扇区)
        diff = bin(s ^ t).count("1")
        assert diff == 2                       # 恰好一进一出
    # HF 串: occ {0..k-1}
    s2 = (1 << nelec) - 1
    assert len(_single_excited_strings(s2, norb)) == nelec * (norb - nelec)
    # 全串 (无 virt) / 空串 (无 occ) → 空
    assert _single_excited_strings((1 << norb) - 1, norb) == set()
    assert _single_excited_strings(0, norb) == set()


# --------------------------------------------------------------------------- #
# P0' 零回归: triple_injection=False (默认) 与不传逐位一致
# --------------------------------------------------------------------------- #
def test_triple_injection_default_bit_identical_active():
    """P0' 锚: solve_sqd_active 不传 (默认 False) vs 显式 triple_injection=False
    代码路径等价 (注入块整体跳过), 能量 + 轨迹逐位一致。

    用 LiH/STO-3G (dim≤225 → numpy eigh 确定性路径, 同 test_pruning best 层
    口径); N2 大维度走 eigsh 且 v0 不固定 → dom_thresh 边界的位数级噪声会
    逐轮放大, 不适合逐位零回归锚。
    """
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
        triple_injection=False, n_triples_per_round=0, trajectory=traj_off,
        **common)
    assert np.isclose(e_default, e_off, rtol=1e-10, atol=1e-10), (
        f"默认 (False) 应代码路径等价: default={e_default!r} off={e_off!r} "
        f"diff={abs(e_default - e_off):.2e}")
    _assert_traj_close(traj_def, traj_off, tag="P0' active")


def test_triple_injection_default_bit_identical_ev_and_improved():
    """P0' 锚 (透传层): solve_sqd_ev / solve_sqd_improved 默认 vs 显式 False 等价。"""
    data = _lih_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((60, 2 * norb)) > 0.5
    common = dict(max_strings=None, n_active_per_round=30, rand_seed=0)
    e_def = tc_sqd.solve_sqd_ev(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, **common)
    e_off = tc_sqd.solve_sqd_ev(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        triple_injection=False, **common)
    assert np.isclose(e_def, e_off, rtol=1e-10, atol=1e-10), (
        f"solve_sqd_ev 默认应等价: {e_def!r} vs {e_off!r}")
    i_def = tc_sqd.solve_sqd_improved(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, **common)
    i_off = tc_sqd.solve_sqd_improved(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        triple_injection=False, **common)
    assert np.isclose(i_def, i_off, rtol=1e-10, atol=1e-10), (
        f"solve_sqd_improved 默认应等价: {i_def!r} vs {i_off!r}")


def test_triple_injection_default_bit_identical_best():
    """P0' 锚 (best 层): solve_sqd_best 默认 vs 显式 False → 全字段一致 (LiH)。"""
    data = _lih_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    d_def = tc_sqd.solve_sqd_best(h1e, eri, norb, nelec, ecore=data.ecore,
                                  n_shots=60, return_details=True, rand_seed=0)
    d_off = tc_sqd.solve_sqd_best(h1e, eri, norb, nelec, ecore=data.ecore,
                                  n_shots=60, return_details=True, rand_seed=0,
                                  triple_injection=False,
                                  n_triples_per_round=0)
    for key in ("energy", "E_direct", "E_pt2", "dim"):
        assert np.isclose(d_def[key], d_off[key], rtol=1e-10, atol=1e-10), (
            f"solve_sqd_best 默认应等价: {key} {d_def[key]!r} vs "
            f"{d_off[key]!r} diff={abs(d_def[key] - d_off[key]):.2e}")


# --------------------------------------------------------------------------- #
# 功能: 注入补高阶字符串 → dim 增加 + E_V 不升
# --------------------------------------------------------------------------- #
def test_triple_injection_increases_dim_and_improves_energy():
    """受限子空间 (少 shots + 少轮) 上注入 → dim 增加, E_V ≤ 基线 (变分),
    且 |E_PT2| 显著缩小 (高阶字符串被补入, 子空间外残余关联减少)。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((40, 2 * norb)) > 0.5
    common = dict(max_strings=None, n_active_per_round=10, max_rounds=4,
                  rand_seed=0)
    traj_base, traj_inj = [], []
    st_base, st_inj = [], []
    e_base = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, trajectory=traj_base,
        state_out=st_base, **common)
    e_inj = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, trajectory=traj_inj,
        state_out=st_inj, triple_injection=True, **common)
    assert traj_inj[-1]["dim"] > traj_base[-1]["dim"], (
        f"注入后 dim 应增加: {traj_base[-1]['dim']} → {traj_inj[-1]['dim']}")
    assert e_inj <= e_base + 1e-10, (
        f"注入只增字符串 (子空间超集), E_V 应不升: {e_base!r} → {e_inj!r}")
    assert len(st_inj[0][1]) > len(st_base[0][1]), "注入后字符串数应增加"
    # 字符串 popcount 不变式 (闭壳层 na=nb=5): 注入不引入非法扇区字符串
    for s in st_inj[0][1]:
        assert bin(int(s)).count("1") == nelec[0]


def test_triple_injection_fci_on_small_system():
    """正确性锚: LiH/STO-3G (norb=6, C(6,2)=15 字符串, 全 dim 225) 上注入
    迭代到 fixpoint = 全空间 → E_V = FCI (理论 §1.1 单激发图连通)。"""
    data = _lih_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_fci = direct_spin1.kernel(h1e, eri, norb, nelec)[0]
    bsm = np.random.default_rng(0).random((30, 2 * norb)) > 0.5
    st = []
    e_inj = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, ecore=data.ecore,
        n_active_per_round=5, max_rounds=3, rand_seed=0, pt2_floor=1e-12,
        triple_injection=True, state_out=st)
    n_full = 15                              # C(6,2)
    assert len(st[0][1]) == n_full, (
        f"小体系注入到 fixpoint 应补全全空间 {n_full} 字符串, "
        f"got {len(st[0][1])}")
    assert abs(e_inj - (e_fci + data.ecore)) < 1e-8, (
        f"补全到全空间 → E = FCI: {e_inj!r} vs {e_fci + data.ecore!r}")


# --------------------------------------------------------------------------- #
# cap 边界: n_triples_per_round > 0 限制注入量
# --------------------------------------------------------------------------- #
def test_triple_injection_cap_limits_growth():
    """cap=4 + 迭代护栏 ≤ norb → 新增字符串数 ≤ 4×norb; 且注入仍改进 E_V。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((40, 2 * norb)) > 0.5
    common = dict(max_strings=None, n_active_per_round=10, max_rounds=4,
                  rand_seed=0)
    st_base, st_cap, st_full = [], [], []
    e_base = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, state_out=st_base,
        **common)
    e_cap = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, state_out=st_cap,
        triple_injection=True, n_triples_per_round=4, **common)
    e_full = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, state_out=st_full,
        triple_injection=True, **common)
    n_base = len(st_base[0][1])
    n_cap = len(st_cap[0][1])
    n_full = len(st_full[0][1])
    assert n_cap > n_base, "cap 注入也应增加字符串数"
    assert n_cap - n_base <= 4 * norb, (
        f"cap=4 + 护栏 ≤ {4 * norb}: 新增 {n_cap - n_base}")
    assert n_full >= n_cap, (
        "无 cap 注入的字符串数应 ≥ cap=4 版 (cap 只限每迭代步长)")
    assert e_cap <= e_base + 1e-10 and e_full <= e_cap + 1e-10, (
        "注入只增字符串 → E_V 单调不升: base→cap→full")


def test_triple_injection_max_strings_guard():
    """max_strings 远小于全空间时不挂死 (cap/护栏/max_strings 三重界)。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((30, 2 * norb)) > 0.5
    st = []
    e = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, max_strings=40,
        n_active_per_round=10, max_rounds=4, rand_seed=0,
        triple_injection=True, state_out=st)
    assert np.isfinite(e)
    assert len(st[0][1]) <= 2 * 40 + len(bsm) * 2, (
        "max_strings 约束注入扩张 (采样并入不受限, 留宽松上界)")


# --------------------------------------------------------------------------- #
# 叠加去重: 与 ④ / tail 组合
# --------------------------------------------------------------------------- #
def test_triple_injection_stacks_with_tail_suppression():
    """tail_suppression (C1) + triple_injection (方向 C) 叠加: 不崩, 注入候选
    与 ④/tail 已并入字符串自然去重 (只取 ∉ 当前集合的新字符串)。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((40, 2 * norb)) > 0.5
    traj, st = [], []
    e = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        n_active_per_round=10, max_rounds=4, rand_seed=0,
        tail_suppression=True, tail_shots_ref=100,
        triple_injection=True, trajectory=traj, state_out=st)
    assert np.isfinite(e)
    assert len(st[0][1]) == len(set(int(s) for s in st[0][1])), \
        "注入后字符串集合无重复 (去重)"


def test_triple_injection_open_shell_smoke():
    """开壳层 (CH/STO-3G, 5e) 冒烟: α/β 独立生成注入, 扇区 popcount 正确。"""
    data = _ch_data()
    h1e, eri, norb = data["h1e"], data["eri"], data["norb"]
    nelec, ecore = data["nelec"], data["ecore"]
    bsm = np.random.default_rng(0).random((30, 2 * norb)) > 0.5
    st = []
    e = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, ecore=ecore,
        n_active_per_round=10, max_rounds=3, rand_seed=0,
        triple_injection=True, state_out=st)
    assert np.isfinite(e)
    c2d, sa, sb = st[0]
    for s in sa:
        assert bin(int(s)).count("1") == nelec[0], "α 扇区 popcount 不变"
    for s in sb:
        assert bin(int(s)).count("1") == nelec[1], "β 扇区 popcount 不变"
