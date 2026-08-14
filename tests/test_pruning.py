"""tc_sqd.cipsi.solve_sqd_active PT2 排序剪枝测试 (round_007, 方向 B)。

覆盖 (theory.md §4 P0 + 任务单 P0'):
  - P0' 零回归: prune_keep=1.0 (默认) 与不传逐位一致 (active/ev/improved/best)。
  - 剪枝功能: prune_keep=0.6 → dim ≈ 0.36×, E_V 升高 (变分), |E_PT2| 增大
    (theory §1.3 正信号), E_V'+E_PT2 仍准 FCI (PT2 二阶回补, theory §1.2)。
  - 边界: prune_keep ∈ {0, 1.5} → ValueError; 极小 prune_keep 空子空间防护
    (每自旋保留 ≥1 字符串); 闭壳层剪后 str_a == str_b 不变式 (§2.3 合并权重);
    开壳层 α/β 独立剪 (各自 ceil)。
  - 透传: solve_sqd_ev / solve_sqd_improved / solve_sqd_best 透传 prune_keep。

慢测试只放核心锚 (N2/STO-3G 全空间回补); 零回归/边界用 LiH/STO-3G
(dim≤225 → numpy eigh 确定性路径), 控制全文件 wall。
"""
import math

import numpy as np
import pytest

import tc_sqd
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
    """CH/STO-3G 空间积分 (自备, 开壳层 5e; 与 test_open_shell.py 同式)。"""
    mol = gto.M(atom="C 0 0 0; H 0 0 1.1", basis="sto-3g", spin=1, verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl",
                    mol.intor("int2e_sph"), mo, mo, mo, mo)
    return {"h1e": h1e, "eri": eri, "norb": mol.nao_nr(),
            "nelec": (3, 2), "ecore": mf.energy_nuc()}


def _assert_traj_close(ta, tb, tag=""):
    """轨迹逐点逐字段一致 (P0' 锚: 零回归逐位一致, rtol=1e-10 吸收 BLAS ULP 噪声)。"""
    assert len(ta) == len(tb), f"{tag}: 轨迹点数不同 {len(ta)} vs {len(tb)}"
    for k, (pa, pb) in enumerate(zip(ta, tb)):
        for key in ("round", "E", "sigma2", "e_pt2", "dim", "shots"):
            va, vb = float(pa[key]), float(pb[key])
            assert np.isclose(va, vb, rtol=1e-10, atol=1e-10), (
                f"{tag}: 轨迹点[{k}].{key} 不一致: {va!r} vs {vb!r} "
                f"diff={abs(va - vb):.2e}")


# --------------------------------------------------------------------------- #
# P0' 零回归: prune_keep=1.0 (默认) 与不传逐位一致
# --------------------------------------------------------------------------- #
def test_prune_keep_default_bit_identical_active():
    """P0' 锚: solve_sqd_active 不传 prune_keep (默认 1.0) 与显式 prune_keep=1.0
    代码路径等价 (剪枝分支整体跳过), 能量 + 轨迹逐位一致。

    与 test_tail_sampling 默认关回归同口径: rtol=1e-10 远严于真实逻辑分歧 (~1e-4),
    远松于多线程 BLAS eigh 的 ULP 噪声 (~1e-13)。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((100, 2 * norb)) > 0.5
    probs = np.full(100, 1.0 / 100)
    common = dict(max_strings=None, n_active_per_round=50, max_rounds=8,
                  rand_seed=0)
    traj_def, traj_one = [], []
    e_default = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        trajectory=traj_def, **common)
    e_one = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        prune_keep=1.0, trajectory=traj_one, **common)
    assert np.isclose(e_default, e_one, rtol=1e-10, atol=1e-10), (
        f"默认 (1.0) 应代码路径等价: default={e_default!r} one={e_one!r} "
        f"diff={abs(e_default - e_one):.2e}")
    _assert_traj_close(traj_def, traj_one, tag="P0' active")


def test_prune_keep_default_bit_identical_ev_and_improved():
    """P0' 锚 (透传层): solve_sqd_ev 与 solve_sqd_improved 不传 prune_keep vs
    显式 1.0 经透传后代码路径等价 (rtol=1e-10)。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((60, 2 * norb)) > 0.5
    common = dict(max_strings=None, n_active_per_round=30, rand_seed=0)
    e_def = tc_sqd.solve_sqd_ev(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, **common)
    e_one = tc_sqd.solve_sqd_ev(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        prune_keep=1.0, **common)
    assert np.isclose(e_def, e_one, rtol=1e-10, atol=1e-10), (
        f"solve_sqd_ev 默认 (1.0) 应等价: default={e_def!r} one={e_one!r} "
        f"diff={abs(e_def - e_one):.2e}")
    i_def = tc_sqd.solve_sqd_improved(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, **common)
    i_one = tc_sqd.solve_sqd_improved(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        prune_keep=1.0, **common)
    assert np.isclose(i_def, i_one, rtol=1e-10, atol=1e-10), (
        f"solve_sqd_improved 默认 (1.0) 应等价: default={i_def!r} one={i_one!r} "
        f"diff={abs(i_def - i_one):.2e}")


def test_prune_keep_default_bit_identical_best():
    """P0' 锚 (best 层): solve_sqd_best 不传 prune_keep vs 显式 1.0 → 全字段一致。

    LiH/STO-3G (dim≤225 → numpy eigh 确定性路径) 保证逐位可比; N2 全空间走 eigsh
    且 evpt2 外推在近饱和区病态 (round_007 根因本身), alpha 对比无意义, 故不用。
    """
    data = _lih_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    d_def = tc_sqd.solve_sqd_best(h1e, eri, norb, nelec, ecore=data.ecore,
                                  n_shots=60, return_details=True, rand_seed=0)
    d_one = tc_sqd.solve_sqd_best(h1e, eri, norb, nelec, ecore=data.ecore,
                                  n_shots=60, return_details=True, rand_seed=0,
                                  prune_keep=1.0)
    for key in ("energy", "E_direct", "E_pt2", "dim"):
        assert np.isclose(d_def[key], d_one[key], rtol=1e-10, atol=1e-10), (
            f"solve_sqd_best 默认 (1.0) 应等价: {key} {d_def[key]!r} vs "
            f"{d_one[key]!r} diff={abs(d_def[key] - d_one[key]):.2e}")
    assert (d_def["E_evpt2"] == d_one["E_evpt2"]) or np.isclose(
        d_def["E_evpt2"], d_one["E_evpt2"], rtol=1e-10, atol=1e-10), (
        "E_evpt2 应一致 (同退化/同非退化)")
    if d_def["evpt2"] is not None:
        assert np.isclose(d_def["evpt2"]["alpha"], d_one["evpt2"]["alpha"],
                          rtol=1e-10, atol=1e-10)


# --------------------------------------------------------------------------- #
# 剪枝功能 (闭壳层 N2/STO-3G): dim/E_V/E_PT2 + str_a==str_b 不变式 + PT2 回补
# --------------------------------------------------------------------------- #
def test_prune_keep_06_prunes_and_pt2_recovers():
    """剪枝功能 (theory §3 P2 a/b/c + §1.2 回补): prune_keep=0.6 →
      (c) dim' ≈ 0.36×dim 且 keep = ceil(0.6×n_str);
      (a) E_V' ≥ E_V (变分, 子空间变小);
      (b) |E_PT2'| > |E_PT2| (被剪 det 移出子空间进入 PT2 和式, §1.3 正信号);
      (回补) E_V' + E_PT2' 仍近 FCI (PT2 二阶回补被剪权重, §1.2 不变性)。
    闭壳层不变式: 剪后 str_a == str_b (合并权重剪同一集合, §2.3)。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_fci, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    bsm = np.random.default_rng(0).random((100, 2 * norb)) > 0.5
    probs = np.full(100, 1.0 / 100)
    common = dict(max_strings=None, n_active_per_round=50, max_rounds=10,
                  rand_seed=0)

    traj_u, state_u = [], []
    e_u = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        trajectory=traj_u, state_out=state_u, **common)
    traj_p, state_p = [], []
    e_p = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        prune_keep=0.6, trajectory=traj_p, state_out=state_p, **common)

    lu, lp = traj_u[-1], traj_p[-1]
    _, sa_u, sb_u = state_u[0]
    _, sa_p, sb_p = state_p[0]
    n_str_u = len(sa_u)
    assert len(sa_u) == len(sb_u), "闭壳层未剪前 str_a==str_b 前提应成立"

    # (c) keep = ceil(0.6 × n_str), dim' ≈ 0.36×
    keep = int(math.ceil(0.6 * n_str_u))
    assert len(sa_p) == keep and len(sb_p) == keep, (
        f"剪后应每自旋保留 ceil(0.6×{n_str_u})={keep} 个字符串, "
        f"got {len(sa_p)}/{len(sb_p)}")
    assert lp["dim"] == keep * keep
    assert abs(lp["dim"] / lu["dim"] - 0.36) < 0.05, (
        f"dim 应减 ~40% (≈0.36×): {lp['dim']} vs {lu['dim']} "
        f"ratio={lp['dim'] / lu['dim']:.3f}")
    # 闭壳层不变式 + 保留集是原集合子集
    assert np.array_equal(np.asarray(sa_p), np.asarray(sb_p)), (
        "闭壳层剪后 str_a == str_b 不变式被破坏 (§2.3 合并权重)")
    assert set(map(int, sa_p)) <= set(map(int, sa_u)), "保留字符串应来自原集合"

    # (a) E_V 升高 (变分上界: 子空间变小能量必升, 允许 ULP 级容差)
    assert lp["E"] >= lu["E"] - 1e-10, (
        f"剪枝违反变分 (E_V' 应 ≥ E_V): {lp['E']} vs {lu['E']}")
    # (b) |E_PT2| 增大 (更负, theory §1.3 正信号; 未剪近饱和 → 剪后回补被剪关联)
    assert lp["e_pt2"] < lu["e_pt2"], (
        f"|E_PT2'| 应增大 (§1.3 正信号): pruned={lp['e_pt2']:.3e} vs "
        f"unpruned={lu['e_pt2']:.3e}")

    # 回补: E_V' + E_PT2 仍近 FCI (PT2 二阶回补被剪权重; 阈值按 0.6 剪枝实测
    # 量级收紧, 远严于化学精度 1.6e-3)
    err_pt2 = abs((lp["E"] + lp["e_pt2"]) - e_fci)
    assert err_pt2 < 5e-4, (
        f"PT2 未回补被剪权重 (E_V'+E_PT2 err={err_pt2:.2e})")
    # 回补后优于剪后直接变分误差 (机制生效: 净改善来自 PT2 而非 E_V)
    err_var = abs(lp["E"] - e_fci)
    assert err_pt2 <= err_var, (
        f"PT2 修正应不劣于剪后直接变分: pt2_err={err_pt2:.2e} "
        f"var_err={err_var:.2e}")


# --------------------------------------------------------------------------- #
# 边界: 越界报错 + 空子空间防护 + 开壳层独立剪
# --------------------------------------------------------------------------- #
def test_prune_keep_invalid_values_raise():
    """边界 (任务单 P0' #3): prune_keep ∈ {0, 1.5} 及 (0,1] 外任意值 raise ValueError;
    1.0 (默认) 与 (0,1] 内合法值不报错。"""
    data = _lih_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((60, 2 * norb)) > 0.5
    probs = np.full(60, 1.0 / 60)
    kw = dict(bitstring_matrix=bsm, probabilities=probs,
              max_strings=8, n_active_per_round=10, max_rounds=3, rand_seed=0)
    for bad in (0.0, -0.1, 1.5, 2.0):
        with pytest.raises(ValueError):
            tc_sqd.solve_sqd_active(h1e, eri, norb, nelec, prune_keep=bad, **kw)
    # 合法值可运行 (1.0 = 零回归; 0.999 / 0.6 / 0.01 进入剪枝分支但保留 ≥1)
    for ok in (1.0, 0.999, 0.6, 0.01):
        e = tc_sqd.solve_sqd_active(h1e, eri, norb, nelec, prune_keep=ok, **kw)
        assert np.isfinite(e), f"prune_keep={ok} 应返回有限能量"

    # 透传层同样越界报错 (经 solve_sqd_ev / solve_sqd_best 到达 active 校验)
    with pytest.raises(ValueError):
        tc_sqd.solve_sqd_ev(h1e, eri, norb, nelec, prune_keep=1.5, **kw)
    with pytest.raises(ValueError):
        tc_sqd.solve_sqd_best(h1e, eri, norb, nelec, n_shots=60,
                              prune_keep=0.0, rand_seed=0)


def test_prune_keep_tiny_never_empties_subspace():
    """边界: 极小 prune_keep (0.01) 空子空间防护 —— 每自旋保留 ≥1 字符串
    (ceil 兜底 + max(1, ...) 护栏), 不崩、不变式保持。"""
    data = _lih_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((60, 2 * norb)) > 0.5
    probs = np.full(60, 1.0 / 60)
    state = []
    e = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        max_strings=6, n_active_per_round=10, max_rounds=3, rand_seed=0,
        prune_keep=0.01, state_out=state)
    assert np.isfinite(e)
    c2d, sa, sb = state[0]
    assert len(sa) >= 1 and len(sb) >= 1, "剪后每自旋至少保留 1 个字符串"
    assert np.array_equal(sa, sb), "闭壳层剪后 str_a == str_b 不变式"


def test_prune_open_shell_prunes_alpha_beta_independently():
    """开壳层 (CH/STO-3G, na=3/nb=2): α/β 各自按边际权重 (行和/列和) 独立剪,
    各自保留 ceil(0.6×n) 个字符串; 保留集是原集合子集; 变分不破 (§2.3 开壳层)。"""
    d = _ch_data()
    h1e, eri, norb, nelec = d["h1e"], d["eri"], d["norb"], d["nelec"]
    n_samples = 200
    bsm = np.random.default_rng(0).random((n_samples, 2 * norb)) > 0.5
    probs = np.full(n_samples, 1.0 / n_samples)
    common = dict(max_strings=None, n_active_per_round=30, max_rounds=10,
                  rand_seed=0)

    state_u, state_p = [], []
    e_u = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        state_out=state_u, **common)
    e_p = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        prune_keep=0.6, state_out=state_p, **common)

    _, sa_u, sb_u = state_u[0]
    _, sa_p, sb_p = state_p[0]
    ka = int(math.ceil(0.6 * len(sa_u)))
    kb = int(math.ceil(0.6 * len(sb_u)))
    assert len(sa_p) == ka, f"α 应保留 ceil(0.6×{len(sa_u)})={ka}, got {len(sa_p)}"
    assert len(sb_p) == kb, f"β 应保留 ceil(0.6×{len(sb_u)})={kb}, got {len(sb_p)}"
    assert set(map(int, sa_p)) <= set(map(int, sa_u)), "α 保留集应来自原集合"
    assert set(map(int, sb_p)) <= set(map(int, sb_u)), "β 保留集应来自原集合"
    assert e_p >= e_u - 1e-10, (
        f"开壳层剪枝违反变分: pruned={e_p} vs unpruned={e_u}")


# --------------------------------------------------------------------------- #
# 透传: solve_sqd_ev / solve_sqd_best 剪枝生效 (dim 下降 + 能量有限)
# --------------------------------------------------------------------------- #
def test_solve_sqd_ev_prune_passthrough_reduces_dim():
    """透传 (theory §2.1): solve_sqd_ev(prune_keep=0.6) 到达 active 剪枝 →
    details dim 下降; E_V 升高; E_PT2 更负; PT2 修正仍达化学精度。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_fci, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    bsm = np.random.default_rng(0).random((100, 2 * norb)) > 0.5
    probs = np.full(100, 1.0 / 100)
    common = dict(max_strings=None, n_active_per_round=30, rand_seed=0)
    e_u, det_u = tc_sqd.solve_sqd_ev(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        correction="pt2", return_details=True, **common)
    e_p, det_p = tc_sqd.solve_sqd_ev(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        correction="pt2", return_details=True, prune_keep=0.6, **common)
    assert det_p["dim"] < det_u["dim"], (
        f"透传剪枝应降 dim: {det_p['dim']} vs {det_u['dim']}")
    assert det_p["E_direct"] >= det_u["E_direct"] - 1e-10, (
        "剪枝后 E_direct 应升高 (变分)")
    assert det_p["E_PT2"] < det_u["E_PT2"], (
        f"剪枝后 E_PT2 应更负 (§1.3): {det_p['E_PT2']:.3e} vs "
        f"{det_u['E_PT2']:.3e}")
    assert abs(e_p - e_fci) < 1.6e-3, (
        f"剪枝 + PT2 修正仍达化学精度: err={abs(e_p - e_fci):.2e}")


def test_solve_sqd_best_prune_passthrough_reduces_dim():
    """透传 (theory §2.3): solve_sqd_best(prune_keep=0.6) 三个 evpt2 尺度一致剪枝
    (同 prune_keep) → dim 下降; 返回能量有限; details dims 全部 ≤ 未剪 dim。"""
    data = _lih_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    d_u = tc_sqd.solve_sqd_best(h1e, eri, norb, nelec, ecore=data.ecore,
                                n_shots=60, return_details=True, rand_seed=0)
    d_p = tc_sqd.solve_sqd_best(h1e, eri, norb, nelec, ecore=data.ecore,
                                n_shots=60, return_details=True, rand_seed=0,
                                prune_keep=0.6)
    assert np.isfinite(d_p["energy"])
    assert d_p["dim"] < d_u["dim"], (
        f"best 透传剪枝应降 dim: {d_p['dim']} vs {d_u['dim']}")
    # 剪枝后 baseline 变分能量升高 (变分上界); E_V+E_PT2 二阶不变 (§1.2), 不作方向断言
    assert d_p["E_direct"] >= d_u["E_direct"] - 1e-10, (
        f"剪枝后 E_direct 应升高 (变分): {d_p['E_direct']} vs {d_u['E_direct']}")
    if d_p["evpt2"] is not None:
        assert all(dim <= d_u["dim"] for dim in d_p["evpt2"]["dims"]), (
            "三个 evpt2 尺度的 dim 都应 ≤ 未剪 dim (一致剪枝口径)")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_pruning: all PASS")
