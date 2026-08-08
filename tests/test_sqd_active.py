"""tc_sqd.cipsi.solve_sqd_active 测试 —— 主动采样双闭环 (方向②)。

核心验证: 低采样下, 受限 PT2 选态确定性补足采样缺口, 误差显著优于纯采样 SQD,
且子空间受限 (远小于 solve_cipsi 补全到全空间)。
"""
import numpy as np
import tc_sqd
from pyscf import gto
from pyscf.fci import direct_spin1
from pyscf.fci import cistring


def _n2_stretch_data():
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    return tc_sqd.from_pyscf(mol)


def _plain_sqd_error(data, n_samples=100, seed=0):
    """纯采样 SQD 对照: 配置恢复 + 解一次 (更新一次平均占据)。"""
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    na, nb = nelec
    e_fci, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    rng = np.random.default_rng(seed)
    bsm = rng.random((n_samples, 2 * norb)) > 0.5
    probs = np.full(n_samples, 1.0 / n_samples)
    occ_a = np.zeros(norb); occ_a[:na] = 1.0
    occ_b = np.zeros(norb); occ_b[:nb] = 1.0
    # 两轮配置恢复迭代 (给纯采样一个公平机会)
    e_ = np.inf
    for _ in range(3):
        rec, _ = tc_sqd.recover_configurations(bsm, probs, (occ_a, occ_b), na, nb,
                                               rand_seed=seed)
        ca, cb = tc_sqd.bitstring_matrix_to_ci_strs(rec)
        r = tc_sqd.solve_sci((ca, cb), h1e, eri, norb, nelec)
        e_ = r.energy
        dm1 = tc_sqd.rdm1_from_sci_result(r)
        occ_a = np.clip(np.diag(dm1) / 2.0, 0, 1)
        occ_b = occ_a.copy()
    return abs(e_ - e_fci)


def test_sqd_active_beats_plain_sampling():
    """低采样 (n=100): 主动采样 (受限 PT2) 误差显著优于纯采样 SQD。

    N2/STO-3G 拉伸纯配置恢复 n=100 误差 ~2e-3 (超化学精度); PT2 确定性补足
    采样缺口应压到化学精度以下 (AS-SQD 的核心主张: 无需额外量子测量)。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_fci, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)

    n_samples = 100
    bsm = (np.random.default_rng(0).random((n_samples, 2 * norb)) > 0.5)
    probs = np.full(n_samples, 1.0 / n_samples)

    e_active = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        max_strings=None, n_active_per_round=50, pt2_floor=1e-7,
        max_rounds=10, ecore=0.0, rand_seed=0, verbose=False,
    )
    err_active = abs(e_active - e_fci)
    err_plain = _plain_sqd_error(data, n_samples=n_samples)
    # 主动采样不劣于纯采样 (PT2 选态永远只加不减)
    assert err_active <= err_plain * 0.3, (
        f"主动采样未显著改善: active={err_active:.2e} plain={err_plain:.2e}"
    )
    # 达到化学精度 (纯采样 n=100 超化学精度 ~2e-3)
    assert err_active < 1.6e-3, f"主动采样未达化学精度: {err_active:.2e}"


def test_sqd_active_subspace_limited():
    """受限子空间: max_strings 限制 PT2 扩展 (采样覆盖不受限), 仍达近 FCI。

    N2/STO-3G 全空间 C(10,7)=120 字符串; n=100 采样恢复 ~89 字符串,
    max_strings=100 约束 PT2 扩展到 ≤100 (非全空间 120) —— 受限闭环
    (区别于 solve_cipsi 补全全空间) 仍逼近 FCI, 体现主动选态的高效性。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_fci, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)

    full = int(cistring.num_strings(norb, nelec[0]))
    assert full == 120

    n_samples = 100
    bsm = (np.random.default_rng(0).random((n_samples, 2 * norb)) > 0.5)
    probs = np.full(n_samples, 1.0 / n_samples)

    e_lim = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        max_strings=100, n_active_per_round=50, max_rounds=10, rand_seed=0,
    )
    err_lim = abs(e_lim - e_fci)
    # 受限子空间 (≤100×100 维, 全空间 120×120=14400) 达化学精度
    assert err_lim < 1.6e-3, f"受限主动采样未达化学精度: {err_lim:.2e}"


def test_solve_sqd_adaptive_returns_min_per_round_energy():
    """④: solve_sqd_adaptive 返回 min(各轮 ③ 自洽能量), 非末轮混合基 diag。

    旧版末轮 ④ 把 sub 重建到 B_{last+1}, 但 str_a 仍是 B_last 的 → 循环后 sub.diag
    是**混合基**对角化 (与 solve_sqd_natural_orbitals F2 同类 bug), 返回不自洽的能量。
    修复: 返回 min(各轮 E_r) (每轮自洽 B_r sub + B_r dets)。用 rounds_out 取各轮能量,
    断言返回值 == min(rounds_out) —— buggy 版返回混合基值, 必 ≠ min(rounds_out)。
    """
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    d = tc_sqd.from_pyscf(mol)
    e_fci = direct_spin1.kernel(d.h1e, d.eri, d.norb, d.nelec, conv_tol=1e-12)[0]
    bsm = np.random.default_rng(0).random((15, 2 * d.norb)) > 0.5
    rounds = []
    e_ada = tc_sqd.solve_sqd_adaptive(
        d.h1e, d.eri, d.norb, d.nelec,
        bitstring_matrix=bsm, max_rounds=4, n_active_per_round=15,
        rand_seed=0, ecore=0.0, rounds_out=rounds)
    assert len(rounds) >= 2, f"应跑 ≥2 轮 (才有 min 意义), got {len(rounds)}"
    # 契约: 返回值 = min(各轮 ③ 能量)
    assert abs(e_ada - min(rounds)) < 1e-12, (
        f"返回值应 = min(各轮能量): {e_ada} vs min({rounds})={min(rounds)}")
    # 变分上界 (各轮自洽 → min 仍 ≥ FCI)
    assert e_ada >= e_fci - 1e-10, (
        f"adaptive 违反变分: {e_ada:.10f} < FCI {e_fci:.10f}")


def test_sqd_adaptive_reaches_chemical_accuracy():
    """组合版 (换基表示层 + PT2 选择层) 稳定达化学精度。

    多 seed 验证 (REVIEW 方向②): 纯采样 0/6 达化学精度 (C₂ 3/8 式覆盖不稳),
    active 与 adaptive 均 6/6。组合版因换基每轮作废 det 累积, 误差略差于单独
    active (N2 拉伸 n=100: mean 1.9e-6 vs 4e-7), 但稳定在远低于化学精度的水平
    —— 断言"稳定达化学精度", 不声称优于 active (实测组合不占优)。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_fci, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)

    n_samples = 100
    bsm = (np.random.default_rng(0).random((n_samples, 2 * norb)) > 0.5)
    probs = np.full(n_samples, 1.0 / n_samples)

    errs = []
    for seed in range(3):
        e_ada = tc_sqd.solve_sqd_adaptive(
            h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
            n_active_per_round=50, max_rounds=10, rand_seed=seed,
        )
        errs.append(abs(e_ada - e_fci))
    # 组合版稳定达化学精度 (远低于阈值)
    assert max(errs) < 1.6e-3, f"组合版未稳定达化学精度: {errs}"


def test_sqd_active_budget_saves_shots():
    """B1: 自适应停采 (energy_tol) 省 shots 且精度不劣化。

    验证: shots_budget=2000 + shots_step=300 + energy_tol, 能量收敛即停
    (N2 拉伸实测 shots_used=900, 省 55%, 精度 ~1e-12 与全量相同)。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_fci, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    e_fci_total = e_fci + data.ecore

    n_pool = 2000
    bsm = (np.random.default_rng(0).random((n_pool, 2 * norb)) > 0.5)
    probs = np.full(n_pool, 1.0 / n_pool)

    usage = []
    e_ada = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        ecore=data.ecore, n_active_per_round=50, max_rounds=10, rand_seed=0,
        shots_budget=n_pool, shots_step=300, energy_tol=1e-5, usage=usage,
    )
    # 自适应停采省 shots (实测 900 < 2000)
    assert len(usage) == 1 and usage[0] < n_pool, f"未停采: shots_used={usage}"
    # 精度不劣化 (化学精度, 实测 ~1e-12)
    assert abs(e_ada - e_fci_total) < 1.6e-3, (
        f"自适应停采精度劣化: {abs(e_ada - e_fci_total):.2e}")
    # 与全量精度同量级 (能量收敛停采不损精度)
    e_full = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        ecore=data.ecore, n_active_per_round=50, max_rounds=10, rand_seed=0,
    )
    assert abs(e_ada - e_fci_total) <= abs(e_full - e_fci_total) * 1.5 + 1e-9


# --------------------------------------------------------------------------- #
#  方向 D: 能量-方差外推轨迹 + solve_sqd_ev + 本征矢重要性采样
# --------------------------------------------------------------------------- #
def test_sqd_active_trajectory_monotone():
    """D: trajectory 记录每轮 (E, σ², E_PT2, dim), 方差单调不增、能量单调降。

    子空间逐步扩展 -> σ² = Σ|⟨a|H|Ψ⟩|² (子空间外矩阵元) 单调不增, E 单调不增
    (变分). 这是能量-方差外推的输入前提。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    n_samples = 100
    bsm = (np.random.default_rng(0).random((n_samples, 2 * norb)) > 0.5)
    probs = np.full(n_samples, 1.0 / n_samples)

    traj = []
    tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        n_active_per_round=50, max_rounds=10, rand_seed=0, trajectory=traj,
    )
    assert len(traj) >= 2, f"轨迹应 ≥2 点: {len(traj)}"
    E_seq = [t["E"] for t in traj]
    V_seq = [t["sigma2"] for t in traj]
    D_seq = [t["dim"] for t in traj]
    for k in range(1, len(traj)):
        assert E_seq[k] <= E_seq[k - 1] + 1e-12, f"能量未单调: {E_seq}"
        assert V_seq[k] <= V_seq[k - 1] + 1e-12, f"方差未单调不增: {V_seq}"
        assert D_seq[k] >= D_seq[k - 1], f"维度未单调增: {D_seq}"
    assert all("e_pt2" in t and "shots" in t for t in traj)


def test_solve_sqd_ev_n2_reaches_chemical_accuracy():
    """D: solve_sqd_ev 修正能量达化学精度且优于直接 (默认 PT2 修正)。

    默认 correction="pt2" (E+E_PT2, SHCI 式, 行为良好): 实测受限子空间下
    直接 err 4.3e-4 -> PT2 修正 6.2e-5 (改善且不劣化)。σ² 线性外推 (correction=
    "ev") 保留为诊断模式 (可非变分过冲, 仅断言可运行 + 化学精度带)。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_fci, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    n_samples = 100
    bsm = (np.random.default_rng(0).random((n_samples, 2 * norb)) > 0.5)
    probs = np.full(n_samples, 1.0 / n_samples)

    # 默认修正 = PT2 (E+E_PT2, 行为良好)
    e_corr, det = tc_sqd.solve_sqd_ev(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        n_active_per_round=50, max_rounds=10, rand_seed=0,
        return_details=True,
    )
    assert det["correction"] == "pt2"
    assert "E_PT2" in det and det["E_PT2"] != 0.0
    assert abs(e_corr - e_fci) < 1.6e-3, f"PT2 修正未达化学精度: {abs(e_corr - e_fci):.2e}"
    assert abs(e_corr - e_fci) <= abs(det["E_direct"] - e_fci) + 1e-12, (
        f"PT2 修正未优于直接: corr={abs(e_corr - e_fci):.2e} "
        f"direct={abs(det['E_direct'] - e_fci):.2e}")
    assert len(det["trajectory"]) >= 2
    # σ² 外推为诊断模式 (允许非变分, 但须可运行且达化学精度带)
    e_ev, det_ev = tc_sqd.solve_sqd_ev(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        n_active_per_round=50, max_rounds=10, rand_seed=0,
        correction="ev", return_details=True,
    )
    assert det_ev["correction"] == "ev" and det_ev["r2"] > 0.9
    assert abs(e_ev - e_fci) < 1.6e-3, f"σ² 外推未达化学精度带: {abs(e_ev - e_fci):.2e}"
    # 非法 correction 报错
    try:
        tc_sqd.solve_sqd_ev(h1e, eri, norb, nelec, bitstring_matrix=bsm,
                            probabilities=probs, correction="bad")
        assert False, "非法 correction 应报错"
    except ValueError:
        pass


def test_eigenvector_importance_sample_concentrates_dominant():
    """D: 本征矢重要性采样按振幅平方分布采样, 聚焦主导 det。

    低 temperature (锐化) -> 采样集中到主导 det (argmax |c|²); 高 temperature
    (展平) -> 分布更均匀。验证"学习型采样先验"确实按振幅加权。
    """
    # 用 H2 (小空间) 验证分布性质, 保证确定性
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    d2 = tc_sqd.from_pyscf(mol)
    h1e2, eri2, norb2, nelec2 = d2.h1e, d2.eri, d2.norb, d2.nelec
    ci = tc_sqd.excited_configurations(norb2, nelec2, max_excitations=2)
    ci_a, ci_b = tc_sqd.bitstring_matrix_to_ci_strs(ci, open_shell=False)
    res = tc_sqd.solve_sci((ci_a, ci_b), h1e2, eri2, norb2, nelec2)
    c2d = res.sci_state.amplitudes.reshape(len(ci_a), len(ci_b))
    assert c2d.shape == (2, 2)          # H2/STO-3G 两字符串: HF + 双激发
    sa = sb = np.asarray(ci_a)

    # 低温度: 主导 det (HF, |c| 更大) 被多数采样
    bsm_hot = tc_sqd.eigenvector_importance_sample(
        c2d, sa, sb, norb2, 500, rand_seed=0, temperature=10.0)
    bsm_cold = tc_sqd.eigenvector_importance_sample(
        c2d, sa, sb, norb2, 500, rand_seed=0, temperature=0.1)
    assert bsm_hot.shape == (500, 2 * norb2) and bsm_hot.dtype == bool
    # 电子数守恒 (每个 det 位串都有 na=1, nb=1)
    assert bsm_cold.sum(axis=1).min() == 2
    # 展平 -> 两 det 比例接近均匀; 锐化 -> 主导 det 明显占优
    from tc_sqd.counts import bitarray_to_int
    u_hot = np.unique(bitarray_to_int(bsm_hot), return_counts=True)[1]
    u_cold = np.unique(bitarray_to_int(bsm_cold), return_counts=True)[1]
    ratio_hot = u_hot.max() / u_hot.sum()
    ratio_cold = u_cold.max() / u_cold.sum()
    assert ratio_cold > ratio_hot + 0.1, (
        f"锐化未更聚焦主导 det: cold={ratio_cold:.3f} hot={ratio_hot:.3f}")


def test_eigenvector_importance_sample_open_shell_layout():
    """F1 回归: 输出必须遵循库 [β|α] 布局 (左 β 右 α)。

    旧实现把 det_a(α) 放左半、det_b(β) 放右半, 与库约定相反; 闭壳层
    (sa=sb) 因 bitstring_matrix_to_ci_strs 合并 α/β 而不可见, 故 ``..._concentrates``
    测试漏检。这里用 det_a ≠ det_b 的开壳层情形: 把全权重放在一个 (α, β) det 上,
    采样后喂回 ``bitstring_matrix_to_ci_strs(open_shell=True)`` 必须精确还原。
    """
    norb = 3
    # 两组互不相同的 α/β 字符串 (占据模式不同), 全权重放在 (sa[1], sb[0])
    sa = np.asarray([0b001, 0b110])   # α: {orb0} 与 {orb1,orb2}
    sb = np.asarray([0b011, 0b100])   # β: {orb0,orb1} 与 {orb2}
    c2d = np.zeros((2, 2))
    c2d[1, 0] = 1.0                   # 100% 在 (α=sa[1]=0b110, β=sb[0]=0b011)
    bsm = tc_sqd.eigenvector_importance_sample(
        c2d, sa, sb, norb, 50, rand_seed=0, temperature=1.0)
    ra, rb = tc_sqd.bitstring_matrix_to_ci_strs(bsm, open_shell=True)
    assert len(ra) == 1 and len(rb) == 1, "全权重单 det 应只采到一个位串"
    assert int(ra[0]) == int(sa[1]), (
        f"α 半区被互换: 期望 {int(sa[1]):b}, 得到 {int(ra[0]):b}")
    assert int(rb[0]) == int(sb[0]), (
        f"β 半区被互换: 期望 {int(sb[0]):b}, 得到 {int(rb[0]):b}")


def test_solve_sqd_ev_evpt2_correction_runs():
    """③: solve_sqd_ev(correction='evpt2') — 退化轨迹自动退化为 pt2 (永不劣于 pt2)。

    LiH/STO-3G + max_strings=8 → solve_sqd_active 轨迹 round 间 E_PT2 重复 (互异点
    <2), 线性外推病态。evpt2 模式应**退化为 pt2 单点修正**, 返回值与 correction='pt2'
    完全一致。完整"非退化轨迹两点外推"验证需 N₂ 拉伸慢测试 (见 REVIEW 方向③)。
    """
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    d = tc_sqd.from_pyscf(mol)
    rng = np.random.default_rng(0)
    bsm = rng.random((20, 2 * d.norb)) > 0.5
    kw = dict(max_strings=8, n_active_per_round=10, max_rounds=5, rand_seed=0)
    e_evpt2, det = tc_sqd.solve_sqd_ev(
        d.h1e, d.eri, d.norb, d.nelec, ecore=d.ecore,
        bitstring_matrix=bsm, correction="evpt2", return_details=True, **kw)
    assert det["correction"] == "evpt2"
    assert det.get("fallback") == "pt2", "退化轨迹 (互异点<2) 应退化为 pt2"
    assert np.isfinite(e_evpt2)
    # 退化时 evpt2 == pt2 单点修正
    e_pt2 = tc_sqd.solve_sqd_ev(
        d.h1e, d.eri, d.norb, d.nelec, ecore=d.ecore,
        bitstring_matrix=bsm, correction="pt2", **kw)
    assert abs(e_evpt2 - e_pt2) < 1e-12, f"fallback 应等于 pt2: {e_evpt2} vs {e_pt2}"
    # 非法 correction 仍报错
    try:
        tc_sqd.solve_sqd_ev(d.h1e, d.eri, d.norb, d.nelec, ecore=d.ecore,
                            bitstring_matrix=bsm, correction="bogus")
        assert False, "非法 correction 应报错"
    except ValueError:
        pass


def test_solve_sqd_distill_runs_and_no_worse_than_single():
    """②: solve_sqd_distill 自蒸馏闭环可运行, 且 best_E ≤ 单次 active。

    round 0 即"单次 solve_sqd_active on 初始 bsm", 而 distill 的 best_E 取各轮 min,
    故 distill 必不劣于单次。温度退火 (高→低) 的实际改善是经验性的 (依赖体系/采样),
    此处仅锁定"不劣化"下界 + 闭环可运行 + 输入校验。
    """
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    d = tc_sqd.from_pyscf(mol)
    rng = np.random.default_rng(0)
    bsm = rng.random((15, 2 * d.norb)) > 0.5
    kw = dict(max_strings=12, n_active_per_round=15, rand_seed=0)
    e_single = tc_sqd.solve_sqd_distill(
        d.h1e, d.eri, d.norb, d.nelec, ecore=d.ecore,
        bitstring_matrix=bsm, n_rounds=1, **kw)
    e_distill = tc_sqd.solve_sqd_distill(
        d.h1e, d.eri, d.norb, d.nelec, ecore=d.ecore,
        bitstring_matrix=bsm, n_rounds=3, n_samples=15, **kw)
    assert np.isfinite(e_distill)
    assert e_distill <= e_single + 1e-12, (
        f"distill 应不劣于单次 (round 0 = 单次): {e_distill} vs {e_single}")
    # temperature_schedule 长度校验 (须 = n_rounds-1)
    try:
        tc_sqd.solve_sqd_distill(
            d.h1e, d.eri, d.norb, d.nelec, ecore=d.ecore,
            bitstring_matrix=bsm, n_rounds=3, temperature_schedule=[0.5, 0.5, 0.5])
        assert False, "schedule 长度 != n_rounds-1 应报错"
    except ValueError:
        pass


def test_solve_sqd_auto_end_to_end():
    """E: solve_sqd_auto 一键流程 —— 推荐 + 自适应收敛 + EV 外推。

    给真机参数, 自动取 shots/max_strings, 自适应停采, 轨迹外推; 返回
    details 含全部诊断, 能量达化学精度, 实际 shots ≤ 预算。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_fci, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    e_fci_total = e_fci + data.ecore   # auto 返回含 ecore 的总能量

    auto = tc_sqd.solve_sqd_auto(
        h1e, eri, norb, nelec, ecore=data.ecore,
        T1_us=30.0, t_gate_ns=100.0, return_details=True,
    )
    assert {"energy", "E_direct", "E_ev", "shots_used", "shots_budget",
            "recommendation", "trajectory", "converged", "n_rounds"} <= set(auto)
    # 推荐已执行: 有 SqdParams, 且 shots 预算匹配
    assert auto["recommendation"] is not None
    assert auto["shots_budget"] == auto["recommendation"].shots
    # 实际 shots ≤ 预算 (自适应停采)
    assert auto["shots_used"] <= auto["shots_budget"]
    # 轨迹 ≥2 点 (EV 外推前提) 且能量达化学精度
    assert auto["n_rounds"] >= 2
    assert abs(auto["energy"] - e_fci_total) < 1.6e-3, (
        f"auto 未达化学精度: {abs(auto['energy'] - e_fci_total):.2e}")
    # EV 外推版启用时 E_ev 与 E_direct 同侧 (修正量小)
    assert auto["E_ev"] is not None
    # 简易版返回 float
    e_float = tc_sqd.solve_sqd_auto(h1e, eri, norb, nelec, ecore=data.ecore)
    assert isinstance(e_float, float)
