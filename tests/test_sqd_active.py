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
