"""tc_sqd T1 感知配置恢复测试 —— estimate_true_occupancies 反卷积。"""
import numpy as np
import tc_sqd
from tc_sqd.configuration_recovery import estimate_true_occupancies


def _simulate_t1_samples(n_samples=20000, seed=0):
    """采样真实位串 -> 施加 per-qubit 不均匀 T1 -> 观测位串。

    返回 (real_bsm, obs_bsm, avg_occ_true, gamma_col)。
    """
    rng = np.random.default_rng(seed)
    norb, na, nb = 6, 3, 3
    avg_occ_true = np.clip(np.linspace(0.95, 0.05, norb), 0.02, 0.98)
    # 逐位不均匀 γ (bsm 列序 [β_{n-1}..β0 | α_{n-1}..α0])
    gamma = np.array([0.02, 0.10, 0.15, 0.05, 0.25, 0.30,
                      0.08, 0.12, 0.03, 0.20, 0.18, 0.06])

    rows = []
    for _ in range(n_samples):
        a = rng.choice(norb, size=na, replace=False,
                       p=avg_occ_true / avg_occ_true.sum())
        b = rng.choice(norb, size=nb, replace=False,
                       p=avg_occ_true / avg_occ_true.sum())
        row = np.zeros(2 * norb, dtype=bool)
        for p in a:
            row[norb + (norb - 1 - p)] = True      # α 轨道 p
        for p in b:
            row[norb - 1 - p] = True               # β 轨道 p
        rows.append(row)
    real = np.array(rows)

    obs = real.copy()
    for col in range(2 * norb):
        obs[(rng.random(n_samples) < gamma[col]) & obs[:, col], col] = False
    return real, obs, avg_occ_true, gamma


def test_estimate_true_occupancies_beats_naive_mean():
    """T1 反卷积: 校正 avg_occ 比直接观测平均更接近真实 (RMSE 降)。"""
    _, obs, avg_true, gamma = _simulate_t1_samples()
    norb = 6
    na = nb = 3

    est_a, est_b = tc_sqd.estimate_true_occupancies(obs, na, nb, gamma,
                                                    norb=norb)

    # 校正 avg_occ (α 自旋) 与真实对比
    rmse_est = float(np.sqrt(np.mean((est_a - avg_true) ** 2)))
    # 观测平均 (轨道序) 作基线
    obs_mean_a = obs[:, norb:].mean(0)[::-1]
    rmse_obs = float(np.sqrt(np.mean((obs_mean_a - avg_true) ** 2)))
    assert rmse_est < rmse_obs, (
        f"T1 反卷积未优于观测平均: est={rmse_est:.4f}, naive={rmse_obs:.4f}")

    # 校正后 avg_occ 总和 = 电子数 (normalize=True)
    assert abs(est_a.sum() - na) < 1e-8
    assert abs(est_b.sum() - nb) < 1e-8


def test_estimate_true_occupancies_uniform_gamma_degenerates():
    """均匀 γ: 校正为整体缩放, 归一后等价观测平均 (保序)。"""
    _, obs, avg_true, _ = _simulate_t1_samples()
    norb = 6
    na = nb = 3
    # 均匀 γ
    est_a, _ = tc_sqd.estimate_true_occupancies(obs, na, nb, 0.1, norb=norb)
    # 归一化观测平均 (均匀 γ 只整体缩放, 归一后相同)
    obs_mean_a = obs[:, norb:].mean(0)[::-1]
    obs_norm = obs_mean_a * na / obs_mean_a.sum()
    assert np.allclose(est_a, np.clip(obs_norm, 0, 1), atol=1e-3)


def test_estimate_true_occupancies_gamma_edge():
    """γ 边界: γ=1 无 NaN, γ=0 退化为归一化观测平均。"""
    _, obs, _, _ = _simulate_t1_samples(n_samples=200)
    norb, na, nb = 6, 3, 3

    # γ=1 (完全衰减): 有限, 观测>0 的位估计为 1
    est1_a, est1_b = tc_sqd.estimate_true_occupancies(obs, na, nb, 1.0, norb=norb)
    assert np.all(np.isfinite(est1_a)) and np.all(np.isfinite(est1_b))
    assert np.all(est1_a >= 0) and np.all(est1_a <= 1)

    # γ=0 (无 T1): 反卷积 = 观测平均的粒子数归一
    est0_a, _ = tc_sqd.estimate_true_occupancies(obs, na, nb, 0.0, norb=norb)
    obs_a = obs[:, norb:].mean(0)[::-1]
    expect = np.clip(obs_a * na / obs_a.sum(), 0, 1)
    assert np.allclose(est0_a, expect, atol=1e-6)


def test_estimate_true_occupancies_validation():
    """非法 t1_gamma / 布局显式报错。"""
    _, obs, _, _ = _simulate_t1_samples(n_samples=50)
    # γ 越界
    try:
        estimate_true_occupancies(obs, 3, 3, np.full(12, 1.5))
        assert False, "γ>1 应报错"
    except ValueError:
        pass
    # γ 长度不对
    try:
        estimate_true_occupancies(obs, 3, 3, np.ones(10))
        assert False, "γ 长度错误应报错"
    except ValueError:
        pass
    # norb 不一致
    try:
        estimate_true_occupancies(obs, 3, 3, 0.1, norb=5)
        assert False, "norb 不一致应报错"
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_t1_recovery: all PASS")
