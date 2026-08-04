"""tc_sqd.diagnostics 模块测试 —— 采样质量诊断报告。"""
import numpy as np
import tc_sqd
from pyscf import gto


def _h2_data():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    return tc_sqd.from_pyscf(mol)


def test_shannon_entropy_bounds():
    """熵: 确定性=0, 均匀最大, 正概率域。"""
    assert tc_sqd.shannon_entropy(np.array([1.0, 0.0])) == 0.0
    h_unif = tc_sqd.shannon_entropy(np.array([0.5, 0.5]))
    h_skew = tc_sqd.shannon_entropy(np.array([0.9, 0.1]))
    assert h_unif > h_skew > 0
    assert abs(h_unif - np.log(2)) < 1e-12      # 均匀 2 态熵 = ln2


def test_subspace_dimension():
    """去重字符串数与行列式对。"""
    # norb=2: [β1β0|α1α0]
    bsm = np.array([[0, 1, 0, 1],      # HF α0β0
                    [1, 0, 1, 0],      # 双激发 α1β1
                    [0, 1, 0, 1]],     # 重复
                   dtype=bool)
    na, nb, dim = tc_sqd.subspace_dimension(bsm)
    assert na == 2 and nb == 2 and dim == 4


def test_sampling_report_h2():
    """完整报告: 字段齐全, 熵>0, 能量收敛到 FCI 附近。"""
    data = _h2_data()
    bsm = np.array([[0, 1, 0, 1],      # HF
                    [1, 0, 1, 0],      # 双激发
                    [0, 1, 0, 1]],     # 重复 → 去重后 2 个
                   dtype=bool)
    probs = np.array([0.6, 0.3, 0.1])

    rep = tc_sqd.sampling_report(
        data.h1e, data.eri, data.norb, data.nelec, bsm,
        probs=probs, ecore=data.ecore, max_iterations=2)

    assert rep["n_samples"] == 3
    assert rep["n_unique"] == 2
    assert rep["subspace_dim"] == 4          # 2 α x 2 β
    assert rep["entropy_nat"] > 0
    assert len(rep["top_configs"]) == 2
    assert rep["top_configs"][0]["bitstring"] == 5   # 0b0101 = HF determinant
    conv = rep["energy_convergence"]
    assert conv["shots"] == sorted(set(conv["shots"]))
    assert len(conv["energies"]) == len(conv["shots"])
    # HF+双激发两行列式对角化 = FCI
    assert abs(conv["converged_energy"] - (-1.13728383)) < 1e-3


def test_sampling_report_entropy_grows_with_diversity():
    """配置越多样, 采样熵越大。"""
    data = _h2_data()
    bsm_1 = np.array([[0, 1, 0, 1]] * 4, dtype=bool)          # 全 HF
    bsm_2 = np.array([[0, 1, 0, 1], [1, 0, 1, 0],
                      [0, 1, 1, 0], [1, 1, 0, 1]], dtype=bool)  # 多样

    # 只测熵 (不跑收敛, 快)
    from tc_sqd.diagnostics import shannon_entropy
    from tc_sqd.counts import bitarray_to_int
    p1 = np.ones(4) / 4
    p2 = np.ones(4) / 4
    ints1 = bitarray_to_int(bsm_1)
    uniq1 = np.unique(ints1)
    w1 = np.ones(4)
    merged1 = np.zeros(len(uniq1))
    for i, u in enumerate(uniq1):
        merged1[i] = np.sum(w1[ints1 == u])
    h1 = shannon_entropy(merged1)

    ints2 = bitarray_to_int(bsm_2)
    assert len(np.unique(ints1)) == 1
    assert len(np.unique(ints2)) == 4
    assert h1 == 0.0
    assert tc_sqd.shannon_entropy(p2) > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_diagnostics: all PASS")


def test_extrapolate_infinite_samples_fit():
    """A1: 合成 1/√S 数据 (统计量), 外推 E∞ 与斜率正确恢复。

    注: 该外推对 SQD 子空间能量不适用 (非统计量, A1 验证证伪, 见 docstring);
    测试用合成统计量数据验证拟合数学正确性。
    """
    e_inf_true, a_true = -1.2345, 0.1
    shots = np.array([100.0, 200.0, 400.0, 800.0, 1600.0])
    energies = e_inf_true + a_true / np.sqrt(shots)
    e_inf, a, r2, fit_std = tc_sqd.extrapolate_infinite_samples(energies, shots)
    assert abs(e_inf - e_inf_true) < 1e-10, f"E∞ 未恢复: {e_inf}"
    assert abs(a - a_true) < 1e-10
    assert r2 > 0.999
    # 不足 2 点报错
    try:
        tc_sqd.extrapolate_infinite_samples([1.0], [10.0])
        assert False, "单点应报错"
    except ValueError:
        pass
