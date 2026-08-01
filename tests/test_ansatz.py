"""tc_sqd SQD+VQE 混合优化测试 —— theta_list 变分入口 + optimize_ansatz_parameters。"""
import numpy as np
import tc_sqd
from pyscf import gto


def _h2_data():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    return tc_sqd.from_pyscf(mol)


def _lih_data():
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    return tc_sqd.from_pyscf(mol)


def test_build_lucj_theta_list():
    """theta_list 覆盖角度 + 长度校验。"""
    data = _h2_data()
    c = tc_sqd.build_lucj_circuit(data.mf, 2, (1, 1), theta_list=[0.5, 0.5])
    stats = tc_sqd.circuit_stats(c)
    assert stats["n_2q"] == 2                 # 2 entries, 2 cnot

    # 长度不符显式报错
    try:
        tc_sqd.build_lucj_circuit(data.mf, 2, (1, 1), theta_list=[0.5])
        assert False, "theta_list 长度不符应报错"
    except ValueError:
        pass


def test_optimize_ansatz_h2_reaches_fci():
    """H2: SQD+VQE 优化后能量 ≤ 固定 CCSD-LUCJ, 且 = FCI。"""
    data = _h2_data()
    norb, nelec = data.norb, data.nelec
    e_fci = data.solve(method="fci")

    # 固定 CCSD-LUCJ 基线 (固定 seed 采样, 公平对比)
    c0 = tc_sqd.build_lucj_circuit(data.mf, norb, nelec, ccsd_scale=1.0)
    bsm, probs = tc_sqd.sample(c0, 2000, backend="tc")
    e_fixed = data.solve(method="sqd", bitstring_matrix=bsm,
                         probabilities=probs, max_iterations=3)

    res = tc_sqd.optimize_ansatz_parameters(
        data.mf, data.h1e, data.eri, norb, nelec,
        ecore=data.ecore, n_samples=1500, num_restarts=2,
        maxiter=20, seed=42, max_iterations=3)

    assert res["method"] == "sqd+vqe"
    assert res["n_params"] == 2
    assert res["energy"] <= e_fixed + 1e-6
    assert abs(res["energy"] - e_fci) < 1e-3
    # theta 与 build_lucj_circuit(theta_list=...) 兼容
    c_opt = tc_sqd.build_lucj_circuit(data.mf, norb, nelec,
                                      theta_list=list(res["theta"]))
    assert c_opt._nqubits == 2 * norb


def test_optimize_ansatz_lih_improves():
    """LiH: SQD+VQE 变分优化能量 ≤ 固定 CCSD-LUCJ (改善 ~mHa 量级)。"""
    data = _lih_data()
    norb, nelec = data.norb, data.nelec

    c0 = tc_sqd.build_lucj_circuit(data.mf, norb, nelec, ccsd_scale=0.5,
                                   max_excitations=6)
    bsm, probs = tc_sqd.sample(c0, 2000, backend="tc")
    e_fixed = data.solve(method="sqd", bitstring_matrix=bsm,
                         probabilities=probs, max_iterations=3)

    res = tc_sqd.optimize_ansatz_parameters(
        data.mf, data.h1e, data.eri, norb, nelec,
        ecore=data.ecore, n_samples=1500, max_excitations=6,
        num_restarts=1, maxiter=15, seed=42, max_iterations=3)

    assert res["energy"] <= e_fixed + 1e-6, (
        f"SQD+VQE 未改善: opt={res['energy']:.6f}, fixed={e_fixed:.6f}")


def test_optimize_ansatz_validation():
    """无可优化参数显式报错。"""
    data = _h2_data()
    # nelec=(0,0) -> K=0 无法优化 (用直接调用触发)
    try:
        tc_sqd.optimize_ansatz_parameters(
            data.mf, data.h1e, data.eri, 2, (0, 0),
            ecore=data.ecore, n_samples=100, num_restarts=1, maxiter=1)
        assert False, "K=0 应报错"
    except ValueError:
        pass
    # n_seeds < 1 报错
    try:
        tc_sqd.optimize_ansatz_parameters(
            data.mf, data.h1e, data.eri, 2, (1, 1),
            ecore=data.ecore, n_samples=100, num_restarts=1,
            maxiter=1, n_seeds=0)
        assert False, "n_seeds=0 应报错"
    except ValueError:
        pass


def test_optimize_ansatz_n_seeds_reproducible():
    """n_seeds>1 消除单 seed 过拟合: 目标 = 多 seed 平均, 可复现。"""
    data = _h2_data()
    res1 = tc_sqd.optimize_ansatz_parameters(
        data.mf, data.h1e, data.eri, 2, (1, 1),
        ecore=data.ecore, n_samples=800, num_restarts=1,
        maxiter=10, seed=42, n_seeds=3)
    res2 = tc_sqd.optimize_ansatz_parameters(
        data.mf, data.h1e, data.eri, 2, (1, 1),
        ecore=data.ecore, n_samples=800, num_restarts=1,
        maxiter=10, seed=42, n_seeds=3)
    assert res1["energy"] == res2["energy"]        # 确定性
    assert res1["n_params"] == 2


def test_include_excitations_reaches_fci():
    """include 单双激发 -> SQD 精确复现 FCI (误差优化核心发现)。

    LiH: 93 个单双激发配置确定性覆盖全部相关空间, 采样仅提供权重,
    即使 1000 shots 也精确到浮点舍入, 且跨 seed 零波动。
    """
    data = _lih_data()
    norb, nelec = data.norb, data.nelec
    e_fci = data.solve(method="fci")

    exc = tc_sqd.excited_configurations(norb, nelec, max_excitations=2)
    c = tc_sqd.build_lucj_circuit(data.mf, norb, nelec, ccsd_scale=0.5)

    es = []
    for s in (7, 123, 2024):
        bsm, probs = tc_sqd.sample(c, 1000, backend="tc")
        e = data.solve(method="sqd", bitstring_matrix=bsm, probabilities=probs,
                       max_iterations=3, include_configurations=exc)
        es.append(e)
    es = np.array(es)
    assert np.allclose(es, e_fci, atol=1e-8), (
        f"include 单双激发未达 FCI: {es}")
    assert es.std() < 1e-9, "应零统计波动"


def test_include_excitations_not_fci_strong_correlation():
    """强相关/高占据 (N2, 7e/spin): include(S+D) ≠ FCI (误差 > 1e-3)。

    与 test_include_excitations_reaches_fci (弱相关, ≤2 occ/spin) 成**对照对**:
    证明 include 单双激发 = FCI 的前提是每自旋 ≤2 电子 (单双激发穷尽该自旋
    行列式), **不是 "CISD 精确"**。锁住这条认知, 防未来误改。
    """
    mol = gto.M(atom="N 0 0 0; N 0 0 2.1", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    norb, nelec = data.norb, data.nelec
    assert nelec[0] > 2, "反例需每自旋 >2 电子"

    e_fci = data.solve(method="fci")
    exc = tc_sqd.excited_configurations(norb, nelec, max_excitations=2)
    c = tc_sqd.build_lucj_circuit(data.mf, norb, nelec, ccsd_scale=0.5,
                                  max_excitations=8)
    bsm, probs = tc_sqd.sample(c, 2000, backend="tc")
    e_inc = data.solve(method="sqd", bitstring_matrix=bsm, probabilities=probs,
                       max_iterations=3, include_configurations=exc)

    assert abs(e_inc - e_fci) > 1e-3, (
        f"include(S+D) 不应达 FCI: inc={e_inc:.6f}, FCI={e_fci:.6f}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_ansatz: all PASS")
