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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_ansatz: all PASS")
