"""tc_sqd.lucj 模块测试 —— LUCJ 电路构建 + 真机深度预算报告。"""
import tensorcircuit as tc
import tc_sqd
from pyscf import gto, scf


def test_circuit_stats():
    """1Q/2Q/multi 门统计正确。"""
    c = tc.Circuit(4)
    c.x(0); c.cnot(0, 1); c.ry(2, theta=0.5); c.cnot(1, 2); c.h(3)
    s = tc_sqd.circuit_stats(c)
    assert s["n_qubits"] == 4
    assert s["n_1q"] == 3          # x, ry, h
    assert s["n_2q"] == 2          # 2 cnot
    assert s["n_multi"] == 0
    assert s["n_gates"] == 5
    assert s["gate_summary"]["cnot"] == 2


def test_lucj_report_h2():
    """H2 LUCJ: 2 个 occ-vir entry (alpha+beta) → 2 cnot; 深度预算检查。"""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()

    r = tc_sqd.lucj_report(mf, 2, (1, 1))
    assert r["depth_proxy"] == 2                    # 2 cnot (2Q 保守深度)
    assert r["circuit_stats"]["n_2q"] == 2
    assert r["circuit_stats"]["n_1q"] == 6          # 2 x (HF) + 4 ry (每 entry 2)
    assert r["circuit_stats"]["n_gates"] == 8
    assert r["within_budget"] is None               # 未给预算

    # 预算太紧: within_budget False, max_entries 按 2Q 门数预算给
    r2 = tc_sqd.lucj_report(mf, 2, (1, 1), max_depth=1)
    assert r2["within_budget"] is False
    assert r2["max_entries_by_2q_budget"] == 1

    # 真机典型预算 1500: 通过
    r3 = tc_sqd.lucj_report(mf, 2, (1, 1), max_depth=1500)
    assert r3["within_budget"] is True


def test_lucj_report_max_excitations_controls_cnot():
    """max_excitations 裁剪 entry 数 → 控制 2Q 门数 (每 entry 1 cnot)。"""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()

    r1 = tc_sqd.lucj_report(mf, 2, (1, 1), max_excitations=1)
    assert r1["depth_proxy"] == 1                   # 只留 1 个 entry → 1 cnot
    assert r1["circuit_stats"]["n_1q"] == 2 + 2     # 2 x + 1 entry*2 ry

    r_all = tc_sqd.lucj_report(mf, 2, (1, 1))       # 全 entries (2 个)
    assert r_all["depth_proxy"] == 2


def test_lucj_report_sampling_still_correct():
    """LUCJ (全 entries) 采样 → SQD 复现 H2 FCI (深度预算与精度路径兼容)。"""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    data = tc_sqd.from_pyscf(mf)

    c = tc_sqd.build_lucj_circuit(mf, data.norb, data.nelec,
                                  max_excitations=2, ccsd_scale=1.0)
    bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=3000)
    e = data.solve(method="sqd", bitstring_matrix=bsm,
                   probabilities=probs, max_iterations=3)
    assert abs(e - (-1.13728383)) < 2e-3


def test_ucj_decomposition_properties():
    """P2-2a: UCJ 分解性质 — kappa 实 anti-Hermitian, J 对称, 分层剥离。"""
    import numpy as np
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    nocc = data.nelec[0]
    _t1, t2, _ = tc_sqd.get_ccsd_amplitudes(data.mf)
    layers = tc_sqd.ucj_decomposition(t2, data.norb, nocc, nlayers=2)
    assert len(layers) == 2
    for kappa, J in layers:
        assert np.allclose(kappa, -kappa.T, atol=1e-12)   # anti-Hermitian (实)
        assert np.allclose(J, J.T, atol=1e-12)            # 对称
        assert kappa.shape == J.shape == (data.norb, data.norb)
    # 参数非零 (CCSD t2 驱动)
    assert np.linalg.norm(layers[0][0]) > 1e-6


def test_ucj_subspace_energy_h2_fci():
    """P2-2a: UCJ 子空间对角化 (确定性 SQD) H2 = FCI。"""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    nocc = data.nelec[0]
    _t1, t2, _ = tc_sqd.get_ccsd_amplitudes(data.mf)
    e_fci = data.solve(method="fci")
    layers = tc_sqd.ucj_decomposition(t2, data.norb, nocc, nlayers=1, scale=10)
    e = tc_sqd.ucj_subspace_energy(layers, data.h1e, data.eri,
                                   data.norb, data.nelec) + data.ecore
    assert abs(e - e_fci) < 1e-8


def test_ucj_subspace_energy_lih():
    """P2-2a: LiH UCJ 子空间随 scale 增大趋近 FCI (覆盖更全)。"""
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    nocc = data.nelec[0]
    _t1, t2, _ = tc_sqd.get_ccsd_amplitudes(data.mf)
    e_fci = data.solve(method="fci")
    e_small = tc_sqd.ucj_subspace_energy(
        tc_sqd.ucj_decomposition(t2, data.norb, nocc, scale=1),
        data.h1e, data.eri, data.norb, data.nelec) + data.ecore
    e_large = tc_sqd.ucj_subspace_energy(
        tc_sqd.ucj_decomposition(t2, data.norb, nocc, scale=50),
        data.h1e, data.eri, data.norb, data.nelec) + data.ecore
    assert e_small >= e_fci - 1e-8       # 变分下界
    assert e_large >= e_fci - 1e-8
    assert abs(e_large - e_fci) < 1e-4   # 大 scale 覆盖更全, 接近 FCI


def test_ucj_circuit_h2_fci():
    """P2-2b: UCJ 电路采样 -> SQD, H2 = FCI。"""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    e_fci = data.solve(method="fci")
    circ = tc_sqd.build_ucj_circuit(data.mf, data.norb, data.nelec, scale=10)
    bsm, probs = tc_sqd.sample(circ, 2000)
    e = data.solve(method="sqd", bitstring_matrix=bsm, probabilities=probs,
                   max_iterations=3)
    assert abs(e - e_fci) < 1e-6


def test_ucj_circuit_lih_better_than_lucj():
    """P2-2b: LiH UCJ 电路 SQD 误差 < 简化 LUCJ (7.5e-4), 且深度可接受。"""
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    e_fci = data.solve(method="fci")

    circ = tc_sqd.build_ucj_circuit(data.mf, data.norb, data.nelec, scale=5)
    bsm, probs = tc_sqd.sample(circ, 3000)
    e_ucj = data.solve(method="sqd", bitstring_matrix=bsm, probabilities=probs,
                       max_iterations=3)
    # 简化 LUCJ 基线
    c_l = tc_sqd.build_lucj_circuit(data.mf, data.norb, data.nelec, ccsd_scale=1.0)
    bsm_l, probs_l = tc_sqd.sample(c_l, 3000)
    e_lucj = data.solve(method="sqd", bitstring_matrix=bsm_l,
                        probabilities=probs_l, max_iterations=3)

    assert abs(e_ucj - e_fci) < abs(e_lucj - e_fci), (
        f"UCJ 未优于 LUCJ: ucj_err={abs(e_ucj-e_fci):.2e}, "
        f"lucj_err={abs(e_lucj-e_fci):.2e}")
    # 深度预算可接受
    stats = tc_sqd.circuit_stats(circ)
    assert stats["n_2q"] <= 50   # LiH 6 MO: 2*nocc*nvir = 16


def test_ucj_assisted_n2_strong_correlation():
    """UCJ 辅助: N2 强关联 (7e/spin), S+D 平台 ~2.25e-2 -> UCJ 补充化学精度。

    方向 A 验证: UCJ 电路采样产生超出单双激发的高激发 det, 补充后
    SQD 误差 ~1e-3 (化学精度)。断言: 显著优于 S+D, 且 < 5e-3。
    """
    mol = gto.M(atom="N 0 0 0; N 0 0 2.1", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    norb, nelec = data.norb, data.nelec
    e_fci = data.solve(method="fci")

    # S+D 基准
    exc = tc_sqd.excited_configurations(norb, nelec, max_excitations=2)
    e_sd = data.solve(method="sqd", bitstring_matrix=exc, probabilities=None,
                      max_iterations=3, include_configurations=exc)
    err_sd = abs(e_sd - e_fci)

    # UCJ 辅助 (两次调用验证稳定性: CCSD 浮点微差下误差稳健)
    for seed in (42, 7):
        e_ucj = tc_sqd.solve_ucj_assisted(
            data.h1e, data.eri, norb, nelec, ecore=data.ecore, mf=data.mf,
            scale=10, n_samples=5000, seed=seed)
        err_ucj = abs(e_ucj - e_fci)
        assert err_ucj < err_sd * 0.5, (
            f"UCJ 补充未显著改善: ucj_err={err_ucj:.2e}, sd_err={err_sd:.2e}")
        assert err_ucj < 5e-3, f"UCJ 辅助应化学精度, got {err_ucj:.2e}"


def test_ucj_assisted_lih_weak_correlation():
    """UCJ 辅助弱关联 (LiH): 单双激发已穷尽, UCJ 补充不破坏 (=FCI)。"""
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    e_fci = data.solve(method="fci")
    e = tc_sqd.solve_ucj_assisted(
        data.h1e, data.eri, data.norb, data.nelec, ecore=data.ecore,
        mf=data.mf, scale=10, n_samples=3000, seed=42)
    assert abs(e - e_fci) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_lucj: all PASS")
