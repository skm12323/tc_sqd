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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_lucj: all PASS")
