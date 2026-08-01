"""tc_sqd.hardware 模块测试 — mock tc.cloud, 不碰真机。

覆盖 A 要求的四个点:
  (a) select_qubits 全路径 (BFS 序 + 连通失败 raise)
  (b) bitstring_matrix_to_energy
  (c) sample_on_hw 的 REM 失败回退分支 (warning + 原始 counts)
  (d) sample_on_hw 的字节序自动校正分支 (e_hf_ref -> reverse_key)
"""
import warnings
from unittest import mock

import numpy as np
import tc_sqd
from pyscf import gto


def _fake_cal():
    """6-qubit 链 calibration, T2 分布让 2/5/3/0 最优。"""
    T2 = [20, 10, 50, 30, 5, 40]
    qubits = {
        i: {"T1_us": 15, "T2_us": T2[i], "readout_f0": 0.01,
            "readout_f1": 0.02, "sq_err": 0.01, "sq_gate_ns": 30,
            "freq_mhz": 5000}
        for i in range(6)
    }
    edges = {f"{i}-{i + 1}": {"cz_err": 0.01, "cz_gate_ns": 30}
             for i in range(5)}
    return {"device": "fake", "n_qubits": 6, "n_links": 5,
            "qubits": qubits, "edges": edges,
            "topology": [[i, i + 1] for i in range(5)]}


def _h2_data():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    return tc_sqd.from_pyscf(mol)


def test_select_qubits_full_path():
    """(a) select_qubits: 选 T2 最大连通子图, BFS 序, 连通失败 raise。"""
    cal = _fake_cal()
    pq = tc_sqd.select_qubits(cal, 4)
    assert len(pq) == 4
    assert len(set(pq)) == 4
    assert 2 in pq                        # T2 最高 (50) 必被选
    # 链式拓扑下连通子图 = 连续区间 (最优 {0,1,2,3}, minT2=10)
    s = sorted(pq)
    assert s == list(range(s[0], s[0] + 4))

    # 连通失败 (要 7 个但只有 6)
    try:
        tc_sqd.select_qubits(cal, 7)
        assert False, "连通失败应 raise"
    except ValueError:
        pass


def test_bitstring_matrix_to_energy_h2():
    """(b) HF bitstring -> SQD 能量 = E_HF。"""
    data = _h2_data()
    bsm = np.array([[0, 1, 0, 1]], dtype=bool)     # HF determinant [β1β0|α1α0]
    e = tc_sqd.bitstring_matrix_to_energy(
        bsm, data.h1e, data.eri, data.norb, data.nelec, data.ecore)
    assert abs(e - data.mf.e_tot) < 1e-6, (
        f"HF bsm 应给 E_HF: got {e:.6f}, HF {data.mf.e_tot:.6f}")


def test_sample_on_hw_rem_fallback_and_endianness():
    """(c)(d) mock tc.cloud: REM 失败回退 + 字节序自校正。"""
    data = _h2_data()
    norb, nelec = data.norb, data.nelec
    circ = tc_sqd.build_lucj_circuit(data.mf, norb, nelec, ccsd_scale=1.0)

    # 假真机 counts: 以 HF 位串 (0b0101) 为主
    counts = {"0101": 700, "1010": 200, "0100": 100}
    nq = 2 * norb

    class FakeDevice:
        def topology(self):
            return [[0, 1], [1, 2], [2, 3]]

    class FakeTask:
        def results(self):
            return counts

    # (c) REM 失败回退分支
    with mock.patch("tensorcircuit.cloud.apis.get_device",
                    return_value=FakeDevice()), \
         mock.patch("tensorcircuit.cloud.apis.submit_task",
                    return_value=FakeTask()), \
         mock.patch("tensorcircuit.compiler.default_compile",
                    return_value=(None, {})), \
         mock.patch("tensorcircuit.results.rem.ReadoutMit",
                    side_effect=RuntimeError("no calibration")):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = tc_sqd.sample_on_hw(
                "fake", circ, norb=norb, h1e=data.h1e, eri=data.eri,
                nelec=nelec, ecore=data.ecore, e_hf_ref=data.mf.e_tot)

    # REM 失败应触发 warning, 且 counts 未被校正 (原始)
    assert any("REM" in str(x.message) for x in w), "REM 失败应 warning"
    assert r["counts"] == counts

    # (d) 字节序自校正: reverse_key 是 bool; bsm 布局 [β|α]
    assert isinstance(r["reverse_key"], bool)
    assert r["bsm"].shape[1] == nq
    # 假 counts 含 HF (0b0101) + 双激发 (0b1010) → 覆盖全空间 → e_sqd ≈ FCI
    assert r["e_sqd"] is not None
    assert abs(r["e_sqd"] - data.solve(method="fci")) < 1e-3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_hardware: all PASS")
