"""tc_sqd 开壳层 (n_α≠n_β) 测试 —— P2-1 期 1 + 期 2。

回归体系 CH/STO-3G (5e, na=3/nb=2, 双自由基)。from_pyscf 拒奇电子,
故空间积分在测试内自备 (ROHF 式: 单 h1e + 共用 eri)。
"""
import numpy as np
import tensorcircuit as tc
import tc_sqd
from pyscf import gto, scf, fci


def _ch_data():
    """CH/STO-3G 空间积分 (自备, 开壳层 5e)。"""
    mol = gto.M(atom="C 0 0 0; H 0 0 1.1", basis="sto-3g", spin=1, verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl",
                    mol.intor("int2e_sph"), mo, mo, mo, mo)
    return {"h1e": h1e, "eri": eri, "norb": mol.nao_nr(),
            "nelec": (3, 2), "ecore": mf.energy_nuc()}


def test_open_shell_fci_direct_match():
    """期1: CH (3,2) fci/direct 路径 = PySCF FCI (direct 的 divmod na!=nb 索引正确)。"""
    d = _ch_data()
    e_ref = fci.direct_spin1.kernel(d["h1e"], d["eri"], d["norb"],
                                    d["nelec"])[0] + d["ecore"]
    e_fci = tc_sqd.compute_ground_state_energy(
        d["h1e"], d["eri"], d["norb"], d["nelec"], ecore=d["ecore"],
        method="fci")
    e_dir = tc_sqd.compute_ground_state_energy(
        d["h1e"], d["eri"], d["norb"], d["nelec"], ecore=d["ecore"],
        method="direct")
    assert abs(e_fci - e_ref) < 1e-8
    assert abs(e_dir - e_ref) < 1e-8


def test_open_shell_sqd_hf_variational():
    """期1: CH (3,2) sqd HF 电路 (右半 α3, 左半 β2) >= FCI (变分下界)。"""
    d = _ch_data()
    norb, nelec = d["norb"], d["nelec"]
    c = tc.Circuit(2 * norb)
    for i in range(nelec[1]):            # β 左半: qubit norb-1..0
        c.x(norb - 1 - i)
    for i in range(nelec[0]):            # α 右半: qubit 2norb-1..norb
        c.x(2 * norb - 1 - i)
    bsm, probs = tc_sqd.sample(c, 2000)
    e = tc_sqd.compute_ground_state_energy(
        d["h1e"], d["eri"], norb, nelec, ecore=d["ecore"], method="sqd",
        bitstring_matrix=bsm, probabilities=probs, max_iterations=3)
    e_ref = fci.direct_spin1.kernel(d["h1e"], d["eri"], norb, nelec)[0] + d["ecore"]
    assert e >= e_ref - 1e-8


def test_open_shell_solve_fermion():
    """期1: solve_fermion open_shell=True 对 (3,2) 走通 (HF bsm)。"""
    d = _ch_data()
    norb, nelec = d["norb"], d["nelec"]
    # HF 位串 [β5..β0 | α5..α0]: β0,β1 占; α0,α1,α2 占
    hf = np.zeros((1, 2 * norb), dtype=bool)
    hf[0, norb - 1] = hf[0, norb - 2] = True          # β0, β1
    hf[0, 2 * norb - 1] = hf[0, 2 * norb - 2] = hf[0, 2 * norb - 3] = True
    e, state, occ, _ = tc_sqd.solve_fermion(
        hf, d["h1e"], d["eri"], open_shell=True, _nelec=nelec)
    e_ref = fci.direct_spin1.kernel(d["h1e"], d["eri"], norb, nelec)[0]
    assert abs(e - e_ref) < 1e-6 or e >= e_ref - 1e-6   # HF 子空间 >= FCI


def test_open_shell_recover_na_nb():
    """期2: recover_configurations na≠nb 粒子数修复 (β=2, α=3)。"""
    d = _ch_data()
    norb = d["norb"]
    occ_a = np.zeros(norb); occ_a[:3] = 1.0
    occ_b = np.zeros(norb); occ_b[:2] = 1.0
    bsm = np.array([
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1],   # HF 合法
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1],   # β 缺 1
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],   # α 多 1
    ], dtype=bool)
    probs = np.ones(3) / 3
    rec, _ = tc_sqd.recover_configurations(bsm, probs, (occ_a, occ_b),
                                           3, 2, rand_seed=42)
    for row in rec:
        assert row[:norb].sum() == 2, "β 半应恰好 2 电子"
        assert row[norb:].sum() == 3, "α 半应恰好 3 电子"


def test_open_shell_estimate_true_occupancies():
    """期2: estimate_true_occupancies na≠nb 分归一 (α→3, β→2)。"""
    d = _ch_data()
    norb = d["norb"]
    bsm = np.tile(
        np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool),
        (20, 1))
    est_a, est_b = tc_sqd.estimate_true_occupancies(bsm, 3, 2, 0.1, norb=norb)
    assert abs(est_a.sum() - 3) < 1e-8
    assert abs(est_b.sum() - 2) < 1e-8


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_open_shell: all PASS")
