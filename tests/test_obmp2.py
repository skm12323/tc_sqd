"""tc_sqd.obmp2 模块测试 —— OBMP2 一体相关势 + 自洽求解。

验证 (全部为 2026-08-10 实测锁定):
- 1st BCH 势对称、E_OBMP2(0) = E_HF 恒等式 (⟨Φ|V̂|Φ⟩ + C' = 0)。
- 归一化 (A_D=½T): v_1st×½, v_2nd×(-1/8) 后 E_OBMP2(0) 精确 = E_MP2 (N₂/H₂O/STO-3G)。
- 自洽 OBMP2 收敛, E 介于 E_HF 与 FCI 之间, ≈ CCSD (N₂/STO-3G 差 < 1 mHa)。
"""
import numpy as np
from pyscf import gto, scf, mp, cc

import tc_sqd
from tc_sqd.obmp2 import obmp2_potential, solve_obmp2


def _h2o_mf():
    return scf.RHF(gto.M(atom="O 0 0 0; H 0 0.96 0; H 0 -0.96 0", basis="sto-3g", verbose=0)).run()


def _n2_mf():
    return scf.RHF(gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", verbose=0)).run()


def test_obmp2_potential_symmetric():
    """1st BCH 势对称 (自旋求和空间势)。"""
    mf = _n2_mf()
    norb = mf.mo_coeff.shape[1]
    nocc = mf.mol.nelectron // 2
    mo = np.asarray(mf.mo_coeff)
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", mf.mol.intor("int2e_sph"), mo, mo, mo, mo, optimize=True)
    v, Cp, _ = obmp2_potential(h1e, eri, mf.mo_energy, norb, nocc,
                               include_2nd_bch=True, fock=np.diag(mf.mo_energy))
    assert np.allclose(v, v.T), "OBMP2 势应对称"


def test_obmp2_reproduces_mp2_at_hf():
    """E_OBMP2(0) = E_HF + 2·Tr_occ(v) + C' 精确 = E_MP2 (归一化正确)。"""
    for mf in (_n2_mf(), _h2o_mf()):
        norb = mf.mo_coeff.shape[1]
        nocc = mf.mol.nelectron // 2
        mo = np.asarray(mf.mo_coeff)
        h1e = mo.T @ mf.get_hcore() @ mo
        eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", mf.mol.intor("int2e_sph"),
                        mo, mo, mo, mo, optimize=True)
        v, Cp, _ = obmp2_potential(h1e, eri, mf.mo_energy, norb, nocc,
                                   include_2nd_bch=True, fock=np.diag(mf.mo_energy))
        Tr_v = float(np.sum(np.diag(v)[:nocc]))
        e_obmp2_0 = mf.e_tot + 2.0 * Tr_v + Cp
        e_mp2 = mp.MP2(mf).run().e_tot
        assert abs(e_obmp2_0 - e_mp2) < 1e-6, f"{e_obmp2_0:.8f} vs MP2 {e_mp2:.8f}"


def test_obmp2_scf_between_hf_and_fci():
    """自洽 OBMP2 能量介于 E_HF 与 FCI 之间 (相关能正确、不过校正)。"""
    from pyscf import fci as fci_mod
    mf = _n2_mf()
    r = solve_obmp2(mf, max_iter=80, tol=1e-9)
    e_fci = fci_mod.FCI(mf).kernel()[0]
    assert r.converged
    assert r.energy < mf.e_tot, "OBMP2 应低于 E_HF (捕获相关能)"
    assert r.energy > e_fci - 1e-4, "OBMP2 不应低于 FCI (微扰方法不越过变分下界)"
    assert r.energy < mf.e_tot, "OBMP2 相关能应为负"


def test_obmp2_scf_matches_ccsd():
    """N₂/STO-3G 平衡: E_OBMP2 ≈ CCSD (差 < 1 mHa)。"""
    mf = _n2_mf()
    r = solve_obmp2(mf, max_iter=80, tol=1e-9)
    e_ccsd = cc.CCSD(mf).run().e_tot
    assert abs(r.energy - e_ccsd) < 1e-3, f"OBMP2 {r.energy:.8f} vs CCSD {e_ccsd:.8f}"


def test_obmp2_v_ext_active_range():
    """active_range 外部限制: 返回活性块 v^ext (形状/有限/非平凡)。"""
    mf = _n2_mf()
    norb = mf.mo_coeff.shape[1]
    nocc = mf.mol.nelectron // 2
    mo = np.asarray(mf.mo_coeff)
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", mf.mol.intor("int2e_sph"),
                    mo, mo, mo, mo, optimize=True)
    v_ext, _, _ = obmp2_potential(h1e, eri, mf.mo_energy, norb, nocc,
                                  include_2nd_bch=True, fock=np.diag(mf.mo_energy),
                                  active_range=(2, norb - 1))
    assert v_ext.shape == (norb - 3, norb - 3)
    assert np.all(np.isfinite(v_ext))
    assert np.allclose(v_ext, v_ext.T)
    assert np.abs(v_ext).max() > 1e-8, "外部限制 v^ext 应非平凡"


def test_obdf_downfold_structure():
    """obdf_downfold 结构: h1e_downfolded = h1e + scale·v^ext, eri/nelec 不变。"""
    mf = scf.RHF(gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)).run()
    r = tc_sqd.obdf_downfold(mf, n_core=1, n_virtual=0, scale=0.1)
    assert r.norb == mf.mo_coeff.shape[1] - 1
    assert r.nelec == (1, 1)
    assert r.n_core == 1 and r.n_virtual == 0
    assert r.v_ext.shape == (r.norb, r.norb)
    assert np.allclose(r.h1e_downfolded, r.h1e + 0.1 * r.v_ext)
    assert np.allclose(r.eri, r.eri)  # eri 不变 (仅改 h1e)


def test_obdf_downfold_differs_from_cas():
    """OBDF 能量与 plain CAS 不同 (非平凡效应), 且仅改 h1e。"""
    mf = scf.RHF(gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)).run()
    r = tc_sqd.obdf_downfold(mf, n_core=1, n_virtual=0, scale=0.1)
    e_cas = r.solve(downfolded=False)
    e_obdf = r.solve(downfolded=True)
    assert abs(e_obdf - e_cas) > 1e-6, "OBDF 应改变活性能量"
    # 对照: 手写加 v^ext
    e_manual = tc_sqd.compute_ground_state_energy(
        r.h1e + 0.1 * r.v_ext, r.eri, r.norb, r.nelec, ecore=r.ecore, method="fci")
    assert abs(e_obdf - e_manual) < 1e-10


def test_obdf_downfold_validation():
    """OBDF 校验: 无外部空间 / 开壳层报错。"""
    # n_core + n_virtual = 0
    mf = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)).run()
    try:
        tc_sqd.obdf_downfold(mf, n_core=0, n_virtual=0)
        raise AssertionError("无外部空间应报错")
    except ValueError:
        pass
    # 开壳层
    mol = gto.M(atom="C 0 0 0; H 0 0 1.1", basis="sto-3g", spin=1, verbose=0)
    mf_open = scf.ROHF(mol).run()
    try:
        tc_sqd.obdf_downfold(mf_open, n_core=1, n_virtual=0)
        raise AssertionError("开壳层应报错")
    except ValueError:
        pass


if __name__ == "__main__":
    import tc_sqd  # noqa
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_obmp2: all PASS")
