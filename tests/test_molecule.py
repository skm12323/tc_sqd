"""tc_sqd.molecule 模块测试 —— from_pyscf 一键分子接口。"""
import numpy as np
import tc_sqd
from pyscf import gto, scf


def test_from_pyscf_h2_matches_manual():
    """from_pyscf 返回的积分/核能/电子数与手写转换一致。"""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)          # 传 mol, 自动跑 RHF

    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    h1e_ref = mo.T @ mf.get_hcore() @ mo
    eri_ref = np.einsum("pqrs,pi,qj,rk,sl->ijkl",
                        mol.intor("int2e_sph"), mo, mo, mo, mo)

    assert data.norb == 2
    assert data.nelec == (1, 1)
    assert data.n_core == 0
    assert abs(data.ecore - mf.energy_nuc()) < 1e-12
    assert np.allclose(data.h1e, h1e_ref)
    assert np.allclose(data.eri, eri_ref)


def test_from_pyscf_accepts_scf_object():
    """传已收敛的 scf.RHF 对象等价于传 mol。"""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    data_mol = tc_sqd.from_pyscf(mol)
    data_mf = tc_sqd.from_pyscf(mf)
    assert np.allclose(data_mol.h1e, data_mf.h1e)
    assert data_mol.nelec == data_mf.nelec


def test_from_pyscf_solve_fci():
    """data.solve(method="fci") 复现手写 compute_ground_state_energy(fci)。"""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    e = data.solve(method="fci")
    e_ref = tc_sqd.compute_ground_state_energy(
        data.h1e, data.eri, data.norb, data.nelec,
        ecore=data.ecore, method="fci")
    assert abs(e - e_ref) < 1e-10
    assert abs(e - (-1.13728383)) < 1e-5      # H2/STO-3G FCI 已知值


def test_from_pyscf_active_space_lif():
    """n_active 冻结 core: 与"MO0 严格双占据的受限 full 对角化"精确一致。

    LiH/STO-3G: 6 MO / 4 e, Li 1s (MO0) 是深 core。frozen-core 近似把 MO0
    固定为双占据; 参考解 = 在 ``alpha=beta = bit0 + 1 活性 bit`` 的受限子空间
    上直接对角化 full 哈密顿量 (6 MO, 4e), 应**精确等于** from_pyscf(n_active=5)
    的活性 FCI (同一物理, 不同实现)。
    """
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    h1e_full = mo.T @ mf.get_hcore() @ mo
    eri_full = np.einsum("pqrs,pi,qj,rk,sl->ijkl",
                         mol.intor("int2e_sph"), mo, mo, mo, mo)
    nuc = mf.energy_nuc()

    data_full = tc_sqd.from_pyscf(mf)
    data_act = tc_sqd.from_pyscf(mf, n_active=data_full.norb - 1)

    assert data_act.norb == data_full.norb - 1
    assert data_act.nelec == (data_full.nelec[0] - 1, data_full.nelec[1] - 1)
    assert data_act.n_core == 1

    # 参考: MO0 (bit0) 恒占据 + 在 MO1..5 选 1 个活性轨道, α/β 各 2 电子
    ci_strs = np.array([1 | (1 << i) for i in range(1, 6)], dtype=np.int64)
    res = tc_sqd.solve_sci((ci_strs, ci_strs), h1e_full, eri_full, 6, (2, 2))
    e_restricted = res.energy + nuc

    e_act = data_act.solve(method="fci")
    assert abs(e_act - e_restricted) < 1e-8, (
        f"frozen-core 实现不精确: 活性 FCI={e_act:.8f}, "
        f"受限 full={e_restricted:.8f}")

    # frozen-core 冻结 core-valence 关联, vs 全空间 FCI 差 ~2e-4 Ha (量级校验)
    e_full = data_full.solve(method="fci")
    assert abs(e_full - e_act) < 1e-3, (
        f"frozen-core 偏差异常大: 活性 FCI={e_act:.8f}, 全空间 FCI={e_full:.8f}")


def test_from_pyscf_validation():
    """非法输入显式报错。"""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    # n_active 超范围
    try:
        tc_sqd.from_pyscf(mol, n_active=99)
        assert False, "n_active 超范围应报错"
    except ValueError:
        pass
    # 非 Mole/SCF 输入
    try:
        tc_sqd.from_pyscf("not a molecule")
        assert False, "非法输入应报错"
    except ValueError:
        pass


def test_from_pyscf_open_shell_ch():
    """P2-1b: CH/STO-3G (4,3) 一键 from_pyscf, fci = PySCF FCI; UHF reject。"""
    from pyscf import fci as fci_mod

    mol = gto.M(atom="C 0 0 0; H 0 0 1.1", basis="sto-3g", spin=1, verbose=0)
    data = tc_sqd.from_pyscf(mol)
    assert data.nelec == (4, 3)               # CH 7e, 双自由基 (2S=1)
    assert type(data.mf).__name__ == "ROHF"   # spin!=0 自动 ROHF

    e = data.solve(method="fci")
    e_ref = fci_mod.direct_spin1.kernel(
        data.h1e, data.eri, data.norb, data.nelec)[0] + data.ecore
    assert abs(e - e_ref) < 1e-8

    # UHF 显式 reject
    mf_u = scf.UHF(mol).run()
    try:
        tc_sqd.from_pyscf(mf_u)
        assert False, "UHF 应显式 reject"
    except ValueError:
        pass


def test_from_pyscf_open_shell_frozen():
    """P2-1b: CH 开壳层冻结 core (n_active), nelec 分减。"""
    mol = gto.M(atom="C 0 0 0; H 0 0 1.1", basis="sto-3g", spin=1, verbose=0)
    data = tc_sqd.from_pyscf(mol)
    data_c = tc_sqd.from_pyscf(mol, n_active=data.norb - 1)   # 冻结 1 MO (C 1s)
    assert data_c.norb == data.norb - 1
    assert data_c.nelec == (3, 2)             # (4,3) - core 双占
    assert data_c.n_core == 1
    # fci 可跑通 (frozen-core 近似)
    assert abs(data_c.solve(method="fci") - data.solve(method="fci")) < 1e-3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_molecule: all PASS")
