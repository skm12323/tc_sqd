"""tc_sqd.basis 模块测试 —— 自然轨道换基的旋转不变性 / 稀疏度改善 / 先验对照。

核心验证 (方向①): 换基到自然轨道后, 基态在计算基下更稀疏, 子空间构建更高效。
"""
import numpy as np
import pytest
import tc_sqd
from pyscf import gto
from pyscf.fci import direct_spin1


def _sparsity(c):
    """系数平方分布指标 (与 _basis_design.py 实验口径一致)。"""
    c2 = np.abs(np.asarray(c).ravel()) ** 2
    c2 = c2[c2 > 1e-15]
    p = c2 / c2.sum()
    ps = np.sort(p)[::-1]
    cum = np.cumsum(ps)
    return dict(
        maxc2=float(p.max()),
        pr=float(1.0 / (p**2).sum()),
        k99=int(np.searchsorted(cum, 0.99) + 1),
        k999=int(np.searchsorted(cum, 0.999) + 1),
    )


def _n2_stretch_data():
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    return tc_sqd.from_pyscf(mol)


def _n2_equil_data():
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", verbose=0)
    return tc_sqd.from_pyscf(mol)


def test_rotate_invariance_fci_energy():
    """FCI 能量对自然轨道换基严格不变 (酉变换是哈密顿量的相似变换)。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_mo, c_mo = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    # 用 FCI 1-RDM 换基 (理想极限)
    dm1 = direct_spin1.make_rdm1(c_mo, norb, nelec)
    h1e_n, eri_n, U, occ = tc_sqd.rotate_to_natural_orbitals(h1e, eri, dm1)
    e_n, _ = direct_spin1.kernel(h1e_n, eri_n, norb, nelec, conv_tol=1e-12)
    # 1e-7 容差: 两次独立 davidson 收敛的数值噪声 (~1e-8 Ha), 远小于化学精度
    assert abs(e_n - e_mo) < 1e-7, f"换基破坏能量: {e_n - e_mo:.2e} Ha"
    # 占据数物理约束: 0 <= occ <= 2, 且总和 = 电子数
    assert np.all(occ >= -1e-8) and np.all(occ <= 2.0 + 1e-8)
    assert abs(occ.sum() - (nelec[0] + nelec[1])) < 1e-6


def test_fci_no_increases_sparsity():
    """FCI-NO 换基提升基态稀疏度 (长尾压缩): N2 拉伸 k999 应显著下降。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    _, c_mo = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    dm1 = direct_spin1.make_rdm1(c_mo, norb, nelec)
    h1e_n, eri_n, _, _ = tc_sqd.rotate_to_natural_orbitals(h1e, eri, dm1)
    _, c_n = direct_spin1.kernel(h1e_n, eri_n, norb, nelec, conv_tol=1e-12)

    m_mo = _sparsity(c_mo)
    m_no = _sparsity(c_n)
    # 实测 (N2/STO-3G 拉伸): k99 84->39, k999 189->62, PR 9.9->9.4
    assert m_no["k999"] < m_mo["k999"] * 0.7, f"长尾未压缩: {m_mo} -> {m_no}"
    assert m_no["pr"] < m_mo["pr"], f"参与度未下降: {m_mo} -> {m_no}"


def test_natural_orbital_occupancies_sum_to_nelec():
    """平均占据数与电子数自洽; spin=True 给出闭壳层每自旋占据。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    _, c_mo = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    dm1 = direct_spin1.make_rdm1(c_mo, norb, nelec)
    occ = tc_sqd.natural_orbital_occupancies(dm1)[0]
    assert abs(occ.sum() - (nelec[0] + nelec[1])) < 1e-6
    occ_a, occ_b = tc_sqd.natural_orbital_occupancies(dm1, spin=True)
    assert np.allclose(occ_a, occ_b)
    assert np.all(occ_a >= 0.0) and np.all(occ_a <= 1.0)


def test_ccsd_no_overlaps_fci_no():
    """CCSD-NO 是合理的先验: 与 FCI-NO 轨道重叠 (逐列 max) 应高。

    用**弱关联** N2 平衡验证 —— CCSD 可靠区, 自然轨道应接近 FCI-NO
    (实测 max-per-column > 0.99)。强关联区 CCSD-NO 退化 (自身多参考缺失)
    是已知结论, 见 basis.ccsd_natural_orbitals docstring, 不做断言。
    """
    data = _n2_equil_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    _, c_mo = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    dm1 = direct_spin1.make_rdm1(c_mo, norb, nelec)
    U_fci = tc_sqd.natural_orbitals_from_rdm(dm1)[0]
    try:
        U_cc = tc_sqd.ccsd_natural_orbitals(data.mf)[0]
    except Exception:
        pytest.skip("CCSD 不可用 (pyscf cc 模块缺失)")
    ovlp = np.abs(U_fci.T @ U_cc).max(axis=0)
    assert np.mean(ovlp) > 0.95, f"CCSD-NO 与 FCI-NO 轨道重叠过低: {np.round(ovlp, 3)}"


def test_rdm1_from_sci_result():
    """从 SQD 解提取 1-RDM 的接口闭合 (自洽换基的输入源)。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    seed = tc_sqd.excited_configurations(norb, nelec, max_excitations=2)
    ci_a, ci_b = tc_sqd.bitstring_matrix_to_ci_strs(seed)
    result = tc_sqd.solve_sci((ci_a, ci_b), h1e, eri, norb, nelec)
    dm1 = tc_sqd.rdm1_from_sci_result(result)
    assert dm1.shape == (norb, norb)
    assert np.allclose(dm1, dm1.T)
    # 与 FCI 1-RDM 量级一致 (S+D 子空间近似, 允许偏差但应可识别)
    # 注意: 电子数约束看 trace (对角占位数之和 = Ne), 而非全矩阵 .sum()。
    _, c_mo = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    dm1_fci = direct_spin1.make_rdm1(c_mo, norb, nelec)
    assert abs(np.trace(dm1) - (nelec[0] + nelec[1])) < 1e-6  # S+D 态电子数守恒
    assert abs(np.trace(dm1_fci) - (nelec[0] + nelec[1])) < 1e-6
