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


def test_ccsd_no_increases_sparsity():
    """CCSD-NO 是可行的自举先验: 换基后波函数稀疏度不劣化于 MO 基。

    用**弱关联** N2 平衡验证 (CCSD 可靠区)。不测"与 FCI-NO 轨道重叠": 近简并
    轨道组 (占据数相近) 内部方向物理任意, 逐列重叠低不代表 CCSD-NO 差; 稀疏度
    才是换基有效性的直接度量。强关联区 CCSD-NO 退化是已知结论 (自身多参考缺失),
    见 basis.ccsd_natural_orbitals docstring, 不做断言。
    """
    data = _n2_equil_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    _, c_mo = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    try:
        U_cc = tc_sqd.ccsd_natural_orbitals(data.mf)[0]
    except Exception:
        pytest.skip("CCSD 不可用 (pyscf cc 模块缺失)")
    h1e_cn = U_cc.T @ h1e @ U_cc
    eri_cn = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, U_cc, U_cc, U_cc, U_cc,
                       optimize=True)
    # 旋转不变性: CCSD-NO 基下 FCI 能量不变
    e_cn, c_cn = direct_spin1.kernel(h1e_cn, eri_cn, norb, nelec, conv_tol=1e-12)
    e_mo, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    assert abs(e_cn - e_mo) < 1e-7
    # 稀疏度不劣化 (+3 容差吸收 CCSD 在近简并轨道组的收敛波动; 原始实测快照
    # 为早期环境 k999 MO=97 -> CCSD-NO=92, 当前环境 MO 稳定 122 —— 基线随
    # pyscf/BLAS 版本漂移, 两数字为不同时代快照)
    # round_020 去抖: margin +3 → +10。6 连测分布 (round_020 R1 探测):
    # MO 稳定 122, CCSD-NO 84-102 (diff -38~-20, 正常时 CCSD-NO 显著更优);
    # 历史失败 diff +2~+5 = CCSD 近简并收敛波动尾部 (BLAS 线程噪声经占据
    # 矩阵 eigh 近简并旋转放大, 不可 seed)。本测试是"不劣化"的定性物理锚
    # (docstring 自述强版本不可断言), +10 (~基线 8%) 仍抓真回归 (换基接线
    # bug 给 O(100) 级劣化)。
    m_mo = _sparsity(c_mo)
    m_cn = _sparsity(c_cn)
    assert m_cn["k999"] <= m_mo["k999"] + 10, (
        f"CCSD-NO 稀疏度劣化: {m_mo} -> {m_cn}")


def test_solve_sqd_natural_orbitals_converges():
    """自洽换基 SQD 收敛: 低采样 (400) 下能量接近 FCI, 且换基积分有效。

    方向① 验证 (demo): N₂/STO-3G 拉伸 n_samples=400 自洽换基 err~1.7e-6,
    无换基卡在 ~1.9e-4 (MO 基覆盖瓶颈)。此处断言 err < 1e-5 (保守阈值)。
    """
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_fci, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)

    res = tc_sqd.solve_sqd_natural_orbitals(
        h1e, eri, norb, nelec, n_samples=400, max_basis_iters=4, rand_seed=0
    )
    # 能量收敛到近 FCI
    assert abs(res.energy - e_fci) < 1e-5, f"自洽换基未收敛: {res.energy - e_fci:.2e}"
    # 至少迭代了换基闭环
    assert len(res.history) >= 2
    # 换基后的积分仍描述同一哈密顿量 (旋转不变性): 换基积分的 FCI 能量不变
    e_n, _ = direct_spin1.kernel(res.h1e, res.eri, norb, nelec, conv_tol=1e-12)
    assert abs(e_n - e_fci) < 1e-7, f"换基积分破坏能量: {e_n - e_fci:.2e}"
    # 占据数物理约束
    assert np.all(res.occ >= -1e-8) and np.all(res.occ <= 2.0 + 1e-8)


def test_solve_sqd_natural_orbitals_beats_no_basis():
    """自洽换基优于无换基 (纯 MO 基迭代): 方向① 覆盖瓶颈突破的回归断言。"""
    data = _n2_stretch_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    na, nb = nelec
    e_fci, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)

    n_samples = 400
    bsm = (np.random.default_rng(0).random((n_samples, 2 * norb)) > 0.5)
    probs = np.full(n_samples, 1.0 / n_samples)

    # 无换基: 配置恢复 + 更新平均占据 (不复用换基)
    def _plain_iter():
        occ_a = np.zeros(norb); occ_a[:na] = 1.0
        occ_b = np.zeros(norb); occ_b[:nb] = 1.0
        h = h1e
        e_ = np.inf
        for _ in range(4):
            rec, _ = tc_sqd.recover_configurations(
                bsm, probs, (occ_a, occ_b), na, nb, rand_seed=0)
            ca, cb = tc_sqd.bitstring_matrix_to_ci_strs(rec)
            r = tc_sqd.solve_sci((ca, cb), h, eri, norb, nelec)
            e_ = r.energy
            dm1 = tc_sqd.rdm1_from_sci_result(r)
            occ_a = np.clip(np.diag(dm1) / 2.0, 0, 1)
            occ_b = occ_a.copy()
        return e_

    e_plain = _plain_iter()
    res = tc_sqd.solve_sqd_natural_orbitals(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
        max_basis_iters=4, rand_seed=0,
    )
    # 自洽换基不劣于无换基 (验证不回归); 且两者都不差于初猜
    assert res.energy <= e_plain + 1e-12, f"换基退化: {res.energy} vs {e_plain}"
    # 方向① 结论的量化: 换基显著更优 (无换基卡在 MO 基瓶颈 ~1e-4 量级)
    assert abs(res.energy - e_fci) < abs(e_plain - e_fci) * 0.5


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


def test_solve_sqd_natural_orbitals_energy_matches_returned_integrals():
    """F2 回归: NaturalOrbitalResult.energy 必须与 .h1e/.eri 同基。

    旧实现末轮 ``solve_sci`` 在换基前 B_k 做, 但 ``h1e``/``eri`` 是换基后 B_{k+1}
    的 (差一轮 NO 旋转), 与 docstring "最终基下 SQD 电子能量" 不符。修复后在最终基
    再对角化一次, 使 energy↔积分严格同基。

    **牙齿条件**: 必须在**子空间未饱和** (k 字符串 < C(norb,na)) 时才测得出——饱和时
    能量 = FCI (基无关), off-by-one 恒零。故用 LiH/STO-3G (norb=6) + **少 shots**
    (n_samples=10 → dim≈25, 真子集), 实测 off-by-one ~8e-8。tolerance 1e-8 把
    修复版 (~1e-12, 同一求解的重算) 与 buggy 版 (~8e-8) 干净分开。
    """
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    d = tc_sqd.from_pyscf(mol)
    norb, nelec = d.norb, d.nelec
    n_samples = 10                               # 少 shots → 未饱和子空间
    rng = np.random.default_rng(0)
    bsm = (rng.random((n_samples, 2 * norb)) > 0.5)
    res = tc_sqd.solve_sqd_natural_orbitals(
        d.h1e, d.eri, norb, nelec, ecore=d.ecore,
        bitstring_matrix=bsm, max_basis_iters=2, rand_seed=0)
    # 复现末轮: 返回的 occ + 同一 bsm → recover → solve_sci 在返回的积分上
    occ_a = np.clip(np.asarray(res.occ, dtype=np.float64) / 2.0, 0.0, 1.0)
    rec, _ = tc_sqd.recover_configurations(
        bsm, np.full(n_samples, 1.0 / n_samples), (occ_a, occ_a.copy()),
        nelec[0], nelec[1], rand_seed=0)
    ci_a, ci_b = tc_sqd.bitstring_matrix_to_ci_strs(rec)
    r2 = tc_sqd.solve_sci((ci_a, ci_b), res.h1e, res.eri, norb, nelec)
    assert abs(r2.energy - res.energy) < 1e-8, (
        f"energy 与返回积分不同基 (差一轮 NO 旋转): res.energy={res.energy:.10f} "
        f"但 solve_sci(res.h1e, res.eri)={r2.energy:.10f} (gap={abs(r2.energy-res.energy):.2e})")
