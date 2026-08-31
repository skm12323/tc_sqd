"""round_011 —— 自旋分辨积分 (h_α≠h_β) 支持测试。

P0: 正确性 (≥2 UHF 体系 vs pyscf.fci.direct_uhf ≤1e-10)
P1: 零回归 (legacy collapse 逐位)
P2: from_pyscf(UHF) 五积分转换 (≤1e-12) + 端到端
边界: 派发真值表两 raise + 范围外功能 (_Subspace/csf/GPU/linkstr/frozen-core) raise
"""

import numpy as np
import pytest

from pyscf import fci, gto, scf
from pyscf.fci import cistring

from tc_sqd.matrixfree import (
    sigma_vector,
    prepare_sigma_operators,
    sigma_vector_ops,
    _fock_cross,
    _fock_cross_beta,
    sigma_linkstr_gpu,
    eigsh_linkstr_gpu,
)
from tc_sqd.fermion import (
    solve_sci,
    build_ci_matrix,
    solve_sci_csf,
    compute_ground_state_energy,
)
from tc_sqd.molecule import from_pyscf
from tc_sqd.cipsi import _Subspace


# --------------------------------------------------------------------------- #
#  UHF 体系 fixture
# --------------------------------------------------------------------------- #
def _n2_uhf():
    """N2/STO-3G R=2.5 Å UHF 破坏对称解 (7,7) —— P0 主锚。"""
    mol = gto.M(atom="N 0 0 0; N 0 0 2.5", basis="sto-3g", verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def _ch_uhf():
    """CH/STO-3G UHF (4,3) —— P0 第二锚 (dim 300, 走稠密分支)。"""
    mol = gto.M(atom="C 0 0 0; H 0 0 1.12", basis="sto-3g", spin=1,
                verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def _uhf_integrals(mf):
    """mf → (h1e (2,norb,norb), eri (aa,ab,bb), norb, nelec)。"""
    data = from_pyscf(mf)
    return data.h1e, data.eri, data.norb, data.nelec


@pytest.fixture(scope="module")
def n2():
    return _uhf_integrals(_n2_uhf())


@pytest.fixture(scope="module")
def ch():
    return _uhf_integrals(_ch_uhf())


def _full_strs(norb, nelec):
    sa = cistring.make_strings(range(norb), nelec[0])
    sb = cistring.make_strings(range(norb), nelec[1])
    return sa, sb


def _direct_uhf_ref(h1e, eri, norb, nelec):
    return fci.direct_uhf.kernel(
        (h1e[0], h1e[1]), tuple(eri), norb, nelec,
        conv_tol=1e-12, max_cycle=1000)[0]


# --------------------------------------------------------------------------- #
#  §1.3 陷阱锁定: _fock_cross_beta 方向
# --------------------------------------------------------------------------- #
def test_fock_cross_beta_direction():
    rng = np.random.default_rng(110)
    na, norb = 5, 4
    occ_a = (rng.random((na, norb)) > 0.5).astype(float)
    eri_ab = rng.random((norb,) * 4)            # 随机**不对称** (锁定方向)
    F = _fock_cross_beta(occ_a, eri_ab)
    # 手工: F[i,b,q] = Σ_p occ_a[i,p] eri_ab[p,p,b,q]
    manual = np.einsum("ip,ppbq->ibq", occ_a, eri_ab)
    assert np.allclose(F, manual, atol=0, rtol=0)
    for _ in range(3):                          # 逐元素抽查
        i, b, q = rng.integers(na), rng.integers(norb), rng.integers(norb)
        expect = sum(occ_a[i, p] * eri_ab[p, p, b, q] for p in range(norb))
        assert F[i, b, q] == pytest.approx(expect, abs=1e-14)
    # 退化: eri_ab 为 8-fold 对称单块时与旧 _fock_cross(occ_a, eri) 数值一致
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mo = np.asarray(scf.RHF(mol).run().mo_coeff, dtype=float)
    from pyscf import ao2mo
    eri_sym = ao2mo.restore(1, ao2mo.kernel(mol, mo), mo.shape[1])
    F_new = _fock_cross_beta(occ_a[:, :eri_sym.shape[0]], eri_sym)
    F_old = _fock_cross(occ_a[:, :eri_sym.shape[0]], eri_sym)
    assert np.allclose(F_new, F_old, atol=1e-12)


# --------------------------------------------------------------------------- #
#  σ-vector vs direct_uhf matvec
# --------------------------------------------------------------------------- #
def test_sigma_vector_spin_resolved_vs_direct_uhf_matvec(n2):
    h1e, eri, norb, nelec = n2
    sa, sb = _full_strs(norb, nelec)
    rng = np.random.default_rng(11)
    v = rng.random((len(sa), len(sb)))
    sigma = sigma_vector(v, sa, sb, norb, nelec, h1e, eri)
    # 参考: pyscf direct_uhf C 核 (独立实现)。
    # 注意: 直接 contract_1e + contract_2e(raw) 的组合在本 pyscf 版本与
    # kernel 不自洽 (absorb_h1e 的分配约定不同); kernel/davidson 实际走
    # contract_2e(absorb_h1e(...)), 以下用该口径。
    h2e_abs = fci.direct_uhf.absorb_h1e((h1e[0], h1e[1]), tuple(eri),
                                        norb, nelec, 0.5)
    ref = fci.direct_uhf.contract_2e(h2e_abs, v, norb, nelec)
    assert np.max(np.abs(sigma - ref)) < 1e-12


# --------------------------------------------------------------------------- #
#  P0(a): solve_sci 全空间 vs direct_uhf.kernel
# --------------------------------------------------------------------------- #
def test_solve_sci_spin_resolved_full_space_vs_direct_uhf_kernel():
    # 前置: 防假 UHF (未破坏对称时测试自身无效)
    mf_n2 = _n2_uhf()
    assert not np.allclose(mf_n2.mo_coeff[0], mf_n2.mo_coeff[1])
    mf_ch = _ch_uhf()
    assert not np.allclose(mf_ch.mo_coeff[0], mf_ch.mo_coeff[1])

    for mf in (mf_n2, mf_ch):
        h1e, eri, norb, nelec = _uhf_integrals(mf)
        e_ref = _direct_uhf_ref(h1e, eri, norb, nelec)
        sa, sb = _full_strs(norb, nelec)
        res = solve_sci((sa, sb), h1e, eri, norb, nelec)
        assert abs(res.energy - e_ref) <= 1e-10


# --------------------------------------------------------------------------- #
#  P0(b): FCI 轨道基不变量 (UHF 轨道 vs RHF 轨道)
# --------------------------------------------------------------------------- #
def test_solve_sci_spin_resolved_basis_invariance():
    mol = gto.M(atom="N 0 0 0; N 0 0 2.5", basis="sto-3g", verbose=0)
    mf_uhf = scf.UHF(mol)
    mf_uhf.conv_tol = 1e-12
    mf_uhf.kernel()
    h1e, eri, norb, nelec = _uhf_integrals(mf_uhf)
    sa, sb = _full_strs(norb, nelec)
    e_uhf_basis = solve_sci((sa, sb), h1e, eri, norb, nelec).energy

    mf_rhf = scf.RHF(mol)
    mf_rhf.conv_tol = 1e-12
    mf_rhf.kernel()
    mo = np.asarray(mf_rhf.mo_coeff, dtype=float)
    h1 = mo.T @ mf_rhf.get_hcore() @ mo
    from pyscf import ao2mo
    eri1 = ao2mo.restore(
        1, ao2mo.kernel(mol, mo), norb)
    e_rhf_basis = fci.direct_spin1.kernel(
        h1, eri1, norb, nelec, conv_tol=1e-12, max_cycle=1000)[0]
    assert abs(e_uhf_basis - e_rhf_basis) <= 1e-8


# --------------------------------------------------------------------------- #
#  P0(c): 随机子空间稠密 H vs direct_uhf.contract_2e 切片
# --------------------------------------------------------------------------- #
def test_solve_sci_spin_resolved_subspace_dense(n2):
    h1e, eri, norb, nelec = n2
    sa_full, sb_full = _full_strs(norb, nelec)
    rng = np.random.default_rng(110)
    # 随机子空间 dim ~ 1e3: 35 α × 30 β = 1050 (→ 稠密分支, >1000 走 ops 列)
    sa = np.sort(rng.choice(sa_full, 35, replace=False))
    sb = np.sort(rng.choice(sb_full, 30, replace=False))
    na, nb = len(sa), len(sb)
    dim = na * nb

    ops = prepare_sigma_operators(sa, sb, norb, nelec, h1e, eri)
    H = np.zeros((dim, dim))
    for col in range(dim):
        e = np.zeros(dim)
        e[col] = 1.0
        H[:, col] = sigma_vector_ops(
            e.reshape(na, nb), ops).ravel()

    # 参考: 全空间 direct_uhf.contract_2e(absorb_h1e) 作用到子空间基矢切片
    h2e_abs = fci.direct_uhf.absorb_h1e((h1e[0], h1e[1]), tuple(eri),
                                        norb, nelec, 0.5)
    pos_a = np.searchsorted(sa_full, sa)
    pos_b = np.searchsorted(sb_full, sb)
    H_ref = np.zeros((dim, dim))
    for col in range(dim):
        ia, ib = divmod(col, nb)
        va = np.zeros((len(sa_full), len(sb_full)))
        va[pos_a[ia], pos_b[ib]] = 1.0
        col_full = fci.direct_uhf.contract_2e(h2e_abs, va, norb, nelec)
        H_ref[:, col] = np.asarray(col_full)[pos_a, :][:, pos_b].ravel()
    assert np.max(np.abs(H - H_ref)) <= 1e-10
    e1 = np.linalg.eigvalsh(H)[0]
    e2 = np.linalg.eigvalsh(H_ref)[0]
    assert abs(e1 - e2) <= 1e-10

    # solve_sci 同子空间 (稠密分支) 与显式 H 最小本征值一致
    res = solve_sci((sa, sb), h1e, eri, norb, nelec)
    assert abs(res.energy - e1) <= 1e-10


# --------------------------------------------------------------------------- #
#  P1: legacy collapse 逐位不变
# --------------------------------------------------------------------------- #
def test_solve_sci_legacy_collapse_bit_identical():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
    mf = scf.RHF(mol).run()
    mo = np.asarray(mf.mo_coeff, dtype=float)
    h1 = mo.T @ mf.get_hcore() @ mo
    from pyscf import ao2mo
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mo), mo.shape[1])
    norb, nelec = mo.shape[1], (1, 1)
    sa, sb = _full_strs(norb, nelec)
    rng = np.random.default_rng(1)
    # 固定随机 v: sigma 逐位比较 (全空间 4×4, cistring 标准序)
    v = rng.random((len(sa), len(sb)))
    h1e_3d = np.stack([h1, h1])
    s2d = sigma_vector(v, sa, sb, norb, nelec, h1, eri)
    s3d = sigma_vector(v, sa, sb, norb, nelec, h1e_3d, eri)
    assert np.array_equal(s2d, s3d)                 # 逐位
    # 等块 + 三元组 (aa,ab,bb)=同一块: 与单块 canonical 路径一致 (数值级)
    s_sr = sigma_vector(v, sa, sb, norb, nelec, h1e_3d,
                        (eri, eri, eri))
    assert np.allclose(s2d, s_sr, atol=1e-12)


# --------------------------------------------------------------------------- #
#  P2: from_pyscf(UHF) 五积分 vs 手工 einsum
# --------------------------------------------------------------------------- #
def test_from_pyscf_uhf_five_integrals_vs_einsum():
    mf = _ch_uhf()
    mol = mf.mol
    mo_a, mo_b = (np.asarray(m, dtype=float) for m in mf.mo_coeff)
    hcore = mf.get_hcore()
    eri_ao = mol.intor("int2e_sph")

    h_a_ref = mo_a.T @ hcore @ mo_a
    h_b_ref = mo_b.T @ hcore @ mo_b
    aa_ref = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao,
                       mo_a, mo_a, mo_a, mo_a, optimize=True)
    ab_ref = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao,
                       mo_a, mo_a, mo_b, mo_b, optimize=True)
    bb_ref = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao,
                       mo_b, mo_b, mo_b, mo_b, optimize=True)

    data = from_pyscf(mf)
    assert data.spin_resolved
    assert data.nelec == tuple(mf.nelec)
    assert np.max(np.abs(data.h1e[0] - h_a_ref)) <= 1e-12
    assert np.max(np.abs(data.h1e[1] - h_b_ref)) <= 1e-12
    for got, ref in zip(data.eri, (aa_ref, ab_ref, bb_ref)):
        assert np.max(np.abs(got - ref)) <= 1e-12


def test_from_pyscf_uhf_solve_fci():
    mf = _ch_uhf()
    data = from_pyscf(mf)
    e = data.solve(method="fci")
    e_ref = _direct_uhf_ref(data.h1e, data.eri, data.norb,
                            data.nelec) + data.ecore
    assert abs(e - e_ref) <= 1e-10


def test_from_pyscf_uhf_n_virtual_slice():
    mf = _ch_uhf()
    data = from_pyscf(mf, n_virtual=1)
    assert data.norb == mf.mo_coeff[0].shape[1] - 1
    e = data.solve(method="fci")
    assert np.isfinite(e)


# --------------------------------------------------------------------------- #
#  边界 raise
# --------------------------------------------------------------------------- #
def test_spin_resolved_raises():
    mol = gto.M(atom="C 0 0 0; H 0 0 1.12", basis="sto-3g", spin=1,
                verbose=0)
    mf = _ch_uhf()
    h1e, eri, norb, nelec = _uhf_integrals(mf)
    h_eq = np.stack([h1e[0], h1e[0]])
    sa, sb = _full_strs(norb, nelec)
    ci = (sa, sb)
    eri_nd = np.asarray(eri[0])

    # 真值表两 raise (solve_sci)
    with pytest.raises(ValueError, match="三元组"):
        solve_sci(ci, h1e, eri_nd, norb, nelec)          # 不等 h + 单 eri
    with pytest.raises(ValueError, match="h_alpha, h_beta"):
        solve_sci(ci, h1e[0], eri, norb, nelec)          # 单 h + 三块 eri
    # compute_ground_state_energy 同派发
    with pytest.raises(ValueError):
        compute_ground_state_energy(h1e, eri_nd, norb, nelec)
    with pytest.raises(ValueError):
        compute_ground_state_energy(h1e[0], eri, norb, nelec)
    # build_ci_matrix 同派发
    with pytest.raises(ValueError):
        build_ci_matrix(sa, sb, h1e, eri_nd, norb, nelec)
    with pytest.raises(ValueError):
        build_ci_matrix(sa, sb, h1e[0], eri, norb, nelec)

    # 首期范围外 raise
    with pytest.raises(ValueError, match="spin_sq"):
        solve_sci(ci, h1e, eri, norb, nelec, spin_sq=0.75)
    with pytest.raises(ValueError, match="基态"):
        solve_sci(ci, h1e, eri, norb, nelec, n_roots=2)
    with pytest.raises(ValueError, match="cpu"):
        solve_sci(ci, h1e, eri, norb, nelec, backend="gpu")
    # round_017: _Subspace 三元组已支持 (正向测试见下文 round_017 节);
    # 但派发真值表两 raise 仍在 _Subspace.__init__ 校验 (与 solve_sci 同契约):
    with pytest.raises(ValueError, match="三元组"):
        _Subspace(h1e, eri_nd, norb, nelec)          # 不等 h + 单 eri
    with pytest.raises(ValueError, match="h_alpha, h_beta"):
        _Subspace(h1e[0], eri, norb, nelec)          # 单 h + 三块 eri
    with pytest.raises(ValueError, match="solve_sci_csf|Spin-resolved"):
        solve_sci_csf(ci, h1e, eri, norb, nelec, spin_sq=0.75)
    with pytest.raises(ValueError, match="sigma_linkstr_gpu"):
        sigma_linkstr_gpu(np.zeros((1, 1)), sa[:1], sb[:1], norb, nelec,
                          h1e, eri)
    with pytest.raises(ValueError, match="eigsh_linkstr_gpu"):
        eigsh_linkstr_gpu(sa[:1], sb[:1], norb, nelec, h1e, eri)
    with pytest.raises(ValueError, match="frozen-core"):
        from_pyscf(mf, n_core=1)
    with pytest.raises(ValueError, match="frozen-core"):
        from_pyscf(mf, n_active=2)


# --------------------------------------------------------------------------- #
#  round_017: UHF active 闭环扩展 (_Subspace 三元组 eri 支持)
#
#  P0  : CH/STO-3G UHF (4,3) active 全空间闭环, err ≤1e-8 vs direct_uhf
#  P0' : N2/STO-3G R=2.5 UHF (7,7) @500 shots, active 准 FCI (实测锚定)
#  P2  : CH (4,3) coverage_closure=True → dim=300 全空间, err ≤1e-9
#  契约: backend="gpu" + 三元组 → NotImplementedError (round_019 起转为可用,
#        见本文件末 round_019 节)
# --------------------------------------------------------------------------- #
def test_subspace_spin_resolved_diag_matches_solve_sci_full(ch):
    """round_017 改写锚 (原 _Subspace raise 守卫 → 正向): 三元组 _Subspace
    接受; 全空间 diag 与 solve_sci 自旋分辨路径及 direct_uhf.kernel 一致。"""
    h1e, eri, norb, nelec = ch
    sub = _Subspace(h1e, eri, norb, nelec)
    assert sub._spin_ints is not None            # 三元组 → 自旋分辨分支
    assert sub.h2e is None and sub.myci is None  # 不 absorb_h1e
    sa, sb = _full_strs(norb, nelec)
    E, c2d, sa_o, sb_o = sub.diag(sa, sb)
    res = solve_sci((sa, sb), h1e, eri, norb, nelec)
    assert abs(E - res.energy) <= 1e-10
    assert abs(E - _direct_uhf_ref(h1e, eri, norb, nelec)) <= 1e-10


def test_sqd_active_uhf_ch_full_space_closure():
    """P0: CH/STO-3G UHF (4,3) active 全空间闭环, err ≤1e-8 vs direct_uhf.kernel
    (前置 assert C_α≠C_β 防假 UHF)。"""
    from tc_sqd import solve_sqd_active
    mf = _ch_uhf()
    assert not np.allclose(mf.mo_coeff[0], mf.mo_coeff[1]), (
        "UHF 未破坏对称, 本测试无效 (防假 UHF)")
    h1e, eri, norb, nelec = _uhf_integrals(mf)
    e_ref = _direct_uhf_ref(h1e, eri, norb, nelec)
    bsm = np.random.default_rng(0).random((2000, 2 * norb)) > 0.5
    st = []
    e = solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        n_active_per_round=50, max_rounds=10, rand_seed=0, state_out=st)
    nA_full = int(cistring.num_strings(norb, nelec[0]))
    nB_full = int(cistring.num_strings(norb, nelec[1]))
    assert st[0][1].shape[0] == nA_full and st[0][2].shape[0] == nB_full, (
        f"active 应收敛到全空间 {nA_full}x{nB_full}, "
        f"got {st[0][1].shape[0]}x{st[0][2].shape[0]}")
    err = abs(e - e_ref)
    assert err <= 1e-8, f"CH (4,3) active 全空间 err={err:.2e} > 1e-8"


def test_sqd_active_uhf_n2_quasi_fci(n2):
    """P0': N2/STO-3G R=2.5 UHF (7,7) @500 shots, active 收敛准 FCI
    (阈值 = 实测 + 余量锚定, 预期 ≤1e-6)。"""
    from tc_sqd import solve_sqd_active
    h1e, eri, norb, nelec = n2
    e_ref = _direct_uhf_ref(h1e, eri, norb, nelec)
    bsm = np.random.default_rng(0).random((500, 2 * norb)) > 0.5
    e = solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        n_active_per_round=50, max_rounds=10, rand_seed=0)
    err = abs(e - e_ref)
    assert err <= 1e-6, f"N2 (7,7) @500 active err={err:.2e} > 1e-6"


def test_sqd_active_uhf_coverage_closure_full_space():
    """P2: CH (4,3) coverage_closure=True → BFS 补全全空间 dim=300,
    err ≤1e-9 vs direct_uhf.kernel。

    注: 开壳层下 active 的加串预算门控 (``len(str_a)+len(add) >= max_strings``)
    把 β 新串也计入 α 计数预算 (round_012 遗留口径, 闭壳层 str_a==str_b 不变式下
    精确); 默认 ``max_strings=C(norb,na)=15`` 会在 α 满 15 串后挡住 β 补全。
    显式 ``max_strings=C(norb,max(na,nb))=20`` 松开预算 → BFS 补全两扇区全空间。
    """
    from tc_sqd import solve_sqd_active
    mf = _ch_uhf()
    assert not np.allclose(mf.mo_coeff[0], mf.mo_coeff[1])
    h1e, eri, norb, nelec = _uhf_integrals(mf)
    e_ref = _direct_uhf_ref(h1e, eri, norb, nelec)
    nA_full = int(cistring.num_strings(norb, nelec[0]))   # C(6,4)=15
    nB_full = int(cistring.num_strings(norb, nelec[1]))   # C(6,3)=20
    bsm = np.random.default_rng(0).random((30, 2 * norb)) > 0.5
    st = []
    e = solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        max_strings=max(nA_full, nB_full),
        n_active_per_round=5, max_rounds=3, rand_seed=0,
        coverage_closure=True, state_out=st)
    dim = st[0][1].shape[0] * st[0][2].shape[0]
    assert dim == nA_full * nB_full == 300, (
        f"coverage_closure 应补全到全空间 dim=300, got {dim}")
    err = abs(e - e_ref)
    assert err <= 1e-9, f"coverage_closure 全空间 err={err:.2e} > 1e-9"


def test_sqd_ev_and_best_uhf_smoke(ch):
    """继承链 smoke: solve_sqd_ev / solve_sqd_best UHF 三元组不炸,
    能量与 active/FCI 参考一致。"""
    from tc_sqd import solve_sqd_ev, solve_sqd_best
    h1e, eri, norb, nelec = ch
    e_ref = _direct_uhf_ref(h1e, eri, norb, nelec)
    bsm = np.random.default_rng(0).random((2000, 2 * norb)) > 0.5
    e_ev = solve_sqd_ev(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        n_active_per_round=50, max_rounds=10, rand_seed=0, correction="pt2")
    assert np.isfinite(e_ev)
    assert abs(e_ev - e_ref) <= 1e-8, (
        f"solve_sqd_ev CH (4,3) err={abs(e_ev - e_ref):.2e} > 1e-8")
    d_best = solve_sqd_best(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, n_shots=2000,
        n_active_per_round=50, rand_seed=0, return_details=True)
    assert np.isfinite(d_best["energy"])
    assert abs(d_best["energy"] - e_ref) <= 1e-8, (
        f"solve_sqd_best CH (4,3) err={abs(d_best['energy'] - e_ref):.2e} > 1e-8")


# --------------------------------------------------------------------------- #
#  round_019: 自旋分辨 GPU 化 (hybrid = scipy eigsh + GPU ops matvec)
#
#  P0  : CH (4,3) active backend="gpu" 全空间闭环 err ≤1e-8, 与 CPU |ΔE|≤1e-9
#  P0' : _Subspace 级 N2 (7,7) 子空间 (dim=1600>1000 走 GPU 分支) gpu vs cpu
#  P2  : mock GPU OOM → except 回退 CPU ops (能量 ≤1e-12); cupyx 模式 smoke
#  契约: backend="gpu" + 三元组从 NotImplementedError 转为可用 (无 GPU 静默
#        回退 CPU, round_003 §6.4 语义)
# --------------------------------------------------------------------------- #
def _have_gpu():
    """cupy + 真实 GPU 设备可用?"""
    try:
        import cupy
        if not cupy.cuda.runtime.getDeviceCount():
            return False
        return True
    except Exception:
        return False


def _n2_sub_strs(n2, step=3):
    """N2 (7,7) 全空间 120x120 每 step 取 1 → 40x40 (dim 1600 > 1000,
    触发 GPU 分支 ②-GPU; dense 分支 ≤1000 不触达)。"""
    _, _, norb, nelec = n2
    sa, sb = _full_strs(norb, nelec)
    return np.asarray(sa[::step]), np.asarray(sb[::step])


def test_subspace_spin_resolved_gpu_accepted(ch):
    """round_019 改写锚 (原 gpu_raises): 三元组 + backend='gpu' 不再 raise;
    有 GPU → backend 保持 'gpu', 无 GPU → 静默回退 'cpu' (round_003 §6.4)。"""
    h1e, eri, norb, nelec = ch
    sub = _Subspace(h1e, eri, norb, nelec, backend="gpu")
    assert sub._spin_ints is not None
    if _have_gpu():
        assert sub.backend == "gpu"
    else:
        assert sub.backend == "cpu"


@pytest.mark.skipif(not _have_gpu(), reason="cupy / GPU 不可用")
def test_subspace_spin_resolved_gpu_hybrid_matches_cpu(n2):
    """P0' (_Subspace 级): 同一 N2 (7,7) 子空间 (dim=1600), backend='gpu'
    (hybrid: scipy 引擎 + GPU ops matvec) 与 'cpu' diag 能量一致 ≤1e-10。"""
    h1e, eri, norb, nelec = n2
    sa, sb = _n2_sub_strs(n2)
    E_cpu, c_cpu, _, _ = _Subspace(h1e, eri, norb, nelec,
                                   backend="cpu").diag(sa, sb)
    sub_gpu = _Subspace(h1e, eri, norb, nelec, backend="gpu")
    E_gpu, c_gpu, _, _ = sub_gpu.diag(sa, sb)
    assert sub_gpu.backend == "gpu"
    assert sub_gpu.last_n_mv > 0, "GPU matvec 计数器未动 (未走 GPU 分支?)"
    assert abs(E_gpu - E_cpu) <= 1e-10, (
        f"GPU hybrid E={E_gpu:.12f} != CPU E={E_cpu:.12f} "
        f"(diff={abs(E_gpu - E_cpu):.2e})")


@pytest.mark.skipif(not _have_gpu(), reason="cupy / GPU 不可用")
def test_subspace_spin_resolved_gpu_fallback_oom(n2, monkeypatch):
    """P2: mock GPU matvec OOM → except 回退 CPU ops, 能量与纯 CPU ≤1e-12。

    monkeypatch matrixfree.sigma_vector_ops: xp 非 numpy (cupy) 时抛
    MemoryError; numpy (回退路径) 透传原函数。"""
    from tc_sqd import matrixfree
    orig = matrixfree.sigma_vector_ops

    def _oom_on_gpu(v, ops, xp=np):
        if xp is not np:
            raise MemoryError("mock GPU ops matvec OOM (round_019 回退测试)")
        return orig(v, ops, xp)

    monkeypatch.setattr(matrixfree, "sigma_vector_ops", _oom_on_gpu)
    h1e, eri, norb, nelec = n2
    sa, sb = _n2_sub_strs(n2)
    E_cpu, _, _, _ = _Subspace(h1e, eri, norb, nelec, backend="cpu").diag(sa, sb)
    E_fb, _, _, _ = _Subspace(h1e, eri, norb, nelec, backend="gpu").diag(sa, sb)
    assert abs(E_fb - E_cpu) <= 1e-12, (
        f"OOM 回退 E={E_fb:.12f} != CPU E={E_cpu:.12f}")


@pytest.mark.skipif(not _have_gpu(), reason="cupy / GPU 不可用")
def test_subspace_spin_resolved_cupyx_smoke(n2):
    """P2: gpu_eigsh_mode='cupyx' 诊断基线 smoke (dim=1600 小维度可收敛;
    若 stall 触发 maxiter 护栏则 except 回退, 能量同样正确)。"""
    h1e, eri, norb, nelec = n2
    sa, sb = _n2_sub_strs(n2)
    E_cpu, _, _, _ = _Subspace(h1e, eri, norb, nelec, backend="cpu").diag(sa, sb)
    E_cx, _, _, _ = _Subspace(h1e, eri, norb, nelec, backend="gpu",
                              gpu_eigsh_mode="cupyx").diag(sa, sb)
    assert abs(E_cx - E_cpu) <= 1e-10, (
        f"cupyx 模式 E={E_cx:.12f} != CPU E={E_cpu:.12f}")


@pytest.mark.skipif(not _have_gpu(), reason="cupy / GPU 不可用")
def test_sqd_active_uhf_ch_full_space_closure_gpu():
    """P0 (端到端): CH (4,3) active backend='gpu' 全空间闭环 err ≤1e-8,
    且与同种子 CPU backend |ΔE| ≤1e-9 (dim=300 走 dense 分支, 本测试锚定
    'gpu 三元组可用且结果正确' 的入口契约; GPU matvec 分支由
    test_subspace_spin_resolved_gpu_hybrid_matches_cpu 覆盖)。"""
    from tc_sqd import solve_sqd_active
    mf = _ch_uhf()
    h1e, eri, norb, nelec = _uhf_integrals(mf)
    e_ref = _direct_uhf_ref(h1e, eri, norb, nelec)
    bsm = np.random.default_rng(0).random((2000, 2 * norb)) > 0.5
    common = dict(bitstring_matrix=bsm, n_active_per_round=50,
                  max_rounds=10, rand_seed=0)
    e_gpu = solve_sqd_active(h1e, eri, norb, nelec, backend="gpu", **common)
    e_cpu = solve_sqd_active(h1e, eri, norb, nelec, backend="cpu", **common)
    assert abs(e_gpu - e_ref) <= 1e-8, (
        f"CH (4,3) active gpu err={abs(e_gpu - e_ref):.2e} > 1e-8")
    assert abs(e_gpu - e_cpu) <= 1e-9, (
        f"gpu/cpu |ΔE|={abs(e_gpu - e_cpu):.2e} > 1e-9")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-x"]))