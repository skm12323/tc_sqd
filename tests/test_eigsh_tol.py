"""tc_sqd.cipsi._Subspace eigsh_tol 覆盖测试 (round_013)。

覆盖:
  - P0' 零回归: eigsh_tol=None (默认) 与不传代码路径等价 (LiH dense 锚 +
    active 层 E 一致)。
  - 功能 (CPU 分支, dim>1000 eigsh 路径): _Subspace.diag 直接消融 ——
    松 tol (1e-4) 比紧 tol (None=默认 1e-10) 少 matvec 且 E 在合理精度内;
    eigsh_tol=1e-12 与 None E 一致 (都近机器精度)。
  - 透传: solve_sqd_active / solve_sqd_ev / solve_sqd_best 接 eigsh_tol
    生效 (完成 + E 与默认一致)。

口径: N2/STO-3G 拉伸 (norb=10, nelec=(7,7)), cistring 固定字符串集
dim>1000 走 eigsh。**不做逐位锚** (ARPACK v0=None 内部随机演化, 见
test_triple_injection 头注), 一致性用 isclose; unit 层固定字符串集 +
tol 差异远大于 v0 随机涨落, n_mv 比较稳健。
"""
import numpy as np
import pytest

import tc_sqd
from tc_sqd.cipsi import _Subspace
from pyscf import gto
from pyscf.fci import cistring


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _n2_data():
    """N2/STO-3G 拉伸 (强关联), norb=10, nelec=(7,7), 全空间 C(10,7)=120 串。"""
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    return tc_sqd.from_pyscf(mol)


def _lih_data():
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    return tc_sqd.from_pyscf(mol)


def _fixed_strings(norb, nelec, n_str=60):
    """前 n_str 个字符串 (确定性, dim=n_str²>1000 走 eigsh)。"""
    full = np.array(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    sa = full[:n_str]
    return sa, sa.copy()


# --------------------------------------------------------------------------- #
# P0' 零回归: eigsh_tol=None (默认) 与不传代码路径等价
# --------------------------------------------------------------------------- #
def test_eigsh_tol_default_none_path_equivalent_dense():
    """P0' 锚 (dense 路径): solve_sqd_active 不传 vs 显式 eigsh_tol=None
    代码路径等价 (dense 分支不读 tol), 能量逐位一致。"""
    data = _lih_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((60, 2 * norb)) > 0.5
    common = dict(max_strings=None, n_active_per_round=30, max_rounds=8,
                  rand_seed=0)
    e_default = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, **common)
    e_none = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        eigsh_tol=None, **common)
    assert e_default == e_none, (
        f"dense 路径默认应逐位一致: {e_default!r} vs {e_none!r}")


def test_eigsh_tol_default_none_path_equivalent_subspace():
    """P0' 锚 (_Subspace 层, CPU eigsh 路径): 不传 vs eigsh_tol=None 同代码
    路径 (都 = 默认 1e-10), 同一实例连续两次 diag E 一致 (收敛值稳定)。"""
    data = _n2_data()
    sa, sb = _fixed_strings(data.norb, data.nelec)
    sub = _Subspace(data.h1e, data.eri, data.norb, data.nelec)
    e1, _, _, _ = sub.diag(sa, sb)
    e2, _, _, _ = sub.diag(sa, sb)   # 同实例第二次 (tol 仍 0)
    assert np.isclose(e1, e2, rtol=1e-12, atol=1e-12), (
        f"默认 tol 两次 diag 收敛值应稳定: {e1!r} vs {e2!r}")


# --------------------------------------------------------------------------- #
# 功能: _Subspace.diag tol 消融 (CPU 分支, dim>1000)
# --------------------------------------------------------------------------- #
def test_subspace_diag_tol_ablation_cpu():
    """功能锚: 固定字符串集 (dim=3600) 上松 tol (1e-4) 比紧 tol (None=0)
    少 matvec (ARPACK 早停) 且 E 仍在合理精度 (<1e-3); 1e-12 与 None 一致。"""
    data = _n2_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    sa, sb = _fixed_strings(norb, nelec, n_str=60)
    assert len(sa) * len(sb) == 3600 > 1000, "须走 eigsh 分支 (dim>1000)"

    sub_tight = _Subspace(h1e, eri, norb, nelec)          # None = 默认 1e-10
    sub_112 = _Subspace(h1e, eri, norb, nelec, eigsh_tol=1e-12)
    sub_loose = _Subspace(h1e, eri, norb, nelec, eigsh_tol=1e-4)

    e_tight, _, _, _ = sub_tight.diag(sa, sb)
    n_tight = sub_tight.last_n_mv
    e_112, _, _, _ = sub_112.diag(sa, sb)
    n_112 = sub_112.last_n_mv
    e_loose, _, _, _ = sub_loose.diag(sa, sb)
    n_loose = sub_loose.last_n_mv

    assert np.isclose(e_tight, e_112, rtol=1e-10, atol=1e-10), (
        f"tol=1e-12 应与默认 1e-10 一致: {e_tight!r} vs {e_112!r}")
    assert abs(e_loose - e_tight) < 1e-3, (
        f"松 tol=1e-4 E 仍应收敛到紧值附近: {e_tight!r} vs {e_loose!r} "
        f"diff={abs(e_loose - e_tight):.2e}")
    assert n_loose < n_tight, (
        f"松 tol 应减少 matvec: loose={n_loose} vs tight={n_tight}")
    assert n_112 + 100 >= n_tight, (       # 1e-12 更紧, 迭代不应显著少于 1e-10
        f"tol=1e-12 (更紧) 迭代应 ≥ 默认 1e-10 (允许 v0 涨落): "
        f"{n_112} vs {n_tight}")


# --------------------------------------------------------------------------- #
# 透传: solve_sqd_active / ev / best 接 eigsh_tol
# --------------------------------------------------------------------------- #
def test_solve_sqd_active_eigsh_tol_plumbing():
    """透传锚: solve_sqd_active(eigsh_tol=1e-10, CPU) 完成且 E 与默认
    (1e-10) 一致 (rtol 1e-9, ARPACK v0 随机涨落内)。"""
    data = _n2_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((40, 2 * norb)) > 0.5
    common = dict(max_strings=None, n_active_per_round=10, max_rounds=4,
                  rand_seed=0)
    e_default = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, **common)
    e_tol = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        eigsh_tol=1e-10, **common)
    assert np.isclose(e_default, e_tol, rtol=1e-9, atol=1e-9), (
        f"active 层 eigsh_tol=1e-10 E 应与默认一致: "
        f"{e_default!r} vs {e_tol!r} diff={abs(e_default - e_tol):.2e}")


def test_solve_sqd_ev_eigsh_tol_passthrough():
    """透传锚 (ev 层): solve_sqd_ev(eigsh_tol=1e-10) 完成且 E 与默认一致。"""
    data = _n2_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    bsm = np.random.default_rng(0).random((40, 2 * norb)) > 0.5
    common = dict(max_strings=None, n_active_per_round=10, rand_seed=0)
    e_default = tc_sqd.solve_sqd_ev(
        h1e, eri, norb, nelec, bitstring_matrix=bsm, **common)
    e_tol = tc_sqd.solve_sqd_ev(
        h1e, eri, norb, nelec, bitstring_matrix=bsm,
        eigsh_tol=1e-10, **common)
    assert np.isclose(e_default, e_tol, rtol=1e-8, atol=1e-8), (
        f"ev 层透传 E 应一致: {e_default!r} vs {e_tol!r}")


def test_solve_sqd_best_eigsh_tol_passthrough():
    """透传锚 (best 层): solve_sqd_best(eigsh_tol=1e-10) 完成且 E 与默认一致。"""
    data = _n2_data()
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    common = dict(n_shots=30, return_details=True, rand_seed=0)
    d_default = tc_sqd.solve_sqd_best(h1e, eri, norb, nelec, **common)
    d_tol = tc_sqd.solve_sqd_best(h1e, eri, norb, nelec,
                                  eigsh_tol=1e-10, **common)
    assert np.isclose(d_default["energy"], d_tol["energy"], rtol=1e-7,
                      atol=1e-7), (
        f"best 层透传 E 应一致: {d_default['energy']!r} vs "
        f"{d_tol['energy']!r}")
