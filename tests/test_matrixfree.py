"""tc_sqd.matrixfree 模块测试 —— 向量化 Slater-Condon σ-vector + GPU matrix-free。

验证:
- sigma_vector 对照 build_ci_matrix (稠密, H2/LiH/N2)。
- sigma_vector_ops (预计算算子) == sigma_vector。
- solve_sci(backend="gpu") == backend="cpu" (需 cupy + GPU, 否则 skip)。
"""
import numpy as np
from pyscf import gto, scf
from pyscf.fci import cistring

import tc_sqd
from tc_sqd.matrixfree import (sigma_vector, prepare_sigma_tables,
                               prepare_sigma_operators, sigma_vector_ops)


def _ints(mol, n_core=0):
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    nocc = mol.nelectron // 2
    mo = np.asarray(mf.mo_coeff)
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl",
                    mol.intor("int2e_sph"), mo, mo, mo, mo, optimize=True)
    if n_core:
        from tc_sqd.molecule import _frozen_core_potential
        h1e = h1e[n_core:, n_core:] + _frozen_core_potential(eri, n_core)
        eri = eri[n_core:, n_core:, n_core:, n_core:]
        norb -= n_core
        nocc -= n_core
    return h1e, eri, norb, (nocc, nocc)


def _ci_strs(norb, nocc):
    return np.array(cistring.make_strings(range(norb), nocc), dtype=np.int64)


def test_sigma_vector_matches_dense():
    """σ-vector 对照 build_ci_matrix (稠密 H), 多个体系。"""
    cases = [
        ("H2", gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0), 0),
        ("LiH", gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0), 1),
        ("N2", gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", verbose=0), 0),
    ]
    rng = np.random.default_rng(0)
    for label, mol, n_core in cases:
        h1e, eri, norb, nelec = _ints(mol, n_core)
        ci_a = _ci_strs(norb, nelec[0])
        ci_b = ci_a.copy()
        H = tc_sqd.build_ci_matrix(ci_a, ci_b, h1e, eri, norb, nelec)
        v = rng.standard_normal((len(ci_a), len(ci_b)))
        sv = sigma_vector(v, ci_a, ci_b, norb, nelec, h1e, eri)
        ref = (H @ v.ravel()).reshape(len(ci_a), len(ci_b))
        assert np.abs(sv - ref).max() < 1e-8, label


def test_sigma_vector_ops_matches():
    """预计算算子版 == 直接版。"""
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", verbose=0)
    h1e, eri, norb, nelec = _ints(mol)
    ci_a = _ci_strs(norb, nelec[0])
    ci_b = ci_a.copy()
    rng = np.random.default_rng(1)
    v = rng.standard_normal((len(ci_a), len(ci_b)))
    ref = sigma_vector(v, ci_a, ci_b, norb, nelec, h1e, eri)
    ops = prepare_sigma_operators(ci_a, ci_b, norb, nelec, h1e, eri)
    got = sigma_vector_ops(v, ops)
    assert np.abs(ref - got).max() < 1e-10


def test_solve_sci_gpu_matches_cpu():
    """solve_sci(backend='gpu') == 'cpu' (需 cupy+GPU, 否则 skip)。"""
    try:
        import cupy  # noqa
        if not cupy.cuda.runtime.getDeviceCount():
            raise ImportError("no GPU")
    except Exception:
        import pytest
        pytest.skip("cupy / GPU 不可用")
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", verbose=0)
    h1e, eri, norb, nelec = _ints(mol)
    ci_a = _ci_strs(norb, nelec[0])
    ci_b = ci_a.copy()
    r_cpu = tc_sqd.solve_sci((ci_a, ci_b), h1e, eri, norb, nelec, backend="cpu")
    r_gpu = tc_sqd.solve_sci((ci_a, ci_b), h1e, eri, norb, nelec, backend="gpu")
    assert abs(r_cpu.energy - r_gpu.energy) < 1e-8


def test_solve_sci_gpu_subspace_matches_cpu():
    """solve_sci(backend='gpu') 子空间 == cpu (关键: selected-CI 子空间正确性)。

    此前 linkstr_gpu 子空间错 (丢双激发), 本测试锁定 selected_ci_gpu 子空间正确。
    """
    try:
        import cupy  # noqa
        if not cupy.cuda.runtime.getDeviceCount():
            raise ImportError("no GPU")
    except Exception:
        import pytest
        pytest.skip("cupy / GPU 不可用")
    from pyscf.fci import cistring
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", verbose=0)
    h1e, eri, norb, nelec = _ints(mol)
    full = np.array(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    ci_a = full[:40]          # 子空间 (非全空间)
    ci_b = full[:40]
    r_cpu = tc_sqd.solve_sci((ci_a, ci_b), h1e, eri, norb, nelec, backend="cpu")
    r_gpu = tc_sqd.solve_sci((ci_a, ci_b), h1e, eri, norb, nelec, backend="gpu")
    assert abs(r_cpu.energy - r_gpu.energy) < 1e-8, f"子空间 GPU 能量 {r_gpu.energy} vs CPU {r_cpu.energy}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_matrixfree: all PASS")
