"""tc_sqd.cipsi 模块测试 —— PT2-CIPSI 生成集扩展从种子补全到近 FCI。"""
import numpy as np
import tc_sqd
from pyscf import gto


def test_cipsi_h2_reaches_fci():
    """H2/STO-3G: S+D 种子 CIPSI -> = FCI (小体系快速验证核心正确性)。"""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    e_fci = data.solve(method="fci")
    seed = tc_sqd.excited_configurations(data.norb, data.nelec, max_excitations=2)
    e = tc_sqd.solve_cipsi(data.h1e, data.eri, data.norb, data.nelec,
                           seed_bitstring_matrix=seed, ecore=data.ecore,
                           verbose=False)
    assert abs(e - e_fci) < 1e-6


def test_cipsi_n2_stretch_breaks_sd_platform():
    """N2/STO-3G 拉伸 (强关联): S+D 种子单双激发平台 2.25e-2, CIPSI 补全到近 FCI。

    验证 CIPSI 能突破单双激发覆盖不足——不依赖 UCJ 采样随机性。
    """
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    e_fci = data.solve(method="fci")
    seed = tc_sqd.excited_configurations(data.norb, data.nelec, max_excitations=2)
    e = tc_sqd.solve_cipsi(data.h1e, data.eri, data.norb, data.nelec,
                           seed_bitstring_matrix=seed, ecore=data.ecore,
                           verbose=False)
    assert abs(e - e_fci) < 1e-4
