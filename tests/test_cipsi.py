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


def test_hci_h2_reaches_fci():
    """真正的 HCI (heat-bath 选态): H2/STO-3G 从 HF 出发 = FCI。

    HCI 用 |<j|H|i>| >= eps_hb 选态 (区别于 CIPSI 的 PT2 排序), 小 eps_hb 补全
    全空间 = FCI。
    """
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    e_fci = data.solve(method="fci")
    # 从 HF 出发 (seed=None), 小 eps_hb 补全全空间
    e = tc_sqd.solve_hci(data.h1e, data.eri, data.norb, data.nelec,
                         eps_hb=1e-4, ecore=data.ecore, verbose=False)
    assert abs(e - e_fci) < 1e-6


def test_hci_n2_stretch_reaches_fci():
    """真正的 HCI: N2/STO-3G 拉伸从 HF 出发 (无种子) 经 heat-bath 选态补全到近 FCI。

    验证 HCI 无需 S+D 种子 (HF 单 det 出发), heat-bath 选态自动覆盖强关联所需
    高激发 det。
    """
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    e_fci = data.solve(method="fci")
    e = tc_sqd.solve_hci(data.h1e, data.eri, data.norb, data.nelec,
                         eps_hb=1e-4, max_iter=40, ecore=data.ecore,
                         verbose=False)
    assert abs(e - e_fci) < 1e-4


def test_hci_eps_hb_controls_space():
    """HCI 的 eps_hb 阈值控制子空间规模: 阈值越大空间越小 (heat-bath 筛选强度)。

    验证 HCI 的 eps_hb 是真正的选态控制参数 (区别于 CIPSI 的 PT2 排序)。
    """
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    e_fci = data.solve(method="fci")

    # 大 eps_hb -> 少选态 -> 空间小、误差大
    e_loose = tc_sqd.solve_hci(data.h1e, data.eri, data.norb, data.nelec,
                               eps_hb=5e-2, ecore=data.ecore, verbose=False)
    # 小 eps_hb -> 全选 -> 空间大、误差小
    e_tight = tc_sqd.solve_hci(data.h1e, data.eri, data.norb, data.nelec,
                               eps_hb=1e-4, ecore=data.ecore, verbose=False)
    # 松阈值误差更大 (空间更小), 紧阈值接近 FCI
    assert abs(e_loose - e_fci) > abs(e_tight - e_fci)
    assert abs(e_tight - e_fci) < 1e-4
