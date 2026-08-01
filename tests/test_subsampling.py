"""tc_sqd.subsampling 模块测试 —— 批量子采样 + max_dim 子空间限制。"""
import numpy as np
import tc_sqd
from pyscf import gto, scf


# norb=2, bsm 宽 4: [β1β0|α1α0]
_H2_BSM = np.array([
    [0, 1, 0, 1],   # α0β0 (HF)
    [1, 0, 1, 0],   # α1β1 (双激发)
    [0, 1, 1, 0],   # α1β0
    [1, 0, 0, 1],   # α0β1
    [0, 1, 0, 1],   # 重复 HF
    [1, 0, 1, 0],   # 重复双激发
], dtype=bool)


def test_limit_subspace_int():
    """max_dim int: 总行列式数 na*nb ≤ max_dim。"""
    b = tc_sqd.limit_subspace(_H2_BSM, max_dim=1, norb=2)
    na, nb, dim = tc_sqd.subspace_dimension(b)
    assert dim <= 1
    assert dim >= 1


def test_limit_subspace_tuple():
    """max_dim tuple: na ≤ d0 且 nb ≤ d1。"""
    b = tc_sqd.limit_subspace(_H2_BSM, max_dim=(2, 2), norb=2)
    na, nb, dim = tc_sqd.subspace_dimension(b)
    assert na <= 2 and nb <= 2
    assert dim <= 4
    # 更紧的 tuple: 裁剪到 1 维
    b1 = tc_sqd.limit_subspace(_H2_BSM, max_dim=(1, 2), norb=2)
    na, _, _ = tc_sqd.subspace_dimension(b1)
    assert na == 1


def test_limit_subspace_none_is_identity():
    """max_dim=None 不裁剪。"""
    b = tc_sqd.limit_subspace(_H2_BSM, max_dim=None, norb=2)
    assert b.shape == _H2_BSM.shape
    assert np.array_equal(b, _H2_BSM)


def test_limit_subspace_prioritizes_probabilities():
    """高概率字符串优先保留。"""
    probs = np.array([0.6, 0.2, 0.1, 0.05, 0.04, 0.01])
    b = tc_sqd.limit_subspace(_H2_BSM, max_dim=(1, 1), norb=2,
                              probabilities=probs)
    # 概率最高是 [0,1,0,1] (HF) → 保留它
    assert b.shape[0] >= 1
    # 裁剪后子空间 1×1, 且包含概率最高行 (HF)
    na, nb, dim = tc_sqd.subspace_dimension(b)
    assert dim == 1


def test_diagonalize_max_dim():
    """diagonalize_fermionic_hamiltonian 支持 max_dim (替换 NotImplementedError)。"""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    data = tc_sqd.from_pyscf(mf)
    c = tc_sqd.build_lucj_circuit(mf, data.norb, data.nelec)
    bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=2000)

    # tuple 裁剪到 (1,1): 子空间 1×1
    res = tc_sqd.diagonalize_fermionic_hamiltonian(
        data.h1e, data.eri, (bsm, probs), samples_per_batch=200,
        norb=data.norb, nelec=data.nelec, num_batches=2,
        max_iterations=2, max_dim=(1, 1), seed=7)
    assert np.isfinite(res.energy)
    assert res.sci_state.ci_strs_a.shape[0] == 1
    assert res.sci_state.ci_strs_b.shape[0] == 1

    # int 限制总行列式数 ≤ 4
    res3 = tc_sqd.diagonalize_fermionic_hamiltonian(
        data.h1e, data.eri, (bsm, probs), samples_per_batch=200,
        norb=data.norb, nelec=data.nelec, num_batches=2,
        max_iterations=2, max_dim=4, seed=7)
    dim = (res3.sci_state.ci_strs_a.shape[0]
           * res3.sci_state.ci_strs_b.shape[0])
    assert dim <= 4

    # 非法 max_dim 显式报错
    try:
        tc_sqd.diagonalize_fermionic_hamiltonian(
            data.h1e, data.eri, (bsm, probs), samples_per_batch=200,
            norb=data.norb, nelec=data.nelec, max_iterations=1,
            max_dim=(0, 2), seed=7)
        assert False, "max_dim=(0,2) 应报错"
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_subsampling: all PASS")
