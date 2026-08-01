"""tc_sqd 激发态 SQD 测试 —— excited_configurations 采样策略 + n_roots。"""
import numpy as np
import tc_sqd
from pyscf import gto, scf, fci
from pyscf.fci import cistring


def test_excited_configurations_h2_covers_full_space():
    """H2 (2 MO, 1e/1e) 单双激发枚举覆盖全部 4 个 determinant。"""
    exc = tc_sqd.excited_configurations(2, (1, 1), max_excitations=2)
    assert exc.ndim == 2 and exc.shape[1] == 4
    # 4 个 determinant 全覆盖 (H2 2e 全空间)
    assert len(np.unique(tc_sqd.bitarray_to_int(exc))) == 4
    # HF determinant [0,1,0,1] 在内
    assert np.any(np.all(exc == np.array([False, True, False, True]), axis=1))


def test_excited_configurations_max_excitations_controls():
    """max_excitations=0 只含 HF; =1 加单激发。"""
    exc0 = tc_sqd.excited_configurations(3, (2, 2), max_excitations=0)
    assert len(np.unique(tc_sqd.bitarray_to_int(exc0))) == 1       # 仅 HF
    exc1 = tc_sqd.excited_configurations(3, (2, 2), max_excitations=1)
    n1 = len(np.unique(tc_sqd.bitarray_to_int(exc1)))
    exc2 = tc_sqd.excited_configurations(3, (2, 2), max_excitations=2)
    n2 = len(np.unique(tc_sqd.bitarray_to_int(exc2)))
    assert n1 > 1 and n2 >= n1


def test_excited_sqd_h2_matches_fci_roots():
    """HF + 单双激发子空间 n_roots 对角化 = H2 FCI 前 4 根 (精确)。"""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    data = tc_sqd.from_pyscf(mf)
    norb, nelec = data.norb, data.nelec

    # 参考: FCI 前 4 根 (电子能量, 加 ecore 对齐 SQD 总能量)
    e_fci = fci.direct_spin1.kernel(data.h1e, data.eri, norb, nelec,
                                    nroots=4)[0] + data.ecore

    # SQD: excited_configurations 作子空间 (覆盖全空间) + n_roots
    exc_bsm = tc_sqd.excited_configurations(norb, nelec, max_excitations=2)
    ci_a, ci_b = tc_sqd.bitstring_matrix_to_ci_strs(exc_bsm)
    results = tc_sqd.solve_sci((ci_a, ci_b), data.h1e, data.eri,
                               norb, nelec, n_roots=4)
    e_sqd = np.array([r.energy for r in results]) + data.ecore

    assert np.allclose(e_sqd, e_fci, atol=1e-8)


def test_excited_sqd_lif_with_include_configurations():
    """LiH: 采样 + include_configurations(单双激发) + n_roots 提升激发态精度。

    对比仅基态采样 (无强制配置) 的激发态能量, 强制纳入激发配置后更接近 FCI
    激发态 (变分下界更紧)。
    """
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    data = tc_sqd.from_pyscf(mf)
    norb, nelec = data.norb, data.nelec

    # FCI 激发态参考 (前 3 根)
    e_fci = fci.direct_spin1.kernel(data.h1e, data.eri, norb, nelec,
                                    nroots=3)[0] + data.ecore

    # 采样: LUCJ 电路 (Givens 不保粒子数, 需先恢复)
    c = tc_sqd.build_lucj_circuit(mf, norb, nelec, ccsd_scale=0.5)
    bsm, probs = tc_sqd.sample_from_circuit(c, n_samples=3000)
    occ_a = np.zeros(norb); occ_a[:nelec[0]] = 1.0
    occ_b = np.zeros(norb); occ_b[:nelec[1]] = 1.0
    bsm_rec, probs_rec = tc_sqd.recover_configurations(
        bsm, probs, (occ_a, occ_b), nelec[0], nelec[1], rand_seed=7)

    # 子空间 = 恢复采样 ∪ 单双激发配置 (激发态采样策略)
    exc_bsm = tc_sqd.excited_configurations(norb, nelec, max_excitations=1)
    all_bsm = np.vstack([bsm_rec, exc_bsm])
    ci_a, ci_b = tc_sqd.bitstring_matrix_to_ci_strs(all_bsm)
    results = tc_sqd.solve_sci((ci_a, ci_b), data.h1e, data.eri,
                               norb, nelec, n_roots=3)
    e_sqd = np.array([r.energy for r in results]) + data.ecore

    # 不含激发配置的子空间 (纯恢复采样) 作对照
    ci_a0, ci_b0 = tc_sqd.bitstring_matrix_to_ci_strs(bsm_rec)
    res0 = tc_sqd.solve_sci((ci_a0, ci_b0), data.h1e, data.eri,
                            norb, nelec, n_roots=3)
    e0 = np.array([r.energy for r in res0]) + data.ecore

    # 强制纳入激发配置后, 激发态能量应更接近 FCI (≤ 纯采样)
    err_with = np.abs(e_sqd - e_fci).sum()
    err_without = np.abs(e0 - e_fci).sum()
    assert err_with <= err_without + 1e-8, (
        f"include 激发配置未改善: with={err_with:.5f}, without={err_without:.5f}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_excited: all PASS")
