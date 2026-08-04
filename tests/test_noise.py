"""tc_sqd.noise 模块测试 —— 密度矩阵 Kraus 噪声通道。"""
import numpy as np
import tc_sqd
from pyscf import gto


def test_statevector_to_density():
    """纯态 -> 密度矩阵, 对角 = |ψ|²。"""
    psi = np.array([1, 0, 0, 0], dtype=complex)  # |00>
    rho = tc_sqd.statevector_to_density(psi)
    assert rho.shape == (4, 4)
    assert np.allclose(np.diag(rho).real, [1, 0, 0, 0])


def test_dephasing_diag_unchanged():
    """退相干 (T2) 不改 diag —— SQD 免疫的核心。"""
    psi = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
    rho = tc_sqd.statevector_to_density(psi)
    diag0 = np.diag(rho).real.copy()
    rho_d = tc_sqd.apply_dephasing(rho, p=0.5, nq=2)
    assert np.allclose(np.diag(rho_d).real, diag0, atol=1e-12)


def test_amp_damping_changes_diag():
    """振幅阻尼 (T1) |1> -> |0>, diag 偏移。"""
    psi = np.array([0, 1], dtype=complex)  # |1>
    rho = tc_sqd.statevector_to_density(psi)
    diag0 = np.diag(rho).real.copy()
    rho_a = tc_sqd.apply_amp_damping(rho, gamma=0.5, nq=1)
    diag_a = np.diag(rho_a).real
    assert diag_a[0] > diag0[0] + 0.1   # |0> 占据上升
    assert diag_a[1] < diag0[1] - 0.1   # |1> 占据下降


def test_amp_damping_gamma0_identity():
    """gamma=0 不改变密度矩阵。"""
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = tc_sqd.statevector_to_density(psi)
    rho_a = tc_sqd.apply_amp_damping(rho, gamma=0.0, nq=1)
    assert np.allclose(rho_a, rho)


def test_depolarizing_trace_preserving():
    """去极化保持迹 = 1。"""
    psi = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
    rho = tc_sqd.statevector_to_density(psi)
    rho_d = tc_sqd.apply_depolarizing(rho, p=0.3, nq=2)
    assert abs(np.trace(rho_d).real - 1.0) < 1e-10


def test_density_to_bitstring_matrix():
    """diag -> bsm 采样, 形状 + 只采正概率态。"""
    # norb=1, nq=2: bit0=α0, bit1=β0; diag=[P(00),P(01),P(10),P(11)]
    diag = np.array([0.5, 0.0, 0.0, 0.5])  # |α0β0> 和 |α1β1>
    bsm = tc_sqd.density_to_bitstring_matrix(diag, norb=1, n_samples=200, seed=42)
    assert bsm.shape == (200, 2)
    for row in bsm:
        # 只应是 [F,F] (α0β0) 或 [T,T] (α1β1)
        assert all(row == [False, False]) or all(row == [True, True])


def test_density_to_bitstring_matrix_layout_norb2():
    """norb=2 时 density 计算基 -> bsm 必须符合全库降序 [β1β0|α1α0]。

    锁住 density_to_bitstring_matrix 的列顺序 (曾因升序/降序反转导致 noise->SQD
    链路轨道错乱)。density 约定: bit0=α0,bit1=α1,bit2=β0,bit3=β1。
    """
    # HF 行列式 α0β0 -> density index = bit0(α0) + bit2(β0) = 1 + 4 = 5
    diag_hf = np.zeros(16)
    diag_hf[5] = 1.0
    bsm = tc_sqd.density_to_bitstring_matrix(diag_hf, norb=2, n_samples=64, seed=0)
    assert bsm.shape == (64, 4)
    # 全库降序 [β1,β0,α1,α0]: HF(α0β0 occ) = [0,1,0,1]
    expected_hf = np.array([False, True, False, True])
    assert np.all(bsm == expected_hf), f"HF row wrong: {bsm[0]}"

    # 双激发 α1β1 -> density index = bit1(α1) + bit3(β1) = 2 + 8 = 10
    diag_de = np.zeros(16)
    diag_de[10] = 1.0
    bsm2 = tc_sqd.density_to_bitstring_matrix(diag_de, norb=2, n_samples=64, seed=0)
    expected_de = np.array([True, False, True, False])  # [β1,β0,α1,α0]=[1,0,1,0]
    assert np.all(bsm2 == expected_de), f"double-exc row wrong: {bsm2[0]}"


def _fermion_civec_to_density_psi(civec, ci_strs_a, ci_strs_b, norb):
    """PySCF selected-CI 向量 (determinant 基) -> noise 模块计算基向量 |ψ>。

    density 约定 (与 density_to_bitstring_matrix 解码一致):
        bit orb        (0..norb-1) = α 轨道 orb
        bit (norb+orb)            = β 轨道 orb
    行列式字符串 bit p = 轨道 p 占据。
    """
    nq = 2 * norb
    psi = np.zeros(2 ** nq, dtype=complex)
    civec = np.asarray(civec).reshape(len(ci_strs_a), len(ci_strs_b))
    for ia, sa in enumerate(ci_strs_a):
        for ib, sb in enumerate(ci_strs_b):
            amp = civec[ia, ib]
            if abs(amp) < 1e-15:
                continue
            idx = 0
            for p in range(norb):
                if (int(sa) >> p) & 1:            # α 轨道 p 占据
                    idx |= (1 << p)
            for p in range(norb):
                if (int(sb) >> p) & 1:            # β 轨道 p 占据
                    idx |= (1 << (norb + p))
            psi[idx] += amp
    return psi


def test_density_sqd_end_to_end_h2():
    """H2 FCI 态 -> density -> 无噪声 -> SQD 应复现 FCI。

    端到端验证 statevector_to_density -> apply_amp_damping(gamma=0) ->
    density_to_bitstring_matrix -> compute_ground_state_energy(sqd) 整条链路
    的轨道布局自洽 (任一环节列序错位都会让能量严重偏离 FCI)。
    """
    from pyscf import gto, scf, fci
    from pyscf.fci import cistring

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", mol.intor("int2e_sph"),
                    mo, mo, mo, mo)
    ecore = mf.energy_nuc()
    norb = int(mol.nao_nr())
    nelec = (1, 1)

    e_fci, civec = fci.direct_spin1.kernel(h1e, eri, norb, nelec)
    ci_a = cistring.make_strings(range(norb), nelec[0])
    ci_b = cistring.make_strings(range(norb), nelec[1])

    psi = _fermion_civec_to_density_psi(civec, ci_a, ci_b, norb)
    rho = tc_sqd.statevector_to_density(psi)
    rho = tc_sqd.apply_amp_damping(rho, gamma=0.0, nq=2 * norb)   # 无噪声: 纯布局验证
    diag = np.diag(rho).real

    bsm = tc_sqd.density_to_bitstring_matrix(diag, norb=norb,
                                             n_samples=4000, seed=42)
    e_sqd = tc_sqd.compute_ground_state_energy(
        h1e, eri, norb, nelec, ecore=ecore, method="sqd",
        bitstring_matrix=bsm, max_iterations=3)

    assert abs(e_sqd - (e_fci + ecore)) < 1e-3, (
        f"density->SQD 偏离 FCI: SQD={e_sqd:.6f}, FCI={e_fci + ecore:.6f}")


def test_zero_noise_extrapolate_t1_improves():
    """A3: N2 拉伸 T1 ZNE——外推 γ→0 误差 < 最噪点 (γ=0.2)。

    验证: 多项式外推稳定恢复零噪声能量, 优于单个噪声点 (位串级 T1, 大体系可跑)。
    """
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    e_fci = data.solve(method="fci")

    bsm = (np.random.default_rng(0).random((2000, 2 * data.norb)) > 0.5)
    probs = np.full(2000, 1.0 / 2000)
    gammas = (0.02, 0.05, 0.1, 0.2)
    e_z, ens = tc_sqd.zero_noise_extrapolate_t1(
        data.h1e, data.eri, data.norb, data.nelec,
        bitstring_matrix=bsm, probabilities=probs, gammas=gammas,
        ecore=data.ecore, max_iterations=3, seed=0,
    )
    # 达化学精度 (主断言: ZNE 在 T1 噪声下恢复零噪声精度)
    assert abs(e_z - e_fci) < 1.6e-3, f"外推未达化学精度: {abs(e_z - e_fci):.2e}"
    # 不显著劣于最噪点 (外推收益依赖 E(γ) 曲线形状, 宽容差避免方法固有脆弱)
    assert abs(e_z - e_fci) < abs(ens[-1] - e_fci) * 1.5 + 1e-9, (
        f"外推显著劣化: e_z_err={abs(e_z - e_fci):.2e} "
        f"noisiest_err={abs(ens[-1] - e_fci):.2e}")


def test_solve_sqd_robust_combines_zne_budget():
    """solve_sqd_robust: A3 ZNE 外推 + B1 预算闭环 统一 API。

    验证: ① ZNE 外推误差 ≤ 最噪点 (噪声鲁棒); ② 总 shots < 无预算全量
    (预算高效, 每个 γ 能量收敛即停采)。N2 拉伸实测外推 err ~1e-13,
    total_shots 3600 < 4×2000=8000 (省 55%)。
    """
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    e_fci = data.solve(method="fci")

    n_pool = 2000
    bsm = (np.random.default_rng(0).random((n_pool, 2 * data.norb)) > 0.5)
    probs = np.full(n_pool, 1.0 / n_pool)

    r = tc_sqd.solve_sqd_robust(
        data.h1e, data.eri, data.norb, data.nelec,
        bitstring_matrix=bsm, probabilities=probs, ecore=data.ecore,
        gammas=(0.05, 0.1, 0.2, 0.3),
        shots_budget=n_pool, shots_step=300, energy_tol=1e-3,
        n_active_per_round=50, max_rounds=10, seed=0,
    )
    # ① ZNE 外推达化学精度, 且不显著劣于最噪点 (外推收益依赖 E(γ) 曲线形状)
    assert abs(r["energy"] - e_fci) < 1.6e-3, f"外推未达化学精度: {abs(r['energy'] - e_fci):.2e}"
    assert abs(r["energy"] - e_fci) <= abs(r["energies_by_gamma"][-1] - e_fci) * 1.5 + 1e-9, (
        f"ZNE 外推显著劣化: {abs(r['energy'] - e_fci):.2e} vs "
        f"{abs(r['energies_by_gamma'][-1] - e_fci):.2e}")
    # ② B1 预算省 shots (每个 γ 能量收敛停采, energy_tol=1e-3 确保停采)
    assert r["total_shots"] < len(r["gammas"]) * n_pool, (
        f"预算未省: total_shots={r['total_shots']}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_noise: all PASS")
