"""tc_sqd.noise 模块测试 —— 密度矩阵 Kraus 噪声通道。"""
import numpy as np
import tc_sqd


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_noise: all PASS")
