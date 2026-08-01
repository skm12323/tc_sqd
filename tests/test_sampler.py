"""tc_sqd.sampler 模块测试 —— 统一采样后端。"""
import numpy as np
import tc_sqd
from pyscf import gto, scf


def _h2_circuit():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    return tc_sqd.build_lucj_circuit(mf, 2, (1, 1), ccsd_scale=1.0), mf


def test_sample_tc_backend():
    """tc 后端: 返回 (bsm, probs) 形状/归一正确。"""
    c, _ = _h2_circuit()
    bsm, probs = tc_sqd.sample(c, n_samples=2000, backend="tc")
    assert bsm.ndim == 2 and bsm.shape[1] == 4        # 2*norb = 4
    assert bsm.shape[0] == probs.shape[0]
    assert abs(probs.sum() - 1.0) < 1e-12
    assert np.all((bsm == 0) | (bsm == 1))


def test_sample_tc_drives_sqd():
    """统一采样输出直接喂 SQD 流水线, 复现 H2 FCI。"""
    c, _ = _h2_circuit()
    bsm, probs = tc_sqd.sample(c, n_samples=3000, backend="tc")
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    e = data.solve(method="sqd", bitstring_matrix=bsm,
                   probabilities=probs, max_iterations=3)
    assert abs(e - (-1.13728383)) < 2e-3


def test_sample_invalid_backend():
    """未知 backend 显式报错。"""
    c, _ = _h2_circuit()
    try:
        tc_sqd.sample(c, 100, backend="bogus")
        assert False, "未知 backend 应报错"
    except ValueError:
        pass


def test_sample_qcloud_requires_device():
    """qcloud 后端缺 device_name 显式报错 (本机不连真机)。"""
    c, _ = _h2_circuit()
    try:
        tc_sqd.sample(c, 100, backend="qcloud")
        assert False, "qcloud 缺 device_name 应报错"
    except ValueError:
        pass


def test_sample_n_samples_validation():
    """n_samples 非正显式报错。"""
    c, _ = _h2_circuit()
    try:
        tc_sqd.sample(c, 0)
        assert False, "n_samples=0 应报错"
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_sampler: all PASS")
