"""tc_sqd.predict 模块测试 —— 噪声容限预测器。"""
import tc_sqd


def test_gamma_T1():
    """gamma_T1 边界 + 单调。"""
    assert tc_sqd.gamma_T1(0, 30, 15) == 0.0                 # depth=0 无衰减
    g = tc_sqd.gamma_T1(100, 30, 15)                          # depth=100, t_gate=30ns, T1=15us
    assert 0 < g < 1
    assert tc_sqd.gamma_T1(100000, 30, 15) > 0.99             # 深电路趋近 1
    # T1 越长 gamma 越小
    assert tc_sqd.gamma_T1(100, 30, 30) < tc_sqd.gamma_T1(100, 30, 10)


def test_predict_sqd_error_structure():
    """预测返回合理结构。"""
    r = tc_sqd.predict_sqd_error(T1_us=15, depth=100, t_gate_ns=30,
                                  shots=8000, n_excited=2)
    for k in ("gamma_T1", "eps_sample", "eps_T1_ground", "eps_T2",
              "eps_readout", "ground", "excited", "dominant"):
        assert k in r
    assert len(r["excited"]) == 2
    assert r["eps_T2"] == 0.0          # 退相干免疫
    assert r["eps_readout"] == 0.0     # recover 纠正


def test_excited_more_sensitive():
    """激发态 T1 误差 ~3× 基态 (方向2)。"""
    r = tc_sqd.predict_sqd_error(T1_us=15, depth=100, t_gate_ns=30,
                                  shots=8000, n_excited=1)
    assert r["excited"][0] > r["ground"]
    ratio = r["excited"][0] / r["ground"]
    assert 2.5 < ratio < 3.5           # ~3×


def test_more_shots_less_sample_error():
    """shots 增加, 采样误差降。"""
    r1 = tc_sqd.predict_sqd_error(15, 100, 30, shots=1000)
    r2 = tc_sqd.predict_sqd_error(15, 100, 30, shots=40000)
    assert r2["eps_sample"] < r1["eps_sample"]


def test_max_depth_for_accuracy():
    """反向预测 depth 上限: 激发态 < 基态。"""
    d_g = tc_sqd.max_depth_for_accuracy(15, 30, 8000, excited=False)
    d_e = tc_sqd.max_depth_for_accuracy(15, 30, 8000, excited=True)
    assert d_g > 0
    assert d_e > 0
    assert d_e < d_g                  # 激发态更严


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_predict: all PASS")
