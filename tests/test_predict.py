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


def test_depth_budget_structured():
    """结构化预算: ok / sampling_limited / t1_unlimited 三种状态 + 与 int 封装一致。"""
    # 常规: T1 主导, ok
    b_ok = tc_sqd.depth_budget(15, 30, 8000)
    assert b_ok.status == "ok"
    assert b_ok.max_depth is not None and b_ok.max_depth > 0
    assert b_ok.reason
    # 激发态更严
    assert tc_sqd.depth_budget(15, 30, 8000, excited=True).max_depth < b_ok.max_depth

    # 采样误差已 ≥ target: sampling_limited
    b_samp = tc_sqd.depth_budget(15, 30, shots=4, target=1e-3)
    assert b_samp.status == "sampling_limited"
    assert b_samp.max_depth is None

    # 目标宽松 (10 mHa): T1 任意不超, t1_unlimited
    b_unl = tc_sqd.depth_budget(15, 30, 8000, target=0.01)
    assert b_unl.status == "t1_unlimited"
    assert b_unl.max_depth is None

    # 与向后兼容的 int 封装一致
    assert tc_sqd.max_depth_for_accuracy(15, 30, 8000) == b_ok.max_depth
    assert tc_sqd.max_depth_for_accuracy(15, 30, 4, target=1e-3) == -1
    assert tc_sqd.max_depth_for_accuracy(15, 30, 8000, target=0.01) == -2


def test_plan_sampling():
    """plan_sampling: 可行方案按 cost 升序, 全部 < target, 激发态更难。"""
    r = tc_sqd.plan_sampling(15, 30, target=1.6e-3)
    assert {"all", "feasible", "best", "target"} <= set(r)
    assert r["feasible"], "默认网格应有化学精度内方案"
    assert r["best"] is not None

    # cost 升序
    costs = [p.cost for p in r["feasible"]]
    assert costs == sorted(costs)

    # 每个可行方案误差 < target; 不行的都 >= target
    for p in r["all"]:
        if p.chemical:
            assert p.error < r["target"]
        else:
            assert p.error >= r["target"]

    # best 有完整字段
    b = r["best"]
    for k in ("shots", "depth", "error", "eps_sample", "eps_T1",
              "dominant", "chemical", "cost"):
        assert hasattr(b, k)

    # 激发态 (3×) 更难: 可行方案不会比基态更多
    re = tc_sqd.plan_sampling(15, 30, target=1.6e-3, excited=True)
    assert len(re["feasible"]) <= len(r["feasible"])

    # shots 权重主导时, 最优方案优先小 shots (深度换精度)
    r_w = tc_sqd.plan_sampling(15, 30, target=1.6e-3,
                               shots_cost=1.0, depth_cost=1e-4)
    assert r_w["best"].shots > 0 and r_w["best"].depth > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_predict: all PASS")
