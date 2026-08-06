"""tc_sqd.predict 模块测试 —— 噪声容限预测器。"""
import tc_sqd
from pyscf import gto


def test_calibrate_dual_mode():
    """calibrate 双模式: circuit 非零 KS/KT1; fci_density 高 shots KS≈0 (benchmark)。"""
    mol = gto.M(atom="H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6",
                basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    circ = tc_sqd.build_lucj_circuit(data.mf, data.norb, data.nelec,
                                     ccsd_scale=1.0)

    # circuit 模式: 实际电路采样 + 位串级 T1 -> 非零 KS/KT1
    r_c = tc_sqd.calibrate(data.h1e, data.eri, data.norb, data.nelec,
                           ecore=data.ecore, circuit=circ,
                           shots_grid=[2000, 4000], gamma_grid=[0.1, 0.3])
    assert r_c["mode"] == "circuit"
    assert abs(r_c["KS"]) > 1e-3, f"circuit 模式 KS 应非零, got {r_c['KS']:.2e}"
    assert abs(r_c["KT1"]) > 1e-4, f"circuit 模式 KT1 应非零, got {r_c['KT1']:.2e}"

    # fci_density 模式: 高 shots 覆盖饱和 -> KS≈0 (benchmark)
    r_f = tc_sqd.calibrate(data.h1e, data.eri, data.norb, data.nelec,
                           ecore=data.ecore,
                           shots_grid=[4000, 8000], gamma_grid=[0.1, 0.3])
    assert r_f["mode"] == "fci_density"
    assert abs(r_f["KS"]) < 1e-3, f"fci_density 高 shots KS 应≈0, got {r_f['KS']:.2e}"


def test_apply_t1_bitstrings():
    """位串级 T1: gamma=0 不变, gamma 翻 1->0, 校验 γ 范围。"""
    import numpy as np
    bsm = np.array([[True, True], [True, False], [False, False]], dtype=bool)
    # gamma=0: 不变
    out0 = tc_sqd.apply_t1_bitstrings(bsm, 0.0, seed=0)
    assert np.array_equal(out0, bsm)
    # gamma=1: 全 1 -> 0
    out1 = tc_sqd.apply_t1_bitstrings(bsm, 1.0, seed=0)
    assert not np.any(out1)
    # 越界
    try:
        tc_sqd.apply_t1_bitstrings(bsm, 1.5)
        assert False, "gamma>1 应报错"
    except ValueError:
        pass


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


def test_recommend_sqd_params():
    """recommend_sqd_params: 组装 plan_sampling + 子空间启发式, 结构完整。

    给定 N2/STO-3G (norb=10, nelec=(5,5)) + 真机参数, 应给出 shots/depth/
    max_strings/n_active_per_round, 且子空间上限 ≤ 全空间 C(10,5)=252。
    """
    rec = tc_sqd.recommend_sqd_params(10, (5, 5), T1_us=30.0, t_gate_ns=100.0)
    assert isinstance(rec, tc_sqd.SqdParams)
    for k in ("shots", "depth", "max_strings", "n_active_per_round",
              "dom_thresh", "pt2_floor", "predicted_error", "dominant",
              "feasible", "reason"):
        assert hasattr(rec, k), f"缺字段 {k}"
    assert rec.feasible, "默认硬件参数下应存在可行方案"
    assert rec.shots > 0 and rec.depth > 0
    assert rec.max_strings <= 252, f"max_strings 超全空间: {rec.max_strings}"
    assert rec.max_strings >= 50
    assert rec.n_active_per_round >= 10
    assert len(rec.reason) > 0
    # 手动覆盖子空间上限
    rec2 = tc_sqd.recommend_sqd_params(
        10, (5, 5), T1_us=30.0, t_gate_ns=100.0, max_strings_override=80)
    assert rec2.max_strings == 80
    # 极紧目标 -> feasible=False (目标过紧无可行组合)
    rec3 = tc_sqd.recommend_sqd_params(
        10, (5, 5), T1_us=30.0, t_gate_ns=100.0, target=1e-9,
        shots_max=1000)
    assert not rec3.feasible
    assert "无可行" in rec3.reason


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
    print("test_predict: all PASS")
