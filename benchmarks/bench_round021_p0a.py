"""round_021 P0a 前提检查（theory §3 P0a ①②；③ 由 ms=1500 试点 run 兼任）。

① 全空间 11,778,624 维单 matvec wall（CPU linkstr C 核，与 _Subspace 分支②-CPU
  同路径：selected_ci._all_linkstr_index + SCI.contract_2e）+ linkstr 构建耗时
  + 峰值 RSS。门槛：min(matvec) ≤ 150s（参考 run 外推 ≤4h 的前提）。
② ms=200 @500 shots 双臂（closure on/max_rounds=10 vs off/max_rounds=50）短跑：
  无 OOM/异常 + 采样地板 F1（首轮轨迹 dim 开方）实测 + |c_HF|² 强关联检验
  （theory R4 风险：|c_HF|² > 0.95 → 换 R=3.5 Å）。

P0a 实测定版记录（2026-08-31）：
- ① 全空间单 matvec 实测 1.8s（R1 外推 77-153s 高估 ~50×：12,12 参考的 "3 分钟"
  是 eigsh 迭代累计而非 per-mv）→ 参考 run 预算从 1.5-3.5h 修正为 ~10 min。
- ② 首版配方带 tail_suppression=True（theory §2.2 沿用 12,12 联合配方）实测
  **尾部洪流**：discover_tail_pool 每轮抽新随机位串（round_seed 逐轮推进,
  cipsi.py:1364-1371），采样串不受 max_strings 门控（cipsi.py:1386），
  (14,14) 串池 3432 在 10-50 轮内不饱和 → 每轮 +~270 串 → 串数 ~465→3165,
  GPU t1 196×3165²×8B=15.7GB 逼近 16GB 显存上限（实测 nvidia-smi 15.6GB
  吻合），晚轮 diag 退化为小时级。**本轮配方改 tail_suppression=False**：
  tail 是 shots 经济装置（替代更多量子采样），在 (14,14) 上会把经典子空间
  冲过被研究的预算语义，污染被隔离变量（预算分配机制）。12,12 上 tail 快速
  饱和于 924 串池故无害——这是大体系特有的配方边界。
  （theory R5「采样地板吞噬预算」的动态版实现：地板随轮次上涨而非固定。）

输出 benchmarks/_round021_p0a.json（分阶段增量落盘）。
用法: python benchmarks/bench_round021_p0a.py
"""
import json
import os
import resource
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "benchmarks", "_round021_p0a.json")
INTS = os.path.join(BASE, "_n2_1414_ints.npz")
NORB, NELEC = 14, (7, 7)


def _save(res):
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def stage1(h1e, eri):
    """全空间单 matvec 计时（CPU linkstr 流式 C 核）。"""
    from pyscf import ao2mo
    from pyscf.fci import cistring, direct_spin1, selected_ci

    sa = cistring.make_strings(range(NORB), NELEC[0])
    sb = sa.copy()
    dim = int(len(sa)) * int(len(sb))
    t0 = time.perf_counter()
    link = selected_ci._all_linkstr_index((sa, sb), NORB, NELEC)
    t_link = time.perf_counter() - t0
    h2e = ao2mo.restore(
        1, direct_spin1.absorb_h1e(h1e, eri, NORB, NELEC, 0.5), NORB)
    myci = selected_ci.SCI()
    v = np.random.default_rng(0).standard_normal(dim)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        myci.contract_2e(h2e, selected_ci._as_SCIvector(v, (sa, sb)),
                         NORB, NELEC, link)
        ts.append(time.perf_counter() - t0)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return dict(dim=dim, n_strings_sector=int(len(sa)),
                t_linkstr_s=round(t_link, 2),
                matvec_s=[round(t, 2) for t in ts],
                matvec_min_s=round(min(ts), 2),
                peak_rss_gb=round(peak_gb, 2),
                pass_mv=bool(min(ts) <= 150.0))


def stage2(h1e, eri, ecore):
    """ms=200 @500 双臂短跑（地板档语义对照 + 地板实测 + 强关联检验）。"""
    import tc_sqd

    s = 500
    rng = np.random.default_rng(0)
    bsm = rng.random((s, 2 * NORB)) > 0.5
    probs = np.full(s, 1.0 / s)
    recipe = dict(max_strings=200, n_active_per_round=90, dom_thresh=1e-3,
                  pt2_floor=1e-7, tail_suppression=False,
                  warm_start=True, eigsh_tol=1e-6, rand_seed=0, ecore=ecore,
                  backend="gpu", verbose=False)
    out = {}
    hf = 0b1111111  # 最低 7 轨道全占 = HF 串
    for arm, clo, mr in [("on", True, 10), ("off", False, 50)]:
        traj, state = [], []
        t0 = time.perf_counter()
        e = tc_sqd.solve_sqd_active(
            h1e, eri, NORB, NELEC, bitstring_matrix=bsm, probabilities=probs,
            coverage_closure=clo, max_rounds=mr, trajectory=traj,
            state_out=state, **recipe)
        wall = time.perf_counter() - t0
        c2d, sa, sb = state[0]
        ia = int(np.searchsorted(sa, hf))
        ib = int(np.searchsorted(sb, hf))
        c_hf2 = float(c2d[ia, ib] ** 2) if (ia < len(sa) and sa[ia] == hf
                                            and ib < len(sb) and sb[ib] == hf) else 0.0
        out[arm] = dict(
            E_total=float(e), wall_s=round(wall, 1),
            n_rounds=len([t for t in traj if t["round"] > 0]),
            dim_round1=traj[0]["dim"], dim_last_round=traj[-2]["dim"],
            dim_final=traj[-1]["dim"],
            floor_F1=round(traj[0]["dim"] ** 0.5, 1),
            strings_final=[int(len(sa)), int(len(sb))],
            sigma2_final=traj[-1]["sigma2"], e_pt2_final=traj[-1]["e_pt2"],
            c_hf2=c_hf2,
            bfs_injected=int(round(traj[-1]["dim"] ** 0.5))
            - int(round(traj[-2]["dim"] ** 0.5)) if clo else 0,
        )
        _save_partial(arm, out[arm])
        print(f"[2] arm={arm} E={e:.8f} wall={wall:.0f}s "
              f"dims={traj[0]['dim']}->{traj[-2]['dim']}->{traj[-1]['dim']} "
              f"|c_HF|2={c_hf2:.4f}", flush=True)
    out["pass_semantics"] = bool(
        abs(out["on"]["E_total"] - out["off"]["E_total"]) < 1e-12
        and out["on"]["dim_final"] == out["off"]["dim_final"]
        and out["on"]["dim_final"] > 200 * 200)
    return out


def _save_partial(arm, rec):
    try:
        res = json.load(open(OUT))
    except Exception:
        res = {}
    res.setdefault("stage2", {})[arm] = rec
    _save(res)


def main():
    d = np.load(INTS)
    h1e, eri, ecore = d["h1e"], d["eri"], float(d["ecore"])
    res = {}
    print("[1] 全空间 matvec 计时 (dim=11,778,624) ...", flush=True)
    res["stage1"] = stage1(h1e, eri)
    _save(res)
    s1 = res["stage1"]
    print(f"[1] dim={s1['dim']} linkstr={s1['t_linkstr_s']}s "
          f"mv={s1['matvec_s']}s peak={s1['peak_rss_gb']}GB "
          f"pass={s1['pass_mv']}", flush=True)
    print("[2] ms=200 双臂短跑 ...", flush=True)
    res["stage2"] = stage2(h1e, eri, ecore)
    _save(res)
    print("[2] pass_semantics =", res["stage2"]["pass_semantics"], flush=True)


if __name__ == "__main__":
    main()
