"""R5 round_002: C1-v2 (budget scaling) A/B/C three-way benchmark (P0/P1/P2).

三方定义 (theory.md §3 / task round_002):
  A (baseline): solve_sqd_improved(..., tail_suppression=False)
  B (C1-v1):    solve_sqd_improved(..., tail_suppression=True,
                 tail_max_draw_factor=10, tail_shots_ref=0)   # 显式 0 = round_001
  C (C1-v2):    solve_sqd_improved(..., tail_suppression=True,
                 tail_max_draw_factor=10, tail_shots_ref=100)  # 预算随 shots 缩放

体系与配置 (task round_002):
  P0 (12,12) 主: shots=500 seed=0  (C 新跑; A/B 复用 round_001 缓存)
  P1 (12,12)  零回归: shots=100 seed=0, C 应与 B 逐位一致 (n_tgt=30)
  P2 诊断: spy discover_tail_pool 记录每轮 (n_tgt, n_drawn) → 验证预算缩放
  P1-nr (10o) 不回归: shots=80 / 500 seed=0, C1-v2 vs round_001 C1-v1

P2 spy: cipsi.py 以模块全局名 discover_tail_pool 调用, 故 monkeypatch
  tc_sqd.cipsi.discover_tail_pool 即可记录每轮调用参数与返回 n_drawn (不改源码)。

结果增量写入 JSON (benchmarks/_round002_c1_results.json, 幂等可续)。
--import-round001: 把 round_001 缓存的 A/B case 复制进 round_002 结果 (供三方对照)。
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import tc_sqd  # noqa: E402
from tc_sqd import cipsi as _cipsi  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "benchmarks", "_round002_c1_results.json")
R1_OUT = os.path.join(BASE, "benchmarks", "_round001_c1_results.json")

SYSTEMS = {
    "1212": dict(norb=12, nelec=(6, 6), ints="_n2_1212_ints.npz", ms=100000,
                 label="N2/cc-pVDZ (12,12)"),
    "10o": dict(norb=10, nelec=(5, 5), ints="_n2_ccpvdz_10o_ints.npz", ms=47000,
                label="N2/cc-pVDZ (10o)"),
}

# ---- C1 参数按 variant 展开 (B/C 的 tail_suppression 均 True) ----
VARIANTS = {
    "A": dict(tail_suppression=False, tail_shots_ref=0),
    "B": dict(tail_suppression=True, tail_shots_ref=0),    # C1-v1: 固定预算 (显式 0)
    "C": dict(tail_suppression=True, tail_shots_ref=100),  # C1-v2: 预算随 shots 缩放
}


def load_ints(system):
    d = np.load(os.path.join(BASE, SYSTEMS[system]["ints"]))
    return (d["h1e"], d["eri"], float(d["ecore"]), float(d["e_ref"]))


_REAL_DISCOVER = None   # install_spy 时保存的原始 discover_tail_pool 引用


def install_spy():
    """monkeypatch tc_sqd.cipsi.discover_tail_pool, 记录每轮 (n_tgt, n_drawn)。

    用闭包保存真实函数, 返回 calls 列表; restore_spy 直接还原引用 (不 reload 模块)。
    """
    global _REAL_DISCOVER
    _REAL_DISCOVER = _cipsi.discover_tail_pool
    calls = []

    def spy(*args, **kwargs):
        res = _REAL_DISCOVER(*args, **kwargs)
        calls.append({
            "call": len(calls),
            "n_tgt": int(kwargs.get("n_target_new")),
            "max_draw_factor": int(kwargs.get("max_draw_factor")),
            "round_seed": kwargs.get("rand_seed"),
            "n_drawn": int(res[2]),
            "n_collected": int(res[0].shape[0]),
        })
        return res

    _cipsi.discover_tail_pool = spy
    return calls


def restore_spy():
    global _REAL_DISCOVER
    if _REAL_DISCOVER is not None:
        _cipsi.discover_tail_pool = _REAL_DISCOVER
        _REAL_DISCOVER = None


def run_case(system, shots, variant, seed=0, verbose=False):
    info = SYSTEMS[system]
    h1e, eri, ecore, e_ref = load_ints(system)
    norb, nelec = info["norb"], info["nelec"]
    bsm = np.random.default_rng(seed).random((shots, 2 * norb)) > 0.5
    probs = np.full(shots, 1.0 / shots)
    vp = VARIANTS[variant]
    calls = install_spy()
    t0 = time.perf_counter()
    try:
        e, det = tc_sqd.solve_sqd_improved(
            h1e, eri, norb, nelec,
            bitstring_matrix=bsm, probabilities=probs,
            max_strings=info["ms"], n_active_per_round=30,
            rand_seed=seed, ecore=ecore, return_details=True, verbose=verbose,
            tail_suppression=vp["tail_suppression"], tail_max_draw_factor=10,
            tail_shots_ref=vp["tail_shots_ref"],
        )
    finally:
        restore_spy()
    wall = time.perf_counter() - t0
    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    traj = det["trajectory"]
    pos = [t for t in traj if t["round"] >= 1]
    nstr = [round(np.sqrt(float(t["dim"]))) for t in pos]
    growth = [nstr[0]] + [nstr[i] - nstr[i - 1] for i in range(1, len(nstr))]
    per_round = []
    for i, (t, ns, g) in enumerate(zip(pos, nstr, growth)):
        per_round.append({
            "round": t["round"], "E": float(t["E"]), "dim": int(t["dim"]),
            "n_str": int(ns), "n_new_proxy": int(g),
            "n_tgt": calls[i]["n_tgt"] if i < len(calls) else None,
            "n_drawn": calls[i]["n_drawn"] if i < len(calls) else None,
        })
    return {
        "system": system, "shots": shots, "variant": variant, "seed": seed,
        "E": float(e), "E_ref": float(e_ref), "err": float(abs(e - e_ref)),
        "E_direct": float(det["E_direct"]),
        "E_PT2": float(det["E_PT2"]), "dim": int(det["dim"]),
        "wall_s": round(wall, 1), "peak_rss_kb": peak_rss_kb,
        "n_tgt_rounds": [c["n_tgt"] for c in calls],
        "n_drawn_rounds": [c["n_drawn"] for c in calls],
        "n_drawn_total": sum(c["n_drawn"] for c in calls),
        "per_round": per_round,
    }


def import_round001():
    """把 round_001 缓存的 A(tail0)/B(tail1) case 映射成 variant, 复制进 round_002 结果。"""
    if not os.path.exists(R1_OUT):
        print("[bench] no round_001 cache, skip import", flush=True)
        return
    with open(R1_OUT, "r") as f:
        r1 = json.load(f)
    if os.path.exists(OUT):
        with open(OUT, "r") as f:
            data = json.load(f)
    else:
        data = {}
    n_imported = 0
    for key, res in r1.items():
        # key = f"{system}_s{shots}_tail{int(tail)}_seed{seed}"
        parts = key.split("_")
        system = parts[0]
        shots = int(parts[1][1:])
        tail = int(parts[2][4:])
        seed = int(parts[3][4:])
        variant = "A" if tail == 0 else "B"
        # 仅 A/B 且体系在 round_002 SYSTEMS 中才导入
        if system not in SYSTEMS:
            continue
        nkey = f"{system}_s{shots}_var{variant}_seed{seed}"
        if nkey in data:
            continue
        # 去掉 round_001 专用的 per_round/n_new_proxy 无 n_tgt/n_drawn → 保留即可
        rec = dict(res)
        rec["variant"] = variant
        rec.pop("tail", None)
        data[nkey] = rec
        n_imported += 1
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(f"[bench] imported {n_imported} A/B cases from round_001 -> {OUT}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["1212", "10o"])
    ap.add_argument("--shots", type=int)
    ap.add_argument("--variant", choices=["A", "B", "C"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", default=False)
    ap.add_argument("--only", action="store_true",
                    help="只跑本 case, 不把结果并入 JSON (调试用)")
    ap.add_argument("--import-round001", action="store_true",
                    help="只导入 round_001 缓存的 A/B case 到 round_002 结果, 不跑 case")
    args = ap.parse_args()

    if args.import_round001:
        import_round001()
        if not (args.system and args.shots and args.variant):
            return

    if not (args.system and args.shots and args.variant):
        ap.error("--system/--shots/--variant 在非 import-only 模式必须给出")

    res = run_case(args.system, args.shots, args.variant, seed=args.seed,
                   verbose=args.verbose)
    print(json.dumps(res, indent=2, default=float), flush=True)

    if args.only:
        return
    key = f"{args.system}_s{args.shots}_var{args.variant}_seed{args.seed}"
    if os.path.exists(OUT):
        with open(OUT, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data[key] = res
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(f"[bench] saved case {key} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
