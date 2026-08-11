"""R5 round_001: C1 tail-discovery sampling A/B benchmark (P0/P1/P2).

A/B 对照设计 (theory.md §3):
  A (baseline): solve_sqd_improved(h1e, eri, norb, nelec, bitstring_matrix=<bsm>,
                  max_strings=MS, n_active_per_round=30, rand_seed=0)
  B (C1):       同 A + tail_suppression=True, tail_max_draw_factor=10
  A/B 同 bsm/seed/max_strings → 差异纯净归因 C1。

体系与配置 (task round_001 · R5):
  P0 (12,12) 主: shots=100 (复现 theory 基线 2.28e-4 @ dim~1e5), ms=100000
  P0 (12,12) 副: shots=500 (task 字面规格), ms=100000
  P1 (10o)   主: shots=80  (复现基线 regime dim~47k, err~9.76e-7), ms=47000
  P1 (10o)   副: shots=500 (task 字面规格), ms=47000

P2 诊断: 每轮 trajectory dim 增长 (closed-shell n_str = sqrt(dim)) 作 n_sampled_new
  代理 (theory §3 P2: "trajectory 的 dim 增长")。

结果增量写入 JSON (每跑完一个 case 落盘), 幂等可续。
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "benchmarks", "_round001_c1_results.json")

SYSTEMS = {
    "1212": dict(norb=12, nelec=(6, 6), ints="_n2_1212_ints.npz", ms=100000,
                 label="N2/cc-pVDZ (12,12)"),
    "10o": dict(norb=10, nelec=(5, 5), ints="_n2_ccpvdz_10o_ints.npz", ms=47000,
                label="N2/cc-pVDZ (10o)"),
}


def load_ints(system):
    d = np.load(os.path.join(BASE, SYSTEMS[system]["ints"]))
    return (d["h1e"], d["eri"], float(d["ecore"]), float(d["e_ref"]))


def run_case(system, shots, tail, seed=0, verbose=False):
    info = SYSTEMS[system]
    h1e, eri, ecore, e_ref = load_ints(system)
    norb, nelec = info["norb"], info["nelec"]
    bsm = np.random.default_rng(seed).random((shots, 2 * norb)) > 0.5
    probs = np.full(shots, 1.0 / shots)
    t0 = time.perf_counter()
    e, det = tc_sqd.solve_sqd_improved(
        h1e, eri, norb, nelec,
        bitstring_matrix=bsm, probabilities=probs,
        max_strings=info["ms"], n_active_per_round=30,
        rand_seed=seed, ecore=ecore, return_details=True, verbose=verbose,
        tail_suppression=tail, tail_max_draw_factor=10,
    )
    wall = time.perf_counter() - t0
    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    traj = det["trajectory"]
    # 每轮 (round>=1) dim 增长 → n_sampled_new 代理 (closed-shell: n_str=sqrt(dim))
    pos = [t for t in traj if t["round"] >= 1]
    nstr = [round(np.sqrt(float(t["dim"]))) for t in pos]
    growth = [nstr[0]] + [nstr[i] - nstr[i - 1] for i in range(1, len(nstr))]
    per_round = []
    for t, ns, g in zip(pos, nstr, growth):
        per_round.append({"round": t["round"], "E": float(t["E"]),
                          "dim": int(t["dim"]), "n_str": int(ns),
                          "n_new_proxy": int(g)})
    return {
        "system": system, "shots": shots, "tail": bool(tail), "seed": seed,
        "E": float(e), "E_ref": float(e_ref), "err": float(abs(e - e_ref)),
        "E_direct": float(det["E_direct"]),
        "E_PT2": float(det["E_PT2"]), "dim": int(det["dim"]),
        "wall_s": round(wall, 1), "peak_rss_kb": peak_rss_kb,
        "per_round": per_round,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["1212", "10o"], required=True)
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--tail", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", default=False)
    ap.add_argument("--only", action="store_true",
                    help="只跑本 case, 不把结果并入 JSON (调试用)")
    args = ap.parse_args()

    res = run_case(args.system, args.shots, args.tail, seed=args.seed,
                   verbose=args.verbose)
    print(json.dumps(res, indent=2, default=float), flush=True)

    if args.only:
        return
    # 幂等并入 JSON (按 case key 覆盖)
    key = f"{args.system}_s{args.shots}_tail{int(args.tail)}_seed{args.seed}"
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
