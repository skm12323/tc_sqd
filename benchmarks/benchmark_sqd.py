"""tc_sqd 性能基准: 各求解方法 vs 体系 / shots 的耗时与峰值内存。

测量对象 (均含对角化, 口径一致):
  - traditional SQD (solve_sqd, 单次恢复 + 对角化)
  - active SQD     (solve_sqd_active, 受限 PT2 双闭环)
  - CIPSI          (solve_cipsi, 补全到近 FCI —— 高精度 refine 层)
  - HCI/SHCI       (solve_hci, heat-bath 选态 + PT2 修正)
  - EV             (solve_sqd_ev, 能量-方差外推 —— 同一 active 流程 + 外推)

测量指标:
  - wall (s): 时间.perf_counter 墙钟
  - peak (MB): resource.getrusage().ru_maxrss (进程峰值 RSS, Linux/WSL)
  - dim / err: 对角化维度与对 FCI 误差 (参考)

用法:
    python benchmarks/benchmark_sqd.py [--quick] [--out bench.csv] [--shots 500,2000,8000]

说明: 基准在 WSL (Linux) 下跑 (resource 依赖); Windows 原生无 ru_maxrss。
结果写 CSV + 打印 Markdown 表格 (方便贴进 REVIEW)。
"""
from __future__ import annotations

import argparse
import csv
import resource
import time

import numpy as np
from pyscf import gto


def _data(name: str):
    """返回 (h1e, eri, norb, nelec, ecore, e_fci)。"""
    if name == "h2":
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    elif name == "n2":
        mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    elif name == "lih":
        mol = gto.M(atom="Li 0 0 0; H 0 0 1.60", basis="sto-3g", verbose=0)
    else:
        raise ValueError(name)
    import tc_sqd
    data = tc_sqd.from_pyscf(mol)
    e_fci = data.solve(method="fci")
    return (data.h1e, data.eri, data.norb, data.nelec, data.ecore, e_fci)


def _random_bsm(norb, nelec, shots, seed=0):
    """均匀随机位串 + 归一化概率。

    solve_sqd / solve_sqd_active / solve_cipsi **内部**都会做配置恢复
    (对原始位串矩阵), 所以这里传原始位串 + 匹配长度的 probs 即可。
    """
    rng = np.random.default_rng(seed)
    raw = rng.random((shots, 2 * norb)) > 0.5
    probs = np.full(shots, 1.0 / shots)
    return raw, probs


def _measure(fn):
    """执行 fn, 返回 (wall_s, peak_mb, 结果)。"""
    t0 = time.perf_counter()
    out = fn()
    wall = time.perf_counter() - t0
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return wall, peak_kb / 1024.0, out


def _bench_method(name, data, shots, seed=0):
    """按方法名运行并返回 (wall, peak, dim, err)。"""
    import tc_sqd
    h1e, eri, norb, nelec, ecore, e_fci = data
    bsm, probs = _random_bsm(norb, nelec, shots, seed)
    if name == "traditional":
        def run():
            r = tc_sqd.solve_sqd(h1e, eri, norb, nelec, ecore=ecore,
                                 bitstring_matrix=bsm, probabilities=probs,
                                 mode="single", seed=seed)
            return r.energy + ecore
    elif name == "active":
        def run():
            return tc_sqd.solve_sqd_active(
                h1e, eri, norb, nelec, bitstring_matrix=bsm,
                probabilities=probs, ecore=ecore, rand_seed=seed)
    elif name == "ev":
        def run():
            return tc_sqd.solve_sqd_ev(
                h1e, eri, norb, nelec, bitstring_matrix=bsm,
                probabilities=probs, ecore=ecore, rand_seed=seed)
    elif name == "cipsi":
        def run():
            return tc_sqd.solve_cipsi(
                h1e, eri, norb, nelec, seed_bitstring_matrix=bsm,
                ecore=ecore, verbose=False)
    elif name == "hci":
        def run():
            return tc_sqd.solve_hci(
                h1e, eri, norb, nelec, eps_hb=1e-3, ecore=ecore,
                verbose=False)
    else:
        raise ValueError(name)

    wall, peak, e = _measure(run)
    return wall, peak, abs(e - e_fci)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="快跑 (少 shots/少体系)")
    ap.add_argument("--out", default="bench_sqd.csv", help="CSV 输出路径")
    ap.add_argument("--shots", default=None, help="shots 网格, 逗号分隔 (默认 500,2000,8000)")
    args = ap.parse_args()

    systems = ["h2", "n2", "lih"] if not args.quick else ["h2", "n2"]
    methods = ["traditional", "active", "ev", "cipsi", "hci"]
    if args.quick:
        # quick 模式跳过 cipsi/hci (补全到全空间/大变分空间, 慢)
        methods = ["traditional", "active", "ev"]
    shots_list = ([int(s) for s in args.shots.split(",")] if args.shots
                  else ([500] if args.quick else [500, 2000, 8000]))

    rows = []
    for sys_name in systems:
        data = _data(sys_name)
        print(f"\n== {sys_name} (FCI={data[5]:.6f}) ==")
        for shots in shots_list:
            for m in methods:
                try:
                    wall, peak, err = _bench_method(m, data, shots)
                except Exception as ex:                      # noqa: BLE001
                    print(f"  {m:12s} shots={shots:6d}: FAIL {type(ex).__name__}")
                    rows.append([sys_name, m, shots, "fail", 0, 0, 0])
                    continue
                print(f"  {m:12s} shots={shots:6d}: "
                      f"wall={wall:7.2f}s peak={peak:6.1f}MB err={err:.2e}")
                rows.append([sys_name, m, shots, "ok", wall, peak, err])

    # CSV + Markdown 表
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "method", "shots", "status", "wall_s",
                    "peak_mb", "err_Ha"])
        w.writerows(rows)
    print(f"\nCSV 写入 {args.out}")
    print("\n| system | method | shots | wall_s | peak_MB | err |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        s, m, sh, st = r[0], r[1], r[2], r[3]
        if st != "ok":
            print(f"| {s} | {m} | {sh} | FAIL | - | - |")
            continue
        wall, peak, err = r[4], r[5], r[6]
        print(f"| {s} | {m} | {sh} | {wall:.2f} | {peak:.1f} | {err:.1e} |")


if __name__ == "__main__":
    main()
