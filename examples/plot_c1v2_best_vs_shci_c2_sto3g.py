"""fig_c1v2_best_vs_shci_c2_sto3g.png: C1-v2 + best vs SHCI (C2/STO-3G).

round_006 全量 plot 轮: 验证 "C1-v2 + best" 组合跨体系不回归。
体系: C2/STO-3G @ R=1.24 A (强关联), 全空间 44100 = C(10,6)^2 = 210^2。
活性空间核对 (theory §2.3.6 坑 2): **10 轨道 nelec=(6,6)** (非任务单标注的 6o)。

4 曲线 (误差 vs 实际对角化维度 dim, log-log):
  - **C1-v2 + best** (红 ^ 实线): solve_sqd_best(tail_suppression=True,
    tail_shots_ref=100, backend="gpu") —— C1 采样覆盖 + evpt2 精修 (核心)
  - **best** (绿 o 实线): solve_sqd_best(tail_suppression=False, backend="gpu")
    —— SQD 最优 baseline (evpt2, 无 C1 tail)
  - **improved** (绿 -- 虚线): solve_sqd_improved(backend="gpu")
    —— active+PT2 (evpt2 前, 对照 best 展示 evpt2 增益)
  - **SHCI** (棕 v 实线): solve_hci(backend="gpu") —— 经典对照

x 轴口径 (theory §2.3.3 坑 1): **一律用各方法回报的实际 dim** (= len(str_a)*len(str_b)),
绝不用 max_strings (那是 α 字符串数上界, 单位不同; C1 tail 发现的 det 不受 max_strings gate,
C1-v2 实际 dim 可超过 max_strings^2)。

参考能量 (theory §2.3.5): 库全空间对角化真基态 (from_pyscf + direct_spin1.kernel, 禁 CASCI)。
对照语义 (图注): 红 vs 绿实 = C1 tail 增益; 绿实 vs 绿虚 = evpt2 增益; 全部 vs 棕 = vs SHCI。

用法:
  python examples/plot_c1v2_best_vs_shci_c2_sto3g.py          # 收集 + 出图
  python examples/plot_c1v2_best_vs_shci_c2_sto3g.py --plot   # 只用缓存出图
英文标签 (matplotlib 缺 CJK 字体)。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import tc_sqd  # noqa: E402
from pyscf import gto  # noqa: E402
from pyscf.fci import direct_spin1, cistring  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEM = 1.6e-3
NPY = os.path.join(BASE, "_plot_data_c1v2_best_vs_shci_c2_sto3g.npy")
FIG = os.path.join(BASE, "fig_c1v2_best_vs_shci_c2_sto3g.png")
SHOTS = 500                  # 固定 shots (theory §2.3.4: 高采样端单 seed=0)
MAX_STRINGS = [10, 15, 22, 32, 48, 70, 100, 145, 210]  # theory §2.3.4 网格 (full=210)
EPS_HB = [1e-1, 8e-2, 6e-2, 5e-2, 4.5e-2, 4e-2, 3.5e-2, 3e-2,
          2e-2, 1e-2, 5e-3, 1e-3]   # 沿用 c2_sto3g 旧脚本 eps 网格


def _system():
    """C2/STO-3G: 10 轨道 nelec=(6,6), 全空间 44100; 参考用库 FCI (禁 CASCI)。"""
    mol = gto.M(atom="C 0 0 0; C 0 0 1.24", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_fci, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    e_ref = float(e_fci + data.ecore)
    full = int(cistring.num_strings(norb, nelec[0])) ** 2
    # 坑 2 自洽核对
    assert (norb, nelec) == (10, (6, 6)), f"坑2: 期望 (10,(6,6)), 得到 ({norb},{nelec})"
    assert full == 44100, f"坑2: full 期望 44100, 得到 {full}"
    return h1e, eri, norb, nelec, data.ecore, e_ref, full


def _collect():
    h1e, eri, norb, nelec, ecore, e_ref, full = _system()
    print(f"C2/STO-3G 真基态参考 = {e_ref:.10f}  full={full}  norb={norb} nelec={nelec}")
    P = {"c1v2_best": [], "best": [], "improved": [], "shci": []}
    if os.path.exists(NPY):
        P = np.load(NPY, allow_pickle=True).item()

    # ---- SHCI 曲线 (棕) ----
    if not P["shci"]:
        for eps in EPS_HB:
            e_t, e_pt2, dim = tc_sqd.solve_hci(
                h1e, eri, norb, nelec, eps_hb=eps, max_iter=20,
                ecore=ecore, return_details=True, verbose=False, backend="gpu")
            P["shci"].append((int(dim), float(abs(e_t - e_ref))))
            print(f"  SHCI eps={eps:.1e}: dim={dim} err={P['shci'][-1][1]:.2e}",
                  flush=True)
        np.save(NPY, P, allow_pickle=True)

    # ---- improved SQD 曲线 (绿虚): active + PT2 ----
    if not P["improved"]:
        for ms in MAX_STRINGS:
            b = np.random.default_rng(0).random((SHOTS, 2 * norb)) > 0.5
            p = np.full(SHOTS, 1.0 / SHOTS)
            e_c, det = tc_sqd.solve_sqd_improved(
                h1e, eri, norb, nelec, bitstring_matrix=b, probabilities=p,
                max_strings=ms, n_active_per_round=30, rand_seed=0, ecore=ecore,
                return_details=True, verbose=False, backend="gpu")
            dim = int(det["dim"])
            P["improved"].append((dim, float(abs(e_c - e_ref))))
            print(f"  improved ms={ms}: dim={dim} err={P['improved'][-1][1]:.2e}",
                  flush=True)
        np.save(NPY, P, allow_pickle=True)

    # ---- best 曲线 (绿实): solve_sqd_best 无 C1 tail ----
    if not P["best"]:
        for ms in MAX_STRINGS:
            out = tc_sqd.solve_sqd_best(
                h1e, eri, norb, nelec, ecore=ecore, n_shots=SHOTS,
                max_strings=ms, rand_seed=0, return_details=True, verbose=False,
                tail_suppression=False, backend="gpu")
            dim = int(out["dim"])
            P["best"].append((dim, float(abs(out["energy"] - e_ref))))
            print(f"  best ms={ms}: dim={dim} err={P['best'][-1][1]:.2e}", flush=True)
        np.save(NPY, P, allow_pickle=True)

    # ---- C1-v2 + best 曲线 (红, 核心): solve_sqd_best + C1 tail ----
    if not P["c1v2_best"]:
        for ms in MAX_STRINGS:
            out = tc_sqd.solve_sqd_best(
                h1e, eri, norb, nelec, ecore=ecore, n_shots=SHOTS,
                max_strings=ms, rand_seed=0, return_details=True, verbose=False,
                tail_suppression=True, tail_shots_ref=100, backend="gpu")
            dim = int(out["dim"])
            P["c1v2_best"].append((dim, float(abs(out["energy"] - e_ref))))
            print(f"  C1v2+best ms={ms}: dim={dim} err={P['c1v2_best'][-1][1]:.2e}",
                  flush=True)
        np.save(NPY, P, allow_pickle=True)
    return P


def _uniform(points, targets):
    """对每个 log 均匀目标维度取最近点 (去重, 保证均匀覆盖)。"""
    arr = sorted(set((round(float(d), 0), float(e)) for d, e in points))
    out, used = [], set()
    for t in targets:
        best = min(arr, key=lambda de: abs(de[0] - t))
        if best[0] in used:
            continue
        used.add(best[0])
        out.append((best[0], best[1]))
    return sorted(out)


def _plot(P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = list(np.geomspace(200, 44000, 12))
    c1v2 = _uniform(P["c1v2_best"], targets)
    best = _uniform(P["best"], targets)
    improved = _uniform(P["improved"], targets)
    shci = _uniform(P["shci"], targets)

    fig, ax = plt.subplots(figsize=(8.5, 6))

    def plot(pts, label, color, marker, ls="-", z=3, ms=6):
        d = np.array(pts)
        if d.size == 0:
            return
        ax.plot(d[:, 0], d[:, 1], marker=marker, ls=ls, color=color,
                label=label, ms=ms, zorder=z, lw=1.8)

    # 4 曲线 (theory §2.3.2 颜色/线型规格)
    plot(c1v2, "C1-v2 + best (tail+C1, evpt2)", "#d62728", "^", ls="-", z=5)
    plot(best, "best (evpt2, no C1)", "#2ca02c", "o", ls="-", z=4)
    plot(improved, "improved (active+PT2)", "#2ca02c", "o", ls="--", z=3)
    plot(shci, "SHCI E_V+E_PT2 (solve_hci)", "#8c564b", "v", ls="-", z=2)

    ax.axvline(44100, color="grey", ls="--", lw=0.8)
    ax.text(46000, 3e-9, "full space\n44100", fontsize=7, color="grey")
    ax.axhline(CHEM, color="red", ls="--", lw=1.0, alpha=0.7)
    ax.text(220, CHEM * 1.6, "chemical accuracy 1.6 mHa", fontsize=8, color="red")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("subspace dimension dim = len(str_a) * len(str_b)")
    ax.set_ylabel("energy error vs FCI (Ha)")
    ax.set_title("C1-v2 + best vs SHCI (C2/STO-3G, strongly correlated)",
                 fontsize=12)
    # 对照语义图注 (theory §2.3.2)
    ax.text(0.02, 0.02,
            "red vs green = C1 tail gain\ngreen solid vs dashed = evpt2 gain\nall vs brown = vs SHCI",
            transform=ax.transAxes, fontsize=7, color="grey", va="bottom")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(150, 60000); ax.set_ylim(1e-12, 1.0)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout(); fig.savefig(FIG, dpi=150)
    print(f"\nsaved {FIG}")
    print("C1-v2+best points:", c1v2)
    print("best points:", best)
    print("SHCI points:", shci)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true",
                    help="只用缓存数据出图 (跳过耗时数据收集)")
    args = ap.parse_args()
    if args.plot:
        if not os.path.exists(NPY):
            raise SystemExit(f"无缓存数据 {NPY}, 请先不带 --plot 跑一次收集。")
        P = np.load(NPY, allow_pickle=True).item()
    else:
        P = _collect()
    _plot(P)
