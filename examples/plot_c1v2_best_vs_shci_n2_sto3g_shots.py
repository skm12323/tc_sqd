"""fig_c1v2_best_vs_shci_n2_sto3g_shots.png: C1-v2 + best vs SHCI (N2/STO-3G), 扫 shots。

round_006-r3b shots 扫描轮: 旧 max_strings 版里 C1-v2+best 的 tail 发现不受
max_strings gate, 所有点 dim 相同 (全空间饱和), 曲线退化为单点。改为扫 shots:
不同 shots 产生不同 dim → 曲线分散可读。

体系: N2/STO-3G @ R=2.0 A (拉伸强关联), 全空间 14400 = C(10,7)^2 = 120^2。
活性空间核对 (theory §2.3.6 坑 2): **10 轨道 nelec=(7,7)** (非任务单标注的 7o)。

扫描轴: SHOTS_LIST = [10, 30, 50, 100, 300, 1000] (覆盖 2 个数量级)。
  - SQD 三曲线 (improved/best/C1-v2+best) 扫 shots, max_strings=None (不限,
    让采样自然决定 dim); 不同 shots → 不同 dim。
  - SHCI 不变 (确定性选态, 与 shots 无关, 仍扫 eps_hb)。

4 曲线 (误差 vs 实际对角化维度 dim, log-log):
  - **C1-v2 + best** (红 ^ 实线): solve_sqd_best(tail_suppression=True,
    tail_shots_ref=100, backend="gpu") —— C1 采样覆盖 + evpt2 精修 (核心)
  - **best** (绿 o 实线): solve_sqd_best(tail_suppression=False, backend="gpu")
  - **improved** (绿 -- 虚线): solve_sqd_improved(backend="gpu")
  - **SHCI** (棕 v 实线): solve_hci(backend="gpu")

x 轴口径 (theory §2.3.3 坑 1): 一律用实际 dim, 不用 shots。
y 轴扩展到 1e-15~1e0 (让 C1-v2+best 的低 err 可见); C1-v2+best 点 err < y 下限时
用红色下箭头标注。
参考 (theory §2.3.5): 库全空间对角化真基态 (from_pyscf + direct_spin1.kernel, 禁 CASCI)。

用法:
  python examples/plot_c1v2_best_vs_shci_n2_sto3g_shots.py          # 收集 + 出图
  python examples/plot_c1v2_best_vs_shci_n2_sto3g_shots.py --plot   # 只用缓存出图
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
NPY = os.path.join(BASE, "_plot_data_c1v2_best_vs_shci_n2_sto3g_shots.npy")
FIG = os.path.join(BASE, "fig_c1v2_best_vs_shci_n2_sto3g_shots.png")
SHOTS_LIST = [10, 30, 50, 100, 300, 1000]   # 扫 shots, 覆盖 2 个数量级; max_strings=None
EPS_HB = [1e-1, 5e-2, 3e-2, 2e-2, 1.5e-2, 1e-2, 5e-3, 1e-3]   # 沿用 n2_sto3g 旧脚本
Y_LO, Y_HI = 1e-15, 1e0   # y 轴扩展 (让 C1-v2+best 的低 err 可见)


def _system():
    """N2/STO-3G: 10 轨道 nelec=(7,7), 全空间 14400; 参考用库 FCI (禁 CASCI)。"""
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_fci, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    e_ref = float(e_fci + data.ecore)
    full = int(cistring.num_strings(norb, nelec[0])) ** 2
    assert (norb, nelec) == (10, (7, 7)), f"坑2: 期望 (10,(7,7)), 得到 ({norb},{nelec})"
    assert full == 14400, f"坑2: full 期望 14400, 得到 {full}"
    return h1e, eri, norb, nelec, data.ecore, e_ref, full


def _collect():
    h1e, eri, norb, nelec, ecore, e_ref, full = _system()
    print(f"N2/STO-3G 真基态参考 = {e_ref:.10f}  full={full}  norb={norb} nelec={nelec}")
    P = {"c1v2_best": [], "best": [], "improved": [], "shci": []}
    if os.path.exists(NPY):
        P = np.load(NPY, allow_pickle=True).item()

    # ---- SHCI 曲线 (棕): 确定性选态, 与 shots 无关, 仍扫 eps_hb ----
    if not P["shci"]:
        for eps in EPS_HB:
            e_t, e_pt2, dim = tc_sqd.solve_hci(
                h1e, eri, norb, nelec, eps_hb=eps, max_iter=20,
                ecore=ecore, return_details=True, verbose=False, backend="gpu")
            P["shci"].append((int(dim), float(abs(e_t - e_ref))))
            print(f"  SHCI eps={eps:.1e}: dim={dim} err={P['shci'][-1][1]:.2e}",
                  flush=True)
        np.save(NPY, P, allow_pickle=True)

    # ---- improved SQD 曲线 (绿虚): 扫 shots, max_strings=None ----
    if not P["improved"]:
        for shots in SHOTS_LIST:
            b = np.random.default_rng(0).random((shots, 2 * norb)) > 0.5
            p = np.full(shots, 1.0 / shots)
            e_c, det = tc_sqd.solve_sqd_improved(
                h1e, eri, norb, nelec, bitstring_matrix=b, probabilities=p,
                max_strings=None, n_active_per_round=30, rand_seed=0, ecore=ecore,
                return_details=True, verbose=False, backend="gpu")
            dim = int(det["dim"])
            P["improved"].append((dim, float(abs(e_c - e_ref))))
            print(f"  improved shots={shots}: dim={dim} err={P['improved'][-1][1]:.2e}",
                  flush=True)
        np.save(NPY, P, allow_pickle=True)

    # ---- best 曲线 (绿实): 扫 shots, max_strings=None ----
    if not P["best"]:
        for shots in SHOTS_LIST:
            out = tc_sqd.solve_sqd_best(
                h1e, eri, norb, nelec, ecore=ecore, n_shots=shots,
                max_strings=None, rand_seed=0, return_details=True, verbose=False,
                tail_suppression=False, backend="gpu")
            dim = int(out["dim"])
            P["best"].append((dim, float(abs(out["energy"] - e_ref))))
            print(f"  best shots={shots}: dim={dim} err={P['best'][-1][1]:.2e}",
                  flush=True)
        np.save(NPY, P, allow_pickle=True)

    # ---- C1-v2 + best 曲线 (红, 核心): 扫 shots, max_strings=None ----
    if not P["c1v2_best"]:
        for shots in SHOTS_LIST:
            out = tc_sqd.solve_sqd_best(
                h1e, eri, norb, nelec, ecore=ecore, n_shots=shots,
                max_strings=None, rand_seed=0, return_details=True, verbose=False,
                tail_suppression=True, tail_shots_ref=100, backend="gpu")
            dim = int(out["dim"])
            P["c1v2_best"].append((dim, float(abs(out["energy"] - e_ref))))
            print(f"  C1v2+best shots={shots}: dim={dim} err={P['c1v2_best'][-1][1]:.2e}",
                  flush=True)
        np.save(NPY, P, allow_pickle=True)
    return P


def _uniform(points, targets):
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

    targets = list(np.geomspace(150, 14000, 11))
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

    plot(c1v2, "C1-v2 + best (tail+C1, evpt2)", "#d62728", "^", ls="-", z=5)
    plot(best, "best (evpt2, no C1)", "#2ca02c", "o", ls="-", z=4)
    plot(improved, "improved (active+PT2)", "#2ca02c", "o", ls="--", z=3)
    plot(shci, "SHCI E_V+E_PT2 (solve_hci)", "#8c564b", "v", ls="-", z=2)

    ax.axvline(14400, color="grey", ls="--", lw=0.8)
    ax.text(15000, 3e-9, "full space\n14400", fontsize=7, color="grey")
    ax.axhline(CHEM, color="red", ls="--", lw=1.0, alpha=0.7)
    ax.text(160, CHEM * 1.6, "chemical accuracy 1.6 mHa", fontsize=8, color="red")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("subspace dimension dim = len(str_a) * len(str_b)")
    ax.set_ylabel("energy error vs FCI (Ha)")
    ax.set_title("C1-v2 + best vs SHCI (N2/STO-3G stretch), shots scan",
                 fontsize=12)
    ax.text(0.02, 0.02,
            "red vs green = C1 tail gain\ngreen solid vs dashed = evpt2 gain\nall vs brown = vs SHCI",
            transform=ax.transAxes, fontsize=7, color="grey", va="bottom")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(120, 20000); ax.set_ylim(Y_LO, Y_HI)
    ax.grid(True, which="both", alpha=0.3)

    # C1-v2+best 点 err < y 下限时用红色下箭头标注 (实际 err 太低, 落在轴外)
    for d, e in c1v2:
        if e < Y_LO:
            ax.annotate("", xy=(d, Y_LO), xytext=(d, Y_LO * 30),
                        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6))
            ax.text(d, Y_LO * 60, f"{e:.1e}", fontsize=6, color="#d62728",
                    ha="center", va="bottom", rotation=90)

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
