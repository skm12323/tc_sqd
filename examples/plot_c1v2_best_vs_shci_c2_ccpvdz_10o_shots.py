"""fig_c1v2_best_vs_shci_c2_ccpvdz_10o_shots.png: C1-v2 + best vs SHCI (C2/cc-pVDZ 10o), 扫 shots。

round_006-r3b shots 扫描轮: 旧 max_strings 版里 C1-v2+best 的 tail 发现不受
max_strings gate, 所有点 dim 相同 (全空间饱和), 曲线退化为单点。改为扫 shots:
不同 shots 产生不同 dim → 曲线分散可读。

体系: C2/cc-pVDZ @ R=1.24 A (强关联), 全空间 63504 = C(10,5)^2 = 252^2。
活性空间核对 (theory §2.3.6): 10 轨道 nelec=(5,5)。

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
参考 (theory §2.3.5): **库内全空间对角化真基态** (solve_sci 全空间,
**禁 CASCI**——C2/cc-pVDZ 近简并, CASCI/davidson 会收敛到虚高 9.3 mHa 的第二根)。

用法:
  python examples/plot_c1v2_best_vs_shci_c2_ccpvdz_10o_shots.py
  python examples/plot_c1v2_best_vs_shci_c2_ccpvdz_10o_shots.py --plot
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import tc_sqd  # noqa: E402
import pyscf, pyscf.scf, pyscf.mcscf, pyscf.ao2mo  # noqa: E402
from pyscf.fci import cistring  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEM = 1.6e-3
NPY = os.path.join(BASE, "_plot_data_c1v2_best_vs_shci_c2_ccpvdz_10o_shots.npy")
FIG = os.path.join(BASE, "fig_c1v2_best_vs_shci_c2_ccpvdz_10o_shots.png")
NCAS, NELEC = 10, 10       # 10 轨道 (5,5) per spin
SHOTS_LIST = [10, 30, 50, 100, 300, 1000]   # 扫 shots, 覆盖 2 个数量级; max_strings=None
EPS_HB = [1e-1, 8e-2, 6e-2, 5e-2, 3e-2, 2e-2, 1.5e-2, 1e-2,
          5e-3, 1e-3]   # 沿用 c2_ccpvdz 旧脚本 eps 网格
Y_LO, Y_HI = 1e-15, 1e0   # y 轴扩展 (让 C1-v2+best 的低 err 可见)


def _integrals():
    """C2/cc-pVDZ 10o CAS: 库内全空间 solve_sci 真基态 (规避 CASCI 近简并跳根)。"""
    m = pyscf.M(atom="C 0 0 0; C 0 0 1.24", basis="cc-pVDZ", spin=0, verbose=0)
    mf = pyscf.scf.RHF(m); mf.kernel()
    cas = pyscf.mcscf.CASCI(mf, ncas=NCAS, nelecas=NELEC)
    h1e, ecore = cas.h1e_for_cas()
    ncore = int((mf.mo_occ.sum() - NELEC) // 2)
    eri = pyscf.ao2mo.full(m, cas.mo_coeff[:, ncore:ncore + NCAS],
                           aosym="1").reshape([NCAS] * 4)
    return h1e, eri, ecore


def _true_gs(h1e, eri, ecore):
    """库内全空间对角化真基态 (规避 CASCI 近简并跳根)。"""
    all_str = cistring.make_strings(range(NCAS), NELEC // 2)
    res = tc_sqd.solve_sci((all_str, all_str), h1e, eri, NCAS, (5, 5))
    return float(res.energy + ecore)


def _collect():
    h1e, eri, ecore = _integrals()
    e_ref = _true_gs(h1e, eri, ecore)
    full = int(cistring.num_strings(NCAS, NELEC // 2)) ** 2
    assert full == 63504, f"坑2: full 期望 63504, 得到 {full}"
    print(f"C2/cc-pVDZ 10o 真基态参考 = {e_ref:.10f}  full={full}")
    P = {"c1v2_best": [], "best": [], "improved": [], "shci": []}
    if os.path.exists(NPY):
        P = np.load(NPY, allow_pickle=True).item()

    # ---- SHCI 曲线 (棕): 确定性选态, 与 shots 无关, 仍扫 eps_hb ----
    if not P["shci"]:
        for eps in EPS_HB:
            e_t, e_pt2, dim = tc_sqd.solve_hci(
                h1e, eri, NCAS, (5, 5), eps_hb=eps, max_iter=15,
                ecore=ecore, return_details=True, verbose=False, backend="gpu")
            P["shci"].append((int(dim), float(abs(e_t - e_ref))))
            print(f"  SHCI eps={eps:.3e}: dim={dim} err={P['shci'][-1][1]:.2e}",
                  flush=True)
        np.save(NPY, P, allow_pickle=True)

    # ---- improved SQD 曲线 (绿虚): 扫 shots, max_strings=None ----
    if not P["improved"]:
        for shots in SHOTS_LIST:
            b = np.random.default_rng(0).random((shots, 2 * NCAS)) > 0.5
            p = np.full(shots, 1.0 / shots)
            e_c, det = tc_sqd.solve_sqd_improved(
                h1e, eri, NCAS, (5, 5), bitstring_matrix=b, probabilities=p,
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
                h1e, eri, NCAS, (5, 5), ecore=ecore, n_shots=shots,
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
                h1e, eri, NCAS, (5, 5), ecore=ecore, n_shots=shots,
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

    targets = list(np.geomspace(500, 63000, 12))
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

    ax.axvline(63504, color="grey", ls="--", lw=0.8)
    ax.text(66000, 3e-4, "full space\n63504", fontsize=7, color="grey")
    ax.axhline(CHEM, color="red", ls="--", lw=1.0, alpha=0.7)
    ax.text(550, CHEM * 1.6, "chemical accuracy 1.6 mHa", fontsize=8, color="red")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("subspace dimension dim = len(str_a) * len(str_b)")
    ax.set_ylabel("energy error vs true ground state (Ha)")
    ax.set_title("C1-v2 + best vs SHCI (C2/cc-pVDZ, 10o active), shots scan",
                 fontsize=11)
    ax.text(0.02, 0.02,
            "red vs green = C1 tail gain\ngreen solid vs dashed = evpt2 gain\nall vs brown = vs SHCI",
            transform=ax.transAxes, fontsize=7, color="grey", va="bottom")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(400, 80000); ax.set_ylim(Y_LO, Y_HI)
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
