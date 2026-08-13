"""fig_c1v2_best_vs_shci_n2_ccpvdz_10o.png: C1-v2 + best vs SHCI (N2/cc-pVDZ 10o).

round_006 全量 plot 轮: 验证 "C1-v2 + best" 组合跨体系不回归。
体系: N2/cc-pVDZ @ R=3.0 A (强关联近解离), 全空间 63504 = C(10,5)^2 = 252^2。
活性空间核对 (theory §2.3.6): 10 轨道 nelec=(5,5)。

4 曲线 (误差 vs 实际对角化维度 dim, log-log):
  - **C1-v2 + best** (红 ^ 实线): solve_sqd_best(tail_suppression=True,
    tail_shots_ref=100, backend="gpu") —— C1 采样覆盖 + evpt2 精修 (核心)
  - **best** (绿 o 实线): solve_sqd_best(tail_suppression=False, backend="gpu")
  - **improved** (绿 -- 虚线): solve_sqd_improved(backend="gpu")
  - **SHCI** (棕 v 实线): solve_hci(backend="gpu")

x 轴口径 (theory §2.3.3 坑 1): 一律用实际 dim, 不用 max_strings。
参考 (theory §2.3.5): 库全空间对角化真基态 (compute_ground_state_energy method="fci",
与各方法同哈密顿量, 禁 CASCI)。

用法:
  python examples/plot_c1v2_best_vs_shci_n2_ccpvdz_10o.py
  python examples/plot_c1v2_best_vs_shci_n2_ccpvdz_10o.py --plot
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
NPY = os.path.join(BASE, "_plot_data_c1v2_best_vs_shci_n2_ccpvdz_10o.npy")
FIG = os.path.join(BASE, "fig_c1v2_best_vs_shci_n2_ccpvdz_10o.png")
NCAS, NELEC = 10, 10       # 10 轨道 (5,5) per spin
SHOTS = 500
MAX_STRINGS = [12, 18, 27, 40, 60, 90, 130, 190, 252]   # theory §2.3.4 (full=252)
# eps 网格沿用 n2_ccpvdz 旧脚本 (敏感区间 7.5e-2..6.5e-2 加密)
EPS_HB = [1e-1, 9e-2, 8.5e-2, 8e-2, 7.5e-2, 7.3e-2, 7.2e-2, 7.1e-2,
          7.05e-2, 7.0e-2, 6.95e-2, 6.92e-2, 6.90e-2, 6.88e-2,
          6.86e-2, 6.84e-2, 6.82e-2, 6.80e-2, 6.6e-2, 6.4e-2,
          6.0e-2, 4e-2, 2e-2, 1e-2]


def _integrals():
    """N2/cc-pVDZ 10o CAS: 库自洽 FCI 参考 (与各方法同哈密顿量, 禁 CASCI 作参考)。"""
    m = pyscf.M(atom="N 0 0 -1.5; N 0 0 1.5", basis="cc-pVDZ", spin=0, verbose=0)
    mf = pyscf.scf.RHF(m); mf.kernel()
    cas = pyscf.mcscf.CASCI(mf, ncas=NCAS, nelecas=NELEC)
    h1e, ecore = cas.h1e_for_cas()
    ncore = int((mf.mo_occ.sum() - NELEC) // 2)
    eri = pyscf.ao2mo.full(m, mf.mo_coeff[:, ncore:ncore + NCAS],
                           aosym="1").reshape([NCAS] * 4)
    # 参考用库自身全空间 FCI (与各方法同哈密顿量, µHa floor 归零)
    e_fci = tc_sqd.compute_ground_state_energy(
        h1e, eri, NCAS, (NELEC // 2, NELEC // 2), ecore=ecore, method="fci")
    print(f"[ref] library self-consistent FCI={e_fci:.10f}")
    return h1e, eri, ecore, float(e_fci)


def _collect():
    h1e, eri, ecore, e_ref = _integrals()
    full = int(cistring.num_strings(NCAS, NELEC // 2)) ** 2
    assert full == 63504, f"坑2: full 期望 63504, 得到 {full}"
    print(f"N2/cc-pVDZ 10o 真基态参考 = {e_ref:.10f}  full={full}")
    P = {"c1v2_best": [], "best": [], "improved": [], "shci": []}
    if os.path.exists(NPY):
        P = np.load(NPY, allow_pickle=True).item()

    if not P["shci"]:
        for eps in EPS_HB:
            e_t, e_pt2, dim = tc_sqd.solve_hci(
                h1e, eri, NCAS, (NELEC // 2, NELEC // 2), eps_hb=eps,
                max_iter=15, ecore=ecore, return_details=True, verbose=False,
                backend="gpu")
            P["shci"].append((int(dim), float(abs(e_t - e_ref))))
            print(f"  SHCI eps={eps:.3e}: dim={dim} err={P['shci'][-1][1]:.2e}",
                  flush=True)
        np.save(NPY, P, allow_pickle=True)

    if not P["improved"]:
        for ms in MAX_STRINGS:
            b = np.random.default_rng(0).random((SHOTS, 2 * NCAS)) > 0.5
            p = np.full(SHOTS, 1.0 / SHOTS)
            e_c, det = tc_sqd.solve_sqd_improved(
                h1e, eri, NCAS, (NELEC // 2, NELEC // 2),
                bitstring_matrix=b, probabilities=p, max_strings=ms,
                n_active_per_round=30, rand_seed=0, ecore=ecore,
                return_details=True, verbose=False, backend="gpu")
            dim = int(det["dim"])
            P["improved"].append((dim, float(abs(e_c - e_ref))))
            print(f"  improved ms={ms}: dim={dim} err={P['improved'][-1][1]:.2e}",
                  flush=True)
        np.save(NPY, P, allow_pickle=True)

    if not P["best"]:
        for ms in MAX_STRINGS:
            out = tc_sqd.solve_sqd_best(
                h1e, eri, NCAS, (NELEC // 2, NELEC // 2), ecore=ecore,
                n_shots=SHOTS, max_strings=ms, rand_seed=0,
                return_details=True, verbose=False,
                tail_suppression=False, backend="gpu")
            dim = int(out["dim"])
            P["best"].append((dim, float(abs(out["energy"] - e_ref))))
            print(f"  best ms={ms}: dim={dim} err={P['best'][-1][1]:.2e}", flush=True)
        np.save(NPY, P, allow_pickle=True)

    if not P["c1v2_best"]:
        for ms in MAX_STRINGS:
            out = tc_sqd.solve_sqd_best(
                h1e, eri, NCAS, (NELEC // 2, NELEC // 2), ecore=ecore,
                n_shots=SHOTS, max_strings=ms, rand_seed=0,
                return_details=True, verbose=False,
                tail_suppression=True, tail_shots_ref=100, backend="gpu")
            dim = int(out["dim"])
            P["c1v2_best"].append((dim, float(abs(out["energy"] - e_ref))))
            print(f"  C1v2+best ms={ms}: dim={dim} err={P['c1v2_best'][-1][1]:.2e}",
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

    targets = list(np.geomspace(300, 63000, 15))
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
    ax.text(320, CHEM * 1.6, "chemical accuracy 1.6 mHa", fontsize=8, color="red")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("subspace dimension dim = len(str_a) * len(str_b)")
    ax.set_ylabel("energy error vs FCI (Ha)")
    ax.set_title("C1-v2 + best vs SHCI (N2/cc-pVDZ, R=3.0 A, 10o active)",
                 fontsize=12)
    ax.text(0.02, 0.02,
            "red vs green = C1 tail gain\ngreen solid vs dashed = evpt2 gain\nall vs brown = vs SHCI",
            transform=ax.transAxes, fontsize=7, color="grey", va="bottom")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(300, 80000); ax.set_ylim(2e-7, 0.5)
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
