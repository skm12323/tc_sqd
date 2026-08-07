"""fig_improved_sqd_vs_shci_c2_ccpvdz.png: C2/cc-pVDZ (10e,10o) 强关联。

与 N2/cc-pVDZ 版同构, 关键差异:
- **参考 = 库内全空间对角化 (真基态)**: C2/cc-pVDZ 近简并, CASCI/davidson
  收敛到虚高 9.3 mHa 的第二根, 必须用库 eigsh 全空间对角化取真基态作参考。
- **improved SQD 多 seed (3) 取 mean±std**: 采样驱动有涨落, 画误差带。
- SHCI (solve_hci) 确定性单次。

用法: python examples/plot_improved_sqd_vs_shci_c2_ccpvdz.py
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
NPY = os.path.join(BASE, "_plot_data_sqd_vs_shci_c2_ccpvdz.npy")
NCAS, NELEC = 10, 10
N_SEED = 3


def _integrals():
    m = pyscf.M(atom="C 0 0 0; C 0 0 1.24", basis="cc-pVDZ", spin=0, verbose=0)
    mf = pyscf.scf.RHF(m); mf.kernel()
    cas = pyscf.mcscf.CASCI(mf, ncas=NCAS, nelecas=NELEC)
    h1e, ecore = cas.h1e_for_cas()
    ncore = int((mf.mo_occ.sum() - NELEC) // 2)
    eri = pyscf.ao2mo.full(m, cas.mo_coeff[:, ncore:ncore + NCAS],
                           aosym="1").reshape([NCAS] * 4)
    return m, h1e, eri, ecore


def _true_gs(m, h1e, eri, ecore):
    """库内全空间对角化 (真基态, 规避 CASCI 近简并跳根)。"""
    all_str = cistring.make_strings(range(NCAS), NELEC // 2)
    res = tc_sqd.solve_sci((all_str, all_str), h1e, eri, NCAS, (5, 5))
    return res.energy + ecore


def _collect():
    m, h1e, eri, ecore = _integrals()
    e_ref = _true_gs(m, h1e, eri, ecore)
    full = int(cistring.num_strings(NCAS, NELEC // 2)) ** 2
    P = {"shci": [], "hci_ev": [], "active_pt2": [], "active_pt2_std": [],
         "active": [], "active_std": []}
    if os.path.exists(NPY):
        P = np.load(NPY, allow_pickle=True).item()
    print(f"C2/cc-pVDZ 真基态参考 = {e_ref:.10f}  full={full}")

    if not P["shci"]:
        for eps in [1e-1, 8e-2, 6e-2, 5e-2, 3e-2, 2e-2, 1.5e-2, 1e-2,
                    5e-3, 1e-3]:
            e_t, e_pt2, dim = tc_sqd.solve_hci(
                h1e, eri, NCAS, (5, 5), eps_hb=eps, max_iter=15,
                ecore=ecore, return_details=True, verbose=False)
            P["shci"].append((dim, abs(e_t - e_ref)))
            P["hci_ev"].append((dim, abs((e_t - e_pt2) - e_ref)))
            print(f"SHCI eps={eps:.3e}: dim={dim} err={P['shci'][-1][1]:.2e}",
                  flush=True)
        np.save(NPY, P, allow_pickle=True)

    if not P["active_pt2"]:
        grid = [(3, 40), (5, 70), (8, 100), (15, 150), (30, None), (60, None),
                (120, None), (300, None)]
        for s, ms in grid:
            errs_pt2, errs_direct, dims = [], [], []
            for seed in range(N_SEED):
                b = np.random.default_rng(seed).random((s, 2 * NCAS)) > 0.5
                p = np.full(s, 1.0 / s)
                e_c, det = tc_sqd.solve_sqd_ev(
                    h1e, eri, NCAS, (5, 5), bitstring_matrix=b,
                    probabilities=p, max_strings=ms, n_active_per_round=30,
                    rand_seed=seed, ecore=ecore, correction="pt2",
                    return_details=True, verbose=False)
                dims.append(det["dim"])
                errs_pt2.append(abs(e_c - e_ref))
                errs_direct.append(abs(det["E_direct"] - e_ref))
            dim = int(np.median(dims))
            P["active_pt2"].append((dim, float(np.mean(errs_pt2))))
            P["active_pt2_std"].append(float(np.std(errs_pt2)))
            P["active"].append((dim, float(np.mean(errs_direct))))
            P["active_std"].append(float(np.std(errs_direct)))
            print(f"SQD shots={s} ms={ms}: dim={dim} "
                  f"mean_err={np.mean(errs_pt2):.2e} "
                  f"std={np.std(errs_pt2):.1e}", flush=True)
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
    shci = _uniform(P["shci"], targets)
    hci_ev = _uniform(P["hci_ev"], targets)
    a_pt2 = sorted(zip(P["active_pt2"], P["active_pt2_std"]))
    active = sorted(zip(P["active"], P["active_std"]))

    fig, ax = plt.subplots(figsize=(8.5, 6))

    def plot(pts, label, color, marker, ls="-", z=3, ms=6):
        d = np.array(pts)
        if d.size == 0:
            return
        ax.plot(d[:, 0], d[:, 1], marker=marker, ls=ls, color=color,
                label=label, ms=ms, zorder=z, lw=1.8)

    # 多 seed 误差带 (mean ± std); pairs = [((dim, mean), std), ...]
    for pairs, color in [(active, "#2ca02c"), (a_pt2, "#2ca02c")]:
        if not pairs:
            continue
        dims = np.array([p[0][0] for p in pairs])
        means = np.array([p[0][1] for p in pairs])
        stds = np.array([p[1] for p in pairs])
        ax.fill_between(dims, np.maximum(means - stds, 1e-12),
                        means + stds, color=color, alpha=0.15, zorder=0)

    plot([p[0] for p in active], "solve_sqd_active (variational, 3-seed mean)",
         "#2ca02c", "o", ls=":", z=1)
    plot([p[0] for p in a_pt2],
         "improved SQD: active+PT2 (3-seed mean ± std)", "#2ca02c", "^", z=4)
    plot(hci_ev, "HCI variational E_V", "#8c564b", "v", ls=":", z=1)
    plot(shci, "SHCI E_V+E_PT2 (solve_hci / Dice)", "#8c564b", "v", z=3)

    ax.axvline(63504, color="grey", ls="--", lw=0.8)
    ax.text(66000, 3e-4, "full space\n63504", fontsize=7, color="grey")
    ax.axhline(CHEM, color="red", ls="--", lw=1.0, alpha=0.7)
    ax.text(550, CHEM * 1.6, "chemical accuracy 1.6 mHa", fontsize=8, color="red")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("subspace dimension (strings_a x strings_b)")
    ax.set_ylabel("energy error vs true ground state (Ha)")
    ax.set_title("improved SQD vs SHCI (C2/cc-pVDZ, 10o active)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(400, 80000)
    ax.set_ylim(1e-8, 0.5)
    ax.grid(True, which="both", alpha=0.3)

    out = os.path.join(BASE, "fig_improved_sqd_vs_shci_c2_ccpvdz.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("\nsaved", out)
    print("SHCI points:", shci)
    print("improved SQD (dim, mean, std):", a_pt2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="只用缓存出图")
    args = ap.parse_args()
    if args.plot:
        if not os.path.exists(NPY):
            raise SystemExit(f"无缓存 {NPY}, 先不带 --plot 跑收集。")
        P = np.load(NPY, allow_pickle=True).item()
    else:
        P = _collect()
    _plot(P)
