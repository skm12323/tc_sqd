"""fig_improved_sqd_vs_shci_n2_ccpvdz.png: N2/cc-pVDZ (10e,10o) @ R=3.0A。

强关联近解离, 全空间 63504。SHCI 用库内 solve_hci (pyhci/Dice 等价)。
取点均匀: 目标 log 均匀维度 + 最近点匹配 (SHCI eps 网格含敏感区间加密)。
注意: N2/cc-pVDZ 的 SHCI 维度对 eps_hb 极敏感 (7e-2->3481, 6.9e-2->40401,
6.8e-2->57600), 故 eps 网格在 7.2e-2..6.5e-2 加密。
用法: python examples/plot_improved_sqd_vs_shci_n2_ccpvdz.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import tc_sqd  # noqa: E402
import pyscf, pyscf.scf, pyscf.mcscf, pyscf.ao2mo  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEM = 1.6e-3
NPY = os.path.join(BASE, "_plot_data_sqd_vs_shci_n2_ccpvdz.npy")
NCAS, NELEC = 10, 10


def _integrals():
    m = pyscf.M(atom="N 0 0 -1.5; N 0 0 1.5", basis="cc-pVDZ", spin=0, verbose=0)
    mf = pyscf.scf.RHF(m); mf.kernel()
    cas = pyscf.mcscf.CASCI(mf, ncas=NCAS, nelecas=NELEC)
    h1e, ecore = cas.h1e_for_cas()
    ncore = int((mf.mo_occ.sum() - NELEC) // 2)
    eri = pyscf.ao2mo.full(m, mf.mo_coeff[:, ncore:ncore + NCAS],
                           aosym="1").reshape([NCAS] * 4)
    e_fci, _, _, _, _ = cas.kernel(verbose=0)
    return h1e, eri, ecore, e_fci


def _collect():
    h1e, eri, ecore, e_fci = _integrals()
    full = int(__import__('pyscf').fci.cistring.num_strings(NCAS, NELEC // 2)) ** 2
    P = {"shci": [], "hci_ev": [], "active_pt2": [], "active": []}
    if os.path.exists(NPY):
        P = np.load(NPY, allow_pickle=True).item()
    print(f"N2/cc-pVDZ R=3.0 FCI={e_fci:.6f} full={full}")

    if not P["shci"]:
        for eps in [1e-1, 8e-2, 7.2e-2, 7e-2, 6.95e-2, 6.9e-2,
                    6.85e-2, 6.8e-2, 6.5e-2, 3e-2]:
            e_t, e_pt2, dim = tc_sqd.solve_hci(
                h1e, eri, NCAS, (NELEC // 2, NELEC // 2), eps_hb=eps,
                max_iter=15, ecore=ecore, return_details=True, verbose=False)
            P["shci"].append((dim, abs(e_t - e_fci)))
            P["hci_ev"].append((dim, abs((e_t - e_pt2) - e_fci)))
            print(f"SHCI eps={eps:.1e}: dim={dim} "
                  f"errSHCI={P['shci'][-1][1]:.2e}", flush=True)
        np.save(NPY, P, allow_pickle=True)

    if not P["active_pt2"]:
        grid = [(4, 60), (8, 90), (15, 120), (30, 200), (60, None),
                (120, None), (300, None), (800, None)]
        for s, ms in grid:
            b = np.random.default_rng(0).random((s, 2 * NCAS)) > 0.5
            p = np.full(s, 1.0 / s)
            e_c, det = tc_sqd.solve_sqd_ev(
                h1e, eri, NCAS, (NELEC // 2, NELEC // 2),
                bitstring_matrix=b, probabilities=p, max_strings=ms,
                n_active_per_round=30, rand_seed=0, ecore=ecore,
                correction="pt2", return_details=True, verbose=False)
            dim = det["dim"]
            P["active_pt2"].append((dim, abs(e_c - e_fci)))
            P["active"].append((dim, abs(det["E_direct"] - e_fci)))
            print(f"SQD shots={s} ms={ms}: dim={dim} "
                  f"errPT2={P['active_pt2'][-1][1]:.2e} "
                  f"errDirect={P['active'][-1][1]:.2e}", flush=True)
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

    targets = list(np.geomspace(300, 63000, 10))
    shci = _uniform(P["shci"], targets)
    hci_ev = _uniform(P["hci_ev"], targets)
    a_pt2 = _uniform(P["active_pt2"], targets)
    active = _uniform(P["active"], targets)

    fig, ax = plt.subplots(figsize=(8.5, 6))

    def plot(pts, label, color, marker, ls="-", z=3, ms=6):
        d = np.array(pts)
        if d.size == 0:
            return
        ax.plot(d[:, 0], d[:, 1], marker=marker, ls=ls, color=color,
                label=label, ms=ms, zorder=z, lw=1.8)

    plot(active, "solve_sqd_active (variational)", "#2ca02c", "o", ls=":", z=1)
    plot(a_pt2, "improved SQD: active + PT2 (E+E_PT2)", "#2ca02c", "^", z=4)
    plot(hci_ev, "HCI variational E_V", "#8c564b", "v", ls=":", z=1)
    plot(shci, "SHCI E_V+E_PT2 (pyhci/Dice equivalent)", "#8c564b", "v", z=3)

    ax.axvline(63504, color="grey", ls="--", lw=0.8)
    ax.text(66000, 3e-4, "full space\n63504", fontsize=7, color="grey")
    ax.axhline(CHEM, color="red", ls="--", lw=1.0, alpha=0.7)
    ax.text(320, CHEM * 1.6, "chemical accuracy 1.6 mHa", fontsize=8, color="red")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("subspace dimension (strings_a x strings_b)")
    ax.set_ylabel("energy error vs FCI (Ha)")
    ax.set_title("improved SQD vs SHCI (N2/cc-pVDZ, R=3.0 A)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(300, 80000)
    ax.set_ylim(1e-5, 0.5)
    ax.grid(True, which="both", alpha=0.3)

    out = os.path.join(BASE, "fig_improved_sqd_vs_shci_n2_ccpvdz.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("\nsaved", out)
    print("SHCI points:", shci)
    print("improved SQD points:", a_pt2)


if __name__ == "__main__":
    _plot(_collect())
