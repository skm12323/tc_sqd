"""fig_improved_sqd_vs_shci_n2.png: improved SQD vs SHCI (N2/STO-3G 拉伸)。

与 C2 版 (plot_improved_sqd_vs_shci.py) 同构, 体系换 N2/STO-3G 拉伸
(全空间 14400)。SHCI 用库内 `solve_hci`——与 pyhci/Dice 原生 SHCI 算法一致
(Holmes 2016 / Sharma 2017, heat-bath 选态 + PT2 修正)。
取点均匀: 目标 log 均匀维度 + 最近点匹配。
用法: python examples/plot_improved_sqd_vs_shci_n2.py
英文标签 (matplotlib 缺 CJK 字体)。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import tc_sqd  # noqa: E402
from pyscf import gto  # noqa: E402
from pyscf.fci import direct_spin1  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEM = 1.6e-3
NPY = os.path.join(BASE, "_plot_data_sqd_vs_shci_n2.npy")


def _collect():
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_fci, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    full = int(__import__('pyscf').fci.cistring.num_strings(norb, nelec[0])) ** 2

    P = {"shci": [], "hci_ev": [], "active_pt2": [], "active": []}
    if os.path.exists(NPY):
        P = np.load(NPY, allow_pickle=True).item()
    print(f"N2 FCI={e_fci:.6f} full_dim={full}")

    # SHCI / HCI E_V: eps_hb 网格 (维度 ~2500 -> 12544)
    if not P["shci"]:
        for eps in [1e-1, 5e-2, 3e-2, 2e-2, 1.5e-2, 1e-2, 5e-3, 1e-3]:
            e_t, e_pt2, dim = tc_sqd.solve_hci(
                h1e, eri, norb, nelec, eps_hb=eps, max_iter=20,
                return_details=True, verbose=False)
            P["shci"].append((dim, abs(e_t - e_fci)))
            P["hci_ev"].append((dim, abs((e_t - e_pt2) - e_fci)))
            print(f"SHCI eps={eps:.1e}: dim={dim} errSHCI={P['shci'][-1][1]:.2e}")
        np.save(NPY, P, allow_pickle=True)

    # improved SQD: (shots, max_strings) 网格
    if not P["active_pt2"]:
        grid = [(4, 50), (4, 80), (4, 120), (6, None), (10, None), (25, None),
                (80, None), (150, None), (400, None), (1000, None), (2000, None)]
        for s, ms in grid:
            b = np.random.default_rng(0).random((s, 2 * norb)) > 0.5
            p = np.full(s, 1.0 / s)
            e_c, det = tc_sqd.solve_sqd_ev(
                h1e, eri, norb, nelec, bitstring_matrix=b, probabilities=p,
                max_strings=ms, n_active_per_round=30, rand_seed=0,
                correction="pt2", return_details=True, verbose=False)
            dim = det["dim"]
            P["active_pt2"].append((dim, abs(e_c - e_fci)))
            P["active"].append((dim, abs(det["E_direct"] - e_fci)))
            print(f"SQD shots={s} ms={ms}: dim={dim} "
                  f"errPT2={P['active_pt2'][-1][1]:.2e} "
                  f"errDirect={P['active'][-1][1]:.2e}")
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

    targets = list(np.geomspace(1200, 14000, 11))
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

    ax.axvline(14400, color="grey", ls="--", lw=0.8)
    ax.text(14800, 3e-9, "full space\n14400", fontsize=7, color="grey")
    ax.axhline(CHEM, color="red", ls="--", lw=1.0, alpha=0.7)
    ax.text(1300, CHEM * 1.6, "chemical accuracy 1.6 mHa", fontsize=8, color="red")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("subspace dimension (strings_a x strings_b)")
    ax.set_ylabel("energy error vs FCI (Ha)")
    ax.set_title("improved SQD vs SHCI (N2/STO-3G stretch)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(1000, 20000)
    ax.set_ylim(1e-10, 0.5)
    ax.grid(True, which="both", alpha=0.3)

    out = os.path.join(BASE, "fig_improved_sqd_vs_shci_n2.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("\nsaved", out)
    print("SHCI points:", shci)
    print("improved SQD points:", a_pt2)


if __name__ == "__main__":
    _plot(_collect())
