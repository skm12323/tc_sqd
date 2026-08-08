"""fig_improved_sqd_vs_shci_n2_1212.png: N2/cc-pVDZ (12e,12o) 强关联 (85万维)。

- 参考 = 库全空间对角化真基态 (存 _n2_1212_ints.npz, 3 分钟算得)
- SHCI (solve_hci) 确定性单次; SQD 多 seed (3) mean±std
- 展示库内方法在 12 轨道活性空间的能力边界 (受限 max_strings 下 PT2 枚举可控)

用法: python examples/plot_improved_sqd_vs_shci_n2_1212.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import tc_sqd  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEM = 1.6e-3
NPY = os.path.join(BASE, "_plot_data_sqd_vs_shci_n2_1212.npy")
INTS = os.path.join(BASE, "_n2_1212_ints.npz")
NCAS, NELEC = 12, 12
N_SEED = 3


def _load():
    d = np.load(INTS)
    return d["h1e"], d["eri"], float(d["ecore"]), float(d["e_ref"])


def _collect():
    h1e, eri, ecore, e_ref = _load()
    full = int(__import__('pyscf').fci.cistring.num_strings(NCAS, 6)) ** 2
    P = {"shci": [], "hci_ev": [], "active_pt2": [], "active_pt2_std": [],
         "active": [], "active_std": []}
    if os.path.exists(NPY):
        P = np.load(NPY, allow_pickle=True).item()
    print(f"N2 (12,12) 真基态 = {e_ref:.10f}  full={full}")

    if not P["shci"]:
        for eps in [5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 1e-3]:
            e_t, e_pt2, dim = tc_sqd.solve_hci(
                h1e, eri, NCAS, (6, 6), eps_hb=eps, max_iter=12,
                ecore=ecore, return_details=True, verbose=False)
            P["shci"].append((dim, abs(e_t - e_ref)))
            P["hci_ev"].append((dim, abs((e_t - e_pt2) - e_ref)))
            print(f"SHCI eps={eps:.3e}: dim={dim} err={P['shci'][-1][1]:.2e}",
                  flush=True)
        np.save(NPY, P, allow_pickle=True)

    if not P["active_pt2"]:
        grid = [(10, 100), (20, 150), (20, 200), (40, None), (80, None),
                (200, None), (500, None)]
        for s, ms in grid:
            errs_pt2, errs_direct, dims = [], [], []
            for seed in range(N_SEED):
                b = np.random.default_rng(seed).random((s, 2 * NCAS)) > 0.5
                p = np.full(s, 1.0 / s)
                e_c, det = tc_sqd.solve_sqd_ev(
                    h1e, eri, NCAS, (6, 6), bitstring_matrix=b,
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
                  f"mean={np.mean(errs_pt2):.2e} std={np.std(errs_pt2):.1e}",
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

    targets = list(np.geomspace(1000, 850000, 12))
    shci = _uniform(P["shci"], targets)
    hci_ev = _uniform(P["hci_ev"], targets)
    active = sorted(zip(P["active"], P["active_std"]))
    a_pt2 = sorted(zip(P["active_pt2"], P["active_pt2_std"]))

    fig, ax = plt.subplots(figsize=(8.5, 6))

    for pairs, color in [(active, "#2ca02c"), (a_pt2, "#2ca02c")]:
        if not pairs:
            continue
        dims = np.array([p[0][0] for p in pairs])
        means = np.array([p[0][1] for p in pairs])
        stds = np.array([p[1] for p in pairs])
        ax.fill_between(dims, np.maximum(means - stds, 1e-12),
                        means + stds, color=color, alpha=0.15, zorder=0)

    def plot(pts, label, color, marker, ls="-", z=3, ms=6):
        d = np.array(pts)
        if d.size == 0:
            return
        ax.plot(d[:, 0], d[:, 1], marker=marker, ls=ls, color=color,
                label=label, ms=ms, zorder=z, lw=1.8)

    plot([p[0] for p in active], "solve_sqd_active (variational, 3-seed)",
         "#2ca02c", "o", ls=":", z=1)
    plot([p[0] for p in a_pt2],
         "improved SQD: active+PT2 (3-seed mean ± std)", "#2ca02c", "^", z=4)
    plot(hci_ev, "HCI variational E_V", "#8c564b", "v", ls=":", z=1)
    plot(shci, "SHCI E_V+E_PT2 (solve_hci)", "#8c564b", "v", z=3)

    # Dice 交叉验证标注: shciscf 无变分空间维度 API, 无法画完整曲线;
    # Dice 收敛到 E=-108.768185 (eps->1e-6 平台), 与库参考差 ~0.5 mHa
    # (853,776 维 FCI 参考口径/收敛差异; 小规模 C2/cc-pVDZ 曾一致到 0.003 mHa)。
    ax.text(7e3, 1.5e-4,
            "Dice SHCI (cross-check):\nE -> -108.768185\n(~0.5 mHa above ref.:\n853k-dim FCI precision)",
            fontsize=7, color="blue", ha="left", va="top")

    ax.axvline(853776, color="grey", ls="--", lw=0.8)
    ax.text(880000, 3e-3, "full space\n853776", fontsize=7, color="grey")
    ax.axhline(CHEM, color="red", ls="--", lw=1.0, alpha=0.7)
    ax.text(1100, CHEM * 1.6, "chemical accuracy 1.6 mHa", fontsize=8, color="red")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("subspace dimension (strings_a x strings_b)")
    ax.set_ylabel("energy error vs true ground state (Ha)")
    ax.set_title("improved SQD vs SHCI (N2/cc-pVDZ, 12o active, 85.4e4)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(900, 1e6)
    ax.set_ylim(1e-9, 0.5)
    ax.grid(True, which="both", alpha=0.3)

    out = os.path.join(BASE, "fig_improved_sqd_vs_shci_n2_1212.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("\nsaved", out)
    print("SHCI points:", shci)
    print("SQD (dim, mean, std):", a_pt2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="只用缓存出图")
    args = ap.parse_args()
    if args.plot:
        if not os.path.exists(NPY):
            raise SystemExit(f"无缓存 {NPY}")
        P = np.load(NPY, allow_pickle=True).item()
    else:
        P = _collect()
    _plot(P)
