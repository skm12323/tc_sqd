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
    # ⑥ 自洽参考: 库自身全空间 FCI 在 (h1e, eri) 上 —— 与各方法同哈密顿量, µHa floor 归零。
    #    (旧版用 cas.kernel() 作参考, 其内部 eri 路径与库 eri 差 ~µHa → 虚假 common floor。
    #     见 REVIEW "µHa 哈密顿量构造偏移诊断"。)
    e_fci = tc_sqd.compute_ground_state_energy(
        h1e, eri, NCAS, (NELEC // 2, NELEC // 2), ecore=ecore, method="fci")
    # CASCI 独立交叉校验 (其内部 eri 与库 eri 差 ~µHa, 仅作一致性印证, 不作参考)
    e_casci = cas.kernel(verbose=0)[0]
    print(f"[ref] library self-consistent FCI={e_fci:.10f}  "
          f"CASCI cross-check={e_casci:.10f}  |diff|={abs(e_fci - e_casci):.2e} (µHa, 预期)")
    return h1e, eri, ecore, e_fci


def _collect():
    h1e, eri, ecore, e_fci = _integrals()
    full = int(__import__('pyscf').fci.cistring.num_strings(NCAS, NELEC // 2)) ** 2
    P = {"shci": [], "hci_ev": [], "active_pt2": [], "active": []}
    if os.path.exists(NPY):
        P = np.load(NPY, allow_pickle=True).item()
    print(f"N2/cc-pVDZ R=3.0 FCI={e_fci:.6f} full={full}")

    if not P["shci"]:
        # SHCI: eps 加密 (敏感区间 7.5e-2..6.5e-2 细扫, 覆盖 324->58564 中间维度)
        for eps in [1e-1, 9e-2, 8.5e-2, 8e-2, 7.5e-2, 7.3e-2, 7.2e-2, 7.1e-2,
                    7.05e-2, 7.0e-2, 6.95e-2, 6.92e-2, 6.90e-2, 6.88e-2,
                    6.86e-2, 6.84e-2, 6.82e-2, 6.80e-2, 6.6e-2, 6.4e-2,
                    6.0e-2, 4e-2, 2e-2, 1e-2]:
            e_t, e_pt2, dim = tc_sqd.solve_hci(
                h1e, eri, NCAS, (NELEC // 2, NELEC // 2), eps_hb=eps,
                max_iter=15, ecore=ecore, return_details=True, verbose=False)
            P["shci"].append((dim, abs(e_t - e_fci)))
            P["hci_ev"].append((dim, abs((e_t - e_pt2) - e_fci)))
            print(f"SHCI eps={eps:.3e}: dim={dim} "
                  f"errSHCI={P['shci'][-1][1]:.2e}", flush=True)
        np.save(NPY, P, allow_pickle=True) 

    if not P["active_pt2"]:
        # SQD: shots x max_strings 加密 (维度 ~2500 -> 63000)
        grid = [(3, 40), (4, 60), (5, 80), (8, 100), (12, 130), (15, 160),
                (20, 200), (30, 250), (40, None), (60, None), (80, None),
                (120, None), (200, None), (400, None), (800, None)]
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

    targets = list(np.geomspace(300, 63000, 15))
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
    # 下界放到数据最小量级以下 (min ~2e-6), 避免裁剪高维度趋同点
    ax.set_ylim(2e-7, 0.5)
    # 参考 = 库自洽全空间 FCI (与各方法同哈密顿量, 无虚假 floor);
    # CASCI 独立交叉校验与库一致到 µHa (eri 构造路径差异, 见 REVIEW ⑥)。
    ax.text(1.5e4, 9e-6,
            "reference: library\nself-consistent FCI\n(CASCI agrees\nwithin ~µHa)",
            fontsize=7, color="grey", ha="center", va="center")
    ax.grid(True, which="both", alpha=0.3)

    out = os.path.join(BASE, "fig_improved_sqd_vs_shci_n2_ccpvdz.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("\nsaved", out)
    print("SHCI points:", shci)
    print("improved SQD points:", a_pt2)


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
