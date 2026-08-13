"""fig_c1v2_best_vs_shci_n2_ccpvdz_10o_shots.png: C1-v2 + best vs SHCI (N2/cc-pVDZ 10o), 扫 shots。

round_006-r3b shots 扫描轮: 旧 max_strings 版里 C1-v2+best 的 tail 发现不受
max_strings gate, 所有点 dim 相同 (全空间饱和), 曲线退化为单点。改为扫 shots:
不同 shots 产生不同 dim → 曲线分散可读。

体系: N2/cc-pVDZ @ R=3.0 A (强关联近解离), 全空间 63504 = C(10,5)^2 = 252^2。
活性空间核对 (theory §2.3.6): 10 轨道 nelec=(5,5)。

扫描轴: SHOTS_LIST = [10, 30, 50, 100, 300, 1000] (覆盖 2 个数量级)。
  - SQD 三曲线 (improved/best/C1-v2+best) 扫 shots, max_strings=None (不限,
    让采样自然决定 dim); 不同 shots → 不同 dim。
  - SHCI 不变 (确定性选态, 与 shots 无关, 仍扫 eps_hb)。

自适应 seed 逻辑 (M0): SQD 三曲线在每个 shots 点先跑 3 个 seed (0,1,2),
按 max(err)/min(err) 比值判断稳定性; 比值 < 5 → 后续 shots 点只跑 seed=0。
  - 3-seed 点画 mean±std 误差带 (log 轴下限裁正);
  - 单 seed 点画单值 (无误差带);
  - 一旦稳定, 后续不再回 3-seed。SHCI 不受影响 (确定性单次)。

4 曲线 (误差 vs 实际对角化维度 dim, log-log):
  - **C1-v2 + best** (红 ^ 实线): solve_sqd_best(tail_suppression=True,
    tail_shots_ref=100, backend="gpu") —— C1 采样覆盖 + evpt2 精修 (核心)
  - **best** (绿 o 实线): solve_sqd_best(tail_suppression=False, backend="gpu")
  - **improved** (绿 -- 虚线): solve_sqd_improved(backend="gpu")
  - **SHCI** (棕 v 实线): solve_hci(backend="gpu")

x 轴口径 (theory §2.3.3 坑 1): 一律用实际 dim, 不用 shots。
y 轴扩展到 1e-15~1e0 (让 C1-v2+best 的低 err 可见); C1-v2+best 点 err < y 下限时
用红色下箭头标注。
参考 (theory §2.3.5): 库全空间对角化真基态 (compute_ground_state_energy method="fci",
与各方法同哈密顿量, 禁 CASCI)。

用法:
  python examples/plot_c1v2_best_vs_shci_n2_ccpvdz_10o_shots.py
  python examples/plot_c1v2_best_vs_shci_n2_ccpvdz_10o_shots.py --plot
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
NPY = os.path.join(BASE, "_plot_data_c1v2_best_vs_shci_n2_ccpvdz_10o_shots.npy")
FIG = os.path.join(BASE, "fig_c1v2_best_vs_shci_n2_ccpvdz_10o_shots.png")
NCAS, NELEC = 10, 10       # 10 轨道 (5,5) per spin
SHOTS_LIST = [10, 30, 50, 100, 300, 1000]   # 扫 shots, 覆盖 2 个数量级; max_strings=None
EVAL_SEEDS = [0, 1, 2]          # 稳定前 3-seed 评估
SEED_RATIO_STABLE = 5.0         # max(err)/min(err) < 5 → 稳定 → 后续单 seed
# eps 网格沿用 n2_ccpvdz 旧脚本 (敏感区间 7.5e-2..6.5e-2 加密)
EPS_HB = [1e-1, 9e-2, 8.5e-2, 8e-2, 7.5e-2, 7.3e-2, 7.2e-2, 7.1e-2,
          7.05e-2, 7.0e-2, 6.95e-2, 6.92e-2, 6.90e-2, 6.88e-2,
          6.86e-2, 6.84e-2, 6.82e-2, 6.80e-2, 6.6e-2, 6.4e-2,
          6.0e-2, 4e-2, 2e-2, 1e-2]
Y_LO, Y_HI = 1e-15, 1e0   # y 轴扩展 (让 C1-v2+best 的低 err 可见)


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


def _collect_adaptive(P, key, runner):
    """收集一条 SQD 曲线 (自适应 3-seed→1-seed)。

    runner(shots, seed) -> (dim, err)。每点存 (dim_mean, err_mean, err_std, n_seeds):
    稳定前每 shots 点跑 EVAL_SEEDS (3 seed), 算 max/min err 比值; < SEED_RATIO_STABLE
    则置 stable, 后续单 seed。整条曲线跑完才落盘 (与旧脚本一致, 避免半成品缓存)。
    """
    if P[key]:
        return
    stable = False
    for shots in SHOTS_LIST:
        seeds = [0] if stable else list(EVAL_SEEDS)
        dims, errs = [], []
        for seed in seeds:
            dim, err = runner(shots, seed)
            dims.append(dim); errs.append(err)
            print(f"  {key} shots={shots} seed={seed}: dim={dim} err={err:.2e}",
                  flush=True)
        dim_m = float(np.mean(dims))
        err_m = float(np.mean(errs))
        n = len(seeds)
        if n >= 2:
            err_std = float(np.std(errs))
            ratio = (max(errs) / min(errs)) if min(errs) > 0 else float("inf")
            if ratio < SEED_RATIO_STABLE:
                stable = True
            print(f"    -> {key} shots={shots}: 3-seed ratio={ratio:.2f} "
                  f"stable={'Y' if stable else 'N'}", flush=True)
        else:
            err_std = 0.0
        P[key].append((dim_m, err_m, err_std, n))
    np.save(NPY, P, allow_pickle=True)


def _collect():
    h1e, eri, ecore, e_ref = _integrals()
    full = int(cistring.num_strings(NCAS, NELEC // 2)) ** 2
    assert full == 63504, f"坑2: full 期望 63504, 得到 {full}"
    print(f"N2/cc-pVDZ 10o 真基态参考 = {e_ref:.10f}  full={full}")
    P = {"c1v2_best": [], "best": [], "improved": [], "shci": []}
    if os.path.exists(NPY):
        P = np.load(NPY, allow_pickle=True).item()

    # ---- SHCI 曲线 (棕): 确定性选态, 与 shots/seed 无关, 仍扫 eps_hb ----
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

    # ---- improved SQD (绿虚): runner 自适应 ----
    def _run_improved(shots, seed):
        b = np.random.default_rng(seed).random((shots, 2 * NCAS)) > 0.5
        p = np.full(shots, 1.0 / shots)
        e_c, det = tc_sqd.solve_sqd_improved(
            h1e, eri, NCAS, (NELEC // 2, NELEC // 2),
            bitstring_matrix=b, probabilities=p, max_strings=None,
            n_active_per_round=30, rand_seed=seed, ecore=ecore,
            return_details=True, verbose=False, backend="gpu")
        return int(det["dim"]), float(abs(e_c - e_ref))

    # ---- best (绿实): runner 自适应 ----
    def _run_best(shots, seed):
        out = tc_sqd.solve_sqd_best(
            h1e, eri, NCAS, (NELEC // 2, NELEC // 2), ecore=ecore,
            n_shots=shots, max_strings=None, rand_seed=seed,
            return_details=True, verbose=False,
            tail_suppression=False, backend="gpu")
        return int(out["dim"]), float(abs(out["energy"] - e_ref))

    # ---- C1-v2 + best (红, 核心): runner 自适应 ----
    def _run_c1v2(shots, seed):
        out = tc_sqd.solve_sqd_best(
            h1e, eri, NCAS, (NELEC // 2, NELEC // 2), ecore=ecore,
            n_shots=shots, max_strings=None, rand_seed=seed,
            return_details=True, verbose=False,
            tail_suppression=True, tail_shots_ref=100, backend="gpu")
        return int(out["dim"]), float(abs(out["energy"] - e_ref))

    _collect_adaptive(P, "improved", _run_improved)
    _collect_adaptive(P, "best", _run_best)
    _collect_adaptive(P, "c1v2_best", _run_c1v2)
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


def _dedup_by_dim(points):
    """(dim, err_mean, err_std, n_seeds) 按 round(dim) 去重, 保留最低 dim (≈最早 shots)。"""
    seen, out = set(), []
    for d, em, es, n in sorted(points, key=lambda x: float(x[0])):
        dk = round(float(d), 0)
        if dk in seen:
            continue
        seen.add(dk)
        out.append((float(d), float(em), float(es), int(n)))
    return out


def _plot(P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = list(np.geomspace(300, 63000, 15))
    c1v2 = _dedup_by_dim(P["c1v2_best"])
    best = _dedup_by_dim(P["best"])
    improved = _dedup_by_dim(P["improved"])
    shci = _uniform(P["shci"], targets)

    fig, ax = plt.subplots(figsize=(8.5, 6))

    def plot(pts, label, color, marker, ls="-", z=3, ms=6):
        """画 mean 折线; 对 3-seed 点叠加 mean±std 垂直误差带 (log 轴下限裁正)。
        返回 3-seed 点数。"""
        if not pts:
            return 0
        d = np.array([p[0] for p in pts])
        em = np.array([p[1] for p in pts])
        ax.plot(d, em, marker=marker, ls=ls, color=color, label=label,
                ms=ms, zorder=z, lw=1.8)
        multi = [(p[0], p[1], p[2]) for p in pts if p[3] >= 2 and p[2] > 0]
        if multi:
            dm = np.array([m[0] for m in multi])
            emm = np.array([m[1] for m in multi])
            esm = np.array([m[2] for m in multi])
            low = np.maximum(emm - esm, Y_LO * 0.5)   # 裁正, 防 log 轴非正值
            yerr = np.vstack([emm - low, esm])
            ax.errorbar(dm, emm, yerr=yerr, fmt="none", ecolor=color,
                        elinewidth=1.0, capsize=3, zorder=z - 1, alpha=0.85)
        return len(multi)

    n_c1v2 = plot(c1v2, "C1-v2 + best (tail+C1, evpt2)", "#d62728", "^", ls="-", z=5)
    n_best = plot(best, "best (evpt2, no C1)", "#2ca02c", "o", ls="-", z=4)
    n_imp = plot(improved, "improved (active+PT2)", "#2ca02c", "o", ls="--", z=3)
    # SHCI 单值曲线 (无误差带)
    sh = np.array(shci)
    if sh.size:
        ax.plot(sh[:, 0], sh[:, 1], marker="v", ls="-", color="#8c564b",
                label="SHCI E_V+E_PT2 (solve_hci)", ms=6, zorder=2, lw=1.8)

    ax.axvline(63504, color="grey", ls="--", lw=0.8)
    ax.text(66000, 3e-4, "full space\n63504", fontsize=7, color="grey")
    ax.axhline(CHEM, color="red", ls="--", lw=1.0, alpha=0.7)
    ax.text(320, CHEM * 1.6, "chemical accuracy 1.6 mHa", fontsize=8, color="red")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("subspace dimension dim = len(str_a) * len(str_b)")
    ax.set_ylabel("energy error vs FCI (Ha)")
    ax.set_title("C1-v2 + best vs SHCI (N2/cc-pVDZ, R=3.0 A, 10o active), shots scan",
                 fontsize=11)
    ax.text(0.02, 0.02,
            "red vs green = C1 tail gain\ngreen solid vs dashed = evpt2 gain\nall vs brown = vs SHCI",
            transform=ax.transAxes, fontsize=7, color="grey", va="bottom")
    ax.text(0.02, 0.98,
            f"adaptive seeds: 3 seeds {EVAL_SEEDS} -> single once max/min err < {SEED_RATIO_STABLE:.0f}x\n"
            f"3-seed pts: C1v2={n_c1v2} best={n_best} improved={n_imp}\n"
            f"bands = mean +/- std (log lower clipped)",
            transform=ax.transAxes, fontsize=6, color="grey", va="top")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(300, 80000); ax.set_ylim(Y_LO, Y_HI)
    ax.grid(True, which="both", alpha=0.3)

    # C1-v2+best 点 err < y 下限时用红色下箭头标注 (实际 err 太低, 落在轴外)
    for d, em, es, n in c1v2:
        if em < Y_LO:
            ax.annotate("", xy=(d, Y_LO), xytext=(d, Y_LO * 30),
                        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6))
            ax.text(d, Y_LO * 60, f"{em:.1e}", fontsize=6, color="#d62728",
                    ha="center", va="bottom", rotation=90)

    fig.tight_layout(); fig.savefig(FIG, dpi=150)
    print(f"\nsaved {FIG}")
    print("C1-v2+best points:", c1v2)
    print("best points:", best)
    print("improved points:", improved)
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
