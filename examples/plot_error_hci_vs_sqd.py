"""再生成 `fig_error_hci_vs_sqd.png`: HCI/SHCI vs SQD 家族 (N2/STO-3G 拉伸)。

曲线 (误差 vs 子空间维度, log-log, FCI 参考):
  - HCI variational E_V (虚线) 与 SHCI E_V+E_PT2 (实线): `solve_hci` eps_hb 网格
  - traditional SQD: `solve_sqd` 不同 shots
  - solve_sqd_active (变分): max_strings 网格
  - **improved SQD: active + PT2 (E+E_PT2)** (`solve_sqd_ev` correction="pt2",
    方向 D 改进): 同 max_strings 下误差低于 active 直接
  - FCI-NO top-K: 自然轨道基 top-K det (确定性选态上限)
  - CCSD 参照 (err 0.10 Ha, 单参考拉伸失效)

用法:
    python examples/plot_error_hci_vs_sqd.py        # 数据收集 + 出图 (~15 min)
    python examples/plot_error_hci_vs_sqd.py --plot # 只用缓存数据出图
英文标签 (matplotlib 缺 CJK 字体)。
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import tc_sqd  # noqa: E402
from pyscf import gto  # noqa: E402
from pyscf.fci import direct_spin1, cistring  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPY = os.path.join(BASE, "_plot_data_hci_v2.npy")
CHEM = 1.6e-3


def _collect():
    """确定性 (seed 0) 收集各曲线数据, 返回 dict {key: [(dim, err), ...]}。"""
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", verbose=0)
    data = tc_sqd.from_pyscf(mol)
    h1e, eri, norb, nelec = data.h1e, data.eri, data.norb, data.nelec
    e_fci, _ = direct_spin1.kernel(h1e, eri, norb, nelec, conv_tol=1e-12)
    na, nb = nelec

    P = {"hci_ev": [], "shci": [], "sqd": [], "active": [], "active_pt2": [],
         "fci_no": []}
    if os.path.exists(NPY):
        P = np.load(NPY, allow_pickle=True).item()

    n_shots = 100
    bsm = np.random.default_rng(0).random((n_shots, 2 * norb)) > 0.5
    probs = np.full(n_shots, 1.0 / n_shots)

    if not P["shci"]:
        # SHCI/HCI: 更密 eps_hb 网格 (5e-2..1e-3, 覆盖 dim 3481~14400;
        # 5e-2 与 3e-2 之间跳跃陡, 加插值点平滑)
        for eps in [5e-2, 4e-2, 3e-2, 2.5e-2, 2e-2, 1.5e-2, 1e-2, 5e-3, 2e-3, 1e-3]:
            e_t, e_pt2, dim = tc_sqd.solve_hci(
                h1e, eri, norb, nelec, eps_hb=eps, max_iter=20,
                return_details=True, verbose=False)
            P["shci"].append((dim, abs(e_t - e_fci)))
            P["hci_ev"].append((dim, abs((e_t - e_pt2) - e_fci)))
            print(f"HCI eps={eps:.1e}: dim={dim} "
                  f"errV={P['hci_ev'][-1][1]:.2e} errSHCI={P['shci'][-1][1]:.2e}")
        np.save(NPY, P, allow_pickle=True)

    if not P["sqd"]:
        # traditional SQD: 更多 shots
        for s in [60, 100, 200, 500, 1000, 2000]:
            b = np.random.default_rng(0).random((s, 2 * norb)) > 0.5
            p = np.full(s, 1.0 / s)
            r = tc_sqd.solve_sqd(h1e, eri, norb, nelec, bitstring_matrix=b,
                                 probabilities=p, mode="single", seed=0)
            dim = len(r.sci_state.ci_strs_a) * len(r.sci_state.ci_strs_b)
            P["sqd"].append((dim, abs(r.energy - e_fci)))
            print(f"SQD shots={s}: dim={dim} err={P['sqd'][-1][1]:.2e}")
        np.save(NPY, P, allow_pickle=True)

    if not P["active"]:
        # active / active+PT2: 扫采样量 shots (维度随覆盖自然变化,
        # 避免 max_strings 被采样字符串主导而重复维度)
        for s in [30, 60, 100, 200, 400]:
            b = np.random.default_rng(0).random((s, 2 * norb)) > 0.5
            p = np.full(s, 1.0 / s)
            traj = []
            e_a = tc_sqd.solve_sqd_active(
                h1e, eri, norb, nelec, bitstring_matrix=b,
                probabilities=p, max_strings=None, n_active_per_round=30,
                rand_seed=0, trajectory=traj, verbose=False)
            dim = traj[-1]["dim"]
            P["active"].append((dim, abs(e_a - e_fci)))
            e_c, det = tc_sqd.solve_sqd_ev(
                h1e, eri, norb, nelec, bitstring_matrix=b,
                probabilities=p, max_strings=None, n_active_per_round=30,
                rand_seed=0, correction="pt2", return_details=True,
                verbose=False)
            P["active_pt2"].append((dim, abs(e_c - e_fci)))
            print(f"active shots={s}: dim={dim} err={P['active'][-1][1]:.2e} "
                  f"err_PT2={P['active_pt2'][-1][1]:.2e}")
        np.save(NPY, P, allow_pickle=True)

    if not P["fci_no"]:
        h1e_nat, eri_nat, *_ = tc_sqd.natural_orbital_basis_from_fci(
            h1e, eri, norb, nelec)
        _, civec_nat = direct_spin1.kernel(h1e_nat, eri_nat, norb, nelec,
                                           conv_tol=1e-12)
        all_a = cistring.make_strings(range(norb), na)
        all_b = cistring.make_strings(range(norb), nb)
        for K in [20, 35, 50, 75, 100, 150, 200]:
            idx = np.argsort(-np.abs(civec_nat.ravel()))[:K]
            ia, ib = np.unravel_index(idx, civec_nat.shape)
            sa = np.unique(all_a[ia])
            sb = np.unique(all_b[ib])
            res = tc_sqd.solve_sci((sa, sb), h1e_nat, eri_nat, norb, nelec)
            P["fci_no"].append((len(sa) * len(sb), abs(res.energy - e_fci)))
            print(f"FCI-NO K={K}: dim={P['fci_no'][-1][0]} "
                  f"err={P['fci_no'][-1][1]:.2e}")
        np.save(NPY, P, allow_pickle=True)
    return P


def _plot(P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6))

    def plot(key, label, color, marker, ls="-", z=3, ms=5):
        d = np.array(P[key])
        if d.size == 0:
            return
        d = d[np.argsort(d[:, 0])]
        ax.plot(d[:, 0], d[:, 1], marker=marker, ls=ls, color=color,
                label=label, ms=ms, zorder=z, lw=1.8)

    plot("hci_ev", "HCI variational E_V", "#8c564b", "v", ls="--", z=1)
    plot("shci", "SHCI E_V+E_PT2", "#8c564b", "v", z=1)
    plot("sqd", "traditional SQD", "#1f77b4", "s")
    plot("active", "solve_sqd_active (variational)", "#ff7f0e", "o")
    plot("active_pt2", "improved SQD: active + PT2 (E+E_PT2)", "#2ca02c", "^", z=4)
    plot("fci_no", "FCI-NO top-K", "#9467bd", "D", ms=4)

    # 传统 SQD 垂直下降 = 配置恢复恰好补全全空间 (= 直接解 FCI), 标注防误导
    sqd_arr = np.array(P.get("sqd", []))
    if sqd_arr.size:
        d = sqd_arr[np.argsort(sqd_arr[:, 0])]
        last = d[-1]
        if float(last[1]) < 1e-6:      # 误差骤降 -> 达全空间
            ax.annotate("full space reached\n= FCI (recovery\ncompletes all dets)",
                        xy=(float(last[0]), float(last[1])),
                        xytext=(float(last[0]) * 1.15, float(last[1]) * 30),
                        arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
                        fontsize=7, color="grey", ha="left")

    ax.axhline(0.10, color="grey", ls=":", lw=1.2)
    ax.text(1.2e3, 0.14, "CCSD (single-ref fails at stretch)",
            fontsize=8, color="grey")
    ax.axhline(CHEM, color="red", ls="--", lw=1.0, alpha=0.7)
    ax.text(1.2e3, CHEM * 1.7, "chemical accuracy 1.6 mHa",
            fontsize=8, color="red")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("subspace dimension (strings_a x strings_b)")
    ax.set_ylabel("energy error vs FCI (Ha)")
    ax.set_title("HCI / SHCI vs SQD family (N2/STO-3G stretch)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(1e3, 3e4)
    ax.set_ylim(1e-9, 0.5)
    ax.grid(True, which="both", alpha=0.3)

    out = os.path.join(BASE, "fig_error_hci_vs_sqd.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true",
                    help="只用缓存数据出图 (跳过数据收集)")
    args = ap.parse_args()
    if args.plot:
        if not os.path.exists(NPY):
            raise SystemExit(f"无缓存数据 {NPY}, 请先不带 --plot 跑一次收集。")
        P = np.load(NPY, allow_pickle=True).item()
    else:
        P = _collect()
    _plot(P)
