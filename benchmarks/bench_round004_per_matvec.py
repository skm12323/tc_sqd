"""R5 round_004: P0' per-matvec 隔离测 (B+C 可归因验收, theory §3 P0' 最重要锚点)。

设计 (theory §3 P0'):
  - 固定 dim>1e5 子空间 (n_str=317 -> dim=100,489), 绕开 cupyx eigsh 收敛 confound。
  - 固定 50 次随机 matvec, **interleave** 两种 eri 模式 (recompute / cached 交替,
    消除 GPU 热状态顺序效应), 各 50 次计时取中位。
  - 对照: `sigma_selected_ci_gpu(eri1_aaaa=None, eri1_bbaa=None)` (重算, round_003
    现状) vs `eri1_aaaa=缓存, eri1_bbaa=缓存` (round_004 方式 C)。
  - 验收: **cached/recompute 中位 wall ≤0.88** (即 per-matvec ≥1.14×)。

附: sigma 缓存版 vs 重算版 输出 max|Δ| ≤2e-13 (P1 接口级, 排除 _selci_* 接合 bug)。

输出: 增量落盘 `benchmarks/_round004_results.json` {"p0_prime": {...}} + stdout。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from pyscf import ao2mo
from pyscf.fci import cistring, direct_spin1, selected_ci

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tc_sqd.selected_ci_gpu import (  # noqa: E402
    sigma_selected_ci_gpu, _selci_eri_aaaa, _selci_eri_bbaa, _get_kernels)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTS = os.path.join(BASE, "_n2_1212_ints.npz")
OUT = os.path.join(BASE, "benchmarks", "_round004_results.json")

N_MV = 50          # 各模式固定 50 次 (theory §3 P0')
N_WARM = 3         # warm-up 次数 (触发 cupy 上下文 / RawModule 编译 / matmul autotune)
N_STR = 317        # dim ≈ 100,489 > 1e5 (P0' 前置)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_str", type=int, default=N_STR)
    ap.add_argument("--n_mv", type=int, default=N_MV)
    ap.add_argument("--only", action="store_true", help="不落盘 (调试)")
    args = ap.parse_args()
    n_str, n_mv = args.n_str, args.n_mv

    import cupy as cp
    from pyscf.fci import selected_ci as _sci

    d = np.load(INTS)
    h1e, eri = d["h1e"], d["eri"]
    norb, nelec = 12, (6, 6)
    full = np.array(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    sa = sb = full[:n_str]
    na = nb = len(sa)
    dim = na * nb
    assert dim > 1e5, f"P0' 子空间 dim={dim} 须 >1e5"

    links = [_sci.des_des_linkstr(sa, norb, nelec[0], True),
             _sci.des_des_linkstr(sb, norb, nelec[1], True),
             _sci.cre_des_linkstr(sa, norb, nelec[0], True),
             _sci.cre_des_linkstr(sb, norb, nelec[1], True)]
    kernels = _get_kernels()

    # 预算缓存 (== _Subspace.__init__ 实例级缓存路径: 基于 self.h2e)
    h2e = ao2mo.restore(1, direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5), norb)
    eri1_aaaa = cp.asarray(_selci_eri_aaaa(h2e, norb))
    eri1_bbaa = cp.asarray(_selci_eri_bbaa(h2e, norb, nelec))

    # ---- sigma 缓存 vs 重算 输出一致性 (P1 接口级) ----
    rng = np.random.default_rng(42)
    v_chk = rng.standard_normal((na, nb))
    s_rec = sigma_selected_ci_gpu(v_chk, sa, sb, norb, nelec, h1e, eri,
                                  links=links, kernels=kernels,
                                  eri1_aaaa=None, eri1_bbaa=None)
    s_cac = sigma_selected_ci_gpu(v_chk, sa, sb, norb, nelec, h1e, eri,
                                  links=links, kernels=kernels,
                                  eri1_aaaa=eri1_aaaa, eri1_bbaa=eri1_bbaa)
    sigma_diff = float(cp.abs(s_cac - s_rec).max())
    print(f"[P0'] sigma 缓存 vs 重算 max|Δ|={sigma_diff:.2e} (阈值 ≤2e-13)",
          flush=True)

    # ---- 50 次随机 matvec, interleave ----
    rng = np.random.default_rng(0)
    vs = [rng.standard_normal((na, nb)) for _ in range(n_mv)]

    def call(v, cached):
        return sigma_selected_ci_gpu(
            v, sa, sb, norb, nelec, h1e, eri, links=links, kernels=kernels,
            eri1_aaaa=eri1_aaaa if cached else None,
            eri1_bbaa=eri1_bbaa if cached else None)

    for v in vs[:N_WARM]:
        call(v, cached=False)
        call(v, cached=True)
    cp.cuda.Stream.null.synchronize()

    t_recompute, t_cached = [], []
    for v in vs:
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        call(v, cached=False)
        cp.cuda.Stream.null.synchronize()
        t_recompute.append(time.perf_counter() - t0)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        call(v, cached=True)
        cp.cuda.Stream.null.synchronize()
        t_cached.append(time.perf_counter() - t0)

    med_recompute = float(np.median(t_recompute))
    med_cached = float(np.median(t_cached))
    tot_recompute = float(np.sum(t_recompute))
    tot_cached = float(np.sum(t_cached))
    ratio_med = med_cached / med_recompute
    ratio_agg = tot_cached / tot_recompute
    speedup_med = med_recompute / med_cached

    print(f"[P0'] dim={dim} n_mv={n_mv} interleave "
          f"med_recompute={med_recompute*1e3:.2f}ms med_cached={med_cached*1e3:.2f}ms "
          f"ratio_med={ratio_med:.3f} speedup_med={speedup_med:.2f}x "
          f"| tot_recompute={tot_recompute*1e3:.0f}ms tot_cached={tot_cached*1e3:.0f}ms "
          f"ratio_agg={ratio_agg:.3f}", flush=True)

    verdict = ("证实 (≤0.88, per-matvec ≥1.14×)" if ratio_med <= 0.88
               else "部分 (0.88-1.0)" if ratio_med <= 1.0
               else "证伪 (>1.0)")
    print(f"[P0'] 判定: {verdict}  (阈值 ratio_med ≤0.88)", flush=True)

    result = {"p0_prime": {
        "n_str": n_str, "dim": dim, "n_mv": n_mv,
        "med_recompute_ms": med_recompute * 1e3,
        "med_cached_ms": med_cached * 1e3,
        "ratio_med": ratio_med, "speedup_med": speedup_med,
        "tot_recompute_ms": tot_recompute * 1e3,
        "tot_cached_ms": tot_cached * 1e3,
        "ratio_agg": ratio_agg,
        "sigma_diff": sigma_diff, "verdict": verdict}}
    if not args.only:
        if os.path.exists(OUT):
            with open(OUT) as f:
                data = json.load(f)
        else:
            data = {}
        data["p0_prime"] = result["p0_prime"]
        with open(OUT, "w") as f:
            json.dump(data, f, indent=2, default=float)
        print(f"[bench] saved p0_prime -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
