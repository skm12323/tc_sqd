"""round_019 P0a 前提验证: 自旋分辨 ops matvec GPU vs CPU 微基准。

判定门槛 (theory §3): GPU per-mv (含逐 mv H2D v / D2H sigma 传输, 即 hybrid
架构真实成本) vs CPU per-mv —— >=3x 继续实现; <2x 前提证伪收档; 2-3x 边界。

两档子空间 (norb=12, 真实 N2 12,12 eri, 全空间 924 串取前 k):
  dim ~ 14.4k (120x120, round_017 N2 (7,7) @500 用例规模)
  dim ~ 90k  (300x300)
另记 prepare_sigma_operators (CPU, 每次 diag 一次性成本) 与 ops H2D (一次性)。
"""
import json
import os
import time

import numpy as np

from tc_sqd import matrixfree

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bench_size(ci_full, h1e, eri, norb, nelec, k, n_rep=7):
    sa, sb = ci_full[:k], ci_full[:k]
    t0 = time.perf_counter()
    ops = matrixfree.prepare_sigma_operators(sa, sb, norb, nelec, h1e, eri)
    t_prep = time.perf_counter() - t0
    na, nb = len(sa), len(sb)
    v = np.random.default_rng(0).random((na, nb))

    # CPU per-mv
    matrixfree.sigma_vector_ops(v, ops)          # 预热
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        matrixfree.sigma_vector_ops(v, ops)
        ts.append(time.perf_counter() - t0)
    cpu_ms = sorted(ts)[n_rep // 2] * 1e3

    # GPU: ops 一次性 H2D
    import cupy as cp
    t0 = time.perf_counter()
    ops_g = {key: cp.asarray(a) for key, a in ops.items() if hasattr(a, "shape")}
    cp.cuda.Stream.null.synchronize()
    t_h2d = time.perf_counter() - t0
    vg = cp.asarray(v)

    def mv_transfer(x):                          # hybrid 真实口径: H2D+D2H 逐 mv
        xg = cp.asarray(x)
        return matrixfree.sigma_vector_ops(xg, ops_g, cp).get()

    mv_transfer(v)                               # 预热 (cuBLAS handle 等)
    cp.cuda.Stream.null.synchronize()
    ts, ts_pure = [], []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        mv_transfer(v)
        ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        matrixfree.sigma_vector_ops(vg, ops_g, cp)
        cp.cuda.Stream.null.synchronize()
        ts_pure.append(time.perf_counter() - t0)
    gpu_ms = sorted(ts)[n_rep // 2] * 1e3
    gpu_pure_ms = sorted(ts_pure)[n_rep // 2] * 1e3

    # 正确性抽查 (GPU vs CPU 逐位口径)
    diff = float(np.abs(mv_transfer(v) - matrixfree.sigma_vector_ops(v, ops)).max())
    return dict(dim=na * nb, na=na, prep_s=t_prep, cpu_ms=cpu_ms,
                gpu_ms=gpu_ms, gpu_pure_ms=gpu_pure_ms, h2d_s=t_h2d,
                ratio=cpu_ms / gpu_ms, ratio_pure=cpu_ms / gpu_pure_ms,
                maxdiff=diff)


def main():
    d = np.load(os.path.join(BASE, "_n2_1212_ints.npz"))
    h1e, eri = d["h1e"], d["eri"]
    norb = h1e.shape[0]
    nelec = (norb // 2, norb // 2)             # N2 12,12 = (6,6)
    from pyscf.fci import cistring
    ci_full = cistring.make_strings(range(norb), nelec[0])   # 全空间 924
    print(f"norb={norb} nelec={nelec} 全空间串数={len(ci_full)}")
    print(f"{'dim':>8} {'prep_s':>7} {'cpu_ms':>8} {'gpu_ms':>8} {'gpu_pure':>9} "
          f"{'ratio':>6} {'pure比':>6} {'h2d_s':>7} {'maxdiff':>9}")
    results = {}
    for k in (120, 300):
        r = bench_size(ci_full, h1e, eri, norb, nelec, k)
        results[f"dim{r['dim']}"] = r
        print(f"{r['dim']:>8} {r['prep_s']:>7.2f} {r['cpu_ms']:>8.2f} "
              f"{r['gpu_ms']:>8.2f} {r['gpu_pure_ms']:>9.2f} {r['ratio']:>6.2f} "
              f"{r['ratio_pure']:>6.2f} {r['h2d_s']:>7.3f} {r['maxdiff']:>9.2e}")
        assert r["maxdiff"] < 1e-10, f"GPU/CPU matvec 不一致: {r['maxdiff']:.2e}"
    out = os.path.join(BASE, "benchmarks", "_round019_p0a_ops_gpu.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {os.path.relpath(out, BASE)}")


if __name__ == "__main__":
    main()
