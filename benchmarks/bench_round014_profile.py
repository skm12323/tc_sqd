"""round_014 P0a: GPU sigma 每 matvec 分段 profile（先 profile 后决策）。

问题: _Subspace.diag 的 GPU hybrid 每 matvec ~43ms (round_009)。sigma 内部
每 matvec 重复做: 4× _links_tril (numpy 打包 + 16 次 H2D) + 3-4× cp.zeros
巨型 t1 工作区 (908 串时 ~1.4GB/次) + H2D v + 转置拷贝。
假设: links 扁平化 + t1 工作区提升到 diag 生命周期可省 ≥10%。

方法: 固定子空间 (N2/cc-pVDZ 12,12 前 908 串, dim=824,464, 同真实 12,12 workload),
复刻 sigma 体并逐段 cp synchronize 计时; 复刻版输出 vs 真 sigma 输出 max diff
锁口径。warm-up 3 次后计时 15 次。

判定: 可消除段 (links+zeros+H2D/转置) 占比 ≥10% → 进 R3 persistent workspace;
<10% → 收档不写优化代码。
"""
import os, sys, time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

import cupy as cp
from pyscf.fci import selected_ci, cistring, direct_spin1
from pyscf import ao2mo

from tc_sqd.selected_ci_gpu import (sigma_selected_ci_gpu, _links_tril,
                                    _get_kernels, _selci_eri_aaaa,
                                    _selci_eri_bbaa)

# ---- 固定输入: 真实 12,12 积分 + 前 908 串 (dim=824,464 同真实 workload) ----
npz = np.load(os.path.join(BASE, "_n2_1212_ints.npz"))
h1e, eri = npz["h1e"], npz["eri"]
norb, nelec = 12, (6, 6)
N_STR = 908
full = np.array(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
sa = full[:N_STR]
sb = sa
na = nb = N_STR
dim = na * nb
print(f"subspace dim={dim} ({na}x{nb})", flush=True)

# ---- 预算 (同 _Subspace.diag 的 GPU 分支) ----
h2e = ao2mo.restore(1, direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5), norb)
eri1_aaaa = cp.asarray(_selci_eri_aaaa(h2e, norb))
eri1_bbaa = cp.asarray(_selci_eri_bbaa(h2e, norb, nelec))
links = [selected_ci.des_des_linkstr(sa, norb, nelec[0], True),
         selected_ci.des_des_linkstr(sb, norb, nelec[1], True),
         selected_ci.cre_des_linkstr(sa, norb, nelec[0], True),
         selected_ci.cre_des_linkstr(sb, norb, nelec[1], True)]
kernels = _get_kernels()
dd_a, dd_b, cd_a, cd_b = links
rng = np.random.default_rng(0)
v = rng.random((na, nb))

def sync():
    cp.cuda.Stream.null.synchronize()

# ---- 复刻 sigma 体 (逐段计时用; 输出与真 sigma 锁口径) ----
def sigma_sections(v, sections):
    """复刻 sigma_selected_ci_gpu (links/kernels/eri1 均缓存路径), 逐段计时。
    sections: dict[str, float] 累加各段秒数。"""
    scat_t1, gath_t1, scat_ba, gath_bb = kernels
    TPB = 256
    nn = norb * (norb - 1) // 2
    npair = norb * (norb + 1) // 2

    def tick(key, t0):
        sync()
        sections[key] = sections.get(key, 0.0) + time.perf_counter() - t0

    t0 = time.perf_counter()
    vg = cp.asarray(np.ascontiguousarray(v), np.float64)
    ci1 = cp.zeros((na, nb))
    tick("h2d_v+zeros_ci1", t0)

    # aaaa α
    t0 = time.perf_counter()
    ca, tra, sa_, sga, nca = _links_tril(dd_a)
    tick("links_tril(aaaa_a)", t0)
    t0 = time.perf_counter()
    t1 = cp.zeros((dd_a.shape[0], nn, nb))
    tick("zeros_t1(aaaa_a)", t0)
    t0 = time.perf_counter()
    scat_t1((nca, (nb + TPB - 1) // TPB, 1), (TPB, 1, 1),
            (ca, tra, sa_, sga, vg, t1, nca, nn, nb))
    vt1 = cp.matmul(eri1_aaaa, t1)
    gath_t1((nca, (nb + TPB - 1) // TPB, 1), (TPB, 1, 1),
            (ca, tra, sa_, sga, vt1, ci1, nca, nn, nb))
    tick("kernels+matmul(aaaa_a)", t0)

    # aaaa β
    t0 = time.perf_counter()
    cb, trb, sb_, sgb, ncb = _links_tril(dd_b)
    tick("links_tril(aaaa_b)", t0)
    t0 = time.perf_counter()
    t1b = cp.zeros((dd_b.shape[0], nn, na))
    vT = cp.ascontiguousarray(vg.T)
    tick("zeros_t1b+transpose(aaaa_b)", t0)
    t0 = time.perf_counter()
    scat_t1((ncb, (na + TPB - 1) // TPB, 1), (TPB, 1, 1),
            (cb, trb, sb_, sgb, vT, t1b, ncb, nn, na))
    vt1b = cp.matmul(eri1_aaaa, t1b)
    ci1T = cp.zeros((nb, na))
    gath_t1((ncb, (na + TPB - 1) // TPB, 1), (TPB, 1, 1),
            (cb, trb, sb_, sgb, vt1b, ci1T, ncb, nn, na))
    ci1 += ci1T.T
    tick("kernels+matmul(aaaa_b)", t0)

    # bbaa
    t0 = time.perf_counter()
    cba, trba, sba_, sgba, ncba = _links_tril(cd_a)
    cbb, trbb, sbb_, sgbb, ncbb = _links_tril(cd_b)
    tick("links_tril(bbaa x2)", t0)
    t0 = time.perf_counter()
    t1c = cp.zeros((na, npair, nb))
    tick("zeros_t1c(bbaa)", t0)
    t0 = time.perf_counter()
    scat_ba((ncba, (nb + TPB - 1) // TPB, 1), (TPB, 1, 1),
            (cba, trba, sba_, sgba, vg, t1c, ncba, npair, nb))
    vt1c = cp.matmul(eri1_bbaa, t1c)
    gath_bb((ncbb, (na + TPB - 1) // TPB, 1), (TPB, 1, 1),
            (cbb, trbb, sbb_, sgbb, vt1c, ci1, ncbb, npair, na, nb))
    tick("kernels+matmul(bbaa)", t0)
    return ci1

# ---- 口径锁: 复刻版 vs 真 sigma ----
sec = {}
sync()
out_real = sigma_selected_ci_gpu(v, sa, sb, norb, nelec, h1e, eri,
                                 links=links, kernels=kernels,
                                 eri1_aaaa=eri1_aaaa, eri1_bbaa=eri1_bbaa)
sync()
out_sec = sigma_sections(v, sec)
sync()
diff = float(cp.abs(out_real - out_sec).max())
print(f"口径锁: |replica - real sigma| max diff = {diff:.2e} (须 ≤1e-10)", flush=True)
assert diff <= 1e-10, "复刻版与真 sigma 不一致, profile 不代表性"

# ---- D2H 成本单独测 (hybrid matvec 的 .get() 在 sigma 外) ----
sync(); t0 = time.perf_counter()
_ = out_real.get()
sync()
t_d2h = time.perf_counter() - t0
print(f"D2H (.get() {dim*8/1e6:.0f}MB): {t_d2h*1e3:.2f} ms", flush=True)

# ---- 计时: warm-up 3 + 计时 15 ----
for _ in range(3):
    sigma_selected_ci_gpu(v, sa, sb, norb, nelec, h1e, eri, links=links,
                          kernels=kernels, eri1_aaaa=eri1_aaaa,
                          eri1_bbaa=eri1_bbaa)
    sync()

N = 15
sections = {}
# 真 sigma 总时间 (同进程背靠背)
t_total = 0.0
for _ in range(N):
    sync(); t0 = time.perf_counter()
    out = sigma_selected_ci_gpu(v, sa, sb, norb, nelec, h1e, eri, links=links,
                                kernels=kernels, eri1_aaaa=eri1_aaaa,
                                eri1_bbaa=eri1_bbaa)
    sync()
    t_total += time.perf_counter() - t0
t_total_ms = t_total / N * 1e3

for _ in range(N):
    sigma_sections(v, sections)

# ---- 汇总 ----
keys = ["h2d_v+zeros_ci1", "links_tril(aaaa_a)", "zeros_t1(aaaa_a)",
        "kernels+matmul(aaaa_a)", "links_tril(aaaa_b)",
        "zeros_t1b+transpose(aaaa_b)", "kernels+matmul(aaaa_b)",
        "links_tril(bbaa x2)", "zeros_t1c(bbaa)", "kernels+matmul(bbaa)"]
t_sum = sum(sections[k] for k in keys) / N * 1e3
print(f"\n=== 每 matvec 分段 (ms, N={N}) ===")
print(f"{'section':<30} {'ms':>8} {'%':>6}")
print("-" * 48)
removable = 0.0
for k in keys:
    ms = sections[k] / N * 1e3
    pct = ms / t_sum * 100
    flag = ""
    if k.startswith("links_tril") or k.startswith("zeros") or k.startswith("h2d"):
        removable += ms
        flag = "  <- 可提升至 diag 生命周期"
    print(f"{k:<30} {ms:>8.2f} {pct:>5.1f}%{flag}")
print("-" * 48)
print(f"复刻分段合计: {t_sum:.2f} ms | 真 sigma 实测: {t_total_ms:.2f} ms "
      f"(差 = 段间 sync 开销)")
print(f"\n可消除段 (links_tril×4 + zeros×4 + h2d_v): {removable:.2f} ms "
      f"= {removable/t_sum*100:.1f}% of {t_sum:.2f} ms")
print(f"D2H .get(): {t_d2h*1e3:.2f} ms (sigma 外, hybrid matvec 另计)")
print(f"\n[P0a 判定] 可消除占比 {removable/t_sum*100:.1f}% -> "
      f"{'≥10%, 进 R3 persistent workspace' if removable/t_sum >= 0.10 else '<10%, 收档不优化'}")
