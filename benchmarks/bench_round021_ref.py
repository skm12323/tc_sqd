"""round_021 参考能量：N₂/cc-pVDZ R=3.0 Å (14e,14o) 全空间 11,778,624 维真基态。

库内 solve_sci 全空间（pyscf selected_ci 流式 C 核，内存 ~300MB 级，
round_021 research.md §A），CPU 一次性后台跑。
结果落 _n2_1414_ref.npz（E_active、E_total=+ecore、wall、n_mv、tol 元数据）。

**tol 勘误（R4 Major-2）**：solve_sci 基态分支（fermion.py 基态 else 分支）
对大维度走 `eigsh(op, k=1, which="SA", maxiter=2000)`，**不转发 **kwargs**
（kwargs 仅 spin_sq 分支转发 kernel_fixed_space）→ 下方传入的 tol=1e-12
被静默忽略，实际按 scipy 默认 `tol=0`（机器精度）收敛。数值偏严 = 对参考
能量是安全方向；npz 以 tol_nominal / tol_actual / tol_note 三元组如实记录
（历史首版 npz 的 `tol=1e-12` 字段为名义值，勘误见
docs/rounds/round_021/implementation.md §5）。

数据纪律（round_016）：参考统一库内 solve_sci 全空间，不用 CASCI/direct_spin1
单档 conv_tol=1e-12（强关联根跳/漂移教训）。

用法: python benchmarks/bench_round021_ref.py
"""
import os
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTS = os.path.join(BASE, "_n2_1414_ints.npz")
OUT = os.path.join(BASE, "_n2_1414_ref.npz")
NORB, NELEC = 14, (7, 7)


def main():
    import sys
    sys.path.insert(0, os.path.join(BASE, "src"))
    import tc_sqd
    from pyscf.fci import cistring

    d = np.load(INTS)
    h1e, eri, ecore = d["h1e"], d["eri"], float(d["ecore"])

    sa = cistring.make_strings(range(NORB), NELEC[0])
    sb = sa.copy()
    dim = int(len(sa)) * int(len(sb))
    print(f"[ref] (14,14) 全空间 dim={dim} 开始 solve_sci "
          f"(tol=1e-12 名义; 基态分支不读 kwargs, 实际 scipy tol=0 机器精度) ...",
          flush=True)

    # n_mv 计数：包一层 pyscf contract_2e（solve_sci 内部经 kernel_fixed_space
    # → myci.contract_2e；模块级 monkeypatch 计数，与本库 hop 同一调用点）
    from pyscf.fci import selected_ci
    counter = {"n": 0}
    _orig = selected_ci.SCI.contract_2e

    def _counted(self, *a, **kw):
        counter["n"] += 1
        return _orig(self, *a, **kw)

    selected_ci.SCI.contract_2e = _counted
    t0 = time.perf_counter()
    try:
        res = tc_sqd.solve_sci(
            (np.asarray(sa), np.asarray(sb)), h1e, eri, NORB, NELEC,
            tol=1e-12)
    finally:
        selected_ci.SCI.contract_2e = _orig
    wall = time.perf_counter() - t0

    e_active = float(res.energy)
    e_total = e_active + ecore
    np.savez(OUT, e_ref_active=e_active, e_ref_total=e_total, ecore=ecore,
             wall_s=wall, n_mv=counter["n"], dim=dim,
             tol_nominal=1e-12, tol_actual=0.0,
             tol_note=("solve_sci 基态分支 eigsh(k=1,which='SA',maxiter=2000) "
                       "不读 **kwargs（仅 spin_sq 分支转发 kernel_fixed_space）；"
                       "tol=1e-12 为名义值被静默忽略，实际 scipy 默认 tol=0 "
                       "机器精度收敛（偏严=安全方向）"))
    print(f"[ref] 完成: E_active={e_active:.12f}  E_total={e_total:.12f}  "
          f"wall={wall:.0f}s  n_mv={counter['n']}  -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
