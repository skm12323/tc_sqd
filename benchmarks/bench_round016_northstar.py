"""round_016: 北极星目标正式验收 (三指标 + 不回归矩阵 + 底噪微实验)。

北极星 (COLLABORATION §0.2): 12,12 err ≤3×SHCI 同维度 + wall ≤3×SHCI(812s) +
shots ≤500, 其余体系不回归 (回归即失败)。
口径判定 (theory.md §2): 全空间达成时改用同任务比较 (同体系同库参考);
残差 = 85 万维 eigsh 数值底噪, 非覆盖误差。
SHCI 参考 (REVIEW.md 历史实测, 引用不重跑): 见 SHCI_REF。

可证伪预测:
  P0: 12,12 @500 closure: dim=853,776 且 err ≤1.14e-10 (=3×3.8e-11) 且 wall ≤2436s
  P1: 其余 4 体系 closure 达全空间且 err ≤ 同 shots 无闭包 baseline (不回归)
  P2: 12,12 closure err 底噪随 tol 收紧 (1e-6→1e-10→0) 系统性降低
      [预期证伪: 底噪=舍入非收敛 → 锁定 @500 配方为验收配置]
"""
import os, sys, time, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
from tc_sqd.cipsi import solve_sqd_active, _Subspace  # noqa: E402
_ORIG = _Subspace.diag

SHCI_REF = {
    "n2_ccpvdz_1212": {"err": 3.8e-11, "dim": 829921, "wall_shci": 812.0,
                       "cite": "REVIEW.md L1586 (round_006 eps 扫最优点)"},
    "n2_ccpvdz_10o":  {"err": 9.5e-11, "dim": 63504,
                       "cite": "REVIEW.md L1584 round_006 末点"},
    "c2_sto3g":       {"err": 9.3e-11, "dim": 44100,
                       "cite": "REVIEW.md L735-744 全空间末点"},
    "n2_sto3g":       {"err": 8.4e-07, "dim": 3000,
                       "cite": "REVIEW.md L761-765 (无全空间点)"},
    "c2_ccpvdz_10o":  {"err": 2.6e-13, "dim": 63504,
                       "cite": "REVIEW.md L826-832 全空间末点"},
}
FULL = {"n2_ccpvdz_1212": 853776, "n2_ccpvdz_10o": 63504,
        "c2_sto3g": 44100, "n2_sto3g": 14400, "c2_ccpvdz_10o": 63504}


def run(h1e, eri, norb, nelec, ecore, shots, seed=0, *,
        coverage_closure=True, eigsh_tol=None, n_active=90, backend="gpu"):
    n_mv = [0]
    n_diag = [0]

    def wrapped(self, sa, sb):
        n_diag[0] += 1
        r = _ORIG(self, sa, sb)
        n = getattr(self, "last_n_mv", 0)
        if n:
            n_mv[0] += n
        return r
    _Subspace.diag = wrapped
    try:
        traj = []
        kw = dict(ecore=ecore, max_strings=None, n_active_per_round=n_active,
                  rand_seed=seed, tail_suppression=True, tail_shots_ref=100,
                  warm_start=True, backend=backend, verbose=False,
                  trajectory=traj, coverage_closure=coverage_closure)
        if eigsh_tol is not None:
            kw["eigsh_tol"] = eigsh_tol
        t0 = time.perf_counter()
        E = solve_sqd_active(
            h1e, eri, norb, nelec,
            bitstring_matrix=np.random.default_rng(seed).random(
                (shots, 2 * norb)) > 0.5,
            probabilities=np.full(shots, 1.0 / shots), **kw)
        wall = time.perf_counter() - t0
    finally:
        _Subspace.diag = _ORIG
    final = traj[-1] if traj else {}
    return {"E": float(E), "wall": round(wall, 1),
            "dim": int(final.get("dim", -1)),
            "n_mv": n_mv[0], "n_diag": n_diag[0]}


def load_npz(name):
    z = np.load(os.path.join(BASE, f"{name}.npz"))
    return z["h1e"], z["eri"], float(z["ecore"]), float(z["e_ref"])


def pyscf_from_mol(atom, basis):
    """全价空间: from_pyscf + 库内 solve_sci 全空间真基态。

    注: direct_spin1 conv_tol=1e-12 作参考生成器在强关联体系不可靠——
    N2/STO-3G (7,7) 跨运行漂移 ~1e-9 (三次三个值), C2/cc-pVDZ (5,5) 根跳 9.3e-3。
    solve_sci 与严格 direct_spin1 一致 (~1e-13), 沿用 plot 脚本既有口径。
    """
    import tc_sqd
    from pyscf import gto
    from pyscf.fci import cistring
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    d = tc_sqd.from_pyscf(mol)
    all_str = cistring.make_strings(range(d.norb), d.nelec[0])
    res = tc_sqd.solve_sci((all_str, all_str), d.h1e, d.eri, d.norb, d.nelec)
    full = int(cistring.num_strings(d.norb, d.nelec[0])) ** 2
    return (d.h1e, d.eri, d.norb, d.nelec, d.ecore,
            float(res.energy + d.ecore), full)


def cas_c2_ccpvdz_10o():
    """C2/cc-pVDZ 10o CAS(10,10): 沿用 plot 脚本口径 (RHF→CASCI h1e + ao2mo)。

    参考用库内 solve_sci 全空间: pyscf direct_spin1 conv_tol=1e-12 在 C2 近简并
    (5,5) 上根跳 (-75.5509, 高 9.3e-3); conv_tol=1e-13 才收敛到真基态。
    """
    import pyscf, pyscf.scf, pyscf.mcscf, pyscf.ao2mo
    from pyscf.fci import cistring
    import tc_sqd
    m = pyscf.M(atom="C 0 0 0; C 0 0 1.24", basis="cc-pVDZ", spin=0, verbose=0)
    mf = pyscf.scf.RHF(m)
    mf.kernel()
    cas = pyscf.mcscf.CASCI(mf, ncas=10, nelecas=10)
    h1e, ecore = cas.h1e_for_cas()
    ncore = int((mf.mo_occ.sum() - 10) // 2)
    eri = pyscf.ao2mo.full(m, cas.mo_coeff[:, ncore:ncore + 10],
                           aosym="1").reshape([10] * 4)
    all_str = cistring.make_strings(range(10), 5)
    res = tc_sqd.solve_sci((all_str, all_str), h1e, eri, 10, (5, 5))
    full = int(cistring.num_strings(10, 5)) ** 2
    return h1e, eri, 10, (5, 5), float(ecore), float(res.energy + ecore), full


res = {"shci_ref": SHCI_REF, "full": FULL}

# ---------- Part A: 12,12 @500 三指标 (P0) + tol 底噪微实验 (P2) ----------
h1e, eri, ec, eref = load_npz("_n2_1212_ints")
print("=== Part A: 12,12 @500 closure (n_active=90, GPU) tol 变体 ===", flush=True)
rows_a = []
for tol in [1e-6, 1e-10, 0.0]:
    r = run(h1e, eri, 12, (6, 6), ec, 500, eigsh_tol=tol)
    r.update(tol=tol, err=abs(r["E"] - eref))
    rows_a.append(r)
    print(f"  tol={tol:g}: E={r['E']:.10f} err={r['err']:.2e} dim={r['dim']} "
          f"n_mv={r['n_mv']} n_diag={r['n_diag']} wall={r['wall']}s", flush=True)
res["n2_1212_tol_scan"] = {"e_ref": eref, "rows": rows_a}

r12 = rows_a[0]  # 验收配置 = 配方 tol=1e-6
p0 = (r12["dim"] == FULL["n2_ccpvdz_1212"]
      and r12["err"] <= 3 * SHCI_REF["n2_ccpvdz_1212"]["err"]
      and r12["wall"] <= 3 * SHCI_REF["n2_ccpvdz_1212"]["wall_shci"])

# ---------- Part B: 4 体系不回归矩阵 (P1) ----------
def small_systems():
    h1e10, eri10, ec10, eref10 = load_npz("_n2_ccpvdz_10o_ints")
    yield "n2_ccpvdz_10o", (h1e10, eri10, 10, (5, 5), ec10, eref10, 63504)
    for name, atom, basis, expect in [
            ("c2_sto3g", "C 0 0 0; C 0 0 1.24", "sto-3g", 44100),
            ("n2_sto3g", "N 0 0 0; N 0 0 2.0", "sto-3g", 14400)]:
        h, e, no, ne, ecx, erx, fx = pyscf_from_mol(atom, basis)
        assert fx == expect, f"{name} full 期望 {expect}, 得到 {fx}"
        np.savez(os.path.join(BASE, f"_{name}_ints.npz"),
                 h1e=h, eri=e, ecore=ecx, e_ref=erx)
        yield name, (h, e, no, ne, ecx, erx, fx)
    h, e, no, ne, ecx, erx, fx = cas_c2_ccpvdz_10o()
    assert fx == 63504
    np.savez(os.path.join(BASE, "_c2_ccpvdz_10o_ints.npz"),
             h1e=h, eri=e, ecore=ecx, e_ref=erx)
    yield "c2_ccpvdz_10o", (h, e, no, ne, ecx, erx, fx)

print("\n=== Part B: 4 体系 @500 closure vs baseline (不回归) ===", flush=True)
rows_b = {}
for name, (h, e, no, ne, ecx, erx, fx) in small_systems():
    c = run(h, e, no, ne, ecx, 500)                       # closure, tol 默认
    b = run(h, e, no, ne, ecx, 500, coverage_closure=False, n_active=30)
    c.update(err=abs(c["E"] - erx), full=fx)
    b.update(err=abs(b["E"] - erx), full=fx)
    rows_b[name] = {"closure": c, "baseline": b}
    print(f"  {name}: closure dim={c['dim']} err={c['err']:.2e} wall={c['wall']}s"
          f" | baseline dim={b['dim']} err={b['err']:.2e}", flush=True)
res["small_systems"] = rows_b

FLOOR_TOL = 1e-12  # 全空间底噪处 ARPACK 轨迹间差异 ~1e-13, 逐位比较会误判 (theory.md)
p1 = all(v["closure"]["dim"] == v["closure"]["full"]
         and v["closure"]["err"] <= v["baseline"]["err"] + FLOOR_TOL
         for v in rows_b.values())

# ---------- P2: tol 收紧能否系统性降低底噪 ----------
e_by_tol = {r["tol"]: r["err"] for r in rows_a}
p2 = e_by_tol[1e-10] < e_by_tol[1e-6] and e_by_tol[0.0] < e_by_tol[1e-6]

# ---------- 汇总 ----------
print("\n" + "=" * 74)
print("=== round_016 北极星验收汇总 ===")
print("=" * 74)
print(f"{'体系':<18}{'full':>9}{'clos_dim':>9}{'clos_err':>11}"
      f"{'SHCI 参考':>11}{'比值':>8}")
print(f"{'n2_ccpvdz_1212':<18}{853776:>9}{r12['dim']:>9}{r12['err']:>11.2e}"
      f"{3.8e-11:>11.2e}{r12['err']/3.8e-11:>8.2f}")
for name, v in rows_b.items():
    c = v["closure"]
    sr = SHCI_REF[name]["err"]
    print(f"{name:<18}{c['full']:>9}{c['dim']:>9}{c['err']:>11.2e}"
          f"{sr:>11.2e}{c['err']/sr:>8.2f}")
print(f"\nP0 (12,12 三指标: dim/err≤1.14e-10/wall≤2436s): {'PASS' if p0 else 'FAIL'}")
print(f"   err={r12['err']:.2e} = {r12['err']/3.8e-11:.2f}×SHCI; "
      f"wall={r12['wall']}s vs 3×812s")
print(f"P1 (4 体系不回归: 全空间 + err≤baseline):       {'PASS' if p1 else 'FAIL'}")
print(f"P2 (tol 收紧降低底噪):                          {'PASS' if p2 else 'FAIL'}")
print(f"   tol=1e-6: {e_by_tol[1e-6]:.2e}  1e-10: {e_by_tol[1e-10]:.2e}  "
      f"0: {e_by_tol[0.0]:.2e}")

out = os.path.join(BASE, "benchmarks", "_round016_northstar.json")
with open(out, "w") as f:
    json.dump(res, f, indent=2)
print(f"\nsaved {out}")
