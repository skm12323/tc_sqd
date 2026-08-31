"""round_018: 开壳层预算门控按扇区修正验收。

修复: cipsi.py 门控开壳层按扇区精确计数 + 默认上限 max(C(na),C(nb))。
对照: 同一脚本在 `git stash` 旧代码下跑出"修复前默认"行 (预期 270)。

可证伪预测 (docs/rounds/round_018/theory.md §3):
  P0 : CH (4,3) coverage_closure 默认上限 → dim=300 (修复前 270), err ≤1e-9
  P0': solve_cipsi 开壳层默认上限修复后 E = FCI (修复前 β 被挡)
  P1 : 零回归 (全库 + GPU 拆分, 另跑)
"""
import os, sys, time, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
import tc_sqd
from pyscf import gto, scf, fci


def ch73():
    """CH/STO-3G ROHF 积分, nelec=(4,3): α 全空间 15 < β 全空间 20。"""
    mol = gto.M(atom="C 0 0 0; H 0 0 1.1", basis="sto-3g", spin=1, verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl",
                    mol.intor("int2e_sph"), mo, mo, mo, mo)
    e_ref = float(fci.direct_spin1.kernel(h1e, eri, mol.nao_nr(), (4, 3),
                                          conv_tol=1e-12)[0])
    return h1e, eri, mol.nao_nr(), (4, 3), e_ref


h1e, eri, norb, nelec, e_ref = ch73()
res = {"e_ref": e_ref, "full": 300}

# 合法电子数种子 (cipsi 种子不经粒子数修复)
rng = np.random.default_rng(0)
rows = []
for _ in range(40):
    a = np.zeros(norb, dtype=bool)
    a[rng.choice(norb, nelec[0], replace=False)] = True
    b = np.zeros(norb, dtype=bool)
    b[rng.choice(norb, nelec[1], replace=False)] = True
    rows.append(np.concatenate([b, a]))
seed = np.array(rows)

for label, kw in [
    ("closure_default", dict(coverage_closure=True)),
    ("closure_cap15", dict(coverage_closure=True, max_strings=15)),
]:
    traj = []
    t0 = time.perf_counter()
    e = tc_sqd.solve_sqd_active(
        h1e, eri, norb, nelec, bitstring_matrix=seed[:30],
        n_active_per_round=5, max_rounds=3, rand_seed=0,
        trajectory=traj, **kw)
    w = time.perf_counter() - t0
    dim = traj[-1]["dim"]
    res[label] = {"E": float(e), "err": abs(e - e_ref), "dim": dim,
                  "wall": round(w, 1)}
    print(f"{label}: dim={dim}/300 err={res[label]['err']:.2e} wall={w:.1f}s",
          flush=True)

t0 = time.perf_counter()
e = tc_sqd.solve_cipsi(h1e, eri, norb, nelec, seed_bitstring_matrix=seed,
                       pt2_floor=0.0, max_iter=12)
res["cipsi_default"] = {"E": float(e), "err": abs(e - e_ref),
                        "wall": round(time.perf_counter() - t0, 1)}
print(f"cipsi_default: err={res['cipsi_default']['err']:.2e}", flush=True)

p0 = (res["closure_default"]["dim"] == 300
      and res["closure_default"]["err"] <= 1e-9)
p0p = res["cipsi_default"]["err"] <= 1e-9
print(f"\nP0 (closure 默认补全 300): {'PASS' if p0 else 'FAIL'}")
print(f"P0' (cipsi 默认=FCI):       {'PASS' if p0p else 'FAIL'}")

out = os.path.join(BASE, "benchmarks", "_round018_budget.json")
with open(out, "w") as f:
    json.dump(res, f, indent=2)
print(f"saved {os.path.relpath(out, BASE)}")
