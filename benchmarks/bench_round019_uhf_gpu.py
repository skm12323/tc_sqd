"""round_019 R5 验收: UHF active GPU 化 A/B (CPU vs GPU backend)。

P0 : CH/STO-3G UHF (4,3) active 全空间闭环 (gpu err ≤1e-8, |ΔE cpu-gpu| ≤1e-9)
P0': N2/STO-3G R=2.5 UHF (7,7) @500 shots (gpu err ≤1e-6, |ΔE| ≤1e-8,
     wall 加速 ≥1.5×; round_017 CPU 锚 314s / err 3.47e-08)

判定打印 + JSON 落盘 _round019_uhf_gpu.json。
"""
import json
import os
import time

import numpy as np

from tc_sqd import solve_sqd_active
from tc_sqd.molecule import from_pyscf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _uhf_integrals(atom, spin):
    from pyscf import gto, scf
    mol = gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    d = from_pyscf(mf)
    return d.h1e, d.eri, d.norb, d.nelec


def _ref(h1e, eri, norb, nelec):
    from pyscf import fci
    return fci.direct_uhf.kernel(
        (h1e[0], h1e[1]), tuple(eri), norb, nelec,
        conv_tol=1e-12, max_cycle=1000)[0]


def run(tag, h1e, eri, norb, nelec, n_shots):
    e_ref = _ref(h1e, eri, norb, nelec)
    bsm = np.random.default_rng(0).random((n_shots, 2 * norb)) > 0.5
    common = dict(bitstring_matrix=bsm, n_active_per_round=50,
                  max_rounds=10, rand_seed=0)
    out = {"e_ref": e_ref}
    for be in ("cpu", "gpu"):
        t0 = time.perf_counter()
        e = solve_sqd_active(h1e, eri, norb, nelec, backend=be, **common)
        wall = time.perf_counter() - t0
        out[be] = dict(E=float(e), err=abs(e - e_ref), wall=wall)
        print(f"[{tag}] {be}: E={e:.12f} err={abs(e - e_ref):.2e} "
              f"wall={wall:.1f}s")
    out["dE"] = abs(out["gpu"]["E"] - out["cpu"]["E"])
    out["speedup"] = out["cpu"]["wall"] / out["gpu"]["wall"]
    print(f"[{tag}] |ΔE cpu-gpu|={out['dE']:.2e}  speedup={out['speedup']:.2f}×")
    return out


def main():
    res = {}
    res["ch43"] = run("CH(4,3)", *_uhf_integrals("C 0 0 0; H 0 0 1.12", 1),
                      2000)
    res["n2_77"] = run("N2(7,7)@500", *_uhf_integrals("N 0 0 0; N 0 0 2.5", 0),
                       500)

    ch, n2 = res["ch43"], res["n2_77"]
    p0 = ch["gpu"]["err"] <= 1e-8 and ch["dE"] <= 1e-9
    p0p = (n2["gpu"]["err"] <= 1e-6 and n2["dE"] <= 1e-8
           and n2["speedup"] >= 1.5)
    print(f"\nP0  (CH gpu 闭环 + 一致性):   {'PASS' if p0 else 'FAIL'}")
    print(f"P0' (N2 gpu err+一致+≥1.5×):  {'PASS' if p0p else 'FAIL'}")

    out = os.path.join(BASE, "benchmarks", "_round019_uhf_gpu.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {os.path.relpath(out, BASE)}")


if __name__ == "__main__":
    main()
