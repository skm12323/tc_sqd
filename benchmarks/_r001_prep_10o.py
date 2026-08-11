"""R5 prep: construct & cache N2/cc-pVDZ (10o) R=3.0 integrals + FCI reference.

新缓存名 _n2_ccpvdz_10o_ints.npz (协议 §4.3.2: 方法演进另写缓存, 不覆盖旧缓存)。
参考 = 库全空间对角化真基态 (compute_ground_state_energy, 禁用 CASCI)。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import tc_sqd  # noqa: E402
import pyscf
import pyscf.ao2mo
import pyscf.mcscf
import pyscf.scf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "_n2_ccpvdz_10o_ints.npz")
NCAS, NELEC = 10, 10

if os.path.exists(OUT):
    d = np.load(OUT)
    print(f"[prep] 缓存已存在 {OUT}: e_ref={float(d['e_ref']):.10f}")
    raise SystemExit(0)

m = pyscf.M(atom="N 0 0 -1.5; N 0 0 1.5", basis="cc-pVDZ", spin=0, verbose=0)
mf = pyscf.scf.RHF(m)
mf.kernel()
cas = pyscf.mcscf.CASCI(mf, ncas=NCAS, nelecas=NELEC)
h1e, ecore = cas.h1e_for_cas()
ncore = int((mf.mo_occ.sum() - NELEC) // 2)
eri = pyscf.ao2mo.full(m, mf.mo_coeff[:, ncore:ncore + NCAS], aosym="1").reshape([NCAS] * 4)
e_fci = tc_sqd.compute_ground_state_energy(
    h1e, eri, NCAS, (NELEC // 2, NELEC // 2), ecore=ecore, method="fci")
print(f"[prep] FCI={e_fci:.10f}")

np.savez(OUT, h1e=h1e, eri=eri, ecore=ecore, e_ref=e_fci)
print(f"[prep] saved {OUT}")
