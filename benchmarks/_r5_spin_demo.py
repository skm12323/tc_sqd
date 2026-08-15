"""round_011 R5: UHF 端到端验证——stretched N2 对称破缺场景."""
import sys
sys.path.insert(0, "/mnt/d/tc_sqd/src")
import numpy as np
import tc_sqd
from pyscf import gto, scf, fci

# 强拉伸 N2: RHF 无法描述三键解离, UHF 破缺对称显著 (h_alpha != h_beta 本质)
mol = gto.M(atom="N 0 0 0; N 0 0 3.0", basis="sto-3g", verbose=0)
mf_uhf = scf.UHF(mol).run()
print(f"UHF converged: E={mf_uhf.e_tot:.6f}")
h_a, h_b = mf_uhf.mo_coeff[0].T @ mf_uhf.get_hcore() @ mf_uhf.mo_coeff[0], mf_uhf.mo_coeff[1].T @ mf_uhf.get_hcore() @ mf_uhf.mo_coeff[1]
print(f"h_alpha != h_beta: {not np.allclose(h_a, h_b)} (自旋分辨本质)")

d = tc_sqd.from_pyscf(mf_uhf)  # UHF 分支 -> 五积分
print(f"from_pyscf(UHF): spin_resolved={d.spin_resolved}, norb={d.norb}, nelec={d.nelec}")

# solve_sci 自旋分辨路径 (全空间)
E = tc_sqd.compute_ground_state_energy(d.h1e, d.eri, d.norb, d.nelec, ecore=d.ecore, method="fci")
print(f"\nsolve_sci (spin-resolved) E = {E:.8f}")

# 参考: PySCF direct_uhf 全空间
from pyscf.fci import direct_uhf
norb, ne = d.norb, d.nelec
ci = direct_uhf.FCISolver(mol)
ci.norb = norb
ef = fci.direct_uhf.kernel((d.h1e[0], d.h1e[1]), tuple(d.eri), norb, ne, ecore=d.ecore, conv_tol=1e-12, max_cycle=1000)[0]
print(f"PySCF direct_uhf  E = {ef:.8f}")
print(f"\ndiff = {abs(E-ef):.2e}  ({'PASS <=1e-10' if abs(E-ef)<=1e-10 else 'FAIL'})")
print(f"vs UHF mean-field  = {mf_uhf.e_tot:.6f} (相关能 {E-mf_uhf.e_tot:.6f})")
