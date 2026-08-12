import numpy as np
from pyscf.fci import cistring
from tc_sqd.cipsi import _Subspace
from tc_sqd.noise import has_gpu

d = np.load('_n2_1212_ints.npz')
print('npz keys', list(d.keys()))
print('has_gpu', has_gpu())
norb, nelec = 12, (6, 6)
full = np.array(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
sa = full[:40]
sb = full[:40]
sub_cpu = _Subspace(d['h1e'], d['eri'], norb, nelec, backend='cpu')
sub_gpu = _Subspace(d['h1e'], d['eri'], norb, nelec, backend='gpu')
print('backend_cpu', sub_cpu.backend, 'backend_gpu', sub_gpu.backend)
E_cpu, c_cpu, _, _ = sub_cpu.diag(sa, sb)
E_gpu, c_gpu, _, _ = sub_gpu.diag(sa, sb)
print('E_cpu', E_cpu, 'E_gpu', E_gpu, 'diff', abs(E_cpu - E_gpu))
print('SMOKE OK')
