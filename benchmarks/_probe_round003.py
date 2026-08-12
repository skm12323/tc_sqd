"""R5 round_003 probe: verify WSL tc env + GPU/cupy availability."""
import sys
import platform

print("python", sys.version)
print("platform", platform.platform())

import numpy as np
print("numpy", np.__version__)

try:
    import cupy
    print("cupy", cupy.__version__)
    print("device_count", cupy.cuda.runtime.getDeviceCount())
    if cupy.cuda.runtime.getDeviceCount() > 0:
        dev = cupy.cuda.runtime.getDeviceProperties(0)
        print("device", dev["name"].decode() if isinstance(dev["name"], bytes) else dev["name"])
        print("mem_total_GB", cupy.cuda.runtime.memGetInfo()[1] / 1e9)
except Exception as e:
    print("cupy import FAIL:", repr(e))

import pyscf
print("pyscf", pyscf.__version__)
import scipy
print("scipy", scipy.__version__)

import tc_sqd
from tc_sqd.noise import has_gpu
print("has_gpu", has_gpu())
print("tc_sqd ok")
