"""R5 round_003: 端到端 GPU arm 单独重跑 (fresh 进程, per-round 计时)。

背景: 全量 bench_round003_gpu.py 的 GPU arm 在跑 ~60min 后进程硬崩 (无 Python traceback,
dmesg 有 WSL2 dxgkrnl ioctl 错误)。本脚本在**全新进程**里重跑 GPU arm,
monkeypatch _Subspace.diag 记录每轮 dim/backend/wall, 定位崩溃点 + 拿最终 E。

配置同 round_002 C1-v2 @500: 12,12 seed=0 shots=500 max_strings=1e5,
tail_suppression=True, tail_max_draw_factor=10, tail_shots_ref=100, backend="gpu"。
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import tc_sqd  # noqa: E402
from tc_sqd import cipsi as _cipsi  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "benchmarks", "_round003_gpu_results.json")
INTS = os.path.join(BASE, "_n2_1212_ints.npz")
E_REF = -108.7686857

d = np.load(INTS)
h1e, eri, ecore, e_ref = d["h1e"], d["eri"], float(d["ecore"]), float(d["e_ref"])
norb, nelec = 12, (6, 6)
shots, seed = 500, 0
bsm = np.random.default_rng(seed).random((shots, 2 * norb)) > 0.5
probs = np.full(shots, 1.0 / shots)

_REAL_DIAG = _cipsi._Subspace.diag
diag_log = []


def spy_diag(self, str_a, str_b):
    t0 = time.perf_counter()
    res = _REAL_DIAG(self, str_a, str_b)
    wall = time.perf_counter() - t0
    E, c2d, sa, sb = res
    dim = len(sa) * len(sb)
    diag_log.append({"dim": int(dim), "backend": self.backend,
                     "E": float(E), "wall": round(wall, 1)})
    print(f"[diag] round#{len(diag_log):2d} backend={self.backend:3s} "
          f"dim={dim:8d} E={E:.10f} wall={wall:.1f}s", flush=True)
    return res


def main():
    _cipsi._Subspace.diag = spy_diag
    t0 = time.perf_counter()
    e, det = tc_sqd.solve_sqd_improved(
        h1e, eri, norb, nelec,
        bitstring_matrix=bsm, probabilities=probs,
        max_strings=100000, n_active_per_round=30,
        rand_seed=seed, ecore=ecore, return_details=True,
        tail_suppression=True, tail_max_draw_factor=10, tail_shots_ref=100,
        backend="gpu",
    )
    wall = time.perf_counter() - t0
    print(f"[e2e-gpu] wall={wall:.1f}s E={e:.10f} dim={det['dim']} "
          f"err={abs(e - E_REF):.2e}", flush=True)
    print(f"[e2e-gpu] E_direct={det['E_direct']} E_PT2={det['E_PT2']}", flush=True)

    result = {
        "backend": "gpu", "wall": wall, "E": float(e),
        "E_direct": float(det["E_direct"]), "E_PT2": float(det["E_PT2"]),
        "dim": int(det["dim"]), "err_vs_fci": float(abs(e - E_REF)),
        "diag_calls": diag_log,
    }
    if os.path.exists(OUT):
        with open(OUT) as f:
            data = json.load(f)
    else:
        data = {}
    data.setdefault("e2e", {})["gpu_rerun"] = result
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(f"[bench] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
