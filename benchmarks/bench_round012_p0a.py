"""round_012 P0a: 零代码参数门控——triple 注入 pt2_floor 断链诊断。

问题：round_008 triple_injection=True 但 dim 不变（824,464 = 908²，缺 16 字符串未补全）。
根因假设：cipsi.py:1301 的 ``if abs(v) < pt2_floor: break`` 在默认 pt2_floor=1e-7 处
过滤掉低分中间父串，导致 BFS 链断裂，永远到不了缺失的 16 个字符串。
round_008 从未扫 pt2_floor→0 —— 本实验补这个缺口。

设计：直接走 solve_sqd_active（不绕 solve_sqd_best 的 4× evpt2 放大成本）。
固定：N2/cc-pVDZ (12,12) @500 shots, seed=0, tail_suppression=C1-v2, warm_start, GPU hybrid。
扫描：pt2_floor ∈ {1e-7(默认=round008), 1e-12, 0}，n_triples_per_round=0（无 cap）。
诊断指标（非只看能量）：
  - dim 是否从 824,464(908²) → 853,776(924² = 全空间 FCI)
  - err 是否随之 →0（全空间 = 精确 FCI）
  - triple pass 的 diag 调用串数跳变（instrumented _Subspace.diag）
  - sigma² 是否 →0（子空间外方差）
"""
import os
import sys
import time
import json
import math
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

# ---- 固定输入（禁现跑 SCF，近简并轨道多线程非确定性）----
npz = np.load(os.path.join(BASE, "_n2_1212_ints.npz"))
h1e, eri = npz["h1e"], npz["eri"]
ecore = float(npz["ecore"]) if "ecore" in npz else 0.0
E_FCI = float(npz["e_ref"])  # 全空间 FCI 真基态 -108.76868573746447
NCAS, NELEC = 12, (6, 6)
SHOTS = 500

bsm = np.random.default_rng(0).random((SHOTS, 2 * NCAS)) > 0.5
probs = np.full(SHOTS, 1.0 / SHOTS)

from tc_sqd.cipsi import solve_sqd_active, _Subspace

# ---- instrument _Subspace.diag: 记录每次调用的 (n_str_a, n_str_b, n_mv) ----
_orig_diag = _Subspace.diag
_diag_log = []  # list of (n_str_a, n_str_b, n_mv)


def _instrumented(self, sa, sb):
    r = _orig_diag(self, sa, sb)
    n_mv = getattr(self, "last_n_mv", 0)
    _diag_log.append((len(sa), len(sb), n_mv))
    return r


_Subspace.diag = _instrumented

FULL_DIM = 924 * 924  # 853,776 全空间 C(12,6)²

configs = [
    ("baseline_no_triple", None),   # triple_injection=False 对照
    ("triple_floor1e-7", 1e-7),     # = round_008 设置（默认）
    ("triple_floor1e-12", 1e-12),
    ("triple_floor0", 0.0),
]

results = []
for label, pt2_floor in configs:
    _diag_log.clear()
    traj = []
    triple_kwargs = ({"triple_injection": False} if pt2_floor is None
                     else {"triple_injection": True, "n_triples_per_round": 0,
                           "pt2_floor": pt2_floor})
    t0 = time.perf_counter()
    E = solve_sqd_active(
        h1e, eri, NCAS, NELEC, ecore=ecore,
        bitstring_matrix=bsm, probabilities=probs,
        max_strings=None, n_active_per_round=30, rand_seed=0,
        tail_suppression=True, tail_shots_ref=100,
        backend="gpu", warm_start=True, verbose=False,
        trajectory=traj, **triple_kwargs,
    )
    wall = time.perf_counter() - t0
    err = abs(float(E) - E_FCI)

    final = traj[-1] if traj else {}
    dim = int(final.get("dim", -1))
    sigma2 = float(final.get("sigma2", -1.0))
    e_pt2 = float(final.get("e_pt2", 0.0))
    n_str = int(round(math.sqrt(dim))) if dim > 0 else -1
    n_mv_total = sum(d[2] for d in _diag_log if d[2])
    # triple pass 串数跳变：找 diag_log 中 n_str_a 的最大值（post-triple）
    max_n_str = max((d[0] for d in _diag_log), default=0)

    row = {
        "label": label, "pt2_floor": pt2_floor, "wall_s": round(wall, 1),
        "E": float(E), "err": float(err), "dim": dim, "n_str": n_str,
        "max_n_str_diag": max_n_str, "sigma2": sigma2, "e_pt2": e_pt2,
        "n_mv_total": n_mv_total, "n_diag_calls": len(_diag_log),
    }
    results.append(row)
    print(f"\n=== {label} ===", flush=True)
    print(f"  E={E:.12f}  err={err:.2e}  dim={dim}  n_str={n_str}  "
          f"max_str={max_n_str}  sigma2={sigma2:.2e}  wall={wall:.0f}s  "
          f"n_mv={n_mv_total}  n_diag={len(_diag_log)}", flush=True)
    # 打印 triple 前后的串数跳变（diag_log 的 n_str_a 序列）
    strseq = [d[0] for d in _diag_log]
    print(f"  diag str_a 序列(前12): {strseq[:12]} ... 末3: {strseq[-3:]}", flush=True)

print("\n\n" + "=" * 78)
print("=== P0a 诊断汇总 ===")
print(f"{'label':<22} {'pt2_floor':<11} {'dim':>8} {'n_str':>6} "
      f"{'err':>10} {'sigma2':>10} {'wall':>6}")
print("-" * 78)
for r in results:
    pf = "off" if r["pt2_floor"] is None else f"{r['pt2_floor']:.0e}"
    print(f"{r['label']:<22} {pf:<11} {r['dim']:>8} {r['n_str']:>6} "
          f"{r['err']:>10.2e} {r['sigma2']:>10.2e} {r['wall_s']:>6}")
print(f"\n全空间 FCI: dim={FULL_DIM}, n_str=924, err=0 (精确)")
print(f"当前覆盖:   dim=824464, n_str=908 (缺 16)")
print(f"E_FCI(e_ref) = {E_FCI:.12f}")

# 判定
best = max(results, key=lambda r: r["dim"])
if best["dim"] == FULL_DIM:
    print(f"\n[P0a 判定] dim 补全到全空间（{best['label']}）——覆盖断链被 pt2_floor 门控"
          f"修复；err={best['err']:.2e}（应≈0）")
else:
    print(f"\n[P0a 判定] 最大 dim={best['dim']}（n_str={best['n_str']}）仍未达全空间"
          f" {FULL_DIM}——pt2_floor 扫描不足以补全；需 P0b bounded frontier 代码改动")

out = os.path.join(BASE, "benchmarks", "_round012_p0a_coverage.json")
with open(out, "w") as f:
    json.dump({"e_fci": E_FCI, "full_dim": FULL_DIM, "results": results}, f, indent=2)
print(f"\nsaved {out}")
