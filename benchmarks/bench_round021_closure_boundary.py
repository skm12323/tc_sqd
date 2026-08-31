"""bench_round021_closure_boundary: 截断 coverage_closure 在 11.78M 维全电子活性空间上的行为刻画。

配方与设计遵循 theory.md §2 及其附录 A（P0a 定版修正）：
- 体系: _n2_1414_ints.npz (norb=14, nelec=(7,7), 全空间 3432^2 = 11,778,624)
- 共享参数: n_active_per_round=90, dom_thresh=1e-3, pt2_floor=1e-7,
           tail_suppression=False, warm_start=True, eigsh_tol=1e-6,
           rand_seed=0, backend="gpu"
- 臂间不对称: ON (coverage_closure=True, max_rounds=5) vs OFF (coverage_closure=False, max_rounds=50)
- 实验矩阵:
    * 主网格 @500 shots: ms in {100, 1500, 2000, 2500} x {on, off}
    * 抽查 @100 shots: ms in {548, 1500} x {on, off}
    * HCI 参照: solve_hci eps_hb in {1e-2, 5e-3, 2e-3, 1e-3}
"""

import argparse
import collections
import functools
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def get_git_sha() -> str:
    """获取当前 HEAD 的 short git commit sha（meta 溯源）。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_integrals(ints_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """读取 N2 14o 体系积分与 ecore（_n2_1414_ints.npz，活跃缓存）。"""
    d = np.load(ints_path)
    return d["h1e"], d["eri"], float(d["ecore"])


def load_reference_energy(ref_path: str) -> Optional[float]:
    """读取全空间真基态参考能量（含 ecore），不存在返回 None（err 字段落 null）。"""
    if not os.path.exists(ref_path):
        return None
    d = np.load(ref_path)
    for key in ("e_ref_total", "E", "e"):
        if key in d:
            return float(d[key])
    return None


def generate_bitstrings(norb: int, max_shots: int = 500, seed: int = 0) -> np.ndarray:
    """生成固定 RNG 采样的共享位串矩阵 (max_shots x 2*norb)。

    rng(0) 一次性生成 500x28 >0.5；@100 抽查取前 100 行（两臂共享同一份）。
    """
    rng = np.random.default_rng(seed)
    return rng.random((max_shots, 2 * norb)) > 0.5


def compute_popcount_excitation_histogram(
    strs: Sequence[int], hf_str: int = 0b1111111
) -> Dict[str, int]:
    """计算字符串集合相对 HF 串 (0b1111111) 的激发阶直方图。

    激发阶 = popcount(s XOR hf) / 2（P2c 口径，键为字符串化的整数阶）。
    """
    hist: Dict[int, int] = collections.Counter()
    for s in strs:
        order = bin(int(s) ^ hf_str).count("1") // 2
        hist[order] += 1
    return {str(k): int(v) for k, v in sorted(hist.items())}


class Instrumentation:
    """零 src 改动的 monkeypatch 观测采集（模式同 round_020 测试）。

    三个挂点：
    1. ``tc_sqd.cipsi.eigsh`` 包装：``v0 is None`` 时注入
       ``np.random.default_rng(1234).standard_normal(op.shape[0])``，
       钉死 ARPACK 随机起点 → 双臂全确定性（P1c |ΔE|<1e-12 前提）。
    2. ``_Subspace.diag`` 包装：记录 (n_str_a, n_str_b, wall, n_mv 增量)，
       并把 set(sa)/set(sb) 快照存内存（≤2500 int × ~20 次，开销可忽略）。
       事后可重构：PT2 每轮增长、BFS 进场点（主循环结束后第一次 diag）、
       每层 BFS 注入串集合（相邻快照差集）、注入串激发阶直方图（P2c）。
    3. ``_Subspace.pt2_matrix_elements`` 包装：记录 (n_cand, n_str_a 调用时,
       n_str_b 调用时, wall)。PT2 循环调用 vs BFS 调用靠序号对齐 diag 快照：
       BFS 的 pt2_me 发生在 BFS 进场 diag（diag 计数 > 主循环轮数）之后。
    """

    def __init__(self, norb: int):
        self.norb = norb
        self.diag_records: List[Dict[str, Any]] = []
        self.pt2_records: List[Dict[str, Any]] = []
        self.snapshots: List[Tuple[Set[int], Set[int]]] = []
        self._orig_eigsh = None
        self._orig_diag = None
        self._orig_pt2 = None

    @staticmethod
    def _pinned_v0_eigsh(orig_eigsh):
        @functools.wraps(orig_eigsh)
        def wrapper(op, *args, **kw):
            if kw.get("v0") is None:
                kw["v0"] = np.random.default_rng(1234).standard_normal(op.shape[0])
            return orig_eigsh(op, *args, **kw)
        return wrapper

    def _wrap_diag(self, orig_diag):
        @functools.wraps(orig_diag)
        def wrapper(sub, str_a, str_b, *args, **kw):
            t0 = time.perf_counter()
            E, c2d, sa, sb = orig_diag(sub, str_a, str_b, *args, **kw)
            wall = time.perf_counter() - t0
            self.diag_records.append({
                "n_str_a": int(len(sa)), "n_str_b": int(len(sb)),
                "wall_s": wall, "n_mv": int(sub.last_n_mv),
            })
            self.snapshots.append((set(int(x) for x in sa),
                                   set(int(x) for x in sb)))
            # OOM 护栏 (round_021 主网格实测): cupy 内存池按尺寸分档持有已释放
            # 块, 轮次间 t1=(norb²,na,nb) 随串数增长逐档累积 → 池可胀破 16GB
            # 显存 (1500:on 完成后 1500:off 起步即 OOM)。每次 diag 后释放空闲
            # 块; 存活的 _eri1_* 缓存/当前数组有引用, 不受影响。bench 侧卫生
            # 措施, 与观测语义无关。
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
            return E, c2d, sa, sb
        return wrapper

    def _wrap_pt2(self, orig_pt2):
        @functools.wraps(orig_pt2)
        def wrapper(sub, str_a, str_b, cand, c2d, sa, sb, *args, **kw):
            t0 = time.perf_counter()
            me = orig_pt2(sub, str_a, str_b, cand, c2d, sa, sb, *args, **kw)
            self.pt2_records.append({
                "n_cand": int(len(cand)),
                "n_str_a": int(len(str_a)), "n_str_b": int(len(str_b)),
                "wall_s": time.perf_counter() - t0,
            })
            return me
        return wrapper

    def __enter__(self):
        import tc_sqd.cipsi as cipsi_mod

        self._orig_eigsh = cipsi_mod.eigsh
        self._orig_diag = cipsi_mod._Subspace.diag
        self._orig_pt2 = cipsi_mod._Subspace.pt2_matrix_elements
        cipsi_mod.eigsh = self._pinned_v0_eigsh(self._orig_eigsh)
        cipsi_mod._Subspace.diag = self._wrap_diag(self._orig_diag)
        cipsi_mod._Subspace.pt2_matrix_elements = self._wrap_pt2(self._orig_pt2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import tc_sqd.cipsi as cipsi_mod

        cipsi_mod.eigsh = self._orig_eigsh
        cipsi_mod._Subspace.diag = self._orig_diag
        cipsi_mod._Subspace.pt2_matrix_elements = self._orig_pt2
        return False


def reconstruct_bfs_layers(
    snapshots: List[Tuple[Set[int], Set[int]]],
    n_main: int,
    has_closure: bool,
) -> Tuple[int, int, List[Set[int]], Set[int]]:
    """从 diag 快照序列重构 BFS 层注入（纯函数，可独立合成测试）。

    diag 调用时序（cipsi.py 逐行核对）：
    主循环每轮恰 1 次（:1390，在同轮 PT2 注入 :1445-1453 **之前**）
    → ON 臂播种 diag（:1505，快照 = 主循环退出时串集，含末轮 PT2 新增）
    → 每层生产性注入后 diag（:1563）→ 最终 diag（:1568，重复末层快照）。
    故快照下标：主循环 [0, n_main)，播种 = n_main，
    注入层 = [n_main+1, ...]，末尾最终 diag 与末层同集（差集为空，自动跳过）。
    OFF 臂无播种 diag，快照数恒 = n_main + 1（仅最终 diag）→ 恒返回 0 层。

    Returns
    -------
    (n_productive_layers, n_injected_total, layer_injected_sets, injected_union)
    """
    if not has_closure or len(snapshots) < n_main + 2:
        return 0, 0, [], set()
    prev = snapshots[n_main][0] | snapshots[n_main][1]  # 播种集作 baseline
    layer_sets: List[Set[int]] = []
    injected_all: Set[int] = set()
    for sa_s, sb_s in snapshots[n_main + 1:]:
        cur = sa_s | sb_s
        inj = cur - prev
        prev = cur
        if not inj:
            continue  # 最终 diag / 提前 break 的冗余快照
        layer_sets.append(inj)
        injected_all |= inj
    return len(layer_sets), len(injected_all), layer_sets, injected_all


def selftest_reconstruct_bfs() -> bool:
    """合成快照仿真自测（R4 Blocker 修复验证，覆盖「有注入」形态）。

    smoke 的零注入形态测不出 off-by-one，必须合成：主循环 5 次 diag
    + 播种（含末轮 PT2 新增）+ BFS 3 层注入 + 最终 diag（重复末层），
    验证重构 layers=3 且每层注入串集合精确正确；另测 ON 零注入与 OFF
    （无播种）两形态不误判。
    """
    def snap(*vals):
        s = set(vals)
        return (s, set(s))

    # 形态 1：主 5 轮 + 播种(+90,91) + 3 层注入 {10,11}/{12}/{13,14} + 最终
    main = [snap(*range(1, r + 1)) for r in (1, 2, 3, 4, 5)]
    seed_set = set(range(1, 6)) | {90, 91}
    l1 = seed_set | {10, 11}
    l2 = l1 | {12}
    l3 = l2 | {13, 14}
    snaps = main + [snap(*seed_set), snap(*l1), snap(*l2), snap(*l3), snap(*l3)]
    n_l, n_tot, sets, union = reconstruct_bfs_layers(snaps, 5, True)
    ok1 = (n_l == 3 and n_tot == 5 and union == {10, 11, 12, 13, 14}
           and sets[0] == {10, 11} and sets[1] == {12} and sets[2] == {13, 14})
    # 形态 2：ON 零注入（smoke 形态）→ 0 层
    snaps_z = main + [snap(*seed_set), snap(*seed_set)]
    ok2 = reconstruct_bfs_layers(snaps_z, 5, True)[:2] == (0, 0)
    # 形态 3：OFF（主 5 + 最终，无播种）→ 0 层
    snaps_o = main + [snap(1, 2, 3, 4, 5)]
    ok3 = reconstruct_bfs_layers(snaps_o, 5, False)[:2] == (0, 0)
    return bool(ok1 and ok2 and ok3)


def run_single_sqd_point(
    h1e: np.ndarray,
    eri: np.ndarray,
    ecore: float,
    norb: int,
    nelec: Tuple[int, int],
    bitstrings: np.ndarray,
    shots: int,
    ms: int,
    arm: str,
    e_ref: Optional[float],
    backend: str = "gpu",
) -> Dict[str, Any]:
    """运行单点 SQD active 实验并收集自描述诊断指标。

    arm: "on" -> coverage_closure=True, max_rounds=5;
         "off" -> coverage_closure=False, max_rounds=50。
    其余配方逐位相同（附录 A.4 定版）。
    """
    import tc_sqd

    bsm = bitstrings[:shots]
    probs = np.full(shots, 1.0 / shots)
    recipe = dict(
        max_strings=ms, n_active_per_round=90, dom_thresh=1e-3,
        pt2_floor=1e-7, tail_suppression=False, warm_start=True,
        eigsh_tol=1e-6, rand_seed=0, ecore=ecore, backend=backend,
        verbose=False,
    )
    clo = arm == "on"
    mr = 5 if clo else 50

    traj: List[dict] = []
    state: List = []
    with Instrumentation(norb) as inst:
        t0 = time.perf_counter()
        e_total = tc_sqd.solve_sqd_active(
            h1e, eri, norb, nelec, bitstring_matrix=bsm, probabilities=probs,
            coverage_closure=clo, max_rounds=mr, trajectory=traj,
            state_out=state, **recipe)
        wall = time.perf_counter() - t0

    main_pts = [p for p in traj if p["round"] > 0]
    n_main = len(main_pts)
    final_pt = traj[-1]
    dim_final = int(final_pt["dim"])
    n_str_a, n_str_b = len(state[0][1]), len(state[0][2])
    s_main_end = int(round(main_pts[-1]["dim"] ** 0.5)) if main_pts else 0

    # ---- 从 diag 快照重构 BFS 机制（R4 Blocker 修复版，见 reconstruct_bfs_layers）
    snaps = inst.snapshots
    bfs_layers, bfs_injected_total, bfs_layer_sets, injected_all = \
        reconstruct_bfs_layers(snaps, n_main, clo)
    layer_records: List[Dict[str, Any]] = [
        {"layer": i + 1, "n_injected": int(len(inj)),
         "excitation_hist": compute_popcount_excitation_histogram(inj)}
        for i, inj in enumerate(bfs_layer_sets)
    ]

    # diag/pt2 分段计时与 n_mv 统计
    diag_wall = sum(r["wall_s"] for r in inst.diag_records)
    pt2_wall = sum(r["wall_s"] for r in inst.pt2_records)
    n_mv_total = sum(r["n_mv"] for r in inst.diag_records)
    per_mv_s = diag_wall / n_mv_total if n_mv_total > 0 else None

    # 验收门（记录不阻断）
    budget_fill_ok = bool(
        (0.98 * ms <= n_str_a <= ms) and (0.98 * ms <= n_str_b <= ms))
    zero_activation = bool(clo and bfs_injected_total == 0)

    return {
        "shots": shots, "ms": ms, "arm": arm,
        "coverage_closure": clo, "max_rounds": mr,
        "E_total": float(e_total),
        "err_vs_ref": (abs(float(e_total) - e_ref) if e_ref is not None else None),
        "ref_available": e_ref is not None,
        "wall_s": wall,
        "trajectory": [
            {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
             for k, v in p.items()} for p in traj
        ],
        "n_rounds_main": n_main,
        "dim_final": dim_final,
        "n_str_final": [int(n_str_a), int(n_str_b)],
        "main_loop_end_strings": s_main_end,
        "sigma2_final": float(final_pt["sigma2"]),
        "e_pt2_final": float(final_pt["e_pt2"]),
        # ON 臂 BFS 进场串数 = 播种 diag（:1505）快照；OFF 无 BFS 段 → None
        "bfs_entry_strings": (int(len(snaps[n_main][0]))
                              if clo and len(snaps) > n_main else None),
        # 守卫（R4 Major-1 修正口径）：主循环 diag（:1390）在同轮 PT2 注入
        # （:1445-1453）**之前**，故播种快照 ⊋ 末轮 diag 快照；正确校验是对
        # trajectory 末轮点（:1469 记录注入后 dim）——播种集串数应等于
        # round(sqrt(main_pts[-1].dim))。
        "bfs_seed_matches_traj_final_dim": (
            bool(int(len(snaps[n_main][0])) == s_main_end)
            if clo and len(snaps) > n_main and main_pts else None),
        # 独立交叉校验（review Minor-7）：trajectory 直接读出聚合注入量
        # = 终态串数 − 主循环末串数；应与 bfs_injected_total 相等（闭壳层）。
        "bfs_injected_traj_xcheck": (
            int(n_str_a) - s_main_end if clo and main_pts else None),
        "bfs_layers": bfs_layers,
        "bfs_injected_total": int(bfs_injected_total),
        "bfs_layer_records": layer_records,
        "bfs_excitation_hist": compute_popcount_excitation_histogram(
            injected_all),
        "diag_wall_s": diag_wall, "pt2_wall_s": pt2_wall,
        "n_mv_total": int(n_mv_total), "per_mv_s": per_mv_s,
        "diag_records": inst.diag_records,
        "pt2_records": inst.pt2_records,
        "budget_fill_ok": budget_fill_ok,
        "zero_activation": zero_activation,
    }


def run_hci_point(
    h1e: np.ndarray,
    eri: np.ndarray,
    ecore: float,
    norb: int,
    nelec: Tuple[int, int],
    eps_hb: float,
    e_ref: Optional[float],
    wall_cap_s: float = 1800.0,
) -> Dict[str, Any]:
    """运行单点 solve_hci（CPU，return_details）作为确定性参照曲线。

    单点 wall cap 30 min：超时记录 timed_out=true 后跳过（SIGALRM，POSIX only）。
    """
    import tc_sqd

    def _alarm_handler(signum, frame):
        raise TimeoutError(f"solve_hci eps={eps_hb} 超过 wall cap {wall_cap_s}s")

    rec: Dict[str, Any] = {
        "eps_hb": eps_hb, "backend": "cpu",
        "E_total": None, "e_pt2": None, "dim": None,
        "wall_s": None, "err_vs_ref": None, "timed_out": False,
        "error": None,
    }
    t0 = time.perf_counter()
    try:
        import signal
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(wall_cap_s))
    except (AttributeError, ValueError, ImportError):
        old_handler = None  # 非 POSIX / 主线程外：无 alarm 兜底
    try:
        e_total, e_pt2, dim = tc_sqd.solve_hci(
            h1e, eri, norb, nelec, eps_hb=eps_hb, ecore=ecore,
            return_details=True)
        rec.update(
            E_total=float(e_total), e_pt2=float(e_pt2), dim=int(dim),
            wall_s=time.perf_counter() - t0,
            err_vs_ref=(abs(float(e_total) - e_ref)
                        if e_ref is not None else None),
        )
    except TimeoutError:
        rec["timed_out"] = True
        rec["wall_s"] = time.perf_counter() - t0
    except Exception as exc:  # 记录不阻断
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["wall_s"] = time.perf_counter() - t0
    finally:
        try:
            if old_handler is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass
    return rec


# --------------------------------------------------------------------------- #
#  网格定义与 CLI
# --------------------------------------------------------------------------- #
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTS = "_n2_1414_ints.npz"                    # 相对项目根（禁止绝对路径入库）
REF = "_n2_1414_ref.npz"
OUT = os.path.join("benchmarks", "_round021_closure.json")
NORB, NELEC = 14, (7, 7)
SMOKE = os.environ.get("ROUND021_SMOKE") == "1"

MAIN_GRID_MS = [100, 1500, 2000, 2500]        # 附录 A.4 定版
MAIN_SHOTS = 500
SPOT_GRID_MS = [548, 1500]                    # shots 轴抽查（P1b）
SPOT_SHOTS = 100
HCI_EPS = [1e-2, 5e-3, 2e-3, 1e-3]
HCI_WALL_CAP_S = 1800.0                       # 单点 30 min
ARMS = ("on", "off")


def build_grid() -> List[Tuple[int, int, str]]:
    """返回 (shots, ms, arm) 列表；SMOKE 模式缩为 ms={100}@shots=50 x 两臂。"""
    if SMOKE:
        return [(50, 100, arm) for arm in ARMS]
    grid = [(MAIN_SHOTS, ms, arm) for ms in MAIN_GRID_MS for arm in ARMS]
    grid += [(SPOT_SHOTS, ms, arm) for ms in SPOT_GRID_MS for arm in ARMS]
    return grid


def parse_only(specs: List[str]) -> List[Tuple[int, int, str]]:
    """解析 --only "500:1500,2000:on" / "500:1500,2000:on,off"（可重复）。

    语法: shots:ms[,ms]:arm[,arm]。空臂段 = 两臂。
    """
    keys: List[Tuple[int, int, str]] = []
    for spec in specs:
        for item in spec.split():
            parts = item.split(":")
            shots_s, ms_s = parts[0], parts[1]
            arm_s = parts[2] if len(parts) > 2 else ""
            arms = arm_s.split(",") if arm_s else list(ARMS)
            for ms in ms_s.split(","):
                for arm in arms:
                    keys.append((int(shots_s), int(ms), arm))
    return keys


def main():
    ap = argparse.ArgumentParser(
        description="round_021 截断 coverage_closure 行为刻画 (14,14) 全空间")
    ap.add_argument("--only", action="append", default=[],
                    help='过滤 run，例: --only "500:1500,2000:on"（可重复）')
    ap.add_argument("--skip-hci", action="store_true",
                    help="跳过 HCI 参照曲线（最低优先级）")
    ap.add_argument("--force", action="store_true",
                    help="忽略已有 JSON 键强制重跑")
    ap.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    h1e, eri, ecore = load_integrals(os.path.join(BASE, INTS))
    e_ref = load_reference_energy(os.path.join(BASE, REF))
    bsm500 = generate_bitstrings(NORB, max_shots=500, seed=0)

    if SMOKE:
        # R4 Blocker 修复的合成仿真（smoke 本身是零注入形态，测不出
        # off-by-one；此处覆盖「有注入」3 层形态 + 零注入 + OFF 形态）
        ok = selftest_reconstruct_bfs()
        print(f"[selftest] reconstruct_bfs 合成仿真: "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            sys.exit(1)

    out_path = os.path.join(BASE, OUT)
    res: Dict[str, Any] = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                res = json.load(f)
        except Exception:
            res = {}
    res["meta"] = {
        "date": time.strftime("%Y-%m-%d"),
        "git_sha": get_git_sha(),
        "smoke": SMOKE,
        "config_echo": {
            "system": "N2/cc-pVDZ R=3.0A (14e,14o) full 3432^2=11778624",
            "integrals": INTS, "ref_npz": REF,
            "shared_recipe": dict(
                n_active_per_round=90, dom_thresh=1e-3, pt2_floor=1e-7,
                tail_suppression=False, warm_start=True, eigsh_tol=1e-6,
                rand_seed=0, backend=args.backend),
            "arms": {"on": dict(coverage_closure=True, max_rounds=5),
                     "off": dict(coverage_closure=False, max_rounds=50)},
            "main_grid": dict(shots=MAIN_SHOTS, ms=MAIN_GRID_MS),
            "spot_grid": dict(shots=SPOT_SHOTS, ms=SPOT_GRID_MS),
            "hci_eps": HCI_EPS, "hci_wall_cap_s": HCI_WALL_CAP_S,
            "bitstrings": "rng(0).random((500,28))>0.5, probs uniform; "
                          "@100 = first 100 rows (shared)",
        },
    }
    runs: Dict[str, Any] = res.setdefault("runs", {})

    def _save():
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    grid = build_grid()
    if args.only:
        want = set(parse_only(args.only))
        grid = [k for k in grid if k in want]
    print(f"[grid] {len(grid)} run(s), smoke={SMOKE}, "
          f"ref={'yes' if e_ref is not None else 'null'}", flush=True)

    for shots, ms, arm in grid:
        key = f"{shots}:{ms}:{arm}"
        if key in runs and not args.force:
            print(f"[skip] {key} 已存在（--force 重跑）", flush=True)
            continue
        print(f"[run] {key} ...", flush=True)
        rec = run_single_sqd_point(
            h1e, eri, ecore, NORB, NELEC, bsm500, shots, ms, arm,
            e_ref, backend=args.backend)
        runs[key] = rec
        _save()
        print(f"[done] {key} E={rec['E_total']:.8f} wall={rec['wall_s']:.0f}s "
              f"dim={rec['dim_final']} bfs_layers={rec['bfs_layers']} "
              f"bfs_inj={rec['bfs_injected_total']} "
              f"fill_ok={rec['budget_fill_ok']} "
              f"zero_act={rec['zero_activation']}", flush=True)

    if not args.skip_hci and not SMOKE:
        hci_runs: Dict[str, Any] = res.setdefault("hci", {})
        for eps in HCI_EPS:
            key = f"{eps:g}"
            if key in hci_runs and not args.force:
                print(f"[skip] hci eps={key} 已存在", flush=True)
                continue
            print(f"[hci] eps={key} (CPU, cap {HCI_WALL_CAP_S:.0f}s) ...",
                  flush=True)
            hci_runs[key] = run_hci_point(
                h1e, eri, ecore, NORB, NELEC, eps, e_ref,
                wall_cap_s=HCI_WALL_CAP_S)
            _save()
            r = hci_runs[key]
            print(f"[hci done] eps={key} E={r['E_total']} dim={r['dim']} "
                  f"wall={r['wall_s']} timed_out={r['timed_out']}",
                  flush=True)

    if SMOKE:
        # 自测自检：钉 v0 后两臂应语义相同（P1c）。阈值 1e-11 为 M0 定版修订
        # （A.4 原 1e-12）；实测结构性噪声底 5.5e-12 = ON 臂播种 diag +
        # warm_start 投影链导致的 ARPACK 停机差（确定性、非随机泄漏，
        # review C.1 已接受该归因）。
        try:
            e_on = runs["50:100:on"]["E_total"]
            e_off = runs["50:100:off"]["E_total"]
            d_e = abs(e_on - e_off)
            print(f"[smoke-check] |ΔE(on-off)|={d_e:.3e} "
                  f"(threshold 1e-11: {'PASS' if d_e < 1e-11 else 'FAIL'})",
                  flush=True)
        except KeyError:
            pass

    _save()
    print(f"[exit] 输出 {OUT}", flush=True)


if __name__ == "__main__":
    main()
