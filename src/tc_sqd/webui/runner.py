"""tc_sqd.webui.runner —— 计算任务管理 (单任务线程 + 多 seed + 参考能量)。

设计要点 (与库其余部分同一套约定):
- **单任务串行**: 本地单用户工具, 同一时间只允许一个计算任务 (避免 GPU/CPU
  抢占互相污染计时 —— round_004 教训); 新任务在运行中被拒 (HTTP 409)。
- **取消是协作式**: 只在 seed 之间 / 阶段之间生效 (solver 内部无从中断),
  已完成的 seed 结果保留, UI 如实标注。
- **进度实时可见**: solve_sqd_active 的 ``trajectory`` 是边跑边 append 的
  list, API 线程直接读其长度与末点 → 每轮 E/dim 实时刷新 (GIL 下安全)。
- 多 seed 平均 = 同一配方对 base_seed..base_seed+n-1 各跑一遍 (位串采样
  与 rand_seed 均随 seed 变化), 汇总 mean±std。
"""

from __future__ import annotations

import math
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["Job", "JobManager", "BusyError", "generate_bitstrings"]


class BusyError(RuntimeError):
    """已有一个任务在运行。"""


# ---------------------------------------------------------------------------
#  位串采样 (列约定: 左半 β | 右半 α, 每半从右往左 = 轨道 0..norb-1)
# ---------------------------------------------------------------------------
def _hf_bitrow(norb: int, na: int, nb: int) -> np.ndarray:
    """HF 参考行: 最低 na/nb 个轨道占据 (轨道 0 = 每半最右列)。"""
    row = np.zeros(2 * norb, dtype=bool)
    for j in range(nb):
        row[norb - 1 - j] = True
    for j in range(na):
        row[2 * norb - 1 - j] = True
    return row


def generate_bitstrings(norb: int, na: int, nb: int, shots: int, seed: int,
                        mode: str = "uniform",
                        hf_weight: float = 0.8) -> np.ndarray:
    """生成 (shots, 2*norb) 位串矩阵。

    - ``uniform``: 均匀随机 (round_021 基准口径; 配置恢复会把每行投影回
      正确粒子数扇区, 错误电子数的行不产生非法 det)。
    - ``hf``: 每位 1 的概率 = w·HF 占据 + (1-w)·0.5; w=0 退化为均匀,
      w=1 全 HF 行。w 越大采样越贴近参考态 (单参考体系收敛更快)。
    """
    rng = np.random.default_rng(seed)
    if mode == "hf":
        base = _hf_bitrow(norb, na, nb).astype(np.float64)
        p = float(hf_weight) * base + (1.0 - float(hf_weight)) * 0.5
        return rng.random((shots, 2 * norb)) < p[None, :]
    return rng.random((shots, 2 * norb)) > 0.5


def _project_seed_bitstrings(bsm: np.ndarray, norb: int, na: int, nb: int,
                             seed: int) -> np.ndarray:
    """CIPSI/HCI 种子位串的粒子数投影 (round_018 坑防护)。

    ``solve_cipsi``/``solve_hci`` 的 ``seed_bitstring_matrix`` **不经**
    ``recover_configurations`` 修复直接转 CI 串 —— 随机行携带错误电子数
    会触发 pyscf C 核 native crash (segfault/malloc abort, round_018 实测)。
    SQD active 内部每轮自带恢复故不需要; 这里只对经典方法做投影。
    """
    import tc_sqd

    occ = np.full(norb, 0.5)   # 无偏占据: 仅修粒子数, 不引入轨道倾向
    probs = np.full(bsm.shape[0], 1.0 / bsm.shape[0])
    rec, _ = tc_sqd.recover_configurations(
        bsm, probs, (occ, occ), na, nb, rand_seed=seed)
    return rec


# ---------------------------------------------------------------------------
#  参数读取助手 (表单值都是标量, 统一容错)
# ---------------------------------------------------------------------------
def _i(d: Dict[str, Any], k: str, default: int) -> int:
    v = d.get(k, default)
    return default if v in (None, "") else int(v)


def _f(d: Dict[str, Any], k: str, default: Optional[float]) -> Optional[float]:
    v = d.get(k, default)
    return default if v in (None, "") else float(v)


def _b(d: Dict[str, Any], k: str, default: bool) -> bool:
    v = d.get(k, default)
    return default if v is None else bool(v)


def _pyify(obj: Any) -> Any:
    """numpy 标量/数组 → JSON 安全的原生类型。"""
    if isinstance(obj, dict):
        return {k: _pyify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_pyify(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
#  方法分发
# ---------------------------------------------------------------------------
def run_full_space_reference(sysd: Dict[str, Any]) -> Dict[str, Any]:
    """全空间 SCI 真基态 (round_016 数据纪律: 参考统一库内 solve_sci)。"""
    from pyscf.fci import cistring

    import tc_sqd

    norb, (na, nb) = sysd["norb"], sysd["nelec"]
    t0 = time.perf_counter()
    sa = cistring.make_strings(range(norb), na)
    sb = sa.copy() if na == nb else cistring.make_strings(range(norb), nb)
    res = tc_sqd.solve_sci((sa, sb), sysd["h1e"], sysd["eri"], norb,
                           (na, nb))
    return {
        "E_total": float(res.energy) + sysd["ecore"],
        "e_active": float(res.energy), "ecore": sysd["ecore"],
        "dim": int(len(sa)) * int(len(sb)),
        "wall_s": time.perf_counter() - t0,
        "note": "solve_sci 全空间 (库内统一参考口径)",
    }


def _run_seed(job: "Job", sysd: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """单 seed 单方法一次完整计算 → 结果 dict (含 trajectory)。"""
    import tc_sqd

    p = job.params
    method = p["method"]
    mp: Dict[str, Any] = p.get("params", {})
    sampling: Dict[str, Any] = p.get("sampling", {})
    norb, (na, nb) = sysd["norb"], sysd["nelec"]

    shots = max(1, _i(sampling, "shots", 500))
    hw = _f(sampling, "hf_weight", None)
    use_sampling = method in ("sqd_active", "cipsi") or (
        method == "hci" and _b(mp, "use_seed", False))
    bsm = (generate_bitstrings(
        norb, na, nb, shots, seed,
        mode=str(sampling.get("mode", "uniform")),
        hf_weight=0.8 if hw is None else hw)
        if use_sampling else None)

    traj: List[dict] = []
    usage: List = []
    t0 = time.perf_counter()
    extras: Dict[str, Any] = {}

    if method == "sqd_active":
        kw: Dict[str, Any] = dict(
            bitstring_matrix=bsm, probabilities=np.full(shots, 1.0 / shots),
            ecore=sysd["ecore"], rand_seed=seed,
            trajectory=traj, usage=usage,
            n_active_per_round=_i(mp, "n_active_per_round", 50),
            dom_thresh=_f(mp, "dom_thresh", 1e-3),
            pt2_floor=_f(mp, "pt2_floor", 1e-7),
            max_rounds=_i(mp, "max_rounds", 10),
            tail_suppression=_b(mp, "tail_suppression", False),
            tail_shots_ref=_i(mp, "tail_shots_ref", 0),
            prune_keep=_f(mp, "prune_keep", 1.0),
            warm_start=_b(mp, "warm_start", False),
            coverage_closure=_b(mp, "coverage_closure", False),
            backend=str(mp.get("backend", "cpu")),
            eigsh_tol=_f(mp, "eigsh_tol", None),
        )
        ms = _i(mp, "max_strings", 0)
        if ms > 0:
            kw["max_strings"] = ms
        et = _f(mp, "energy_tol", None)
        if et is not None:
            kw["energy_tol"] = et
        job.live_traj = traj
        E = tc_sqd.solve_sqd_active(
            sysd["h1e"], sysd["eri"], norb, (na, nb), **kw)
        job.live_traj = None
        extras = {"shots_used": usage[0] if usage else shots,
                  "final_dim": traj[-1]["dim"] if traj else None,
                  "n_rounds": len([t for t in traj if t["round"] > 0]),
                  "sigma2_final": traj[-1]["sigma2"] if traj else None,
                  "e_pt2_final": traj[-1]["e_pt2"] if traj else None}
    elif method == "cipsi":
        ms = _i(mp, "max_strings", 0)
        seed_bsm = _project_seed_bitstrings(bsm, norb, na, nb, seed)
        E = tc_sqd.solve_cipsi(
            sysd["h1e"], sysd["eri"], norb, (na, nb),
            seed_bitstring_matrix=seed_bsm,
            max_strings=ms if ms > 0 else None,
            dom_thresh=_f(mp, "dom_thresh", 1e-3),
            pt2_floor=_f(mp, "pt2_floor", 1e-7),
            max_iter=_i(mp, "max_iter", 40),
            ecore=sysd["ecore"], backend=str(mp.get("backend", "cpu")),
        )
    elif method == "hci":
        kw = dict(
            eps_hb=_f(mp, "eps_hb", 1e-3),
            dom_thresh=_f(mp, "dom_thresh", 1e-3),
            pt2_floor=_f(mp, "pt2_floor", 1e-7),
            max_iter=_i(mp, "max_iter", 40),
            ecore=sysd["ecore"], return_details=True,
            backend=str(mp.get("backend", "cpu")),
        )
        if bsm is not None:
            kw["seed_bitstring_matrix"] = _project_seed_bitstrings(
                bsm, norb, na, nb, seed)
        E, e_pt2, dim = tc_sqd.solve_hci(sysd["h1e"], sysd["eri"], norb,
                                         (na, nb), **kw)
        extras = {"e_pt2": float(e_pt2), "dim": int(dim)}
    elif method == "fci":
        ref = run_full_space_reference(sysd)
        E = ref["E_total"]
        extras = {"dim": ref["dim"], "e_active": ref["e_active"]}
    else:
        raise ValueError(f"未知方法: {method}")

    wall = time.perf_counter() - t0
    out: Dict[str, Any] = {
        "seed": seed, "method": method, "E_total": float(E),
        "wall_s": wall, "extras": extras,
    }
    if method == "sqd_active":
        out["trajectory"] = [_pyify(t) for t in traj]
    e_ref = job.e_ref
    out["err_vs_ref"] = (abs(float(E) - e_ref) if e_ref is not None else None)
    return _pyify(out)


# ---------------------------------------------------------------------------
#  Job / JobManager
# ---------------------------------------------------------------------------
class Job:
    def __init__(self, params: Dict[str, Any]):
        self.id = uuid.uuid4().hex[:12]
        self.params = params
        self.status = "running"          # running | done | error | cancelled
        self.phase = "queued"            # queued | integrals | reference | run
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.finished_at: Optional[float] = None
        self.system_info: Optional[Dict[str, Any]] = None
        self.reference: Optional[Dict[str, Any]] = None
        self.e_ref: Optional[float] = None
        self.seed_results: List[Dict[str, Any]] = []
        self.seed_index = 0              # 0-based, 运行中的 seed
        self.n_seeds = 1
        self.live_traj: Optional[list] = None
        self.aggregate: Optional[Dict[str, Any]] = None
        self.cancel_requested = False

    def public(self) -> Dict[str, Any]:
        lt = self.live_traj or []
        live = None
        if lt:
            last = lt[-1]
            live = {"rounds_done": len(lt), "last_E": float(last["E"]),
                    "last_dim": int(last["dim"]),
                    "last_sigma2": float(last["sigma2"]),
                    "trajectory": _pyify(list(lt))}
        return _pyify({
            "job_id": self.id, "status": self.status, "phase": self.phase,
            "created_at": self.created_at, "finished_at": self.finished_at,
            "error": self.error, "cancel_requested": self.cancel_requested,
            "system_info": self.system_info, "reference": self.reference,
            "seed_index": self.seed_index, "n_seeds": self.n_seeds,
            "live": live,
            "seed_results": self.seed_results, "aggregate": self.aggregate,
        })


class JobManager:
    """单计算线程; 提交/查询/取消都从 Flask 线程调用。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.job: Optional[Job] = None

    def submit(self, params: Dict[str, Any]) -> Job:
        with self._lock:
            cur = self.job
            if cur is not None and cur.status == "running":
                raise BusyError("已有一个计算任务在运行; 请等它结束或先取消")
            self.job = Job(params)
        threading.Thread(target=self._run, args=(self.job,),
                         daemon=True, name="tc-sqd-webui-job").start()
        return self.job

    def cancel(self) -> bool:
        job = self.job
        if job is None or job.status != "running":
            return False
        job.cancel_requested = True
        return True

    # ---- 工作线程 ----
    def _run(self, job: Job) -> None:
        from .systems import build_system

        try:
            p = job.params
            job.phase = "integrals"
            sysd = build_system(p["system"])
            job.system_info = sysd["meta"]

            ref = p.get("reference", {})
            mode = str(ref.get("mode", "none"))
            if mode == "manual":
                job.reference = {"mode": "manual", "E_total": float(ref["value"])}
                job.e_ref = float(ref["value"])
            elif mode == "auto":
                limit = int(ref.get("dim_limit", 200000) or 200000)
                dim = int(sysd["meta"]["dim_full"])
                if dim <= limit:
                    job.phase = "reference"
                    r = run_full_space_reference(sysd)
                    r["mode"] = "auto"
                    job.reference, job.e_ref = r, r["E_total"]
                else:
                    job.reference = {
                        "mode": "auto", "skipped": True,
                        "dim_full": dim, "dim_limit": limit,
                        "note": (f"全空间 dim={dim:,} 超过上限 {limit:,}, "
                                 "已跳过参考 (可调大上限或手动输入)"),
                    }

            sampling = p.get("sampling", {})
            if p["method"] == "fci":
                job.n_seeds = 1   # 确定性方法与 seed 无关
            else:
                job.n_seeds = max(1, _i(sampling, "n_seeds", 1))
            base_seed = _i(sampling, "seed", 0)

            for i in range(job.n_seeds):
                if job.cancel_requested:
                    job.status = "cancelled"
                    return
                job.seed_index = i
                job.phase = "run"
                job.seed_results.append(_run_seed(job, sysd, base_seed + i))

            Es = [r["E_total"] for r in job.seed_results]
            agg: Dict[str, Any] = {
                "n_seeds": job.n_seeds,
                "E_mean": float(np.mean(Es)),
                "E_std": float(np.std(Es, ddof=1)) if len(Es) > 1 else 0.0,
                "E_min": float(np.min(Es)), "E_max": float(np.max(Es)),
                "wall_total": float(sum(r["wall_s"] for r in job.seed_results)),
            }
            errs = [r["err_vs_ref"] for r in job.seed_results
                    if r["err_vs_ref"] is not None]
            if errs:
                agg["err_mean"] = float(np.mean(errs))
                agg["err_max"] = float(np.max(errs))
                agg["best_seed"] = int(job.seed_results[
                    int(np.argmin(errs))]["seed"])
            job.aggregate = agg
            job.status = "done"
        except Exception:
            job.status = "error"
            job.error = traceback.format_exc()
        finally:
            job.live_traj = None
            job.finished_at = time.time()
