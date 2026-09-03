"""tc_sqd.webui.app —— Flask 路由层 (静态页 + JSON API)。

API 一览:
- ``GET  /api/capabilities`` GPU 可用性 / 版本信息
- ``GET  /api/presets``      预设体系清单
- ``POST /api/preview``      体系 spec → 维度/电子数预览 (不跑 SCF)
- ``POST /api/run``          提交计算任务 (运行中返回 409)
- ``GET  /api/job``          当前任务状态 + 进度 + 结果 (轮询)
- ``POST /api/job/cancel``   请求取消 (seed 间生效)

计算在服务进程内的单工作线程上执行 (见 :mod:`tc_sqd.webui.runner`),
结果仅存内存 —— 重启服务即清空, 勿当持久化存储用。
"""

from __future__ import annotations

import os
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory

from .runner import BusyError, JobManager
from .systems import (PRESETS, gpu_available, normalize_system_spec,
                      preview_system)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

_METHODS = ("sqd_active", "cipsi", "hci", "fci")


def create_app() -> Flask:
    app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
    manager = JobManager()

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/capabilities")
    def capabilities():
        import tc_sqd
        import pyscf

        return jsonify({
            "gpu": gpu_available(),
            "single_job": True,
            "versions": {
                "tc_sqd": getattr(tc_sqd, "__version__", "0.1.0"),
                "pyscf": pyscf.__version__,
            },
        })

    @app.get("/api/presets")
    def presets():
        return jsonify({"presets": PRESETS, "gpu": gpu_available()})

    @app.post("/api/preview")
    def preview():
        body = _json_body()
        try:
            spec = normalize_system_spec(body.get("system"))
            out = preview_system(spec)
        except (ValueError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pyscf 基组/几何解析错误等
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify({"preview": out})

    @app.post("/api/run")
    def run():
        body = _json_body()
        try:
            params = _validate_params(body)
        except (ValueError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400
        try:
            job = manager.submit(params)
        except BusyError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"job_id": job.id, "status": job.status})

    @app.get("/api/job")
    def job_state():
        job = manager.job
        if job is None:
            return jsonify({"status": "idle"})
        return jsonify(job.public())

    @app.post("/api/job/cancel")
    def cancel():
        ok = manager.cancel()
        return jsonify({"cancel_requested": ok})

    def _json_body() -> Dict[str, Any]:
        body = request.get_json(force=True, silent=True)
        return body if isinstance(body, dict) else {}

    def _validate_params(body: Dict[str, Any]) -> Dict[str, Any]:
        spec = normalize_system_spec(body.get("system"))
        method = str(body.get("method", "sqd_active"))
        if method not in _METHODS:
            raise ValueError(f"method 必须是 {_METHODS} 之一, got {method}")
        sampling = body.get("sampling") or {}
        shots = int(sampling.get("shots", 500) or 500)
        if not 1 <= shots <= 1_000_000:
            raise ValueError(f"shots 须在 [1, 1e6], got {shots}")
        n_seeds = int(sampling.get("n_seeds", 1) or 1)
        if not 1 <= n_seeds <= 50:
            raise ValueError(f"多 seed 个数须在 [1, 50], got {n_seeds}")
        if str(sampling.get("mode", "uniform")) not in ("uniform", "hf"):
            raise ValueError("采样模式只支持 uniform / hf")
        ref = body.get("reference") or {}
        if str(ref.get("mode", "none")) not in ("none", "auto", "manual"):
            raise ValueError("参考模式只支持 none / auto / manual")
        if ref.get("mode") == "manual":
            float(ref["value"])   # 早失败: 非数值直接 400
        params: Dict[str, Any] = body.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params 必须是对象")
        if not gpu_available() and str(params.get("backend", "cpu")) == "gpu":
            raise ValueError("backend=gpu 但本机 cupy 不可用 (装 cupy-cuda12x 或切 cpu)")
        return {"system": spec, "method": method, "sampling": sampling,
                "params": params, "reference": ref}

    return app
