"""tc_sqd.webui —— Web 计算面板的 API/端到端测试。

flask 为可选依赖: 未安装时整文件跳过 (importorskip)。
端到端用 H₂/STO-3G (全空间 dim=4) 走真实线程任务 (SCF→SQD→参考),
单文件 ~30s 量级。积分缓存落仓库根 ``_webui_*_ints.npz`` (gitignored)。
"""

import time

import pytest

flask = pytest.importorskip("flask")  # noqa: F841 — WebUI 唯一额外依赖

from tc_sqd.webui import create_app  # noqa: E402


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _wait_job(client, timeout=180.0):
    """轮询 /api/job 直到任务离开 running 状态。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get("/api/job").get_json()
        if j.get("status") != "running":
            return j
        time.sleep(0.3)
    pytest.fail("任务未在超时内完成")


def _submit(client, body):
    r = client.post("/api/run", json=body)
    assert r.status_code == 200, r.get_json()
    return r.get_json()["job_id"]


H2 = {
    "geometry": "H 0 0 0; H 0 0 0.75",
    "basis": "sto-3g", "charge": 0, "spin": 0,
    "n_core": 0, "n_virtual": 0, "scf": "auto",
}


# ---------------------------------------------------------------- 基础 API
def test_webui_index_and_capabilities(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "tc_sqd" in r.get_data(as_text=True)

    caps = client.get("/api/capabilities").get_json()
    assert "gpu" in caps and "versions" in caps

    presets = client.get("/api/presets").get_json()["presets"]
    assert any(p["id"] == "h2_sto3g" for p in presets)


def test_webui_preview_dim(client):
    r = client.post("/api/preview", json={"system": H2})
    assert r.status_code == 200
    pv = r.get_json()["preview"]
    assert pv["norb"] == 2 and pv["nelec"] == [1, 1]
    assert pv["dim_full"] == 4


def test_webui_preview_rejects_bad_system(client):
    r = client.post("/api/preview", json={"system": {"geometry": ""}})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_webui_run_rejects_bad_params(client):
    r = client.post("/api/run", json={
        "system": H2, "method": "nope",
        "sampling": {}, "params": {}, "reference": {}})
    assert r.status_code == 400
    # manual 参考缺数值 → 400
    r = client.post("/api/run", json={
        "system": H2, "method": "fci", "sampling": {}, "params": {},
        "reference": {"mode": "manual"}})
    assert r.status_code == 400


# ---------------------------------------------------------------- 端到端
def test_webui_sqd_multiseed_end_to_end(client):
    """H₂ SQD active 双 seed + auto 参考全链路 (SCF→采样→PT2→参考)。"""
    _submit(client, {
        "system": H2,
        "method": "sqd_active",
        "sampling": {"shots": 30, "seed": 0, "mode": "uniform",
                     "n_seeds": 2},
        "params": {"max_rounds": 5, "backend": "cpu",
                   "warm_start": True},
        "reference": {"mode": "auto", "dim_limit": 1000},
    })
    j = _wait_job(client)
    assert j["status"] == "done", j.get("error")
    assert j["n_seeds"] == 2 and len(j["seed_results"]) == 2
    assert {r["seed"] for r in j["seed_results"]} == {0, 1}
    # 全空间 4 维, PT2 必然补全 → 准 FCI
    for r in j["seed_results"]:
        assert r["err_vs_ref"] < 1e-8
        assert r["trajectory"] and r["trajectory"][-1]["dim"] == 4
    agg = j["aggregate"]
    assert agg["n_seeds"] == 2 and agg["err_mean"] < 1e-8
    ref = j["reference"]
    assert ref["mode"] == "auto" and ref["dim"] == 4


def test_webui_fci_and_hci_methods(client):
    """fci = 全空间 solve_sci (应与 auto 参考逐位同源); hci 带 PT2 修正。"""
    _submit(client, {
        "system": H2, "method": "fci", "sampling": {},
        "params": {}, "reference": {"mode": "auto", "dim_limit": 1000},
    })
    j = _wait_job(client)
    assert j["status"] == "done", j.get("error")
    r0 = j["seed_results"][0]
    assert r0["method"] == "fci" and r0["err_vs_ref"] < 1e-10

    _submit(client, {
        "system": H2, "method": "hci",
        "sampling": {"shots": 20, "seed": 3},
        "params": {"eps_hb": 1e-4, "use_seed": True, "backend": "cpu"},
        "reference": {"mode": "auto", "dim_limit": 1000},
    })
    j = _wait_job(client)
    assert j["status"] == "done", j.get("error")
    r1 = j["seed_results"][0]
    assert r1["method"] == "hci" and r1["err_vs_ref"] < 5e-3
    assert r1["extras"]["e_pt2"] is not None


def test_webui_reference_skip_over_limit(client):
    """dim 超上限时 auto 参考被跳过 (err=None), 任务照常完成。"""
    _submit(client, {
        "system": H2, "method": "sqd_active",
        "sampling": {"shots": 10, "seed": 0, "n_seeds": 1},
        "params": {"max_rounds": 3, "backend": "cpu"},
        "reference": {"mode": "auto", "dim_limit": 2},   # 4 > 2 → skip
    })
    j = _wait_job(client)
    assert j["status"] == "done", j.get("error")
    assert j["reference"]["skipped"] is True
    assert j["seed_results"][0]["err_vs_ref"] is None


# ---------------------------------------------------------------- 自定义体系
def test_webui_custom_xyz_basis_dict_bohr(client):
    """自定义体系输入增强: .xyz 粘贴块 / 分元素基组 JSON / bohr 单位。"""
    # 标准 .xyz 文件块 (原子数行 + 注释行) + 分元素基组 JSON → H₂O/STO-3G
    r = client.post("/api/preview", json={"system": {
        "geometry": "3\nwater xyz\n"
                    "O 0.0 0.0 0.1173\n"
                    "H 0.0 0.7572 -0.4692\n"
                    "H 0.0 -0.7572 -0.4692",
        "basis": '{"O": "sto-3g", "H": "sto-3g"}',
        "charge": 0, "spin": 0, "n_core": 0, "n_virtual": 0,
        "scf": "auto", "unit": "angstrom",
    }})
    assert r.status_code == 200, r.get_json()
    pv = r.get_json()["preview"]
    assert pv["formula"] == "H2O"
    assert pv["norb"] == 7 and pv["nelec"] == [5, 5]
    assert pv["dim_full"] == 441

    # bohr 单位 (0.75 Å ≈ 1.417 bohr)
    r = client.post("/api/preview", json={"system": {
        "geometry": "H 0 0 0; H 0 0 1.417",
        "basis": "sto-3g", "unit": "bohr",
        "charge": 0, "spin": 0, "n_core": 0, "n_virtual": 0, "scf": "auto",
    }})
    assert r.status_code == 200
    pv = r.get_json()["preview"]
    assert pv["formula"] == "H2" and pv["dim_full"] == 4

    # 坏 JSON 基组 → 400
    r = client.post("/api/preview", json={"system": {
        "geometry": "H 0 0 0; H 0 0 0.75", "basis": '{"O": ',
        "charge": 0, "spin": 0, "n_core": 0, "n_virtual": 0,
        "scf": "auto", "unit": "angstrom"}})
    assert r.status_code == 400


def test_webui_rks_uks_end_to_end(client):
    """DFT (KS) 初轨道: RKS/UKS(pbe) 积分 → 全空间 FCI 轨道旋转不变 → err≈0。"""
    for scf_mode in ("rks", "uks"):
        _submit(client, {
            "system": {**H2, "scf": scf_mode, "xc": "pbe"},
            "method": "fci", "sampling": {},
            "params": {},
            "reference": {"mode": "auto", "dim_limit": 1000},
        })
        j = _wait_job(client)
        assert j["status"] == "done", j.get("error")
        si = j["system_info"]
        assert si["scf_type"].startswith(scf_mode.upper())
        assert si["scf_converged"] is True
        if scf_mode == "uks":
            assert si["spin_resolved"] is True   # 五积分路径 (round_011/017)
        r0 = j["seed_results"][0]
        assert r0["err_vs_ref"] < 1e-8   # 全空间 FCI 对轨道选择不变
