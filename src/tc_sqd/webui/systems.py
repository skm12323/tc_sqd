"""tc_sqd.webui.systems —— 体系预设 + PySCF 积分构建 (带磁盘缓存)。

体系层完全复用 :func:`tc_sqd.from_pyscf` (MO 变换/冻结核修正/开壳层/UHF
五积分等整类易错转换已在那里消掉), 这里只做三件事:

1. 预设体系清单 (项目历史上验证过的体系, 供下拉直接选);
2. ``preview_system``: 只建 Mole (不跑 SCF) 秒出 norb/nelec/全空间维度,
   供前端运行前确认;
3. ``build_system``: 跑 SCF → ``from_pyscf`` → 活性积分, 结果按体系指纹
   缓存为仓库根 ``_webui_<sha>_ints.npz`` (已被 ``_*_ints.npz`` gitignore
   规则覆盖), 换 shots/seed/方法重跑不重复 SCF。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["PRESETS", "build_system", "preview_system", "normalize_system_spec",
           "gpu_available"]

_CACHE_VERSION = 2   # v2: scf 增加 rks/uks + xc 入指纹 (同几何不同泛函不共缓存)

# ---------------------------------------------------------------------------
#  预设体系 (均为项目历轮验证过的; dim_full = C(norb,na)*C(norb,nb))
# ---------------------------------------------------------------------------
PRESETS: List[Dict[str, Any]] = [
    {
        "id": "h2_sto3g",
        "label": "H₂/STO-3G R=0.75 Å (2e,2o)",
        "desc": "最小演示体系, 全空间 dim=4, 秒级跑完",
        "geometry": "H 0 0 0; H 0 0 0.75",
        "basis": "sto-3g", "charge": 0, "spin": 0,
        "n_core": 0, "n_virtual": 0, "scf": "auto",
    },
    {
        "id": "h2_ccpvdz",
        "label": "H₂/cc-pVDZ R=0.75 Å (2e,5o)",
        "desc": "小体系 + 相关一致基组, 全空间 dim=25",
        "geometry": "H 0 0 0; H 0 0 0.75",
        "basis": "cc-pvdz", "charge": 0, "spin": 0,
        "n_core": 0, "n_virtual": 0, "scf": "auto",
    },
    {
        "id": "ch_sto3g",
        "label": "CH/STO-3G R=1.15 Å 开壳层 (7e,6o)",
        "desc": "开壳层 (4,3), 全空间 dim=300 (round_017/018 体系)",
        "geometry": "C 0 0 0; H 0 0 1.15",
        "basis": "sto-3g", "charge": 0, "spin": 1,
        "n_core": 0, "n_virtual": 0, "scf": "auto",
    },
    {
        "id": "n2_sto3g_r10",
        "label": "N₂/STO-3G R=1.0 Å (14e,10o)",
        "desc": "平衡附近, 全空间 dim=14 400",
        "geometry": "N 0 0 0; N 0 0 1.0",
        "basis": "sto-3g", "charge": 0, "spin": 0,
        "n_core": 0, "n_virtual": 0, "scf": "auto",
    },
    {
        "id": "n2_sto3g_r20",
        "label": "N₂/STO-3G R=2.0 Å (14e,10o)",
        "desc": "键拉伸强关联, 全空间 dim=14 400",
        "geometry": "N 0 0 0; N 0 0 2.0",
        "basis": "sto-3g", "charge": 0, "spin": 0,
        "n_core": 0, "n_virtual": 0, "scf": "auto",
    },
    {
        "id": "n2_sto3g_r30",
        "label": "N₂/STO-3G R=3.0 Å (14e,10o)",
        "desc": "强关联极限, 全空间 dim=14 400 (round_016 参考体系)",
        "geometry": "N 0 0 0; N 0 0 3.0",
        "basis": "sto-3g", "charge": 0, "spin": 0,
        "n_core": 0, "n_virtual": 0, "scf": "auto",
    },
    {
        "id": "c2_sto3g",
        "label": "C₂/STO-3G R=1.25 Å (12e,10o)",
        "desc": "经典多参考体系, 全空间 dim=44 100",
        "geometry": "C 0 0 0; C 0 0 1.25",
        "basis": "sto-3g", "charge": 0, "spin": 0,
        "n_core": 0, "n_virtual": 0, "scf": "auto",
    },
    {
        "id": "n2_ccpvdz_r30",
        "label": "N₂/cc-pVDZ R=3.0 Å 全电子 (14e,14o)",
        "desc": "round_021 大体系平台: 全空间 11 778 624 维, 全空间参考不可行",
        "geometry": "N 0 0 0; N 0 0 3.0",
        "basis": "cc-pvdz", "charge": 0, "spin": 0,
        "n_core": 0, "n_virtual": 0, "scf": "auto",
    },
]

_SYSTEM_KEYS = ("geometry", "basis", "charge", "spin", "n_core", "n_virtual",
                "scf", "unit", "xc")
_SCF_MODES = ("auto", "uhf", "rks", "uks")   # auto: RHF/ROHF 按 spin 自动


def normalize_system_spec(body: Any) -> Dict[str, Any]:
    """POST body 的 system 字段 → 完整体系 spec (展开预设 / 补默认值)。

    支持两种形式: ``{"preset": "<id>"}`` (预设字段可被同级键覆盖) 或
    直接的完整 spec dict。几何输入 ";" 或换行分隔 (也接受粘贴的标准
    .xyz 文件块: 首行原子数 + 次行注释自动剥去), 单位 Å / bohr。
    ``basis`` 为基组名, 或分元素字典的 JSON 串 (如
    ``{"O": "sto-3g", "H": "cc-pvdz"}``)。
    """
    if not isinstance(body, dict):
        raise ValueError("system 必须是对象 (preset 或完整字段)")
    spec: Dict[str, Any] = dict(body)
    pid = spec.pop("preset", None)
    if pid is not None:
        base = next((p for p in PRESETS if p["id"] == pid), None)
        if base is None:
            raise ValueError(f"未知预设体系: {pid}")
        merged = {k: base[k] for k in _SYSTEM_KEYS if k in base}
        merged.update({k: v for k, v in spec.items() if v not in (None, "")})
        spec = merged
    spec.setdefault("geometry", "")
    spec.setdefault("basis", "sto-3g")
    spec.setdefault("charge", 0)
    spec.setdefault("spin", 0)
    spec.setdefault("n_core", 0)
    spec.setdefault("n_virtual", 0)
    spec.setdefault("scf", "auto")
    spec.setdefault("unit", "angstrom")
    spec.setdefault("xc", "")
    spec["charge"] = int(spec["charge"])
    spec["spin"] = int(spec["spin"])
    spec["n_core"] = int(spec["n_core"])
    spec["n_virtual"] = int(spec["n_virtual"])
    if spec["scf"] not in _SCF_MODES:
        raise ValueError(f"scf 只支持 {_SCF_MODES}, got {spec['scf']}")
    if spec["unit"] not in ("angstrom", "bohr"):
        raise ValueError("unit 只支持 angstrom / bohr")
    if not isinstance(spec["xc"], str):
        raise ValueError("xc 泛函须是字符串 (如 b3lyp / pbe)")
    return spec


def _atom_block(geometry: str) -> str:
    """几何串 → pyscf atom 块。接受 ";" 或换行分隔; 若首行是纯整数
    (粘贴的标准 .xyz 文件), 剥去原子数行 + 注释行。"""
    parts = [ln.strip() for ln in geometry.replace(";", "\n").splitlines()
             if ln.strip()]
    if not parts:
        raise ValueError("geometry 为空")
    if len(parts) >= 3 and parts[0].isdigit():
        parts = parts[2:]
    return "\n".join(parts)


def _parse_basis(basis: Any) -> Any:
    """基组 → pyscf 基组名或分元素 dict (JSON 串 / 已是 dict)。"""
    if isinstance(basis, dict):
        if not basis:
            raise ValueError("分元素基组不能为空 dict")
        return basis
    s = str(basis).strip()
    if s.startswith("{"):
        try:
            d = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"分元素基组 JSON 解析失败: {exc}") from exc
        if not isinstance(d, dict) or not d:
            raise ValueError('分元素基组 JSON 须形如 {"N": "cc-pvdz", "H": "sto-3g"}')
        return d
    return s


def _formula(mol) -> str:
    """元素计数拼分子式 (pyscf Mole 无 .formula 属性)。"""
    from collections import Counter

    counts = Counter(getattr(mol, "elements", None) or
                     [a[0] for a in getattr(mol, "_atom", [])])
    return "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(counts.items()))


def _active_layout(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Mole 级布局推断 (norb/nelec/dim), 不跑 SCF。"""
    from pyscf import gto

    mol = gto.M(atom=_atom_block(spec["geometry"]),
                basis=_parse_basis(spec["basis"]),
                charge=spec["charge"], spin=spec["spin"],
                unit=spec["unit"], verbose=0)
    norb_full = int(mol.nao)
    na = (mol.nelectron + mol.spin) // 2
    nb = (mol.nelectron - mol.spin) // 2
    n_core, n_virtual = spec["n_core"], spec["n_virtual"]
    if n_core < 0 or n_virtual < 0 or n_core + n_virtual >= norb_full:
        raise ValueError(
            f"n_core+n_virtual 必须在 [0, {norb_full}) 内, "
            f"got n_core={n_core}, n_virtual={n_virtual}")
    if n_core > min(na, nb):
        raise ValueError(f"n_core={n_core} 超过闭壳层占据数 min(na,nb)={min(na, nb)}")
    norb = norb_full - n_core - n_virtual
    nelec = (na - n_core, nb - n_core)
    dim = math.comb(norb, nelec[0]) * math.comb(norb, nelec[1])
    return {
        "formula": _formula(mol),
        "norb_full": norb_full, "norb": norb, "nelec": list(nelec),
        "na": nelec[0], "nb": nelec[1],
        "n_core": n_core, "n_virtual": n_virtual,
        "dim_full": dim,
        "spin_resolved": spec["scf"] in ("uhf", "uks"),
        "scf": spec["scf"], "xc": (spec.get("xc") or "").strip() or None,
    }


def preview_system(spec: Dict[str, Any]) -> Dict[str, Any]:
    """运行前预览: 只建 Mole 出维度/电子数 (秒级, 不跑 SCF)。"""
    out = _active_layout(spec)
    out["warnings"] = []
    if out["dim_full"] > 1_000_000:
        out["warnings"].append(
            f"全空间维度 {out['dim_full']:,} 超过 1e6: 全空间参考/FCI 方法"
            "将非常慢, 建议限制 max_strings 或用 PT2 修正的采样方法")
    return out


def gpu_available() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
#  积分构建 + 缓存
# ---------------------------------------------------------------------------
def _cache_dir() -> str:
    """仓库根 (向上找 pyproject.toml); 找不到 (如 pip 装到 site-packages) 用临时目录。"""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        d, parent = os.path.dirname(d), d
        pj = os.path.join(parent, "pyproject.toml")
        if os.path.isfile(pj):
            try:
                with open(pj, encoding="utf-8") as f:
                    if "tc_sqd" in f.read(400):
                        return parent
            except OSError:
                pass
    return os.path.join(tempfile.gettempdir(), "tc_sqd_webui_cache")


def _cache_path(spec: Dict[str, Any]) -> str:
    fp = json.dumps({k: spec[k] for k in _SYSTEM_KEYS if k in spec},
                    sort_keys=True, ensure_ascii=False)
    sha = hashlib.sha1(f"v{_CACHE_VERSION}:{fp}".encode("utf-8")).hexdigest()[:12]
    return os.path.join(_cache_dir(), f"_webui_{sha}_ints.npz")


_MEM_CACHE: Dict[str, Dict[str, Any]] = {}


def build_system(spec: Dict[str, Any]) -> Dict[str, Any]:
    """SCF → from_pyscf → 活性积分 dict (h1e/eri/ecore/norb/nelec + meta)。

    eri 为闭壳层单块 (norb,)*4 或 UHF 三元组 (eri_aa, eri_ab, eri_bb) ——
    与 :func:`tc_sqd.from_pyscf` 输出约定一致, 直接喂各 solver。
    """
    path = _cache_path(spec)
    if path in _MEM_CACHE:
        return _MEM_CACHE[path]
    if os.path.isfile(path):
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta_json"]))
        sysd = {
            "h1e": d["h1e"],
            "eri": d["eri"] if "eri" in d else (d["eri_aa"], d["eri_ab"], d["eri_bb"]),
            "ecore": float(d["ecore"]),
            "norb": int(d["norb"]), "nelec": (int(d["na"]), int(d["nb"])),
            "spin_resolved": bool(d["spin_resolved"]),
            "meta": meta,
        }
        _MEM_CACHE[path] = sysd
        return sysd

    from pyscf import gto, scf

    import tc_sqd

    mol = gto.M(atom=_atom_block(spec["geometry"]),
                basis=_parse_basis(spec["basis"]),
                charge=spec["charge"], spin=spec["spin"],
                unit=spec["unit"], verbose=0)
    xc = (spec.get("xc") or "").strip() or "b3lyp"
    if spec["scf"] == "uhf":
        mf = scf.UHF(mol).run()
    elif spec["scf"] == "uks":
        mf = scf.UKS(mol)
        mf.xc = xc
        mf.run()
    elif spec["scf"] == "rks":
        mf = scf.RKS(mol)
        mf.xc = xc
        mf.run()
    else:
        # from_pyscf 对 Mole 自动跑 RHF (闭壳层) / ROHF (spin!=0)
        mf = None
    data = tc_sqd.from_pyscf(
        mf if mf is not None else mol,
        n_core=spec["n_core"] or None, n_virtual=spec["n_virtual"] or None,
    )
    mf = mf if mf is not None else data.mf
    meta = _active_layout(spec)
    meta["e_scf"] = float(mf.e_tot)
    meta["scf_converged"] = bool(getattr(mf, "converged", False))
    if spec["scf"] == "auto":
        meta["scf_type"] = "ROHF" if spec["spin"] else "RHF"
    else:
        meta["scf_type"] = spec["scf"].upper() + (f"({xc})" if spec["scf"] in ("rks", "uks") else "")

    sysd = {
        "h1e": np.asarray(data.h1e), "eri": data.eri,
        "ecore": float(data.ecore), "norb": int(data.norb),
        "nelec": (int(data.nelec[0]), int(data.nelec[1])),
        "spin_resolved": bool(data.spin_resolved),
        "meta": meta,
    }
    arrays = {"h1e": sysd["h1e"], "ecore": sysd["ecore"], "norb": sysd["norb"],
              "na": sysd["nelec"][0], "nb": sysd["nelec"][1],
              "spin_resolved": sysd["spin_resolved"],
              "meta_json": json.dumps(meta, ensure_ascii=False)}
    if isinstance(sysd["eri"], tuple):
        arrays.update(eri_aa=sysd["eri"][0], eri_ab=sysd["eri"][1],
                      eri_bb=sysd["eri"][2])
    else:
        arrays["eri"] = sysd["eri"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)
    _MEM_CACHE[path] = sysd
    return sysd
