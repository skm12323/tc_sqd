"""numpy 2.x ↔ tensorcircuit 0.12 兼容补丁。

tensorcircuit 0.12.0 引用了 numpy 在 2.x 中搬走/移除的两个 API:

- ``np.ComplexWarning``: numpy 2.0 起从顶层搬到 ``np.exceptions`` (顶层 alias 删除)
- ``np.reshape(a, newshape=...)``: numpy 2.5 起移除了 ``newshape`` 关键字

本模块在 numpy 顶层补回这两个 API, 使 tensorcircuit 0.12 能在 numpy 2.x 下
正常 import 与运行。补丁对 numpy 1.x 安全 (``hasattr`` 检查, 不会覆盖原有定义),
且幂等 (重复调用无副作用)。

用法 (任选其一, 都必须在 ``import tensorcircuit`` 之前生效)
-----------------------------------------------------------
1. **一劳永逸 (推荐)**: 写入 sitecustomize, 之后该环境所有脚本自动 patch ::

       python -m tc_sqd._compat install

2. 脚本里 ``import tc_sqd`` (本模块在 ``tc_sqd.__init__`` 中已自动 apply),
   只要它在 ``import tensorcircuit`` 之前即可。

3. 显式 ::

       import tc_sqd._compat   # 导入即 apply
"""

from __future__ import annotations

import sys

import numpy as np

__all__ = ["apply"]


def apply() -> None:
    """补回 tensorcircuit 0.12 需要的 numpy 兼容 API (幂等, 对 numpy 1.x 安全)。"""
    # 1) np.ComplexWarning: numpy 2.0 起搬到 np.exceptions, 顶层 alias 删除
    if not hasattr(np, "ComplexWarning"):
        import numpy.exceptions
        np.ComplexWarning = numpy.exceptions.ComplexWarning

    # 2) np.reshape(newshape=): numpy 2.5 起移除 newshape 关键字 (改名为 shape)
    _orig_reshape = np.reshape
    if not getattr(_orig_reshape, "_tc_sqd_compat_patched", False):
        def _reshape(a, *args, **kwargs):
            if "newshape" in kwargs:
                kwargs["shape"] = kwargs.pop("newshape")
            return _orig_reshape(a, *args, **kwargs)
        _reshape._tc_sqd_compat_patched = True  # type: ignore[attr-defined]
        np.reshape = _reshape


# 模块导入时自动执行
apply()


def _install_sitecustomize() -> int:
    """把 apply() 追加写进当前环境的 sitecustomize.py, 返回退出码。"""
    import os
    import site

    for sp in site.getsitepackages():
        if not os.path.isdir(sp):
            continue
        sc = os.path.join(sp, "sitecustomize.py")
        try:
            existing = open(sc, encoding="utf-8").read() if os.path.exists(sc) else ""
        except OSError:
            continue
        if "tc_sqd._compat" in existing:
            print(f"[tc_sqd] sitecustomize 已含补丁, 跳过: {sc}")
            return 0
        snippet = (
            "\n# --- tc_sqd._compat: tensorcircuit 0.12 <-> numpy 2.x ---\n"
            "try:\n"
            "    from tc_sqd._compat import apply as _tc_sqd_apply\n"
            "    _tc_sqd_apply()\n"
            "except Exception:\n"
            "    pass  # 避免 sitecustomize 报错影响所有 python 启动\n"
        )
        try:
            with open(sc, "a", encoding="utf-8") as f:
                f.write(snippet)
        except OSError:
            continue
        print(f"[tc_sqd] 已写入 sitecustomize: {sc}")
        print("[tc_sqd] 该环境现在任何脚本 import tensorcircuit 前都会自动 patch。")
        return 0
    print("[tc_sqd] 未找到可写的 site-packages 目录", file=sys.stderr)
    return 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "install":
        raise SystemExit(_install_sitecustomize())
    print("用法: python -m tc_sqd._compat install  (把补丁写入 sitecustomize.py)")
