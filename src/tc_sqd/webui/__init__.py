"""tc_sqd.webui —— 本地参数化 Web 计算面板 (可选组件)。

克隆本仓库后无需写代码即可在浏览器里调节体系/方法/随机种子/shots/
多 seed 平均等参数, 在**本地设备**上跑 SQD/CIPSI/HCI/全空间 SCI 计算:

    pip install flask                # 唯一额外依赖 (或 pip install -e .[webui])
    python -m tc_sqd.webui           # 打开 http://127.0.0.1:8765

flask 为可选依赖: 未安装时 ``import tc_sqd.webui`` 不报错, 调用
:func:`create_app` 才给出带安装指引的 ImportError。计算本身在本机
CPU/GPU 上执行 (GPU 需 cupy, 与库其余部分一致), 不依赖任何外部服务。
"""

__all__ = ["create_app", "main"]

_INSTALL_HINT = (
    "WebUI 需要 flask: pip install flask (或 pip install -e .[webui])"
)


def create_app(*args, **kwargs):
    """构建 Flask 应用 (见 :func:`tc_sqd.webui.app.create_app`)。"""
    try:
        import flask  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise ImportError(_INSTALL_HINT) from exc
    from .app import create_app as _factory

    return _factory(*args, **kwargs)


def main(argv=None) -> None:
    """``python -m tc_sqd.webui`` 入口 (见 :mod:`tc_sqd.webui.__main__`)。"""
    from .__main__ import main as _main

    _main(argv)
