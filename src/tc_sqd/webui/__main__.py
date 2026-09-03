"""``python -m tc_sqd.webui`` —— 启动本地计算面板。

    python -m tc_sqd.webui                 # http://127.0.0.1:8765
    python -m tc_sqd.webui --host 0.0.0.0  # 局域网可访问
    python -m tc_sqd.webui --port 9000 --no-browser

计算在本机执行 (CPU, 装有 cupy 时可选 GPU 后端); 服务重启后任务历史清空。
"""

from __future__ import annotations

import argparse
import webbrowser


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m tc_sqd.webui",
        description="tc_sqd 本地参数化 Web 计算面板")
    ap.add_argument("--host", default="127.0.0.1",
                    help="监听地址 (默认 127.0.0.1 仅本机; 0.0.0.0 开放局域网)")
    ap.add_argument("--port", type=int, default=8765, help="端口 (默认 8765)")
    ap.add_argument("--no-browser", action="store_true",
                    help="不自动打开浏览器")
    args = ap.parse_args(argv)

    from . import create_app

    app = create_app()
    url = f"http://{args.host}:{args.port}"
    print(f"[tc_sqd webui] serving at {url}  (Ctrl+C 退出)")
    if not args.no_browser and args.host in ("127.0.0.1", "localhost"):
        try:
            webbrowser.open(url)
        except Exception:
            pass
    # 本地单用户工具: threaded 服务静态页/轮询, 计算在单独工作线程 (单任务)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
