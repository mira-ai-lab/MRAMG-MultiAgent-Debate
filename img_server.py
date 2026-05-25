#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时 HTTP Server 启动脚本
用法:
    python start_image_server.py --root /data2/qn/MRAMG/MRAMG-Bench/IMAGE/IMAGE/images --port 0
返回:
    server_url: http://127.0.0.1:随机端口
"""
import os
import socket
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from threading import Thread
from pathlib import Path
import argparse

def start_temp_http_server(root_dir: str, port: int = 0) -> str:
    """
    启动临时 HTTP Server
    """
    root_dir = os.path.abspath(root_dir)
    os.chdir(root_dir)

    # 随机端口
    if port == 0:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()

    server = ThreadingHTTPServer(("127.0.0.1", port), SimpleHTTPRequestHandler)

    def run_server():
        print(f"HTTP server started at http://127.0.0.1:{port}")
        server.serve_forever()

    thread = Thread(target=run_server, daemon=True)
    thread.start()

    return f"http://127.0.0.1:{port}"

def get_local_image_url(img_path: str, root_dir: str, server_url: str) -> str:
    """
    根据本地路径生成 URL
    """
    img_path = Path(img_path).resolve()
    root_dir = Path(root_dir).resolve()
    relative_path = img_path.relative_to(root_dir)
    return f"{server_url}/{relative_path.as_posix()}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="MRAMG-Bench/IMAGE/IMAGE/images/", help="图片根目录，例如 images/")
    parser.add_argument("--port", type=int, default=8009, help="HTTP server 端口，0 表示随机")
    args = parser.parse_args()

    server_url = start_temp_http_server(args.root, args.port)
    print("临时 HTTP Server 已启动！")
    print("根目录:", args.root)
    print("访问 URL 前缀:", server_url)
    print("使用 get_local_image_url(img_path, root_dir, server_url) 获取图片 URL")

    # 阻塞主线程，让 server 持续运行
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Server 停止")