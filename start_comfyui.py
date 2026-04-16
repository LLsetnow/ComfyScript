"""
ComfyUI + Ngrok 启动脚本
启动 ComfyUI 服务器并通过 ngrok 进行内网穿透
"""
import subprocess
import sys
import time
import os
import urllib.request
import json

# ============================================================================
# 配置
# ============================================================================

COMFYUI_PYTHON = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1\python\python.exe"
COMFYUI_MAIN = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1\main.py"
COMFYUI_PORT = 8188
NGROK_EXE = r"E:\Program Files\ngrok-v3-stable-windows-amd64\ngrok.exe"


def get_ngrok_public_url(retries=10, delay=2) -> str:
    """通过 ngrok 本地 API 获取公网地址"""
    api_url = "http://127.0.0.1:4040/api/tunnels"
    for i in range(retries):
        try:
            with urllib.request.urlopen(api_url, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                tunnels = data.get("tunnels", [])
                for tunnel in tunnels:
                    public_url = tunnel.get("public_url", "")
                    if public_url.startswith("https://"):
                        return public_url
        except Exception:
            pass
        time.sleep(delay)
    return ""


def main():
    print("=" * 50)
    print("  ComfyUI + Ngrok 启动脚本")
    print("=" * 50)
    print()

    # 检查文件
    for label, path in [("Python", COMFYUI_PYTHON), ("main.py", COMFYUI_MAIN), ("ngrok", NGROK_EXE)]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} 不存在: {path}")
            input("按回车退出...")
            sys.exit(1)

    # 启动 ComfyUI
    print(f"[1/2] 启动 ComfyUI 服务器 (端口 {COMFYUI_PORT})...")
    comfyui_proc = subprocess.Popen(
        [COMFYUI_PYTHON, COMFYUI_MAIN, "--listen", "0.0.0.0", "--port", str(COMFYUI_PORT)],
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
    )

    # 等待 ComfyUI 启动
    print("      等待 ComfyUI 启动...")
    time.sleep(10)

    # 启动 ngrok
    print("[2/2] 启动 ngrok 内网穿透...")
    ngrok_proc = subprocess.Popen(
        [NGROK_EXE, "http", str(COMFYUI_PORT)],
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
    )

    print()
    print("  等待获取 ngrok 公网地址...")
    public_url = get_ngrok_public_url()

    print()
    print("=" * 50)
    print("  启动完成！")
    print(f"  - ComfyUI 本地地址: http://127.0.0.1:{COMFYUI_PORT}")
    if public_url:
        print(f"  - 公网地址: {public_url}")
    else:
        print("  - 公网地址: 获取失败，请访问 http://127.0.0.1:4040 查看")
    print("=" * 50)
    print()

    try:
        print("按 Ctrl+C 停止所有服务...")
        while True:
            # 检查子进程是否存活
            if comfyui_proc.poll() is not None:
                print("[WARNING] ComfyUI 进程已退出")
                break
            if ngrok_proc.poll() is not None:
                print("[WARNING] ngrok 进程已退出")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止所有服务...")

    # 终止子进程
    for proc in [ngrok_proc, comfyui_proc]:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    print("所有服务已停止")


if __name__ == "__main__":
    main()
