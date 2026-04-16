"""
ComfyUI 本地启动脚本（无需内网穿透）
"""
import subprocess
import sys
import os

# ============================================================================
# 配置
# ============================================================================

COMFYUI_PYTHON = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1\python\python.exe"
COMFYUI_MAIN = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1\main.py"
COMFYUI_PORT = 8188


def main():
    print("=" * 50)
    print("  ComfyUI 本地启动脚本")
    print("=" * 50)
    print()

    for label, path in [("Python", COMFYUI_PYTHON), ("main.py", COMFYUI_MAIN)]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} 不存在: {path}")
            input("按回车退出...")
            sys.exit(1)

    print(f"启动 ComfyUI 服务器 (端口 {COMFYUI_PORT})...")
    comfyui_proc = subprocess.Popen(
        [COMFYUI_PYTHON, COMFYUI_MAIN, "--listen", "0.0.0.0", "--port", str(COMFYUI_PORT)],
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
    )

    print()
    print(f"ComfyUI 本地地址: http://127.0.0.1:{COMFYUI_PORT}")
    print()

    try:
        print("按 Ctrl+C 停止服务...")
        while True:
            if comfyui_proc.poll() is not None:
                print("[WARNING] ComfyUI 进程已退出")
                break
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止服务...")

    try:
        comfyui_proc.terminate()
        comfyui_proc.wait(timeout=5)
    except Exception:
        comfyui_proc.kill()

    print("服务已停止")


if __name__ == "__main__":
    main()
