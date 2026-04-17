"""
ComfyUI + Ngrok 启动脚本
启动 ComfyUI 服务器并通过 ngrok 进行内网穿透，
自动将公网地址写入 config.json5 供 main.py 使用
"""
import subprocess
import sys
import time
import os
import urllib.request
import json
import re

# ============================================================================
# 配置
# ============================================================================

COMFYUI_PYTHON = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1\python\python.exe"
COMFYUI_MAIN = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1\main.py"
COMFYUI_PORT = 8188
NGROK_EXE = r"E:\Program Files\ngrok-v3-stable-windows-amd64\ngrok.exe"
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json5")


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


def update_config_url(public_url: str):
    """将 ngrok 公网地址写入 config.json5 的 comfyUI.url 字段"""
    if not os.path.exists(CONFIG_FILE):
        print(f"[WARNING] 配置文件不存在: {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # 查找 "url": "..." 并替换
    # 匹配 "url": "任意内容" (包括空字符串)
    pattern = r'("url"\s*:\s*)"[^"]*"'
    replacement = f'\\1"{public_url}"'

    new_content, count = re.subn(pattern, replacement, content)

    if count > 0:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  ✅ 已将公网地址写入 config.json5: url = \"{public_url}\"")
    else:
        print(f"  [WARNING] 未在 config.json5 中找到 \"url\" 字段，请手动添加:")
        print(f'    "url": "{public_url}"')


def clear_config_url():
    """清除 config.json5 中的公网地址（停止服务时调用）"""
    if not os.path.exists(CONFIG_FILE):
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'("url"\s*:\s*)"[^"]*"'
    new_content = re.sub(pattern, r'\1""', content)

    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("  ✅ 已清除 config.json5 中的公网地址")


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

    # 自动更新配置文件
    if public_url:
        update_config_url(public_url)
    else:
        print("  ❌ 获取公网地址失败，请访问 http://127.0.0.1:4040 手动获取")
        print("  获取后请手动在 config.json5 中设置 comfyUI.url")

    print()
    print("=" * 50)
    print("  启动完成！")
    print(f"  - ComfyUI 本地地址: http://127.0.0.1:{COMFYUI_PORT}")
    if public_url:
        print(f"  - 公网地址: {public_url}")
        print()
        print("  📋 使用方式：")
        print("    1. config.json5 已自动配置 comfyUI.url")
        print("    2. 在另一台机器上运行 main.py 即可连接此 ComfyUI")
        print(f"    3. 另一台机器的 config.json5 中设置:")
        print(f'       "url": "{public_url}"')
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

    # 清除配置中的公网地址
    clear_config_url()

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
