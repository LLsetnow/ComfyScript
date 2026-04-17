"""
ComfyUI 客户端模块
提供 ComfyUI 服务器连接、工作流执行、图像处理等功能
"""
import json
import time
import os
import sys
import shutil
import random
import subprocess
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

# ============================================================================
# 配置管理
# ============================================================================

class ComfyUIConfig:
    """ComfyUI 配置类"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """从 config.json5 加载配置"""
        from dotenv import load_dotenv
        load_dotenv()
        
        # 尝试加载 config.json5
        config_path = os.path.join(os.path.dirname(__file__), "config.json5")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 移除注释
                lines = []
                for line in content.split('\n'):
                    if '//' in line and not line.strip().startswith('"'):
                        line = line.split('//')[0]
                    lines.append(line)
                content = '\n'.join(lines)
                
                self._config = json.loads(content)
            except Exception as e:
                print(f"[ComfyUI] 加载配置文件失败: {e}")
                self._config = {}
        else:
            self._config = {}
    
    def get(self, key: str, default=None):
        """获取配置值"""
        return self._config.get(key, default) if self._config else default
    
    @property
    def api_url(self) -> str:
        """获取 ComfyUI API URL"""
        comfyui_config = self._config.get("comfyUI", {}) if self._config else {}
        # 支持直接配置完整 URL（如 ngrok 地址 https://xxxx.ngrok-free.app）
        url = comfyui_config.get('url', '')
        if url:
            return url.rstrip('/')
        host = comfyui_config.get('host', '127.0.0.1')
        port = comfyui_config.get('port', '8188')
        return f"http://{host}:{port}"

    @property
    def proxy_settings(self) -> dict:
        """获取代理设置，用于 requests 调用"""
        proxy_config = self._config.get("proxy", {}) if self._config else {}
        if not proxy_config.get("use_proxy", False):
            return {"http": None, "https": None}
        http_proxy = proxy_config.get("http", "")
        https_proxy = proxy_config.get("https", "")
        return {"http": http_proxy, "https": https_proxy} if http_proxy or https_proxy else {"http": None, "https": None}
    
    @property
    def folder(self) -> str:
        """获取 ComfyUI 文件夹路径"""
        comfyui_config = self._config.get("comfyUI", {}) if self._config else {}
        return comfyui_config.get('folder', 'ComfyUI')
    
    @property
    def python_exe(self) -> str:
        """获取 Python 可执行文件路径"""
        comfyui_config = self._config.get("comfyUI", {}) if self._config else {}
        folder = comfyui_config.get('folder', 'ComfyUI')
        if os.name == 'nt':
            return os.path.join(folder, 'python', 'python.exe')
        return os.path.join(folder, 'python', 'python')
    
    @property
    def main_py(self) -> str:
        """获取 main.py 路径"""
        comfyui_config = self._config.get("comfyUI", {}) if self._config else {}
        folder = comfyui_config.get('folder', 'ComfyUI')
        return os.path.join(folder, 'main.py')
    
    @property
    def input_folder(self) -> str:
        """获取输入文件夹"""
        return os.path.join(self.folder, "input")
    
    @property
    def output_folder(self) -> str:
        """获取输出文件夹"""
        return os.path.join(self.folder, "output", "FeiShuBot")
    
    @property
    def workflow_configs(self) -> Dict:
        """获取工作流配置"""
        workflows_config = self._config.get("workflows", {}) if self._config else {}
        workflow_configs = {}
        
        for name, workflow_data in workflows_config.items():
            config_dict = {
                "seed_id": str(workflow_data.get("seed_id")),
                "input_image_id": str(workflow_data.get("input_image_id")),
                "output_image_id": str(workflow_data.get("output_image_id")),
                "workflow": workflow_data.get("workflow", ""),
                "points_cost": workflow_data.get("points_cost", 10),
                "remove_iterations": workflow_data.get("remove_iterations", 1)
            }
            if "prompt_node_id" in workflow_data:
                config_dict["prompt_node_id"] = workflow_data["prompt_node_id"]
            workflow_configs[name] = config_dict
        
        return workflow_configs
    
    @property
    def default_workflow(self) -> str:
        """获取默认工作流"""
        return self._config.get("default_workflow", "Qwen_remove") if self._config else "Qwen_remove"
    
    @property
    def text_to_image_config(self) -> Dict:
        """获取文生图配置"""
        return self._config.get("text_to_image", {}) if self._config else {}


# 全局配置实例
config = ComfyUIConfig()


# ============================================================================
# ComfyUI 工作流处理器
# ============================================================================

class ComfyUIWorkflow:
    """ComfyUI 工作流处理类"""
    
    def __init__(self, seed_id: int, input_image_id: Optional[int],
                 output_image_id: int, workflow: str, prompt_node_id: Optional[int] = None):
        """
        初始化工作流处理器
        :param seed_id: 随机种子节点ID
        :param input_image_id: 输入图像节点ID，None表示不需要输入图像
        :param output_image_id: 输出图像节点ID
        :param workflow: 工作流文件名
        :param prompt_node_id: 提示词节点ID，用于图像编辑模式
        """
        self.seed_id = str(seed_id)
        self.input_image_id = str(input_image_id) if input_image_id else None
        self.output_image_id = str(output_image_id)
        self.prompt_node_id = str(prompt_node_id) if prompt_node_id else None
        self.workflow_file = workflow
        self.original_workflow = None
    
    def load_workflow(self) -> bool:
        """加载工作流 JSON 文件"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workflow_path = os.path.join(script_dir, "workflows", self.workflow_file)
        
        if not os.path.exists(workflow_path):
            raise FileNotFoundError(f"找不到工作流文件: {workflow_path}")
        
        with open(workflow_path, 'r', encoding='utf-8') as f:
            self.original_workflow = json.load(f)
        
        return True
    
    def set_seed(self, seed_value: int):
        """设置随机种子"""
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")
        self.original_workflow[self.seed_id]["inputs"]["seed"] = int(seed_value)
    
    def set_input_image(self, image_filename: str):
        """设置输入图像"""
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")
        if self.input_image_id is None:
            print("  警告: 此工作流不需要输入图像")
            return
        self.original_workflow[self.input_image_id]["inputs"]["image"] = image_filename
    
    def set_output_prefix(self, output_prefix: str):
        """设置输出文件前缀"""
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")
        self.original_workflow[self.output_image_id]["inputs"]["filename_prefix"] = output_prefix
    
    def set_prompt(self, prompt: str, prompt_node_id: Optional[str] = None):
        """设置提示词（用于文生图，使用 text 字段）"""
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")
        
        node_id = prompt_node_id if prompt_node_id else self.prompt_node_id
        
        if node_id is None:
            raise RuntimeError("未指定提示词节点ID")
        
        self.original_workflow[node_id]["inputs"]["text"] = prompt
    
    def set_prompt_field(self, prompt: str, prompt_node_id: Optional[str] = None):
        """设置提示词字段（用于图像编辑，使用 prompt 字段）"""
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")
        
        node_id = prompt_node_id if prompt_node_id else self.prompt_node_id
        
        if node_id is None:
            raise RuntimeError("未指定提示词节点ID")
        
        self.original_workflow[node_id]["inputs"]["prompt"] = prompt
    
    def get_workflow(self) -> Dict:
        """获取当前工作流配置"""
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")
        return self.original_workflow
    
    def create_workflow_copy(self) -> Dict:
        """创建工作流的深拷贝"""
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")
        return json.loads(json.dumps(self.original_workflow))


# ============================================================================
# 工具函数
# ============================================================================

def generate_random_seed() -> int:
    """生成15位随机数种子"""
    return random.randint(10**14, 10**15 - 1)


def save_image_with_unique_name(source_path: str, target_folder: str) -> str:
    """保存图像文件到指定文件夹，如果文件名重复则使用随机种子重命名"""
    original_filename = os.path.basename(source_path)
    file_ext = os.path.splitext(original_filename)[1]
    
    target_path = os.path.join(target_folder, original_filename)
    
    if not os.path.exists(target_path):
        shutil.copy2(source_path, target_path)
        return original_filename
    
    random_seed = generate_random_seed()
    new_filename = f"{random_seed}{file_ext}"
    target_path = os.path.join(target_folder, new_filename)
    
    while os.path.exists(target_path):
        random_seed = generate_random_seed()
        new_filename = f"{random_seed}{file_ext}"
        target_path = os.path.join(target_folder, new_filename)
    
    shutil.copy2(source_path, target_path)
    return new_filename


# ============================================================================
# ComfyUI 客户端
# ============================================================================

class ComfyUIClient:
    """ComfyUI 客户端"""
    
    def __init__(self, api_url: str = None):
        """
        初始化 ComfyUI 客户端
        :param api_url: ComfyUI API 地址，如 http://127.0.0.1:8188
        """
        self.api_url = api_url or config.api_url
        self._running = False
        self._process = None
    
    @property
    def is_remote(self) -> bool:
        """判断是否为远程服务器（非 localhost）"""
        return not (self.api_url.startswith("http://127.0.0.1") or
                    self.api_url.startswith("http://localhost"))
    
    @property
    def is_running(self) -> bool:
        """检查服务器是否运行"""
        return self.check_server()
    
    def check_server(self, max_attempts: int = 3, check_delay: int = 2) -> bool:
        """检查 ComfyUI 服务器是否可访问"""
        try:
            from urllib import request, error as urllib_error
        except ImportError:
            return False
        
        for attempt in range(max_attempts):
            try:
                request.urlopen(f"{self.api_url}/system_stats", timeout=3)
                return True
            except urllib_error.URLError:
                if attempt < max_attempts - 1:
                    time.sleep(check_delay)
                else:
                    return False
        return False

    def upload_image(self, image_path: str, subfolder: str = "", overwrite: bool = True) -> Optional[str]:
        """
        通过 HTTP API 上传图片到 ComfyUI 服务器。
        本地和远程服务器均可使用，远程时必须用此方法。

        Args:
            image_path: 本地图片路径
            subfolder: 子目录（如 "FeiShuBot"）
            overwrite: 是否覆盖同名文件

        Returns:
            str: 上传后的文件名（不含路径），失败返回 None
        """
        try:
            import requests
            from requests_toolbelt import MultipartEncoder
        except ImportError:
            # 无 requests 库时，本地模式用文件复制
            if not self.is_remote:
                filename = save_image_with_unique_name(image_path, config.input_folder)
                return filename
            print("[ComfyUI] requests 库未安装，无法上传图片到远程服务器")
            return None

        filename = os.path.basename(image_path)

        try:
            with open(image_path, 'rb') as f:
                form = {
                    'image': (filename, f, 'application/octet-stream'),
                    'subfolder': ('', subfolder),
                    'overwrite': ('', str(overwrite).lower()),
                }
                multi_form = MultipartEncoder(form)

                headers = {'Content-Type': multi_form.content_type}
                response = requests.post(
                    f"{self.api_url}/upload/image",
                    headers=headers,
                    data=multi_form,
                    timeout=60,
                    proxies=config.proxy_settings
                )

            if response.status_code != 200:
                print(f"[ComfyUI] 上传图片失败: HTTP {response.status_code}")
                return None

            result = response.json()
            uploaded_name = result.get('name', filename)
            print(f"[ComfyUI] 图片上传成功: {uploaded_name} (subfolder: {subfolder})")
            return uploaded_name

        except Exception as e:
            print(f"[ComfyUI] 上传图片异常: {e}")
            return None

    def download_output(self, filename: str, subfolder: str = "",
                        local_save_path: str = None) -> Optional[str]:
        """
        通过 HTTP API 从 ComfyUI 服务器下载输出图片。

        Args:
            filename: 远程文件名
            subfolder: 子目录
            local_save_path: 本地保存路径（含文件名），默认保存到 output_folder

        Returns:
            str: 本地文件路径，失败返回 None
        """
        # 本地模式：直接检查本地文件
        if not self.is_remote:
            return self.find_output_file(filename.replace(".png", "").replace(".jpg", ""))

        try:
            import requests as req_lib
        except ImportError:
            print("[ComfyUI] requests 库未安装，无法从远程服务器下载图片")
            return None

        try:
            params = {
                "filename": filename,
                "subfolder": subfolder,
                "type": "output",
            }
            response = req_lib.get(
                f"{self.api_url}/view",
                params=params,
                timeout=60,
                proxies=config.proxy_settings
            )

            if response.status_code != 200:
                print(f"[ComfyUI] 下载图片失败: HTTP {response.status_code}")
                return None

            if not local_save_path:
                os.makedirs(config.output_folder, exist_ok=True)
                local_save_path = os.path.join(config.output_folder, filename)

            with open(local_save_path, 'wb') as f:
                f.write(response.content)

            print(f"[ComfyUI] 图片下载成功: {local_save_path}")
            return local_save_path

        except Exception as e:
            print(f"[ComfyUI] 下载图片异常: {e}")
            return None
    
    def start_server(self) -> bool:
        """
        启动 ComfyUI 服务器
        :return: 是否启动成功
        """
        # 远程模式下不启动本地服务器
        if self.is_remote:
            print("[ComfyUI] 远程模式，跳过本地服务器启动")
            return self.check_server()
        
        # 检查是否已经在运行
        if self.check_server(max_attempts=1, check_delay=0):
            print("[ComfyUI] 服务器已在运行")
            return True
        
        try:
            python_exe = config.python_exe
            main_py = config.main_py
            
            # 检查路径是否存在
            if not os.path.exists(python_exe):
                print(f"[ComfyUI] Python 可执行文件不存在: {python_exe}")
                return False
            
            if not os.path.exists(main_py):
                print(f"[ComfyUI] main.py 不存在: {main_py}")
                return False
            
            print(f"[ComfyUI] 正在启动服务器...")
            
            if os.name == 'nt':
                process = subprocess.Popen(
                    f'start "ComfyUI Server" cmd /k ""{python_exe}" "{main_py}" --enable-manager"',
                    cwd=config.folder,
                    shell=True
                )
            else:
                process = subprocess.Popen(
                    [python_exe, main_py, "--enable-manager"],
                    cwd=config.folder
                )
            
            self._process = process
            print(f"[ComfyUI] 进程已启动，PID: {process.pid}")
            
            # 等待服务器启动（最多 150 秒）
            print("[ComfyUI] 等待服务器启动...")
            for i in range(30):
                time.sleep(5)
                if self.check_server(max_attempts=1, check_delay=0):
                    self._running = True
                    print(f"[ComfyUI] 服务器启动成功！")
                    return True
            
            print("[ComfyUI] 服务器启动超时")
            return False
            
        except Exception as e:
            print(f"[ComfyUI] 启动服务器失败: {e}")
            return False
    
    def stop_server(self):
        """停止 ComfyUI 服务器"""
        if self._process and self._process.poll() is None:
            print("[ComfyUI] 正在停止服务器...")
            try:
                self._process.terminate()
                self._process.wait(timeout=10)
                print("[ComfyUI] 服务器已停止")
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
                print("[ComfyUI] 服务器已强制停止")
            except Exception as e:
                print(f"[ComfyUI] 停止服务器时出错: {e}")
            
            self._process = None
            self._running = False
    
    def queue_prompt(self, prompt_workflow: Dict, max_retries: int = 3,
                    retry_delay: int = 2) -> Optional[str]:
        """将 prompt workflow 发送到 ComfyUI 服务器并排队执行"""
        try:
            from urllib import request, error as urllib_error
        except ImportError:
            print("[ComfyUI] urllib 不可用")
            return None
        
        p = {"prompt": prompt_workflow}
        data = json.dumps(p).encode('utf-8')
        req = request.Request(f"{self.api_url}/prompt", data=data)
        
        for attempt in range(max_retries):
            try:
                print(f"    正在提交工作流...")
                response = request.urlopen(req, timeout=10)
                result = json.loads(response.read().decode('utf-8'))
                prompt_id = result.get('prompt_id')
                print(f"    工作流已提交，prompt_id: {prompt_id}")
                return prompt_id
            except urllib_error.URLError as e:
                print(f"    URL错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return None
            except Exception as e:
                print(f"    发送失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return None
    
    def wait_for_completion(self, prompt_id: str, check_interval: int = 5,
                          timeout: int = 120) -> bool:
        """轮询检查任务完成状态"""
        try:
            from urllib import request, error as urllib_error
        except ImportError:
            return False
        
        start_time = time.time()
        check_count = 0
        initial_interval = 15
        initial_phase = True
        
        while time.time() - start_time < timeout:
            check_count += 1
            current_interval = initial_interval if initial_phase else check_interval
            
            try:
                req = request.Request(f"{self.api_url}/history/{prompt_id}")
                response = request.urlopen(req, timeout=5)
                result = json.loads(response.read().decode('utf-8'))
                
                if prompt_id in result:
                    history_data = result[prompt_id]
                    status = history_data.get('status', {}).get('completed', False)
                    if status:
                        print(f"    任务已完成 (耗时: {int(time.time() - start_time)}秒)")
                        return True
                    
                    exec_info = history_data.get('status', {}).get('exec_info', None)
                    if exec_info and 'error' in str(exec_info).lower():
                        print(f"    任务执行出错: {exec_info}")
                        return False
                
                if check_count % 5 == 0:
                    elapsed = int(time.time() - start_time)
                    print(f"    等待任务完成... (已等待 {elapsed}秒)")
                
            except urllib_error.HTTPError as e:
                if e.code == 404:
                    if check_count <= 3 or check_count % 10 == 0:
                        print(f"    任务尚未开始 (检查次数: {check_count})")
                    pass
                else:
                    print(f"    HTTP错误: {e.code} - {e}")
            except Exception as e:
                print(f"    检查状态时出错: {e}")
            
            if time.time() - start_time >= 30:
                initial_phase = False
            
            time.sleep(current_interval)
        
        print(f"    等待超时 (超过 {timeout} 秒)")
        return False
    
    def find_output_file(self, search_pattern: str, output_folder: str = None) -> Optional[str]:
        """查找输出文件"""
        output_folder = output_folder or config.output_folder
        output_file = None
        
        print(f"  搜索输出文件: {search_pattern}")
        print(f"  搜索目录: {output_folder}")
        
        time.sleep(1)
        
        found_files = []
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                if search_pattern in file:
                    full_path = os.path.join(root, file)
                    found_files.append(full_path)
                    output_file = full_path
                    print(f"  找到匹配文件: {full_path}")
            if output_file:
                break
        
        if output_file and os.path.exists(output_file):
            return output_file
        
        # 尝试查找最新的文件
        print(f"  尝试查找最近的图像文件...")
        all_files = []
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    full_path = os.path.join(root, file)
                    mtime = os.path.getmtime(full_path)
                    all_files.append((mtime, full_path))
        
        all_files.sort(reverse=True, key=lambda x: x[0])
        
        if all_files:
            newest_file = all_files[0][1]
            if time.time() - all_files[0][0] < 60:
                print(f"  找到最近创建的文件: {newest_file}")
                return newest_file
        
        return None


# ============================================================================
# 图像处理器
# ============================================================================

class ImageProcessor:
    """图像处理器"""
    
    def __init__(self, client: ComfyUIClient = None):
        self.client = client or ComfyUIClient()
    
    def _get_remote_output(self, prompt_id: str, search_pattern: str) -> Optional[str]:
        """
        从远程 ComfyUI 服务器获取输出图片。
        通过 /history API 获取输出文件信息，再通过 /view API 下载。

        Args:
            prompt_id: 工作流 prompt ID
            search_pattern: 搜索模式（seed 值）

        Returns:
            str: 下载后的本地文件路径，失败返回 None
        """
        try:
            import requests as req_lib
        except ImportError:
            print("[ComfyUI] requests 库未安装，无法从远程获取输出")
            return None

        try:
            # 从 history 获取输出信息
            response = req_lib.get(
                f"{self.client.api_url}/history/{prompt_id}",
                timeout=10,
                proxies=config.proxy_settings
            )
            if response.status_code != 200:
                print(f"[ComfyUI] 获取历史记录失败: HTTP {response.status_code}")
                return None

            history = response.json()
            if prompt_id not in history:
                print(f"[ComfyUI] 历史记录中未找到 prompt_id: {prompt_id}")
                return None

            outputs = history[prompt_id].get('outputs', {})

            # 遍历输出节点查找图片
            for node_id, node_output in outputs.items():
                images = node_output.get('images', [])
                for img_info in images:
                    filename = img_info.get('filename', '')
                    subfolder = img_info.get('subfolder', '')
                    # 检查文件名是否匹配
                    if search_pattern in filename:
                        return self.client.download_output(filename, subfolder)

            # 如果没有精确匹配，尝试下载第一张图
            for node_id, node_output in outputs.items():
                images = node_output.get('images', [])
                for img_info in images:
                    filename = img_info.get('filename', '')
                    subfolder = img_info.get('subfolder', '')
                    if filename:
                        print(f"[ComfyUI] 未精确匹配，下载第一张输出: {filename}")
                        return self.client.download_output(filename, subfolder)

            print("[ComfyUI] 远程输出中未找到图片")
            return None

        except Exception as e:
            print(f"[ComfyUI] 获取远程输出异常: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_image(self, image_path: str, workflow_name: str) -> Optional[str]:
        """
        使用 ComfyUI 处理图像
        :param image_path: 图像文件路径
        :param workflow_name: 工作流名称
        :return: 处理后的图片路径，失败返回 None
        """
        if not self.client.check_server():
            print("  ComfyUI 服务器未运行")
            return None
        
        workflow_configs = config.workflow_configs
        if workflow_name not in workflow_configs:
            print(f"  未知的工作流: {workflow_name}")
            return None
        
        cfg = workflow_configs[workflow_name]
        
        try:
            # 上传/保存图像到 ComfyUI
            print(f"  上传图像到 ComfyUI...")
            if self.client.is_remote:
                image_filename = self.client.upload_image(image_path, subfolder="FeiShuBot")
            else:
                image_filename = save_image_with_unique_name(image_path, config.input_folder)
            if not image_filename:
                print("  图像上传/保存失败")
                return None
            print(f"  图像文件名: {image_filename}")
            
            # 初始化工作流处理器
            prompt_node_id = cfg.get("prompt_node_id")
            workflow_handler = ComfyUIWorkflow(
                seed_id=cfg["seed_id"],
                input_image_id=cfg["input_image_id"],
                output_image_id=cfg["output_image_id"],
                workflow=cfg["workflow"],
                prompt_node_id=prompt_node_id
            )
            
            # 加载工作流
            workflow_handler.load_workflow()
            
            # 设置参数
            seed_value = generate_random_seed()
            prompt_workflow = workflow_handler.create_workflow_copy()
            prompt_workflow[workflow_handler.seed_id]["inputs"]["seed"] = int(seed_value)
            
            output_prefix = f"FeiShuBot\\{seed_value}"
            prompt_workflow[workflow_handler.output_image_id]["inputs"]["filename_prefix"] = output_prefix
            prompt_workflow[workflow_handler.input_image_id]["inputs"]["image"] = image_filename
            
            # 提交工作流
            print(f"  正在提交工作流...")
            prompt_id = self.client.queue_prompt(prompt_workflow)
            if not prompt_id:
                return None
            
            # 等待任务完成
            if not self.client.wait_for_completion(prompt_id, check_interval=2, timeout=300):
                return None
            
            # 获取输出文件
            if self.client.is_remote:
                output_file = self._get_remote_output(prompt_id, str(seed_value))
            else:
                output_file = self.client.find_output_file(str(seed_value))
            if output_file:
                print(f"  处理完成: {output_file}")
                return output_file
            
            return None
            
        except Exception as e:
            print(f"  处理图像时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_text_to_image(self, prompt: str) -> Optional[str]:
        """
        使用 ComfyUI 进行文生图
        :param prompt: 提示词
        :return: 生成的图片路径，失败返回 None
        """
        text_to_image_config = config.text_to_image_config
        if not text_to_image_config:
            print("  文生图配置未找到")
            return None
        
        if not self.client.check_server():
            print("  ComfyUI 服务器未运行")
            return None
        
        try:
            print(f"  开始文生图: {prompt[:50]}...")
            
            prompt_node_id = text_to_image_config.get("prompt_node_id")
            workflow_handler = ComfyUIWorkflow(
                seed_id=str(text_to_image_config["seed_id"]),
                input_image_id=None,
                output_image_id=str(text_to_image_config["output_image_id"]),
                workflow=text_to_image_config["workflow"],
                prompt_node_id=prompt_node_id
            )
            
            workflow_handler.load_workflow()
            workflow_handler.set_prompt(prompt)
            
            seed_value = generate_random_seed()
            workflow_handler.set_seed(seed_value)
            
            output_prefix = f"FeiShuBot\\t2i_{seed_value}"
            workflow_handler.set_output_prefix(output_prefix)
            
            prompt_workflow = workflow_handler.get_workflow()
            
            prompt_id = self.client.queue_prompt(prompt_workflow)
            if not prompt_id:
                return None
            
            if not self.client.wait_for_completion(prompt_id, check_interval=2, timeout=300):
                return None
            
            search_pattern = f"t2i_{seed_value}"
            if self.client.is_remote:
                output_file = self._get_remote_output(prompt_id, search_pattern)
            else:
                output_file = self.client.find_output_file(search_pattern)
            
            return output_file
            
        except Exception as e:
            print(f"  文生图出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_image_with_prompt(self, image_path: str, workflow_name: str, 
                                  prompt: str) -> Optional[str]:
        """
        使用 ComfyUI 处理图像（带提示词，用于图像编辑）
        :param image_path: 图像文件路径
        :param workflow_name: 工作流名称
        :param prompt: 编辑提示词
        :return: 处理后的图片路径，失败返回 None
        """
        if not self.client.check_server():
            print("  ComfyUI 服务器未运行")
            return None
        
        workflow_configs = config.workflow_configs
        if workflow_name not in workflow_configs:
            print(f"  未知的工作流: {workflow_name}")
            return None
        
        cfg = workflow_configs[workflow_name]
        
        try:
            # 上传/保存图像到 ComfyUI
            print(f"  上传图像到 ComfyUI...")
            if self.client.is_remote:
                image_filename = self.client.upload_image(image_path, subfolder="FeiShuBot")
            else:
                image_filename = save_image_with_unique_name(image_path, config.input_folder)
            if not image_filename:
                print("  图像上传/保存失败")
                return None
            print(f"  图像文件名: {image_filename}")
            print(f"  提示词: {prompt[:50]}...")
            
            prompt_node_id = cfg.get("prompt_node_id")
            workflow_handler = ComfyUIWorkflow(
                seed_id=cfg["seed_id"],
                input_image_id=cfg["input_image_id"],
                output_image_id=cfg["output_image_id"],
                workflow=cfg["workflow"],
                prompt_node_id=prompt_node_id
            )
            
            workflow_handler.load_workflow()
            
            seed_value = generate_random_seed()
            prompt_workflow = workflow_handler.create_workflow_copy()
            prompt_workflow[workflow_handler.seed_id]["inputs"]["seed"] = int(seed_value)
            
            output_prefix = f"FeiShuBot\\{seed_value}"
            prompt_workflow[workflow_handler.output_image_id]["inputs"]["filename_prefix"] = output_prefix
            prompt_workflow[workflow_handler.input_image_id]["inputs"]["image"] = image_filename
            
            if prompt_node_id:
                prompt_workflow[workflow_handler.prompt_node_id]["inputs"]["prompt"] = prompt
            
            print(f"  正在提交工作流...")
            
            prompt_id = self.client.queue_prompt(prompt_workflow)
            if not prompt_id:
                return None
            
            if not self.client.wait_for_completion(prompt_id, check_interval=2, timeout=300):
                return None
            
            if self.client.is_remote:
                output_file = self._get_remote_output(prompt_id, str(seed_value))
            else:
                output_file = self.client.find_output_file(str(seed_value))
            if output_file:
                print(f"  处理完成: {output_file}")
                return output_file
            
            return None
            
        except Exception as e:
            print(f"  处理图像时出错: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# 创建必要文件夹
# ============================================================================

os.makedirs(config.input_folder, exist_ok=True)
os.makedirs(config.output_folder, exist_ok=True)
