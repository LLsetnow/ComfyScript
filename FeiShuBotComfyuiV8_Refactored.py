"""
飞书机器人 V8 - 重构版本
基于飞书SDK WebSocket Client 实现长连接监听事件
集成 ComfyUI 图像处理功能和任务队列管理

主要改进：
- 统一的消息去重管理器
- 结构化的任务队列系统
- 清晰的错误处理机制
- 配置管理优化
- 模块化的代码组织
"""
import json
import time
import os
import sys
import shutil
import random
import threading
import subprocess
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 加载 .env 配置
load_dotenv()

# 导入飞书客户端
from feishu_client import FeishuClient, FeishuMessenger, FeishuAPI


# ============================================================================
# 飞书SDK导入
# ============================================================================

try:
    import lark_oapi as lark
    from lark_oapi.ws.client import Client
    from lark_oapi.event.dispatcher_handler import EventDispatcherHandlerBuilder
    from lark_oapi.core.enum import LogLevel
    SDK_AVAILABLE = True
    print("[OK] 飞书SDK已加载")
except ImportError as e:
    SDK_AVAILABLE = False
    print(f"[ERROR] 未安装 lark-oapi SDK: {e}")
    print("请运行: pip install lark-oapi")
    sys.exit(1)

try:
    from urllib import request, error as urllib_error
    URLLIB_AVAILABLE = True
except ImportError:
    URLLIB_AVAILABLE = False


# ============================================================================
# 配置管理模块
# ============================================================================

class ConfigManager:
    """配置文件管理器"""

    @staticmethod
    def load_json5(file_path: str) -> Optional[Dict]:
        """加载JSON5配置文件（支持注释）"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 移除注释
            lines = []
            for line in content.split('\n'):
                # 保留URL中的//
                if '//' in line and not line.strip().startswith('"'):
                    line = line.split('//')[0]
                lines.append(line)
            content = '\n'.join(lines)

            return json.loads(content)
        except Exception as e:
            print(f"[ERROR] 加载配置文件失败: {e}")
            return None

    @staticmethod
    def get_app_config() -> Dict:
        """获取应用配置"""
        config_path = os.path.join(os.path.dirname(__file__), "config.json5")
        config = ConfigManager.load_json5(config_path)

        if config is None:
            print("[ERROR] 无法加载配置文件，程序退出")
            sys.exit(1)

        return config


# 加载应用配置
app_config = ConfigManager.get_app_config()


# 提取各模块配置
class AppConfig:
    """应用配置常量"""

    # 飞书配置
    FEISHU_APP_ID = app_config.get("feishu", {}).get("app_id", "")
    FEISHU_APP_SECRET = app_config.get("feishu", {}).get("app_secret", "")
    FEISHU_API_BASE = app_config.get("feishu", {}).get("api_base", "https://open.feishu.cn/open-apis")

    # ComfyUI配置
    comfyui_config = app_config.get("comfyUI", {})
    COMFYUI_API_URL = f"http://{comfyui_config.get('host', '127.0.0.1')}:{comfyui_config.get('port', '8188')}"
    COMFYUI_FOLDER = comfyui_config.get('folder', 'D:\\AI_Graph\\ConfyUI-aki\\ComfyUI-aki-v1')
    COMFYUI_PYTHON_EXE = comfyui_config.get('python_exe', os.path.join(COMFYUI_FOLDER, 'python', 'python.exe'))
    COMFYUI_MAIN_PY = comfyui_config.get('main_py', os.path.join(COMFYUI_FOLDER, 'main.py'))
    COMFYUI_INPUT_FOLDER = os.path.join(COMFYUI_FOLDER, "input")
    COMFYUI_OUTPUT_FOLDER = os.path.join(COMFYUI_FOLDER, "output", "FeiShuBot")

    # 工作流配置
    @staticmethod
    def load_workflow_configs() -> Dict:
        """加载工作流配置"""
        workflows_config = app_config.get("workflows", {})
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

    WORKFLOW_CONFIGS = load_workflow_configs.__func__()
    DEFAULT_WORKFLOW = app_config.get("default_workflow", "Qwen_remove")

    # 文生图配置
    TEXT_TO_IMAGE_CONFIG = app_config.get("text_to_image", {})

    # DeepSeek配置
    DEEPSEEK_API_KEY = app_config.get("deepseek", {}).get("api_key", "")
    DEEPSEEK_API_URL = app_config.get("deepseek", {}).get("api_url", "https://api.deepseek.com/v1/chat/completions")


# 创建必要的文件夹
os.makedirs(AppConfig.COMFYUI_INPUT_FOLDER, exist_ok=True)
os.makedirs(AppConfig.COMFYUI_OUTPUT_FOLDER, exist_ok=True)


# ============================================================================
# 消息去重管理器
# ============================================================================

@dataclass
class MessageDeduplicator:
    """消息去重管理器"""

    processed_messages: Dict[str, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    max_cache_size: int = 1000
    cache_ttl: int = 600  # 10分钟

    def is_duplicate(self, message_id: str) -> bool:
        """检查消息是否重复"""
        with self.lock:
            current_time = time.time()

            # 清理过期记录
            self._cleanup_expired(current_time)

            # 检查是否重复
            if message_id in self.processed_messages:
                return True

            # 标记为已处理
            self.processed_messages[message_id] = current_time

            # 限制缓存大小
            self._limit_cache_size()

            return False

    def _cleanup_expired(self, current_time: float):
        """清理过期的消息记录"""
        expired_keys = [
            msg_id for msg_id, timestamp in self.processed_messages.items()
            if current_time - timestamp > self.cache_ttl
        ]
        for key in expired_keys:
            del self.processed_messages[key]
        if expired_keys:
            print(f"  [去重] 清理了 {len(expired_keys)} 条过期消息记录")

    def _limit_cache_size(self):
        """限制缓存大小"""
        if len(self.processed_messages) > self.max_cache_size:
            # 按时间戳排序，保留最新的500条
            sorted_items = sorted(
                self.processed_messages.items(),
                key=lambda x: x[1],
                reverse=True
            )
            to_remove = sorted_items[500:]
            for msg_id, _ in to_remove:
                del self.processed_messages[msg_id]
            print(f"  [去重] 限制缓存大小，保留最新的500条")

    def generate_message_id(self, chat_id: str, content: str) -> str:
        """生成消息ID（当原始ID为空时）"""
        current_time = int(time.time() // 10)  # 10秒精度
        return f"{chat_id}_{content[:50]}_{current_time}"


# 全局消息去重实例
message_deduplicator = MessageDeduplicator()


# ============================================================================
# 任务队列管理模块
# ============================================================================

@dataclass
class Task:
    """任务数据结构"""
    task_number: int
    user_id: str
    created_time: float = field(default_factory=time.time)
    status: str = "pending"  # pending, processing, completed, failed


class TaskQueue:
    """任务队列管理器"""

    def __init__(self):
        self.tasks: Dict[int, Task] = {}
        self.lock = threading.Lock()
        self.counter = 0
        self.counter_lock = threading.Lock()

    def add_task(self, user_id: str) -> int:
        """添加新任务到队列"""
        with self.counter_lock:
            self.counter += 1
            task_number = self.counter

        with self.lock:
            self.tasks[task_number] = Task(task_number=task_number, user_id=user_id)
            print(f"  [队列] 任务 #{task_number} (用户 {user_id}) 已加入队列，队列长度: {len(self.tasks)}")

        return task_number

    def remove_task(self, task_number: int) -> bool:
        """从队列移除任务"""
        with self.lock:
            if task_number in self.tasks:
                del self.tasks[task_number]
                print(f"  [队列] 任务 #{task_number} 已从队列移除，队列长度: {len(self.tasks)}")
                return True
            return False

    def get_task_info(self, task_number: int) -> Optional[Tuple[int, int, int]]:
        """
        获取任务信息
        :return: (位置, 前面等待数, 总任务数)
        """
        with self.lock:
            task_numbers = sorted(self.tasks.keys())
            if task_number not in task_numbers:
                return None

            position = task_numbers.index(task_number) + 1
            waiting = position - 1
            total = len(task_numbers)

            return (position, waiting, total)

    def get_user_tasks(self, user_id: str) -> List[int]:
        """获取用户的所有任务序号"""
        with self.lock:
            return [task_num for task_num, task in self.tasks.items()
                    if task.user_id == user_id]

    def format_user_status(self, user_id: str) -> str:
        """格式化用户的队列状态"""
        user_tasks = self.get_user_tasks(user_id)

        if not user_tasks:
            return "📭 您当前没有在处理的任务"

        task_numbers = sorted(self.tasks.keys())
        total = len(task_numbers)
        status_lines = [f"📊 当前队列总任务数: {total}", ""]

        for task_num in sorted(user_tasks):
            position = task_numbers.index(task_num) + 1
            status_lines.append(f"  任务 #{task_num}: 位置 {position}/{total}")

        return "\n".join(status_lines)


# 全局任务队列实例
task_queue = TaskQueue()


# ============================================================================
# ComfyUI工作流处理器
# ============================================================================

class ComfyUIWorkflow:
    """ComfyUI工作流处理类"""

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
        """加载工作流JSON文件"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workflow_path = os.path.join(script_dir, self.workflow_file)

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

        # 优先使用传入的prompt_node_id，否则使用实例变量
        node_id = prompt_node_id if prompt_node_id else self.prompt_node_id

        if node_id is None:
            raise RuntimeError("未指定提示词节点ID")

        self.original_workflow[node_id]["inputs"]["text"] = prompt

    def set_prompt_field(self, prompt: str, prompt_node_id: Optional[str] = None):
        """设置提示词字段（用于图像编辑，使用 prompt 字段）"""
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")

        # 优先使用传入的prompt_node_id，否则使用实例变量
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
# ComfyUI工具函数
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


def check_comfyui_server(max_attempts: int = 3, check_delay: int = 2) -> bool:
    """检查ComfyUI服务器是否可访问"""
    try:
        from urllib import request, error as urllib_error
    except ImportError:
        return False

    for attempt in range(max_attempts):
        try:
            request.urlopen(f"{AppConfig.COMFYUI_API_URL}/system_stats", timeout=3)
            return True
        except urllib_error.URLError:
            if attempt < max_attempts - 1:
                time.sleep(check_delay)
            else:
                return False
    return False


def start_comfyui_server():
    """
    启动 ComfyUI 服务器
    :return: 进程对象，启动失败返回 None
    """
    global comfyui_process, comfyui_running

    # 检查是否已经在运行
    if check_comfyui_server(max_attempts=1, check_delay=0):
        print("ComfyUI 服务器已在运行")
        return None

    try:
        # 从配置获取 Python 路径和 main.py 路径
        python_exe = AppConfig.COMFYUI_PYTHON_EXE
        main_py = AppConfig.COMFYUI_MAIN_PY

        # 检查路径是否存在
        if not os.path.exists(python_exe):
            print(f"错误: Python 可执行文件不存在: {python_exe}")
            print(f"请检查配置文件 config.json5 中的 comfyUI.python_exe")
            return None

        if not os.path.exists(main_py):
            print(f"错误: main.py 不存在: {main_py}")
            print(f"请检查配置文件 config.json5 中的 comfyUI.main_py")
            return None

        print(f"正在启动 ComfyUI 服务器...")
        print(f"Python 路径: {python_exe}")
        print(f"主程序: {main_py}")
        print(f"提示: ComfyUI 将在新终端窗口中启动，请查看新终端窗口的输出信息")

        # 启动 ComfyUI 进程（启用 manager）
        # 不使用 CREATE_NO_WINDOW，让用户在新终端窗口中看到启动信息
        if os.name == 'nt':
            # Windows: 使用 start 命令在新窗口中启动
            process = subprocess.Popen(
                f'start "ComfyUI Server" cmd /k ""{python_exe}" "{main_py}" --enable-manager"',
                cwd=AppConfig.COMFYUI_FOLDER,
                shell=True
            )
        else:
            # Linux/Mac: 正常启动
            process = subprocess.Popen(
                [python_exe, main_py, "--enable-manager"],
                cwd=AppConfig.COMFYUI_FOLDER
            )

        comfyui_process = process
        print(f"ComfyUI 进程已启动，PID: {process.pid}")

        # 等待服务器启动（最多等待 150 秒）
        print("等待 ComfyUI 服务器启动...")
        for i in range(30):  # 30 * 5 = 150 秒
            time.sleep(5)
            if check_comfyui_server(max_attempts=1, check_delay=0):
                comfyui_running = True
                print(f"✓ ComfyUI 服务器启动成功！(耗时: {i * 5}秒)")
                return process

        # 超时未启动
        print("ComfyUI 服务器启动超时")
        comfyui_process = None
        return None

    except Exception as e:
        print(f"启动 ComfyUI 服务器失败: {e}")
        comfyui_process = None
        return None


def stop_comfyui_server():
    """
    停止 ComfyUI 服务器（如果是我们启动的）
    """
    global comfyui_process, comfyui_running

    if comfyui_process and comfyui_process.poll() is None:
        print("正在停止 ComfyUI 服务器...")
        try:
            comfyui_process.terminate()
            # 等待进程结束（最多 10 秒）
            comfyui_process.wait(timeout=10)
            print("✓ ComfyUI 服务器已停止")
        except subprocess.TimeoutExpired:
            print("进程未正常结束，强制终止...")
            comfyui_process.kill()
            comfyui_process.wait()
            print("✓ ComfyUI 服务器已强制停止")
        except Exception as e:
            print(f"停止 ComfyUI 服务器时出错: {e}")

        comfyui_process = None
        comfyui_running = False


# ComfyUI 服务器状态监控
comfyui_running = False
comfyui_process = None  # ComfyUI 进程对象


def queue_prompt(prompt_workflow: Dict, max_retries: int = 3,
                retry_delay: int = 2) -> Optional[str]:
    """将prompt workflow发送到ComfyUI服务器并排队执行"""
    try:
        from urllib import request, error as urllib_error
    except ImportError:
        print("  [ERROR] urllib不可用")
        return None

    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req = request.Request(f"{AppConfig.COMFYUI_API_URL}/prompt", data=data)

    for attempt in range(max_retries):
        try:
            print(f"    正在提交工作流到 ComfyUI...")
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
                print(f"    工作流提交失败，超过最大重试次数")
                return None
        except Exception as e:
            print(f"    发送失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"    工作流提交失败，超过最大重试次数")
                return None


def wait_for_completion(prompt_id: str, check_interval: int = 5,
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
            req = request.Request(f"{AppConfig.COMFYUI_API_URL}/history/{prompt_id}")
            response = request.urlopen(req, timeout=5)
            result = json.loads(response.read().decode('utf-8'))

            if prompt_id in result:
                history_data = result[prompt_id]
                status = history_data.get('status', {}).get('completed', False)
                if status:
                    print(f"    任务已完成 (耗时: {int(time.time() - start_time)}秒, 检查次数: {check_count})")
                    return True

                exec_info = history_data.get('status', {}).get('exec_info', None)
                if exec_info and 'error' in str(exec_info).lower():
                    print(f"    任务执行出错: {exec_info}")
                    return False

            if check_count % 5 == 0:
                elapsed = int(time.time() - start_time)
                print(f"    等待任务完成... (已等待 {elapsed}秒, 检查次数: {check_count})")

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

    elapsed = int(time.time() - start_time)
    print(f"    等待超时 (超过 {timeout} 秒, 总检查次数: {check_count})")
    return False


def find_output_file(search_pattern: str) -> Optional[str]:
    """查找输出文件"""
    output_file = None
    print(f"  正在搜索输出文件，搜索模式: {search_pattern}")
    print(f"  搜索目录: {AppConfig.COMFYUI_OUTPUT_FOLDER}")

    # 等待一小段时间，确保文件已写入磁盘
    time.sleep(1)

    found_files = []
    for root, dirs, files in os.walk(AppConfig.COMFYUI_OUTPUT_FOLDER):
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

    # 打印输出文件夹中的文件用于调试
    print(f"  未找到输出文件，搜索模式: {search_pattern}")
    print(f"  输出文件夹中的文件:")
    file_count = 0
    for root, dirs, files in os.walk(AppConfig.COMFYUI_OUTPUT_FOLDER):
        for file in files:
            print(f"    {file}")
            file_count += 1
            if file_count > 20:
                print(f"    ... (还有更多文件)")
                break
        if file_count > 20:
            break

    # 尝试查找最新的文件
    print(f"  尝试查找最新的图像文件...")
    all_files = []
    for root, dirs, files in os.walk(AppConfig.COMFYUI_OUTPUT_FOLDER):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                full_path = os.path.join(root, file)
                mtime = os.path.getmtime(full_path)
                all_files.append((mtime, full_path))

    # 按修改时间排序，获取最新的文件
    all_files.sort(reverse=True, key=lambda x: x[0])

    if all_files:
        newest_file = all_files[0][1]
        if time.time() - all_files[0][0] < 60:
            print(f"  找到最近创建的文件: {newest_file}")
            return newest_file

    return None


# ============================================================================
# 图像处理模块
# ============================================================================

class ImageProcessor:
    """图像处理器"""

    @staticmethod
    def process_image(image_path: str, workflow_name: str) -> Optional[str]:
        """
        使用ComfyUI处理图像
        :param image_path: 图像文件路径
        :param workflow_name: 工作流名称
        :return: 处理后的图片路径，失败返回None
        """
        if not check_comfyui_server():
            print("  ComfyUI服务器未运行")
            return None

        if workflow_name not in AppConfig.WORKFLOW_CONFIGS:
            print(f"  未知的工作流: {workflow_name}")
            return None

        config = AppConfig.WORKFLOW_CONFIGS[workflow_name]

        try:
            # 保存图像到ComfyUI input文件夹
            print(f"  保存图像到 input 文件夹...")
            image_filename = save_image_with_unique_name(image_path, AppConfig.COMFYUI_INPUT_FOLDER)

            print(f"  图像文件名: {image_filename}")

            # 初始化工作流处理器
            prompt_node_id = config.get("prompt_node_id")
            workflow_handler = ComfyUIWorkflow(
                seed_id=config["seed_id"],
                input_image_id=config["input_image_id"],
                output_image_id=config["output_image_id"],
                workflow=config["workflow"],
                prompt_node_id=prompt_node_id
            )

            # 加载工作流
            workflow_handler.load_workflow()

            # 进行处理
            seed_value = generate_random_seed()
            prompt_workflow = workflow_handler.create_workflow_copy()
            prompt_workflow[workflow_handler.seed_id]["inputs"]["seed"] = int(seed_value)

            output_prefix = f"FeiShuBot\\{seed_value}"
            prompt_workflow[workflow_handler.output_image_id]["inputs"]["filename_prefix"] = output_prefix

            prompt_workflow[workflow_handler.input_image_id]["inputs"]["image"] = image_filename

            print(f"  正在提交工作流到 ComfyUI...")

            # 提交工作流
            prompt_id = queue_prompt(prompt_workflow)
            if not prompt_id:
                print("  工作流提交失败")
                return None

            # 等待任务完成
            if not wait_for_completion(prompt_id, check_interval=2, timeout=300):
                print("  任务未完成")
                return None

            # 查找输出文件
            output_file = find_output_file(str(seed_value))
            if output_file:
                print(f"  处理完成: {output_file}")
                return output_file
            else:
                print("  未找到输出文件")
                return None

        except Exception as e:
            print(f"  处理图像时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def process_text_to_image(prompt: str) -> Optional[str]:
        """
        使用ComfyUI进行文生图
        :param prompt: 提示词
        :return: 生成的图片路径，失败返回None
        """
        if not AppConfig.TEXT_TO_IMAGE_CONFIG:
            print("  文生图配置未找到")
            return None

        if not check_comfyui_server():
            print("  ComfyUI服务器未运行")
            return None

        try:
            print(f"  开始文生图: {prompt[:50]}...")

            # 初始化工作流处理器
            prompt_node_id = AppConfig.TEXT_TO_IMAGE_CONFIG.get("prompt_node_id")
            workflow_handler = ComfyUIWorkflow(
                seed_id=str(AppConfig.TEXT_TO_IMAGE_CONFIG["seed_id"]),
                input_image_id=None,
                output_image_id=str(AppConfig.TEXT_TO_IMAGE_CONFIG["output_image_id"]),
                workflow=AppConfig.TEXT_TO_IMAGE_CONFIG["workflow"],
                prompt_node_id=prompt_node_id
            )

            # 加载工作流
            workflow_handler.load_workflow()

            # 设置提示词
            workflow_handler.set_prompt(prompt)

            # 设置随机种子
            seed_value = generate_random_seed()
            workflow_handler.set_seed(seed_value)

            # 设置输出文件前缀
            output_prefix = f"FeiShuBot\\t2i_{seed_value}"
            workflow_handler.set_output_prefix(output_prefix)

            # 获取工作流
            prompt_workflow = workflow_handler.get_workflow()

            # 提交到ComfyUI
            prompt_id = queue_prompt(prompt_workflow)
            if not prompt_id:
                print("  工作流提交失败")
                return None

            # 等待任务完成
            if not wait_for_completion(prompt_id, check_interval=2, timeout=300):
                print("  任务未完成")
                return None

            # 查找输出文件
            search_pattern = f"t2i_{seed_value}"
            output_file = find_output_file(search_pattern)

            return output_file

        except Exception as e:
            print(f"  文生图出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def process_image_with_prompt(image_path: str, workflow_name: str, prompt: str) -> Optional[str]:
        """
        使用ComfyUI处理图像（带提示词，用于图像编辑）
        :param image_path: 图像文件路径
        :param workflow_name: 工作流名称
        :param prompt: 编辑提示词
        :return: 处理后的图片路径，失败返回None
        """
        if not check_comfyui_server():
            print("  ComfyUI服务器未运行")
            return None

        if workflow_name not in AppConfig.WORKFLOW_CONFIGS:
            print(f"  未知的工作流: {workflow_name}")
            return None

        config = AppConfig.WORKFLOW_CONFIGS[workflow_name]

        try:
            # 保存图像到ComfyUI input文件夹
            print(f"  保存图像到 input 文件夹...")
            image_filename = save_image_with_unique_name(image_path, AppConfig.COMFYUI_INPUT_FOLDER)

            print(f"  图像文件名: {image_filename}")
            print(f"  提示词: {prompt[:50]}...")

            # 初始化工作流处理器
            prompt_node_id = config.get("prompt_node_id")
            workflow_handler = ComfyUIWorkflow(
                seed_id=config["seed_id"],
                input_image_id=config["input_image_id"],
                output_image_id=config["output_image_id"],
                workflow=config["workflow"],
                prompt_node_id=prompt_node_id
            )

            # 加载工作流
            workflow_handler.load_workflow()

            # 进行处理
            seed_value = generate_random_seed()
            prompt_workflow = workflow_handler.create_workflow_copy()
            prompt_workflow[workflow_handler.seed_id]["inputs"]["seed"] = int(seed_value)

            output_prefix = f"FeiShuBot\\{seed_value}"
            prompt_workflow[workflow_handler.output_image_id]["inputs"]["filename_prefix"] = output_prefix

            prompt_workflow[workflow_handler.input_image_id]["inputs"]["image"] = image_filename

            # 设置提示词（使用 prompt 字段）
            if prompt_node_id:
                prompt_workflow[workflow_handler.prompt_node_id]["inputs"]["prompt"] = prompt
                print(f"  已设置提示词到节点 {prompt_node_id}: {prompt[:50]}...")
            else:
                print(f"  警告: 工作流配置中未找到 prompt_node_id")

            print(f"  正在提交工作流到 ComfyUI...")

            # 提交工作流
            prompt_id = queue_prompt(prompt_workflow)
            if not prompt_id:
                print("  工作流提交失败")
                return None

            # 等待任务完成
            if not wait_for_completion(prompt_id, check_interval=2, timeout=300):
                print("  任务未完成")
                return None

            # 查找输出文件
            output_file = find_output_file(str(seed_value))
            if output_file:
                print(f"  处理完成: {output_file}")
                return output_file
            else:
                print("  未找到输出文件")
                return None

        except Exception as e:
            print(f"  处理图像时出错: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# 自然语言处理模块
# ============================================================================

class NLProcessor:
    """自然语言处理器（DeepSeek）"""

    @staticmethod
    def call_deepseek(user_message: str, chat_id: str) -> Optional[Dict]:
        """
        调用DeepSeek API进行function calling
        :param user_message: 用户消息
        :param chat_id: 聊天ID
        :return: 返回结构化的指令或None
        """
        if not AppConfig.DEEPSEEK_API_KEY:
            print("  DeepSeek API未配置")
            return None

        try:
            import requests
        except ImportError:
            print("  requests库未安装")
            return None

        current_workflow = get_user_workflow(chat_id)

        # 定义可用的functions
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "switch_workflow",
                    "description": "切换图像处理模式。当用户想要切换当前使用的图像处理方式时调用此函数。例如用户说'切换面部重绘'、'改成去除背景'、'我想用服装移除'等。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "workflow_name": {
                                "type": "string",
                                "description": "目标处理模式的名称",
                                "enum": ["FaceFix", "BackgroundRemove", "Qwen_remove", "BoobsFix", "Qwen_edit"]
                            }
                        },
                        "required": ["workflow_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "text_to_image",
                    "description": "【重要】文生图功能。当用户提到以下关键词时必须调用：生成、画、创建、制作、画图、图片、图像。例如：'帮我生成一张美女图'、'画一个风景'、'创建一个动漫角色'、'帮我生成一张中国网红图'、'创建图片'、'画图'等。请将用户描述作为完整的prompt参数传递。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "用于生成图像的详细提示词，应该完整保留用户的原始描述，包含图像的主题、风格、细节等。"
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            }
        ]

        system_prompt = f"""你是一个智能助手，帮助用户使用ComfyUI图像处理机器人。

    当前用户的处理模式是：{current_workflow}

    你有以下工具可以使用：

    1. switch_workflow: 当用户明确要求切换图像处理模式时调用
    - 触发关键词：切换、改成、使用、启用 + 模式名称
    - 可用模式：FaceFix(面部重绘)、BackgroundRemove(去除背景杂物)、Qwen_remove(服装移除)、BoobsFix(胸部重绘)、Qwen_edit(图像编辑)
    - 例如："切换到面部重绘"、"改成去除背景"、"用服装移除模式"

    2. text_to_image: 当用户要求生成图片时调用（重要！）
    - 触发关键词：生成、画、创建、制作、想要...图/图片/图像
    - 例如："帮我生成一张美女图"、"画一个风景"、"创建一个动漫角色"、"帮我生成一张中国网红图"
    - 无论当前是什么模式，只要用户提到"生成图片"就必须调用此函数

    重要规则：
    - 如果用户说"生成"、"画"、"创建"等词汇，无论上下文如何，必须调用 text_to_image 函数
    - 不要告诉用户你"无法生成图片"或"无法直接生成"，直接调用 text_to_image 函数
    - 不要询问用户是否要切换模式，直接根据用户意图调用相应的函数
    - 只有当用户的问题完全与图像处理无关时，才进行普通对话

    输出格式要求：
    - 所有回复必须是纯文本格式，不要使用 Markdown 语法（不要使用 **加粗**、*斜体*、`代码`、>引用 等Markdown标记）
    - 可以使用适当的 emoji 来增强表达
    - 回复简洁友好，直接回答用户问题"""

        # 构建 messages
        example_args_1 = json.dumps({"prompt": "帮我生成一张中国美女图"}, ensure_ascii=False)
        example_args_2 = json.dumps({"workflow_name": "FaceFix"}, ensure_ascii=False)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！我是ComfyUI图像处理助手。我可以帮你处理图片或生成新图片。有什么可以帮你的吗？"},
            {"role": "user", "content": "帮我生成一张中国美女图"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "text_to_image", "arguments": example_args_1}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"status": "success", "message": "已提交文生图任务"}, ensure_ascii=False)},
            {"role": "user", "content": "切换到面部重绘"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "switch_workflow", "arguments": example_args_2}}]},
            {"role": "tool", "tool_call_id": "call_2", "content": json.dumps({"status": "success", "message": "已切换到FaceFix模式"}, ensure_ascii=False)},
            {"role": "user", "content": user_message}
        ]

        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "tools": functions,
            "tool_choice": "auto",
            "temperature": 0.3
        }

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AppConfig.DEEPSEEK_API_KEY}"
            }

            print(f"  DeepSeek请求参数: tool_choice=auto, temperature=0.3")

            response = requests.post(
                AppConfig.DEEPSEEK_API_URL,
                json=payload,
                headers=headers,
                timeout=30
            )

            result = response.json()

            if "choices" not in result or len(result["choices"]) == 0:
                print(f"DeepSeek API返回格式错误: {result}")
                return None

            choice = result["choices"][0]
            message = choice.get("message", {})

            # 检查是否有tool_calls
            if "tool_calls" in message and message["tool_calls"]:
                tool_call = message["tool_calls"][0]
                function = tool_call.get("function", {})
                function_name = function.get("name")
                function_args_str = function.get("arguments", "{}")

                try:
                    function_args = json.loads(function_args_str)
                    print(f"DeepSeek调用function: {function_name} with args: {function_args}")
                    return {
                        "type": "function_call",
                        "name": function_name,
                        "arguments": function_args
                    }
                except json.JSONDecodeError as e:
                    print(f"解析function参数失败: {e}")
                    return None

            # 没有tool_calls，返回文本回复
            content = message.get("content", "")
            if content:
                print(f"DeepSeek文本回复: {content[:100]}...")
                return {
                    "type": "text",
                    "content": content
                }

            return None

        except Exception as e:
            print(f"调用DeepSeek API失败: {e}")
            return None


# ============================================================================
# 用户状态管理
# ============================================================================

user_workflows = {}  # {chat_id: workflow_name}
edit_prompt_states = {}  # {chat_id: {"image_path": str, "task_number": int, "workflow_name": str, "points_cost": int}}


def get_user_workflow(chat_id: str) -> str:
    """获取用户当前的工作流设置"""
    return user_workflows.get(chat_id, AppConfig.DEFAULT_WORKFLOW)


def set_user_workflow(chat_id: str, workflow_name: str) -> bool:
    """设置用户的工作流"""
    if workflow_name in AppConfig.WORKFLOW_CONFIGS:
        user_workflows[chat_id] = workflow_name
        print(f"  用户 {chat_id} 切换到工作流: {workflow_name}")
        return True
    return False


# ============================================================================
# 主消息处理器
# ============================================================================

class MessageHandler:
    """消息处理器"""

    def __init__(self, client):
        self.client = client
        self.messenger = FeishuMessenger(client)
        self.text_request_cache = {}  # 文本请求缓存
        self.text_request_lock = threading.Lock()

    def _check_text_duplicate(self, chat_id: str, text: str) -> bool:
        """检查文本消息是否重复（5秒窗口）"""
        text_request_key = f"{chat_id}_{text}"
        current_time = time.time()

        with self.text_request_lock:
            if text_request_key in self.text_request_cache:
                last_request_time = self.text_request_cache[text_request_key]
                if current_time - last_request_time < 5:
                    print(f"  [去重] 检测到5秒内的重复文本请求，跳过")
                    return True

            # 更新缓存
            self.text_request_cache[text_request_key] = current_time

            # 清理过期的缓存条目（超过30秒的）
            expired_keys = [k for k, v in self.text_request_cache.items()
                           if current_time - v > 30]
            for k in expired_keys:
                del self.text_request_cache[k]

        return False

    def handle(self, data):
        """处理消息事件"""
        try:
            print(f"\n========== 收到新消息 ==========")
            print(f"事件类型: {type(data).__name__}")

            # 解析消息数据
            if hasattr(data, 'message'):
                message = data.message
            elif isinstance(data, dict):
                message = data.get("message", {})
            else:
                print("  数据格式无法识别")
                return

            # 提取消息信息
            chat_id = getattr(message, 'chat_id', '')
            content = getattr(message, 'content', '')
            message_id = getattr(message, 'message_id', '')
            message_type = getattr(message, 'msg_type', '') or getattr(message, 'message_type', '')

            print(f"Chat ID: {chat_id}")
            print(f"Message ID: {message_id}")
            print(f"Message Type: {message_type}")
            print(f"Content: {str(content)[:200]}...")

            # 如果message_id为空，生成唯一标识
            if not message_id:
                message_id = message_deduplicator.generate_message_id(chat_id, content)
                print(f"  使用生成的message_id: {message_id}")

            # 检查是否已处理过该消息
            if message_deduplicator.is_duplicate(message_id):
                print(f"  [去重] 消息已处理过，跳过")
                return

            # 如果 message_type 为空，尝试从 content 推断
            if not message_type and content:
                try:
                    content_json = json.loads(content)
                    if 'text' in content_json:
                        message_type = 'text'
                    elif 'image_key' in content_json:
                        message_type = 'image'
                    print(f"  推断的消息类型: {message_type}")
                except:
                    pass

            # 分发处理
            if message_type == "text":
                self._handle_text_message(chat_id, content)
            elif message_type == "image":
                self._handle_image_message(chat_id, content, message_id)
            else:
                print(f"  不支持的消息类型: {message_type}")

            print(f"====================================\n")

        except Exception as e:
            print(f"  处理消息事件时出错: {e}")
            import traceback
            traceback.print_exc()

    def _handle_text_message(self, chat_id: str, content: str):
        """处理文本消息"""
        try:
            content_json = json.loads(content)
            text = content_json.get("text", "").strip()

            print(f"  收到文本消息: {text}")

            # 处理命令
            if text == "/start":
                self._send_welcome_message(chat_id)

            elif text == "/help":
                self._send_help_message(chat_id)

            elif text == "/queue" or text == "/status":
                queue_status = task_queue.format_user_status(chat_id)
                self.messenger.send_message(chat_id, json.dumps({"text": queue_status}), "text")
                print(f"  已发送队列状态")

            elif text == "/cancel":
                # 取消图像编辑状态
                if chat_id in edit_prompt_states:
                    state = edit_prompt_states[chat_id]
                    task_number = state.get("task_number")
                    image_path = state.get("image_path")

                    # 从队列中移除任务
                    if task_number and task_number in task_queue.tasks:
                        task_queue.remove_task(task_number)

                    # 删除临时图片
                    if image_path and os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                            print(f"  已删除临时图片: {image_path}")
                        except Exception as e:
                            print(f"  删除临时图片失败: {e}")

                    # 清除编辑状态
                    del edit_prompt_states[chat_id]

                    reply_content = json.dumps({
                        "text": "✅ 已取消当前编辑任务\n\n您可以重新发送图片开始新的编辑。"
                    })
                    self.messenger.send_message(chat_id, reply_content, "text")
                    print(f"  已取消用户 {chat_id} 的编辑状态")
                else:
                    reply_content = json.dumps({
                        "text": "ℹ️ 当前没有需要取消的任务"
                    })
                    self.messenger.send_message(chat_id, reply_content, "text")
                    print(f"  用户 {chat_id} 没有待取消的任务")

            # 检查用户是否在等待输入编辑提示词
            elif chat_id in edit_prompt_states:
                state = edit_prompt_states[chat_id]
                print(f"  检测到用户正在等待输入编辑提示词")
                print(f"  状态: {state}")

                # 清除编辑状态
                del edit_prompt_states[chat_id]

                # 提取提示词
                prompt_text = text
                image_path = state["image_path"]
                task_number = state["task_number"]
                workflow_name = state["workflow_name"]
                points_cost = state["points_cost"]

                print(f"  使用提示词处理图像: {prompt_text[:50]}...")

                # 发送处理中消息
                reply_content = json.dumps({
                    "text": f"🎨 正在使用提示词处理图片...\n\n提示词: {prompt_text}\n\n请稍候..."
                })
                self.messenger.send_message(chat_id, reply_content, "text")

                # 使用ComfyUI处理图片（带提示词）
                print(f"  正在处理图片...")
                output_image_path = ImageProcessor.process_image_with_prompt(
                    image_path,
                    workflow_name,
                    prompt_text
                )

                if output_image_path:
                    # 上传并发送原图和处理后的图片
                    print(f"  正在发送图片...")

                    # 先发送文字说明
                    caption_content = json.dumps({"text": f"✅ 处理完成！\n\n任务序号: #{task_number}\n提示词: {prompt_text}\n\n📎 原图："})
                    self.messenger.send_message(chat_id, caption_content, "text")

                    # 上传并发送原图
                    original_image_key = FeishuAPI.upload_image(image_path)
                    if original_image_key:
                        self.messenger.send_image_message(chat_id, original_image_key)

                    # 发送处理后的图片说明
                    processed_caption = json.dumps({"text": "🎨 处理后："})
                    self.messenger.send_message(chat_id, processed_caption, "text")

                    # 上传并发送处理后的图片
                    self.messenger.upload_and_send_image(chat_id, output_image_path)

                    # 清理临时文件
                    try:
                        os.remove(image_path)
                    except:
                        pass

                    # 任务完成，从队列移除
                    task_queue.remove_task(task_number)
                else:
                    reply_content = json.dumps({
                        "text": "❌ 图片处理失败\n\n请检查ComfyUI是否正常运行。"
                    })
                    self.messenger.send_message(chat_id, reply_content, "text")
                    task_queue.remove_task(task_number)

                return

            # 处理命令
            if text == "/start":
                self._send_welcome_message(chat_id)

            elif text == "/help":
                self._send_help_message(chat_id)

            elif text == "/queue" or text == "/status":
                queue_status = task_queue.format_user_status(chat_id)
                self.messenger.send_message(chat_id, json.dumps({"text": queue_status}), "text")
                print(f"  已发送队列状态")

            elif text == "/cancel":
                # 取消图像编辑状态
                if chat_id in edit_prompt_states:
                    state = edit_prompt_states[chat_id]
                    task_number = state.get("task_number")
                    image_path = state.get("image_path")

                    # 从队列中移除任务
                    if task_number and task_number in task_queue.tasks:
                        task_queue.remove_task(task_number)

                    # 删除临时图片
                    if image_path and os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                            print(f"  已删除临时图片: {image_path}")
                        except Exception as e:
                            print(f"  删除临时图片失败: {e}")

                    # 清除编辑状态
                    del edit_prompt_states[chat_id]

                    reply_content = json.dumps({
                        "text": "✅ 已取消当前编辑任务\n\n您可以重新发送图片开始新的编辑。"
                    })
                    self.messenger.send_message(chat_id, reply_content, "text")
                    print(f"  已取消用户 {chat_id} 的编辑状态")
                else:
                    reply_content = json.dumps({
                        "text": "ℹ️ 当前没有需要取消的任务"
                    })
                    self.messenger.send_message(chat_id, reply_content, "text")
                    print(f"  用户 {chat_id} 没有待取消的任务")

            elif text in ["/FaceFix", "/BackgroundRemove", "/Qwen_remove", "/BoobsFix", "/Qwen_edit"]:
                workflow_name = text.replace("/", "")
                if set_user_workflow(chat_id, workflow_name):
                    reply_content = json.dumps({
                        "text": f"✅ 已切换到工作流: {workflow_name}\n\n现在发送图片给我，我会使用{workflow_name}进行处理。"
                    })
                    self.messenger.send_message(chat_id, reply_content, "text")
                else:
                    reply_content = json.dumps({
                        "text": f"❌ 未知的工作流: {text}"
                    })
                    self.messenger.send_message(chat_id, reply_content, "text")

            else:
                # 普通文本消息，尝试使用自然语言处理
                print(f"  尝试使用自然语言处理...")

                # 检查文本重复
                if self._check_text_duplicate(chat_id, text):
                    return

                # 调用DeepSeek进行function calling
                deepseek_result = NLProcessor.call_deepseek(text, chat_id)

                if deepseek_result:
                    result_type = deepseek_result.get("type")

                    if result_type == "function_call":
                        self._handle_function_call(chat_id, deepseek_result)
                    elif result_type == "text":
                        reply_content = json.dumps({
                            "text": deepseek_result.get("content", "")
                        })
                        self.messenger.send_message(chat_id, reply_content, "text")
                        print(f"  已回复AI消息")
                    else:
                        reply_content = json.dumps({
                            "text": f"收到消息: {text}\n\n发送 /help 查看使用说明"
                        })
                        self.messenger.send_message(chat_id, reply_content, "text")
                else:
                    reply_content = json.dumps({
                        "text": f"收到消息: {text}\n\n发送 /help 查看使用说明"
                    })
                    self.messenger.send_message(chat_id, reply_content, "text")
                    print(f"  已回复: {text[:30]}...")

        except json.JSONDecodeError:
            reply_content = json.dumps({"text": "消息格式错误"})
            self.messenger.send_message(chat_id, reply_content, "text")

    def _handle_image_message(self, chat_id: str, content: str, message_id: str):
        """处理图片消息"""
        try:
            content_json = json.loads(content)
            image_key = content_json.get("image_key", "")

            print(f"  收到图片消息，image_key: {image_key}")

            # 检查用户是否已经在等待输入编辑提示词
            if chat_id in edit_prompt_states:
                print(f"  [警告] 用户已在等待输入编辑提示词，忽略新图片")
                reply_content = json.dumps({
                    "text": "⚠️ 您已发送了一张图片，正在等待您输入编辑提示词。\n\n请输入提示词（如：给人物加上墨镜）来描述您想要的修改。\n\n发送 /cancel 可取消当前任务重新开始。"
                })
                self.messenger.send_message(chat_id, reply_content, "text")
                return

            # 检查是否为图像编辑模式
            workflow_name = get_user_workflow(chat_id)
            if workflow_name == "Qwen_edit":
                # 图像编辑模式：先下载图片，然后询问编辑提示词
                print(f"  图像编辑模式，下载图片...")
                temp_image_path = FeishuAPI.download_image(image_key, message_id)

                if temp_image_path:
                    print(f"  图片已下载: {temp_image_path}")
                    print(f"  已保存状态到 edit_prompt_states，等待用户输入提示词")

                    # 获取工作流配置
                    workflow_config = AppConfig.WORKFLOW_CONFIGS.get(workflow_name, {})
                    points_cost = workflow_config.get("points_cost", 2)

                    # 生成任务序号并加入队列
                    task_number = task_queue.add_task(chat_id)
                    task_info = task_queue.get_task_info(task_number)

                    if task_info:
                        position, waiting, total = task_info
                        # 发送等待提示词的消息
                        reply_content = json.dumps({
                            "text": f"📝 已收到图片！\n\n任务序号: #{task_number}\n队列位置: {position}/{total}\n前面等待: {waiting} 个任务\n\n请输入编辑提示词（prompt），描述你想要对图片进行的修改...\n\n发送 /cancel 可取消当前任务"
                        })
                        self.messenger.send_message(chat_id, reply_content, "text")

                    # 将状态保存到edit_prompt_states
                    edit_prompt_states[chat_id] = {
                        "image_path": temp_image_path,
                        "task_number": task_number,
                        "workflow_name": workflow_name,
                        "points_cost": points_cost
                    }
                    print(f"  edit_prompt_states 当前状态: {edit_prompt_states}")
                    print(f"  图像编辑模式处理完成，等待用户输入提示词")
                else:
                    reply_content = json.dumps({
                        "text": "❌ 下载图片失败"
                    })
                    self.messenger.send_message(chat_id, reply_content, "text")
                return

            # 生成任务序号并加入队列
            task_number = task_queue.add_task(chat_id)
            task_info = task_queue.get_task_info(task_number)

            if task_info:
                position, waiting, total = task_info
                # 发送处理中消息
                reply_content = json.dumps({
                    "text": f"📸 收到图片！\n\n任务序号: #{task_number}\n队列位置: {position}/{total}\n前面等待: {waiting} 个任务\n\n正在使用 {get_user_workflow(chat_id)} 处理...\n请稍候..."
                })
                self.messenger.send_message(chat_id, reply_content, "text")

            # 下载图片
            print(f"  正在下载图片...")
            temp_image_path = FeishuAPI.download_image(image_key, message_id)

            if temp_image_path:
                # 使用ComfyUI处理图片
                print(f"  正在处理图片...")
                output_image_path = ImageProcessor.process_image(
                    temp_image_path,
                    get_user_workflow(chat_id)
                )

                if output_image_path:
                    # 上传并发送原图和处理后的图片
                    print(f"  正在发送图片...")

                    # 先发送文字说明
                    caption_content = json.dumps({"text": f"✅ 处理完成！\n\n任务序号: #{task_number}\n\n📎 原图："})
                    self.messenger.send_message(chat_id, caption_content, "text")

                    # 上传并发送原图
                    original_image_key = FeishuAPI.upload_image(temp_image_path)
                    if original_image_key:
                        self.messenger.send_image_message(chat_id, original_image_key)

                    # 发送处理后的图片说明
                    processed_caption = json.dumps({"text": "🎨 处理后："})
                    self.messenger.send_message(chat_id, processed_caption, "text")

                    # 上传并发送处理后的图片
                    self.messenger.upload_and_send_image(chat_id, output_image_path)

                    # 清理临时文件
                    try:
                        os.remove(temp_image_path)
                    except:
                        pass

                    # 任务完成，从队列移除
                    task_queue.remove_task(task_number)
                else:
                    reply_content = json.dumps({
                        "text": "❌ 图片处理失败\n\n请检查ComfyUI是否正常运行。"
                    })
                    self.messenger.send_message(chat_id, reply_content, "text")
                    task_queue.remove_task(task_number)
            else:
                reply_content = json.dumps({
                    "text": "❌ 下载图片失败"
                })
                self.messenger.send_message(chat_id, reply_content, "text")
                task_queue.remove_task(task_number)

        except json.JSONDecodeError:
            reply_content = json.dumps({
                "text": "❌ 图片消息格式错误"
            })
            self.messenger.send_message(chat_id, reply_content, "text")

    def _handle_function_call(self, chat_id: str, function_call: Dict):
        """处理function调用结果"""
        function_name = function_call["name"]
        arguments = function_call["arguments"]

        if function_name == "switch_workflow":
            workflow_name = arguments.get("workflow_name")
            if workflow_name:
                print(f"自然语言切换工作流: {workflow_name}")
                if set_user_workflow(chat_id, workflow_name):
                    # 映射到中文名称用于显示
                    workflow_cn_names = {
                        "FaceFix": "面部重绘",
                        "BackgroundRemove": "去除背景杂物",
                        "Qwen_remove": "服装移除",
                        "BoobsFix": "胸部重绘",
                        "Qwen_edit": "图像编辑"
                    }
                    cn_name = workflow_cn_names.get(workflow_name, workflow_name)
                    reply_content = json.dumps({
                        "text": f"✅ 已切换到「{cn_name}」处理方式"
                    })
                    self.messenger.send_message(chat_id, reply_content, "text")
                else:
                    reply_content = json.dumps({
                        "text": "❌ 切换失败"
                    })
                    self.messenger.send_message(chat_id, reply_content, "text")

        elif function_name == "text_to_image":
            prompt = arguments.get("prompt", "")
            if prompt:
                print(f"文生图请求: {prompt[:100]}...")

                # 发送处理中消息
                reply_content = json.dumps({
                    "text": f"🎨 正在生成图片...\n\n提示词: {prompt}\n\n请稍候..."
                })
                self.messenger.send_message(chat_id, reply_content, "text")

                # 使用ComfyUI生成图片
                output_image_path = ImageProcessor.process_text_to_image(prompt)

                if output_image_path:
                    # 上传并发送生成的图片
                    print(f"  正在发送生成的图片...")
                    self.messenger.upload_and_send_image(chat_id, output_image_path, "✅ 图片生成完成！")
                else:
                    reply_content = json.dumps({
                        "text": "❌ 图片生成失败\n\n请检查ComfyUI是否正常运行。"
                    })
                    self.messenger.send_message(chat_id, reply_content, "text")

    def _send_welcome_message(self, chat_id: str):
        """发送欢迎消息"""
        reply_content = json.dumps({
            "text": """[OK] 飞书机器人已启动！

📸 功能说明：
1. 发送图片进行处理
2. 使用自然语言生成图片（如：帮我生成一张美女图）
3. 使用命令切换工作流

🔧 工作流命令：
/FaceFix - 面部重绘
/BackgroundRemove - 去除背景杂物
/Qwen_remove - 服装移除（默认）
/BoobsFix - 胸部重绘
/Qwen_edit - 图像编辑

📊 任务队列：
/queue - 查看当前任务队列状态
/status - 查看任务队列状态（同/queue）

🎨 文生图示例：
"帮我生成一张美女图"
"画一个风景"
"创建一个动漫角色"

💡 发送 /help 查看更多帮助"""
        })
        self.messenger.send_message(chat_id, reply_content, "text")
        print("  已发送欢迎消息")

    def _send_help_message(self, chat_id: str):
        """发送帮助消息"""
        help_text = """[使用说明]

📱 基础命令：
/start - 启动机器人
/help - 查看帮助
/cancel - 取消当前编辑任务

🎨 工作流命令：
/FaceFix - 面部重绘
/BackgroundRemove - 去除背景杂物
/Qwen_remove - 服装移除（默认）
/BoobsFix - 胸部重绘
/Qwen_edit - 图像编辑

📊 任务队列：
/queue - 查看当前任务队列状态
/status - 查看任务队列状态（同/queue）

🌟 文生图功能：
直接发送描述即可生成图片，例如：
|- 帮我生成一张美女图
|- 画一个风景
|- 创建一个动漫角色
|- 生成一张中国网红图

💡 使用方法：
1. 图片处理：选择工作流 → 发送图片 → 等待处理结果
2. 文生图：发送图片描述 → 自动生成
3. 切换模式：发送命令或自然语言

📝 图像编辑模式：
1. 切换到 /Qwen_edit 模式
2. 发送待编辑的图片
3. 输入编辑提示词（如：给人物加上墨镜）
4. 等待处理结果
5. 发送 /cancel 可取消当前任务重新开始"""
        reply_content = json.dumps({"text": help_text})
        self.messenger.send_message(chat_id, reply_content, "text")
        print("  已发送帮助消息")


# ============================================================================
# 主程序
# ============================================================================

def create_client():
    """创建飞书客户端"""
    return lark.Client.builder() \
        .app_id(AppConfig.FEISHU_APP_ID) \
        .app_secret(AppConfig.FEISHU_APP_SECRET) \
        .build()


def start_long_connection():
    """使用飞书SDK启动长连接监听"""
    print("=" * 60)
    print("飞书机器人 V8 - 长连接监听模式（重构版本）")
    print("=" * 60)
    print(f"应用ID: {AppConfig.FEISHU_APP_ID}")
    print()

    # 创建客户端
    client = create_client()
    print(f"[OK] 飞书客户端创建成功")

    # 创建消息处理器
    message_handler = MessageHandler(client)
    print(f"[OK] 消息处理器创建成功")

    # 创建事件处理器
    print(f"\n正在创建事件处理器...")

    builder = EventDispatcherHandlerBuilder(
        encrypt_key="",
        verification_token=""
    )

    # 注册消息接收事件处理
    builder.register_p2_im_message_receive_v1(
        lambda event: message_handler.handle(event.event)
    )

    event_dispatcher = builder.build()
    print(f"[OK] 事件处理器创建成功")

    # 创建长连接客户端
    print(f"\n正在创建长连接客户端...")
    ws_client = Client(
        app_id=AppConfig.FEISHU_APP_ID,
        app_secret=AppConfig.FEISHU_APP_SECRET,
        log_level=LogLevel.DEBUG,
        event_handler=event_dispatcher,
        domain="https://open.feishu.cn",
        auto_reconnect=True
    )
    print(f"[OK] 长连接客户端创建成功")

    # 启动长连接
    print(f"\n正在启动长连接监听...")
    print(f"[INFO] 机器人已就绪，等待消息...")
    print(f"[提示] 在飞书中发送 /start 测试机器人")
    print(f"[提示] 在飞书开放平台确保已选择「使用长连接接收事件」")
    print()

    try:
        ws_client.start()

    except KeyboardInterrupt:
        print("\n[INFO] 正在停止长连接...")
        print("[INFO] 长连接已停止")

    except Exception as e:
        print(f"\n[ERROR] 长连接运行时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    # 检查飞书SDK
    if not SDK_AVAILABLE:
        print("[ERROR] 飞书SDK未安装")
        print("请运行: pip install lark-oapi")
        return

    # 检查配置
    if not AppConfig.TEXT_TO_IMAGE_CONFIG:
        print("[WARNING] 文生图配置未找到")
    else:
        print(f"[OK] 文生图配置已加载: {AppConfig.TEXT_TO_IMAGE_CONFIG.get('workflow', 'unknown')}")

    if not AppConfig.DEEPSEEK_API_KEY:
        print("[WARNING] 未配置DeepSeek API密钥,自然语言功能不可用")

    # 检查ComfyUI
    if check_comfyui_server():
        print(f"[OK] ComfyUI服务正常: {AppConfig.COMFYUI_API_URL}")
    else:
        print(f"[WARNING] ComfyUI服务未启动或无法访问")
        print(f"   正在尝试启动 ComfyUI 服务器...")
        print()

        # 尝试自动启动 ComfyUI
        comfyui_process = start_comfyui_server()
        if comfyui_process:
            print(f"[OK] ComfyUI 服务器已自动启动")
        else:
            print(f"[ERROR] ComfyUI 服务器启动失败")
            print(f"   请确保ComfyUI运行在: {AppConfig.COMFYUI_API_URL}")
            print(f"   图片处理功能将不可用")
            print()

    # 启动长连接监听
    try:
        start_long_connection()
    finally:
        # 程序退出时停止 ComfyUI（如果是我们启动的）
        if comfyui_process:
            print("\n[INFO] 正在停止 ComfyUI 服务器...")
            stop_comfyui_server()


if __name__ == "__main__":
    main()
