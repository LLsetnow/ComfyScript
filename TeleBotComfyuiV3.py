import json
from urllib import request, error
import time
import os
import shutil
import random
import requests
from typing import Optional, List, Dict
import threading
import io


# 加载配置文件
def load_config():
    """从config.json5加载配置"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json5")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用 json5 库解析支持注释的配置文件
    try:
        import json5
        return json5.loads(content)
    except ImportError:
        # 如果没有安装 json5，尝试去掉注释后用标准 json 解析
        # 简单去除 // 单行注释
        lines = []
        for line in content.split('\n'):
            # 找到 // 注释的开始位置（忽略字符串中的 //）
            in_string = False
            string_char = None
            comment_pos = -1
            for i, char in enumerate(line):
                if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                elif not in_string and char == '/' and i+1 < len(line) and line[i+1] == '/':
                    comment_pos = i
                    break
            if comment_pos >= 0:
                lines.append(line[:comment_pos].rstrip())
            else:
                lines.append(line)
        return json.loads('\n'.join(lines))

CONFIG = load_config()


class ComfyUIWorkflow:
    """ComfyUI工作流处理类"""

    def __init__(self, seed_id=65, input_image_id=41, output_image_id=181, prompt_node_id=None, workflow='Qwen_remove.json'):
        """
        初始化工作流处理器
        :param seed_id: 种子节点ID
        :param input_image_id: 输入图像节点ID
        :param output_image_id: 输出图像节点ID
        :param prompt_node_id: 提示词节点ID（用于图像编辑）
        :param workflow: 工作流JSON文件名
        """
        self.seed_id = str(seed_id)
        self.input_image_id = str(input_image_id)
        self.output_image_id = str(output_image_id)
        self.prompt_node_id = str(prompt_node_id) if prompt_node_id else None
        self.workflow_file = workflow
        self.original_workflow = None

    def load_workflow(self):
        """加载工作流JSON文件"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workflow_path = os.path.join(script_dir, self.workflow_file)

        if not os.path.exists(workflow_path):
            raise FileNotFoundError(f"找不到工作流文件: {workflow_path}")

        with open(workflow_path, 'r', encoding='utf-8') as f:
            self.original_workflow = json.load(f)

        return True

    def set_seed(self, seed_value):
        """
        设置随机种子
        :param seed_value: 种子值
        """
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")

        self.original_workflow[self.seed_id]["inputs"]["seed"] = int(seed_value)

    def set_prompt(self, prompt_value):
        """
        设置提示词（用于图像编辑）
        :param prompt_value: 提示词内容
        """
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")
        if not self.prompt_node_id:
            raise RuntimeError("当前工作流不支持设置提示词")

        self.original_workflow[self.prompt_node_id]["inputs"]["prompt"] = str(prompt_value)

    def set_input_image(self, image_filename):
        """
        设置输入图像
        :param image_filename: 图像文件名
        """
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")

        self.original_workflow[self.input_image_id]["inputs"]["image"] = image_filename

    def set_output_prefix(self, output_prefix):
        """
        设置输出文件前缀
        :param output_prefix: 输出文件前缀
        """
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")

        self.original_workflow[self.output_image_id]["inputs"]["filename_prefix"] = output_prefix

    def get_workflow(self):
        """获取当前工作流配置"""
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")

        return self.original_workflow

    def create_workflow_copy(self):
        """
        创建工作流的深拷贝
        :return: 工作流的副本
        """
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")

        return json.loads(json.dumps(self.original_workflow))


class UserDatabase:
    """用户数据库管理类"""

    def __init__(self, db_file="user_database.json"):
        """
        初始化数据库
        :param db_file: 数据库文件路径
        """
        self.db_file = db_file
        self.lock = threading.Lock()  # 线程锁
        self.load()

    def load(self):
        """加载数据库"""
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = {
                "users": {},
                "keys": {}
            }
            self.save()

    def save(self):
        """保存数据库"""
        # 注意：调用者需要先获取锁
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def add_user(self, user_id: int, username: str = "Unknown") -> bool:
        """
        添加新用户
        :param user_id: 用户ID
        :param username: 用户名
        :return: True表示新添加的用户，False表示已存在的用户
        """
        with self.lock:
            user_id_str = str(user_id)
            if user_id_str not in self.data["users"]:
                self.data["users"][user_id_str] = {
                    "id": user_id,
                    "username": username,
                    "role": "普通用户",  # 默认为普通用户
                    "points": INITIAL_POINTS,  # 新用户初始积分
                    "task_numbers": []  # 当前任务序号列表
                }
                self.save()
                return True  # 新用户
            return False  # 用户已存在

    def get_user(self, user_id: int) -> Optional[Dict]:
        """获取用户信息"""
        user_id_str = str(user_id)
        return self.data["users"].get(user_id_str)

    def update_user_role(self, user_id: int, role: str):
        """更新用户身份"""
        with self.lock:
            user = self.get_user(user_id)
            if user:
                user_id_str = str(user_id)
                self.data["users"][user_id_str]["role"] = role
                self.save()
                return True
            return False

    def add_points(self, user_id: int, points: int):
        """为用户增加积分"""
        with self.lock:
            user = self.get_user(user_id)
            if user:
                user_id_str = str(user_id)
                self.data["users"][user_id_str]["points"] += points
                self.save()
                return True
            return False

    def consume_points(self, user_id: int, points: int) -> bool:
        """消耗用户积分"""
        with self.lock:
            user = self.get_user(user_id)
            if user and user["points"] >= points:
                user_id_str = str(user_id)
                self.data["users"][user_id_str]["points"] -= points
                self.save()
                return True
        return False

    def get_user_points(self, user_id: int) -> int:
        """获取用户积分"""
        user = self.get_user(user_id)
        return user["points"] if user else 0

    def generate_keys(self, count: int = 10) -> List[str]:
        """生成密钥"""
        keys = []
        for _ in range(count):
            key = self._generate_key()
            self.data["keys"][key] = {
                "used": False,
                "used_by": None,
                "used_time": None
            }
            keys.append(key)
        self.save()
        return keys

    def _generate_key(self) -> str:
        """生成单个密钥（16位字母数字混合）"""
        chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # 去除容易混淆的字符
        return ''.join(random.choice(chars) for _ in range(16))

    def validate_key(self, key: str) -> bool:
        """验证密钥是否有效且未使用"""
        key_data = self.data["keys"].get(key)
        return key_data is not None and not key_data["used"]

    def use_key(self, key: str, user_id: int) -> bool:
        """使用密钥"""
        with self.lock:
            if not self.validate_key(key):
                return False

            # 标记密钥为已使用
            self.data["keys"][key]["used"] = True
            self.data["keys"][key]["used_by"] = user_id
            self.data["keys"][key]["used_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # 为用户增加积分（需要独立锁）
        self.add_points(user_id, KEY_REWARD_POINTS)

        # 将用户身份修改为会员，但管理员保持管理员身份（需要独立锁）
        user = self.get_user(user_id)
        if user and user["role"] != "管理员":
            self.update_user_role(user_id, "会员")

        return True

    def get_key_status(self, key: str) -> Optional[Dict]:
        """获取密钥状态"""
        return self.data["keys"].get(key)

    def add_task_number(self, user_id: int, task_number: int):
        """为用户添加任务序号"""
        with self.lock:
            user = self.get_user(user_id)
            if user:
                user_id_str = str(user_id)
                self.data["users"][user_id_str]["task_numbers"].append(task_number)
                self.save()
                return True
        return False


# 从配置文件读取配置
USE_PROXY = CONFIG["proxy"]["use_proxy"]
PROXY_SETTINGS = {
    "http": CONFIG["proxy"]["http"],
    "https": CONFIG["proxy"]["https"]
}

# 积分配置
INITIAL_POINTS = CONFIG["points"]["initial_points"]
KEY_REWARD_POINTS = CONFIG["points"]["key_reward_points"]

# 根据设置决定是否使用代理
if USE_PROXY:
    # 设置环境变量代理
    os.environ['HTTP_PROXY'] = PROXY_SETTINGS["http"]
    os.environ['HTTPS_PROXY'] = PROXY_SETTINGS["https"]
else:
    # 清除环境变量代理
    if 'HTTP_PROXY' in os.environ:
        del os.environ['HTTP_PROXY']
    if 'HTTPS_PROXY' in os.environ:
        del os.environ['HTTPS_PROXY']

# Telegram配置
TELEGRAM_BOT_TOKEN = CONFIG["telegram"]["bot_token"]
AUTHORIZED_USER_IDS = CONFIG["telegram"]["authorized_user_ids"]

# ComfyUI配置
COMFYUI_FOLDER = CONFIG["comfyUI"]["folder"]
COMFYUI_INPUT_FOLDER = os.path.join(COMFYUI_FOLDER, "input")
COMFYUI_OUTPUT_FOLDER = os.path.join(COMFYUI_FOLDER, "output")

# 工作流配置
WORKFLOW_CONFIGS = CONFIG["workflows"]

# 默认工作流配置
DEFAULT_WORKFLOW = CONFIG["default_workflow"]

# DeepSeek 配置
DEEPSEEK_API_KEY = CONFIG["deepseek"]["api_key"]
DEEPSEEK_API_URL = CONFIG["deepseek"]["api_url"]

# 文生图配置
TEXT_TO_IMAGE_CONFIG = CONFIG["text_to_image"]

# 全局任务序号（所有用户共享）
global_task_counter = 0
task_counter_lock = threading.Lock()

# 任务队列（用于排队管理）
task_queue = []  # 任务序号队列
task_queue_lock = threading.Lock()  # 队列锁

# 用户自定义工作流配置
user_workflows = {}  # {chat_id: workflow_name}

# 密钥兑换状态（记录用户是否在兑换密钥流程中）
key_exchange_states = {}  # {chat_id: "waiting_for_key"}

# 图像编辑状态（记录用户是否在输入prompt流程中）
edit_prompt_states = {}  # {chat_id: {"image_path": str, "user_id": int, "task_number": int}}

# 文生图状态
text_to_image_states = {}  # {chat_id: {"user_id": int, "prompt": str, "points_cost": int}}

# 任务队列（存储 {用户id, 任务序号} 的元组）
task_queue = []  # [(user_id, task_number), ...]
task_queue_lock = threading.Lock()  # 队列锁


def get_proxies():
    """根据USE_PROXY设置返回代理配置"""
    return PROXY_SETTINGS if USE_PROXY else None


def generate_random_seed():
    """生成15位随机数种子"""
    return random.randint(10**14, 10**15 - 1)


def add_task_to_queue(user_id: int, task_number: int):
    """
    将任务加入队列
    :param user_id: 用户ID
    :param task_number: 任务序号
    """
    with task_queue_lock:
        task_queue.append((user_id, task_number))
        print(f"任务 {task_number} (用户 {user_id}) 已加入队列，队列长度: {len(task_queue)}")


def remove_task_from_queue(task_number: int):
    """
    从队列中移除任务序号
    :param task_number: 任务序号
    """
    with task_queue_lock:
        # 找到并移除指定任务序号的项
        for i, (uid, tnum) in enumerate(task_queue):
            if tnum == task_number:
                task_queue.pop(i)
                print(f"任务 {task_number} 已从队列移除，队列长度: {len(task_queue)}")
                return


def get_queue_info(user_id: int, task_number: int) -> tuple:
    """
    获取队列信息
    :param user_id: 用户ID
    :param task_number: 当前任务序号
    :return: (当前位置, 前面等待的任务数, 队列总任务数)
    """
    with task_queue_lock:
        # 找到当前任务在队列中的位置
        for i, (uid, tnum) in enumerate(task_queue):
            if tnum == task_number:
                position = i + 1  # 位置从1开始计数
                waiting_count = i  # 前面等待的任务数
                total_count = len(task_queue)  # 队列总任务数
                return (position, waiting_count, total_count)
        
        # 任务不在队列中
        return (0, 0, 0)


def get_user_tasks(user_id: int) -> list:
    """
    获取用户在队列中的所有任务
    :param user_id: 用户ID
    :return: 任务序号列表
    """
    with task_queue_lock:
        user_tasks = [tnum for uid, tnum in task_queue if uid == user_id]
        return user_tasks


def check_comfyui_server(max_attempts=3, check_delay=2):
    """检查ComfyUI服务器是否可访问"""
    for attempt in range(max_attempts):
        try:
            response = request.urlopen("http://127.0.0.1:8188", timeout=3)
            return True
        except error.URLError:
            if attempt < max_attempts - 1:
                time.sleep(check_delay)
            else:
                return False


# ComfyUI 服务器状态监控
comfyui_running = True


def queue_prompt(prompt_workflow, max_retries=3, retry_delay=2):
    """将prompt workflow发送到ComfyUI服务器并排队执行"""
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req = request.Request("http://127.0.0.1:8188/prompt", data=data)

    for attempt in range(max_retries):
        try:
            print(f"    正在提交工作流到 ComfyUI...")
            response = request.urlopen(req, timeout=10)
            result = json.loads(response.read().decode('utf-8'))
            prompt_id = result.get('prompt_id')
            print(f"    工作流已提交，prompt_id: {prompt_id}")
            return prompt_id
        except error.URLError as e:
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


def wait_for_completion(prompt_id, check_interval=2, timeout=120):
    """轮询检查任务完成状态"""
    start_time = time.time()
    check_count = 0

    while time.time() - start_time < timeout:
        check_count += 1
        try:
            req = request.Request(f"http://127.0.0.1:8188/history/{prompt_id}")
            response = request.urlopen(req, timeout=5)
            result = json.loads(response.read().decode('utf-8'))

            if prompt_id in result:
                history_data = result[prompt_id]
                status = history_data.get('status', {}).get('completed', False)
                if status:
                    print(f"    任务已完成 (耗时: {int(time.time() - start_time)}秒, 检查次数: {check_count})")
                    return True

                if history_data.get('status', {}).get('exec_info', None):
                    exec_info = history_data['status'].get('exec_info')
                    if exec_info and 'error' in str(exec_info).lower():
                        print(f"    任务执行出错: {exec_info}")
                        return False

            if check_count % 10 == 0:  # 每20秒打印一次进度
                elapsed = int(time.time() - start_time)
                print(f"    等待任务完成... (已等待 {elapsed}秒, 检查次数: {check_count})")

        except error.HTTPError as e:
            if e.code == 404:
                if check_count <= 5 or check_count % 20 == 0:
                    print(f"    任务尚未开始 (检查次数: {check_count})")
                pass
            else:
                print(f"    HTTP错误: {e.code} - {e}")
        except Exception as e:
            print(f"    检查状态时出错: {e}")

        time.sleep(check_interval)

    elapsed = int(time.time() - start_time)
    print(f"    等待超时 (超过 {timeout} 秒, 总检查次数: {check_count})")
    return False


def save_image_with_unique_name(source_path, target_folder):
    """保存图像文件到指定文件夹，如果文件名重复则使用随机种子重命名"""
    original_filename = os.path.basename(source_path)
    file_ext = os.path.splitext(original_filename)[1]
    image_basename = os.path.splitext(original_filename)[0]

    target_path = os.path.join(target_folder, original_filename)

    if not os.path.exists(target_path):
        shutil.copy2(source_path, target_path)
        return original_filename
    else:
        random_seed = generate_random_seed()
        new_filename = f"{random_seed}{file_ext}"
        target_path = os.path.join(target_folder, new_filename)

        while os.path.exists(target_path):
            random_seed = generate_random_seed()
            new_filename = f"{random_seed}{file_ext}"
            target_path = os.path.join(target_folder, new_filename)

        shutil.copy2(source_path, target_path)
        return new_filename


# Telegram API函数
def send_message(chat_id: str, text: str):
    """发送文本消息到Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text
    }
    try:
        response = requests.post(url, json=data, timeout=10, proxies=get_proxies())
        return response.json()
    except Exception as e:
        print(f"发送消息失败: {e}")
        return None


def send_photo(chat_id: str, photo_path: str, caption: Optional[str] = None, original_path: Optional[str] = None):
    """发送图片到Telegram（支持同时发送原图和处理后的图）"""
    # 如果有原图，使用 sendMediaGroup 发送两张图片
    if original_path and os.path.exists(original_path):
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMediaGroup"
        media = []
        original_bytes = None
        processed_bytes = None

        # 添加原图
        try:
            with open(original_path, 'rb') as f:
                original_bytes = io.BytesIO(f.read())
            media.append({
                "type": "photo",
                "media": f"attach://original",
                "caption": f"📸 原图\n{caption}" if caption else "📸 原图"
            })
        except Exception as e:
            print(f"读取原图失败: {e}")
            original_path = None

        # 添加处理后的图
        try:
            with open(photo_path, 'rb') as f:
                processed_bytes = io.BytesIO(f.read())
            media.append({
                "type": "photo",
                "media": f"attach://processed",
                "caption": "🖼️ 处理结果"
            })
        except Exception as e:
            print(f"读取处理后图片失败: {e}")
            # 如果处理后的图失败，只发送原图
            if original_path:
                return send_photo(chat_id, original_path, caption, None)

        # 只有当两张图片都读取成功时才发送媒体组
        if original_bytes and processed_bytes:
            data = {"chat_id": chat_id, "media": json.dumps(media)}
            files = {
                "original": ("original.jpg", original_bytes, "image/jpeg"),
                "processed": ("processed.jpg", processed_bytes, "image/jpeg")
            }

            try:
                response = requests.post(url, data=data, files=files, timeout=30, proxies=get_proxies())
                return response.json()
            except Exception as e:
                print(f"发送媒体组失败: {e}")
                return None

    # 没有原图，使用原来的 sendPhoto 方法
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption

    try:
        with open(photo_path, 'rb') as photo_file:
            files = {"photo": photo_file}
            response = requests.post(url, data=data, files=files, timeout=30, proxies=get_proxies())
        return response.json()
    except Exception as e:
        print(f"发送图片失败: {e}")
        return None


def download_telegram_photo(file_id: str):
    """从Telegram下载图片并返回保存路径"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile"
        response = requests.get(url, params={"file_id": file_id}, timeout=10, proxies=get_proxies())
        file_info = response.json()

        if not file_info.get("ok"):
            print(f"获取文件信息失败: {file_info}")
            return None

        file_path = file_info["result"]["file_path"]
        download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"

        response = requests.get(download_url, timeout=30, proxies=get_proxies())

        original_filename = os.path.basename(file_path)
        temp_path = os.path.join(COMFYUI_INPUT_FOLDER, f"temp_{original_filename}")

        with open(temp_path, 'wb') as f:
            f.write(response.content)

        return temp_path

    except Exception as e:
        print(f"下载Telegram图片失败: {e}")
        return None


def process_text_to_image(chat_id: str, user_id: int, prompt: str, db: UserDatabase, task_number: int):
    """
    处理文生图请求
    :param chat_id: Telegram聊天ID
    :param user_id: 用户ID
    :param prompt: 图像生成提示词
    :param db: 数据库实例
    :param task_number: 任务序号
    """
    if not comfyui_running:
        send_message(chat_id, "❌ ComfyUI 服务器未运行，无法生成图片")
        return

    # 获取用户信息
    user_info = None
    try:
        user_info = db.get_user(user_id)
    except Exception as e:
        print(f"获取用户信息失败: {e}")
        send_message(chat_id, "❌ 获取用户信息失败")
        return

    try:
        workflow_name = "文生图"
        config = TEXT_TO_IMAGE_CONFIG

        points_cost = config.get("points_cost", 2)

        # 初始化工作流处理器
        workflow_handler = ComfyUIWorkflow(
            seed_id=config["seed_id"],
            input_image_id=None,  # 文生图不需要输入图像
            output_image_id=config["output_image_id"],
            prompt_node_id=config.get("prompt_node_id"),
            workflow=config["workflow"]
        )

        # 加载工作流
        try:
            workflow_handler.load_workflow()
        except FileNotFoundError as e:
            send_message(chat_id, str(e))
            return

        # 生成工作流副本
        prompt_workflow = workflow_handler.create_workflow_copy()
        seed_value = generate_random_seed()
        prompt_workflow[workflow_handler.seed_id]["inputs"]["seed"] = int(seed_value)

        # 设置提示词
        if prompt and workflow_handler.prompt_node_id:
            prompt_workflow[workflow_handler.prompt_node_id]["inputs"]["text"] = prompt
            print(f"    使用提示词: {prompt[:100]}...")

        output_prefix = f"TextToImage\\{seed_value}"
        prompt_workflow[workflow_handler.output_image_id]["inputs"]["filename_prefix"] = output_prefix

        send_message(chat_id, f"🎨 开始生成图片...\n提示词: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")

        # 提交工作流
        prompt_id = queue_prompt(prompt_workflow)

        if not prompt_id:
            send_message(chat_id, f"❌ 提交工作流失败")
            return

        # 等待任务完成
        if not wait_for_completion(prompt_id, check_interval=2, timeout=300):
            send_message(chat_id, f"❌ 任务未完成")
            return

        # 获取并发送结果
        search_pattern = str(seed_value)
        output_file = None

        for root, dirs, files in os.walk(COMFYUI_OUTPUT_FOLDER):
            for file in files:
                if file.startswith(search_pattern):
                    output_file = os.path.join(root, file)
                    break
            if output_file:
                break

        if output_file and os.path.exists(output_file):
            send_photo(chat_id, output_file, f"文生图 - {prompt[:30]}")
            time.sleep(1)

            # 消耗积分（管理员免积分）
            if user_info['role'] == "管理员":
                print(f"    管理员免积分消耗")
            elif db.consume_points(user_id, points_cost):
                print(f"    消耗积分: {points_cost}")
            else:
                send_message(chat_id, f"⚠️ 积分不足")
                return
        else:
            send_message(chat_id, f"❌ 未找到输出文件")

        # 从队列中移除已完成的任务序号
        remove_task_from_queue(task_number)

        send_message(chat_id,
            f"✅ 文生图完成！\n"
            f"💰 本次消耗积分: {points_cost}\n"
            f"🎯 剩余积分: {db.get_user_points(user_id)}"
        )

    except Exception as e:
        error_msg = f"文生图时出错: {str(e)}"
        print(error_msg)
        send_message(chat_id, error_msg)


def process_image(image_path, chat_id: str, workflow_name: str, user_id: int, task_number: int, db: UserDatabase, prompt_text: str = None):
    """
    处理图像并发送结果到Telegram
    :param image_path: 图像文件路径
    :param chat_id: Telegram聊天ID
    :param workflow_name: 工作流名称
    :param user_id: 用户ID
    :param task_number: 任务序号
    :param db: 数据库实例
    :param prompt_text: 提示词（用于图像编辑）
    """
    if not comfyui_running:
        send_message(chat_id, "❌ ComfyUI 服务器未运行，无法处理图片")
        return

    # 获取用户信息（用于判断管理员身份）
    user_info = None
    try:
        user_info = db.get_user(user_id)
    except Exception as e:
        print(f"获取用户信息失败: {e}")
        send_message(chat_id, "❌ 获取用户信息失败")
        return

    try:
        if workflow_name not in WORKFLOW_CONFIGS:
            send_message(chat_id, f"错误: 未知的工作流 {workflow_name}")
            return

        config = WORKFLOW_CONFIGS[workflow_name]
        remove_iterations = config.get("remove_iterations", 1)
        points_cost = config.get("points_cost", 10)

        # send_message(chat_id,
        #     f"开始{workflow_name}处理..."
        #     f"任务编号: {task_number}"
        # )

        # 保存原图路径（用于后续发送时附带原图）
        original_image_path = image_path

        # 保存图像到ComfyUI input文件夹
        print(f"保存图像到 input 文件夹...")
        image_filename = save_image_with_unique_name(image_path, COMFYUI_INPUT_FOLDER)
        image_basename = os.path.splitext(image_filename)[0]

        print(f"图像文件名: {image_filename}")

        # 初始化工作流处理器
        workflow_handler = ComfyUIWorkflow(
            seed_id=config["seed_id"],
            input_image_id=config["input_image_id"],
            output_image_id=config["output_image_id"],
            prompt_node_id=config.get("prompt_node_id"),
            workflow=config["workflow"]
        )

        # 加载工作流
        try:
            workflow_handler.load_workflow()
        except FileNotFoundError as e:
            send_message(chat_id, str(e))
            return

        # 进行多次处理
        suffixes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

        success_count = 0
        total_cost = 0

        for iteration in range(remove_iterations):
            if not comfyui_running:
                send_message(chat_id, f"❌ ComfyUI 服务器已关闭，处理中止（已完成 {iteration}/{remove_iterations}）")
                return

            prompt_workflow = workflow_handler.create_workflow_copy()
            seed_value = generate_random_seed()
            prompt_workflow[workflow_handler.seed_id]["inputs"]["seed"] = int(seed_value)

            # 如果是图像编辑模式且提供了prompt，设置提示词
            if prompt_text and workflow_handler.prompt_node_id:
                prompt_workflow[workflow_handler.prompt_node_id]["inputs"]["prompt"] = prompt_text
                print(f"    使用提示词: {prompt_text}")

            current_suffix = suffixes[iteration]
            output_prefix = f"AutoOutput\\{image_basename}_{current_suffix}"

            prompt_workflow[workflow_handler.input_image_id]["inputs"]["image"] = image_filename
            prompt_workflow[workflow_handler.output_image_id]["inputs"]["filename_prefix"] = output_prefix

            print(f"\n迭代 {iteration+1}/{remove_iterations}: {current_suffix}")
            # send_message(chat_id, f"处理中... {iteration+1}/{remove_iterations} ({current_suffix})")

            # 提交工作流
            prompt_id = queue_prompt(prompt_workflow)

            if not prompt_id:
                send_message(chat_id, f"提交工作流失败，跳过此迭代")
                continue

            # 等待任务完成
            if not wait_for_completion(prompt_id, check_interval=2, timeout=300):
                send_message(chat_id, f"任务未完成，继续下一个迭代")
                continue

            # 获取并发送当前迭代的结果
            search_pattern = f"{image_basename}_{current_suffix}"
            output_file = None

            for root, dirs, files in os.walk(COMFYUI_OUTPUT_FOLDER):
                for file in files:
                    if file.startswith(search_pattern):
                        output_file = os.path.join(root, file)
                        break
                if output_file:
                    break

            if output_file and os.path.exists(output_file):
                send_photo(chat_id, output_file, f"{workflow_name} - 处理结果 {iteration+1}/{remove_iterations}", original_image_path)
                time.sleep(1)

                # 成功生成，消耗积分（管理员免积分）
                if user_info['role'] == "管理员":
                    success_count += 1
                    print(f"    管理员免积分消耗")
                elif db.consume_points(user_id, points_cost):
                    success_count += 1
                    total_cost += points_cost
                    print(f"    消耗积分: {points_cost}")
                else:
                    send_message(chat_id, f"⚠️ 积分不足，已生成{success_count}张")
                    break
            else:
                send_message(chat_id, f"未找到迭代 {iteration+1} 的输出文件")

        # 从队列中移除已完成的任务序号
        remove_task_from_queue(task_number)

        send_message(chat_id,
            f"✅ {workflow_name}处理完成！共发送 {success_count} 张处理结果\n"
            f"💰 本次消耗积分: {total_cost}\n"
            f"🎯 剩余积分: {db.get_user_points(user_id)}"
        )
        # 清理临时文件
        if "temp_" in os.path.basename(image_path):
            try:
                os.remove(image_path)
            except:
                pass

    except Exception as e:
        error_msg = f"处理图像时出错: {str(e)}"
        print(error_msg)
        send_message(chat_id, error_msg)


def is_authorized(user_id: int) -> bool:
    """检查用户是否授权"""
    if not AUTHORIZED_USER_IDS:
        return True
    return user_id in AUTHORIZED_USER_IDS


def get_user_workflow(chat_id: str) -> str:
    """获取用户当前的工作流设置"""
    result = user_workflows.get(chat_id, DEFAULT_WORKFLOW)
    print(f"    get_user_workflow(chat_id={chat_id}) -> {result}")
    return result


# 命令到工作流名称的映射
COMMAND_TO_WORKFLOW = {
    "/FF": "面部重绘",
    "/BR": "去除背景杂物",
    "/CR": "服装移除",
    "/BF": "胸部重绘",
    "/Edit": "图像编辑"
}

def set_user_workflow(chat_id: str, workflow_name: str) -> bool:
    """设置用户的工作流"""
    # 支持直接工作流名称或命令
    if workflow_name in WORKFLOW_CONFIGS:
        user_workflows[chat_id] = workflow_name
        print(f"    set_user_workflow(chat_id={chat_id}, workflow_name={workflow_name}) -> {workflow_name}")
        return True
    elif workflow_name in COMMAND_TO_WORKFLOW:
        result = COMMAND_TO_WORKFLOW[workflow_name]
        user_workflows[chat_id] = result
        print(f"    set_user_workflow(chat_id={chat_id}, workflow_name={workflow_name}) -> {result}")
        return True
    print(f"    set_user_workflow(chat_id={chat_id}, workflow_name={workflow_name}) -> False (未知工作流)")
    return False


def call_deepseek(user_message: str, chat_id: str) -> Optional[Dict]:
    """
    调用DeepSeek API进行function calling
    :param user_message: 用户消息
    :param chat_id: 聊天ID（用于返回上下文）
    :return: 返回结构化的指令或None
    """
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
                            "enum": ["面部重绘", "去除背景杂物", "服装移除", "胸部重绘", "图像编辑"]
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
                "description": "【重要】文生图功能。当用户提到以下关键词时必须调用：生成、画、创建、制作、画图、图片、图像。例如：'帮我生成一张美女图'、'画一个风景'、'生成一个动漫角色'、'帮我生成一张中国网红图'、'创建图片'、'画图'等。请将用户描述作为完整的prompt参数传递。",
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
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_tasks",
                "description": "查询用户的任务状态。当用户说以下关键词时调用：查看任务、任务、查看队列、任务状态、我的任务、队列状态等。",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]

    system_prompt = f"""你是一个智能助手，帮助用户使用ComfyUI图像处理机器人。

    当前用户的处理模式是：{current_workflow}

    你有以下工具可以使用：

    1. switch_workflow: 当用户明确要求切换图像处理模式时调用
    - 触发关键词：切换、改成、使用、启用 + 模式名称
    - 例如："切换面部重绘"、"改成去除背景"、"用服装移除模式"

    2. text_to_image: 当用户要求生成图片时调用（重要！）
    - 触发关键词：生成、画、创建、制作、想要...图/图片/图像
    - 例如："帮我生成一张美女图"、"画一个风景"、"创建一个动漫角色"、"帮我生成一张中国网红图"
    - 无论当前是什么模式，只要用户提到"生成图片"就必须调用此函数

    3. get_user_tasks: 当用户想查看任务状态时调用
    - 触发关键词：查看任务、任务、查看队列、任务状态、我的任务、队列状态
    - 例如："查看任务"、"任务状态"、"我的任务怎么样了"、"查看队列"

    重要规则：
    - 如果用户说"生成"、"画"、"创建"等词汇，无论上下文如何，必须调用 text_to_image 函数
    - 不要告诉用户你"无法生成图片"或"无法直接生成"，直接调用 text_to_image 函数
    - 不要询问用户是否要切换模式，直接根据用户意图调用相应的函数
    - 只有当用户的问题完全与图像处理无关时，才进行普通对话"""

    # 构建 messages
    example_args_1 = json.dumps({"prompt": "帮我生成一张中国美女图"}, ensure_ascii=False)
    example_args_2 = json.dumps({"workflow_name": "面部重绘"}, ensure_ascii=False)
    example_args_3 = json.dumps({}, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！我是ComfyUI图像处理助手。我可以帮你处理图片或生成新图片。有什么可以帮你的吗？"},
        {"role": "user", "content": "帮我生成一张中国美女图"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "text_to_image", "arguments": example_args_1}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"status": "success", "message": "已提交文生图任务"}, ensure_ascii=False)},
        {"role": "user", "content": "切换到面部重绘"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "switch_workflow", "arguments": example_args_2}}]},
        {"role": "tool", "tool_call_id": "call_2", "content": json.dumps({"status": "success", "message": "已切换到面部重绘模式"}, ensure_ascii=False)},
        {"role": "user", "content": "查看任务"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_3", "type": "function", "function": {"name": "get_user_tasks", "arguments": example_args_3}}]},
        {"role": "tool", "tool_call_id": "call_3", "content": json.dumps({"status": "success", "message": "已返回任务列表"}, ensure_ascii=False)},
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
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }

        print(f"  DeepSeek请求参数: tool_choice=auto, temperature=0.3")

        response = requests.post(
            DEEPSEEK_API_URL,
            json=payload,
            headers=headers,
            timeout=30,
            proxies=get_proxies()
        )

        result = response.json()

        if "choices" not in result or len(result["choices"]) == 0:
            print(f"DeepSeek API返回格式错误: {result}")
            return None

        choice = result["choices"][0]
        message = choice.get("message", {})

        # 检查是否有tool_calls (新格式)
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


def handle_function_call(chat_id: str, function_call: Dict, user_id: int, db: UserDatabase):
    """
    处理function calling结果
    :param chat_id: 聊天ID
    :param function_call: function调用信息
    :param user_id: 用户ID
    :param db: 数据库实例
    :return: True表示已处理，False表示需要继续
    """
    function_name = function_call["name"]
    arguments = function_call["arguments"]

    if function_name == "switch_workflow":
        workflow_name = arguments.get("workflow_name")
        if workflow_name:
            # 找到对应的命令
            command_map = {
                "面部重绘": "/FF",
                "去除背景杂物": "/BR",
                "服装移除": "/CR",
                "胸部重绘": "/BF",
                "图像编辑": "/Edit"
            }
            command = command_map.get(workflow_name)
            if command:
                print(f"自然语言切换工作流: {workflow_name}")
                if set_user_workflow(chat_id, command):
                    send_message(chat_id, f"✅ 已切换到「{workflow_name}」处理方式")
                else:
                    send_message(chat_id, "❌ 切换失败")
                return True
        return False

    elif function_name == "text_to_image":
        prompt = arguments.get("prompt", "")
        if prompt:
            print(f"文生图请求: {prompt[:100]}...")

            # 检查积分
            user_info = db.get_user(user_id)
            points_cost = TEXT_TO_IMAGE_CONFIG.get("points_cost", 2)
            if user_info['role'] != "管理员":
                current_points = db.get_user_points(user_id)
                if current_points < points_cost:
                    send_message(chat_id, f"❌ 积分不足！\n当前积分: {current_points}\n需要积分: {points_cost}\n\n发送「/密钥兑换」获取积分")
                    return True

            # 分配任务序号
            try:
                with task_counter_lock:
                    global global_task_counter
                    global_task_counter += 1
                    current_task_number = global_task_counter
                print(f"文生图任务序号: {current_task_number}")
            except Exception as e:
                print(f"分配任务序号失败: {e}")
                send_message(chat_id, "❌ 任务分配失败")
                return True

            # 添加到任务队列
            add_task_to_queue(user_id, current_task_number)
            print(f"任务已加入队列")

            # 计算排队位置
            current_position, waiting_count, total_count = get_queue_info(user_id, current_task_number)
            wait_time = (waiting_count + 1) * 30

            send_message(chat_id,
                f"📊 当前排队序列: {current_task_number} (位置 {current_position}/{total_count})\n"
                f"⏳ 前面还有 {waiting_count} 个待执行任务\n"
                f"⏰ 预计等待时间: {wait_time} 秒 ({wait_time//60}分{wait_time%60}秒)"
            )

            # 启动处理线程
            try:
                thread = threading.Thread(
                    target=process_text_to_image,
                    args=(chat_id, user_id, prompt, db, current_task_number)
                )
                thread.daemon = True
                thread.start()
                print(f"  文生图处理线程已启动")
            except Exception as e:
                print(f"启动处理线程失败: {e}")
                send_message(chat_id, f"❌ 启动处理失败: {e}")

            return True

    elif function_name == "get_user_tasks":
        # 获取用户任务列表
        user_tasks = get_user_tasks(user_id)
        
        with task_queue_lock:
            total_queue_size = len(task_queue)
        
        if user_tasks:
            # 计算每个任务的排队位置
            task_list = []
            for i, (uid, tnum) in enumerate(task_queue):
                if uid == user_id:
                    position = i + 1
                    task_list.append(f"• 任务 {tnum}: 位置 {position}/{total_queue_size}")
            
            task_info = "\n".join(task_list)
            send_message(chat_id,
                f"📋 您的任务列表\n\n"
                f"队列中总任务数: {total_queue_size}\n"
                f"您的未完成任务数: {len(user_tasks)}\n\n"
                f"{task_info}"
            )
        else:
            send_message(chat_id, "📋 您当前没有正在处理的任务")
        
        return True

    return False


def send_welcome_message(chat_id: str, user_info: dict, db: UserDatabase):
    """发送欢迎消息"""
    current_workflow = get_user_workflow(chat_id)
    admin_commands = "\n/generate_keys - 生成新密钥（仅管理员）" if user_info['role'] == "管理员" else ""

    # 生成工作流列表（带积分消耗）
    workflow_list = ""
    for name, config in WORKFLOW_CONFIGS.items():
        cmd_map = {
            "面部重绘": "/FF",
            "去除背景杂物": "/BR",
            "服装移除": "/CR",
            "胸部重绘": "/BF",
            "图像编辑": "/Edit"
        }
        cmd = cmd_map.get(name, f"/{name}")
        total_cost = config["points_cost"] * config["remove_iterations"]
        workflow_list += f"• {cmd}({name}) - {total_cost}积分/张\n"

    send_message(chat_id,
               "🤖 欢迎使用 ComfyUI 图像处理机器人 V3！\n\n"
               f"发送图片给我，将使用「{current_workflow}」处理方式。\n\n"
               f"支持格式: JPG, PNG, JPEG\n\n"
               f"👤 身份: {user_info['role']}\n"
               f"💰 积分: {user_info['points']}\n"
               f"当前图像处理方式: {current_workflow}\n"
               "可用的图像处理方式:\n"
               f"{workflow_list}"
               "点击以切换图像处理方式\n\n"
               "💬 V3新功能：支持自然语言对话！\n"
               "• 可以说「切换面部重绘」、「改成去除背景」等来切换模式\n"
               "• 可以说「帮我生成一张精美中国网红的图」来进行文生图\n"
               "• 其他问题我也会尽力回答\n\n"
               "发送「/key」进行积分兑换\n"
               "发送「/points」查询积分\n"
               "发送「/info」查看个人信息\n"
               "发送「/help」查看使用说明"
               f"{admin_commands}")


def monitor_comfyui_server():
    """持续监控 ComfyUI 服务器状态"""
    global comfyui_running
    while comfyui_running:
        try:
            response = request.urlopen("http://127.0.0.1:8188", timeout=2)
            time.sleep(5)
        except error.URLError:
            print("⚠️ ComfyUI 服务器已关闭！")
            comfyui_running = False
            break
        except Exception as e:
            print(f"⚠️ 检测 ComfyUI 服务器时出错: {e}")
            comfyui_running = False
            break


def main():
    global comfyui_running, global_task_counter

    print("========== Telegram机器人启动 V3 ==========")
    print(f"Bot Token: {TELEGRAM_BOT_TOKEN[:20]}...")
    print(f"DeepSeek API Key: {DEEPSEEK_API_KEY[:20]}...")
    print(f"文生图配置: {TEXT_TO_IMAGE_CONFIG['workflow']}, 节点 {TEXT_TO_IMAGE_CONFIG['prompt_node_id']}, 积分消耗 {TEXT_TO_IMAGE_CONFIG['points_cost']}")

    if AUTHORIZED_USER_IDS:
        print(f"授权用户ID: {AUTHORIZED_USER_IDS}")
    else:
        print(f"授权用户ID: 所有用户（测试模式）")
        print("⚠️  警告: 未设置授权用户ID列表，允许所有用户访问")

    print(f"默认工作流: {DEFAULT_WORKFLOW}")
    print(f"可用工作流: {list(WORKFLOW_CONFIGS.keys())}")

    # 初始化数据库
    print("\n初始化用户数据库...")
    db = UserDatabase("user_database.json")
    print(f"数据库已加载: {db.db_file}")
    print(f"当前用户数: {len(db.data['users'])}")
    print(f"可用密钥数: {sum(1 for k in db.data['keys'].values() if not k['used'])}")

    # 生成密钥（首次运行时）
    if not db.data["keys"]:
        print("\n生成初始密钥...")
        keys = db.generate_keys(10)
        print(f"已生成 {len(keys)} 个密钥:")
        for i, key in enumerate(keys, 1):
            print(f"  {i}. {key}")

    # 检查ComfyUI服务器
    print("\n检查ComfyUI服务器状态...")
    if not check_comfyui_server():
        print("请先启动ComfyUI服务器")
        return
    else:
        print("ComfyUI服务器已运行")

    # 启动 ComfyUI 服务器监控线程
    print("\n启动 ComfyUI 服务器监控...")
    monitor_thread = threading.Thread(target=monitor_comfyui_server, daemon=True)
    monitor_thread.start()

    offset = 0
    print("\n========== 开始监听消息 ==========")

    while True:
        try:
            if not comfyui_running:
                print("⚠️ ComfyUI 服务器未运行，暂停处理新请求...")
                time.sleep(5)
                continue

            updates = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                params={"timeout": 30, "offset": offset if offset else None},
                timeout=40,
                proxies=get_proxies()
            ).json()

            if not updates.get("ok"):
                time.sleep(5)
                continue

            for update in updates["result"]:
                offset = update["update_id"] + 1

                if "message" not in update:
                    continue

                message = update["message"]
                user_id = message.get("from", {}).get("id")
                chat_id = message.get("chat", {}).get("id")
                username = message.get("from", {}).get("username", "Unknown")

                print(f"\n收到消息 - 用户ID: {user_id} (@{username}), 聊天ID: {chat_id}")
                print(f"消息类型: {list(message.keys())}")  # 添加调试输出

                # 检查授权
                if not is_authorized(user_id):
                    print(f"用户 {user_id} 未授权，忽略消息")
                    send_message(chat_id, "❌ 您没有权限使用此机器人")
                    continue

                # 添加或更新用户
                print("添加或更新用户到数据库...")
                is_new_user = db.add_user(user_id, username)
                user_info = db.get_user(user_id)
                print(f"用户信息已获取")

                # 如果是新用户，自动发送欢迎消息
                if is_new_user:
                    print(f"检测到新用户，发送欢迎消息...")
                    send_welcome_message(chat_id, user_info, db)

                # 处理图片消息
                if "photo" in message:
                    print("处理图片消息...")
                    try:
                        photos = message["photo"]
                        largest_photo = max(photos, key=lambda p: p.get("file_size", 0))
                        file_id = largest_photo["file_id"]
                    except Exception as e:
                        print(f"解析photo字段失败: {e}")
                        continue

                    # 检查积分
                    try:
                        user_info = db.get_user(user_id)
                        current_points = db.get_user_points(user_id)
                        workflow_name = get_user_workflow(chat_id)
                        print(f"收到图片，当前工作流: {workflow_name}")
                        config = WORKFLOW_CONFIGS[workflow_name]
                        points_cost = config.get("points_cost", 10) * config.get("remove_iterations", 1)
                        print(f"工作流配置加载成功，积分消耗: {points_cost}")
                    except Exception as e:
                        print(f"获取积分配置失败: {e}")
                        continue

                    # 管理员免积分
                    if user_info['role'] != "管理员":
                        if current_points < points_cost:
                            send_message(chat_id, f"❌ 积分不足！\n当前积分: {current_points}\n需要积分: {points_cost}\n\n发送「/密钥兑换」获取积分")
                            continue

                    # 图像编辑模式需要先询问提示词
                    if workflow_name == "图像编辑":
                        # 先下载图片
                        print(f"  图像编辑模式，下载图片...")
                        image_path = download_telegram_photo(file_id)

                        if image_path:
                            print(f"  图片已下载: {image_path}")
                            print(f"  已保存状态到 edit_prompt_states，等待用户输入提示词")
                            send_message(chat_id, f"📝 已收到图片！\n\n请输入编辑提示词（prompt），描述你想要对图片进行的修改...")
                            # 将状态保存到edit_prompt_states
                            edit_prompt_states[chat_id] = {
                                "image_path": image_path,
                                "user_id": user_id,
                                "workflow_name": workflow_name,
                                "points_cost": points_cost
                            }
                            print(f"  edit_prompt_states 当前状态: {edit_prompt_states}")
                        else:
                            send_message(chat_id, "❌ 下载图片失败")
                        print(f"  图像编辑模式处理完成，即将执行 continue 跳过后续代码")
                        continue

                    # 分配任务序号
                    print(f"  正常模式，开始分配任务序号...")
                    try:
                        with task_counter_lock:
                            global_task_counter += 1
                            current_task_number = global_task_counter
                        print(f"任务序号: {current_task_number}")
                    except Exception as e:
                        print(f"分配任务序号失败: {e}")
                        continue

                    # 添加到任务队列
                    add_task_to_queue(user_id, current_task_number)
                    print(f"任务已加入队列")

                    # 计算排队位置
                    current_position, waiting_count, total_count = get_queue_info(user_id, current_task_number)
                    wait_time = (waiting_count + 1) * 30  # 每张图30秒

                    send_message(chat_id,
                        f"收到图片，正在使用「{workflow_name}」处理...\n"
                        f"📊 当前排队序列: {current_task_number} (位置 {current_position}/{total_count})\n"
                        f"⏳ 前面还有 {waiting_count} 个待执行任务\n"
                        f"⏰ 预计等待时间: {wait_time} 秒 ({wait_time//60}分{wait_time%60}秒)"
                        )

                    print(f"  开始下载图片...")
                    image_path = download_telegram_photo(file_id)

                    if image_path:
                        print(f"  图片已下载: {image_path}")
                        try:
                            thread = threading.Thread(
                                target=process_image,
                                args=(image_path, chat_id, workflow_name, user_id, current_task_number, db)
                            )
                            thread.daemon = True
                            thread.start()
                            print(f"  处理线程已启动")
                        except Exception as e:
                            print(f"启动处理线程失败: {e}")
                            send_message(chat_id, f"❌ 启动处理失败: {e}")
                    else:
                        send_message(chat_id, "❌ 下载图片失败")

                # 处理文本消息
                elif "text" in message:
                    text = message["text"].strip()

                    if text == "/start":
                        user_info = db.get_user(user_id)
                        send_welcome_message(chat_id, user_info, db)

                    elif text == "/help":
                        current_workflow = get_user_workflow(chat_id)
                        user_info = db.get_user(user_id)
                        admin_commands = "\n/generate_keys - 生成新密钥（仅管理员）" if user_info['role'] == "管理员" else ""

                        # 生成工作流列表（带积分消耗）
                        workflow_list = ""
                        for name, config in WORKFLOW_CONFIGS.items():
                            cmd_map = {
                                "面部重绘": "/FF",
                                "去除背景杂物": "/BR",
                                "服装移除": "/CR",
                                "胸部重绘": "/BF",
                                "图像编辑": "/Edit"
                            }
                            cmd = cmd_map.get(name, f"/{name}")
                            total_cost = config["points_cost"] * config["remove_iterations"]
                            workflow_list += f"{cmd}({name}) - {total_cost}积分/张\n"

                        send_message(chat_id,
                                   "📖 使用说明:\n\n"
                                   "1. 发送图片给我\n"
                                   "2. 图片将自动使用当前处理方式处理\n"
                                   "3. 消耗积分生成图片\n"
                                   "4. 等待处理完成\n\n"
                                   "命令:\n"
                                   "/start - 开始\n"
                                   "/help - 帮助\n"
                                   "/info - 查看个人信息\n"
                                   "/points - 查询积分\n"
                                   "切换处理方式:\n"
                                   f"{workflow_list}"
                                   "密钥兑换: 发送「/key」"
                                   f"{admin_commands}\n\n"
                                   "💬 V3新功能 - 自然语言对话:\n"
                                   "• 说「切换面部重绘」、「改成去除背景」等来切换模式\n"
                                   "• 说「帮我生成一张...的图」来进行文生图\n"
                                   "• 其他问题我会尽力回答")

                    elif text == "/info":
                        user_info = db.get_user(user_id)
                        with task_queue_lock:
                            queue_size = len(task_queue)
                        send_message(chat_id,
                                   f"👤 用户信息\n\n"
                                   f"用户ID: {user_id}\n"
                                   f"身份: {user_info['role']}\n"
                                   f"💰 积分: {user_info['points']}\n"
                                   f"📋 全局队列任务: {queue_size} 个\n"
                                   f"当前处理方式: {get_user_workflow(chat_id)}")

                    elif text == "/points":
                        points = db.get_user_points(user_id)
                        send_message(chat_id, f"💰 您的积分: {points}")

                    elif text == "/task":
                        # 获取用户任务列表
                        user_tasks = get_user_tasks(user_id)
                        
                        with task_queue_lock:
                            total_queue_size = len(task_queue)
                        
                        if user_tasks:
                            # 计算每个任务的排队位置
                            task_list = []
                            for i, (uid, tnum) in enumerate(task_queue):
                                if uid == user_id:
                                    position = i + 1
                                    task_list.append(f"• 任务 {tnum}: 位置 {position}/{total_queue_size}")
                            
                            task_info = "\n".join(task_list)
                            send_message(chat_id,
                                f"📋 您的任务列表\n\n"
                                f"队列中总任务数: {total_queue_size}\n"
                                f"您的未完成任务数: {len(user_tasks)}\n\n"
                                f"{task_info}"
                            )
                        else:
                            send_message(chat_id, "📋 您当前没有正在处理的任务")

                    elif text == "/key":
                        key_exchange_states[chat_id] = "waiting_for_key"
                        send_message(chat_id, "🔑 请直接回复密钥进行兑换")

                    elif text in COMMAND_TO_WORKFLOW:
                        # 优先处理工作流切换命令
                        workflow_name = COMMAND_TO_WORKFLOW[text]
                        print(f"检测到工作流切换命令: {text} -> {workflow_name}")
                        print(f"当前工作流: {get_user_workflow(chat_id)}")
                        if set_user_workflow(chat_id, text):
                            print(f"切换后工作流: {get_user_workflow(chat_id)}")
                            send_message(chat_id, f"✅ 已切换到「{workflow_name}」处理方式")
                        else:
                            print(f"切换失败: {text}")
                            send_message(chat_id, "❌ 切换失败")

                    elif chat_id in key_exchange_states and key_exchange_states[chat_id] == "waiting_for_key":
                        # 验证并兑换密钥
                        if db.validate_key(text):
                            if db.use_key(text, user_id):
                                user_role = db.get_user(user_id)['role']
                                message = f"✅ 密钥兑换成功！\n\n💰 获得{KEY_REWARD_POINTS}积分\n"
                                if user_role == "会员":
                                    message += f"👤 身份已升级为「会员」\n"
                                elif user_role == "管理员":
                                    message += f"👤 保持管理员身份\n"
                                message += f"当前积分: {db.get_user_points(user_id)}"
                                send_message(chat_id, message)
                                del key_exchange_states[chat_id]
                            else:
                                send_message(chat_id, "❌ 密钥使用失败")
                        else:
                            send_message(chat_id, "❌ 无效的密钥或密钥已使用")
                            del key_exchange_states[chat_id]

                    elif chat_id in edit_prompt_states:
                        # 处理图像编辑的提示词输入
                        prompt_text = text.strip()
                        if not prompt_text:
                            send_message(chat_id, "❌ 提示词不能为空，请重新输入")
                            continue

                        edit_state = edit_prompt_states[chat_id]
                        send_message(chat_id, f"✅ 收到提示词: {prompt_text[:50]}{'...' if len(prompt_text) > 50 else ''}\n\n开始处理...")

                        # 分配任务序号
                        try:
                            with task_counter_lock:
                                global_task_counter += 1
                                current_task_number = global_task_counter
                            print(f"任务序号: {current_task_number}")
                        except Exception as e:
                            print(f"分配任务序号失败: {e}")
                            send_message(chat_id, "❌ 任务分配失败")
                            del edit_prompt_states[chat_id]
                            continue

                        # 添加到任务队列
                        add_task_to_queue(edit_state["user_id"], current_task_number)
                        print(f"任务已加入队列")

                        # 计算排队位置
                        current_position, waiting_count, total_count = get_queue_info(edit_state["user_id"], current_task_number)
                        wait_time = (waiting_count + 1) * 30

                        send_message(chat_id,
                            f"📊 当前排队序列: {current_task_number} (位置 {current_position}/{total_count})\n"
                            f"⏳ 前面还有 {waiting_count} 个待执行任务\n"
                            f"⏰ 预计等待时间: {wait_time} 秒 ({wait_time//60}分{wait_time%60}秒)"
                        )

                        # 启动处理线程
                        try:
                            thread = threading.Thread(
                                target=process_image,
                                args=(edit_state["image_path"], chat_id, edit_state["workflow_name"],
                                      edit_state["user_id"], current_task_number, db, prompt_text)
                            )
                            thread.daemon = True
                            thread.start()
                            print(f"  编辑处理线程已启动")
                        except Exception as e:
                            print(f"启动处理线程失败: {e}")
                            send_message(chat_id, f"❌ 启动处理失败: {e}")

                        # 清除状态
                        del edit_prompt_states[chat_id]

                    elif text == "/generate_keys":
                        user_info = db.get_user(user_id)
                        if user_info['role'] != "管理员":
                            send_message(chat_id, "❌ 只有管理员可以使用此功能")
                        else:
                            keys = db.generate_keys(1)
                            send_message(chat_id, f"🔑 已生成新密钥:\n\n{keys[0]}")

                    else:
                        # 其他文本消息，使用DeepSeek处理
                        print(f"使用DeepSeek处理文本消息: {text[:50]}...")

                        deepseek_result = call_deepseek(text, chat_id)

                        if deepseek_result:
                            if deepseek_result["type"] == "function_call":
                                # 处理function调用
                                handled = handle_function_call(chat_id, deepseek_result, user_id, db)
                                if handled:
                                    continue
                            elif deepseek_result["type"] == "text":
                                # 返回文本回复
                                send_message(chat_id, deepseek_result["content"])
                                continue

                        # DeepSeek失败或无结果，显示默认消息
                        print(f"未处理的文本消息: {text}")
                        # 生成工作流列表（带积分消耗）
                        workflow_list = ""
                        for name, config in WORKFLOW_CONFIGS.items():
                            cmd_map = {
                                "面部重绘": "/FF",
                                "去除背景杂物": "/BR",
                                "服装移除": "/CR",
                                "胸部重绘": "/BF",
                                "图像编辑": "/Edit"
                            }
                            cmd = cmd_map.get(name, f"/{name}")
                            total_cost = config["points_cost"] * config["remove_iterations"]
                            workflow_list += f"{cmd}({name}) - {total_cost}积分/张\n"

                        send_message(chat_id, f"请发送图片给我，我会处理并返回结果\n\n"
                                               f"切换处理方式:\n"
                                               f"{workflow_list}"
                                               f"💬 提示：可以用自然语言对话，例如「切换面部重绘」或「帮我生成一张美女图」\n\n"
                                               f"发送「/key」进行积分兑换")

                else:
                    # 其他类型的消息（如 service message 等）
                    print(f"未处理的消息类型: {message}")
                    continue

        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            comfyui_running = False
            break
        except requests.exceptions.ProxyError as e:
            print(f"代理错误: {e}")
            print("检查代理设置或禁用代理（USE_PROXY = False）")
            print("按 Ctrl+C 退出，或等待自动重连...")
            time.sleep(5)
        except requests.exceptions.SSLError as e:
            print(f"SSL连接错误: {e}")
            print("尝试重新连接...")
            time.sleep(10)
        except requests.exceptions.ConnectionError as e:
            print(f"连接错误: {e}")
            print("网络连接异常，等待自动重连...")
            time.sleep(5)
        except Exception as e:
            print(f"运行出错: {e}")
            print("按 Ctrl+C 退出，或等待自动重连...")
            time.sleep(5)

    print("程序结束")


if __name__ == "__main__":
    main()
