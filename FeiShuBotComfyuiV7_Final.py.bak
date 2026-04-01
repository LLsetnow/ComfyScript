"""
飞书机器人 V7 - 长连接版本（完整功能）
基于飞书SDK WebSocket Client 实现长连接监听事件
集成 ComfyUI 图像处理功能
"""
import json
import time
import os
import sys
import shutil
import random
import threading
from typing import Optional

# 读取配置文件
def load_config():
    """从config.json5加载配置"""
    config_path = os.path.join(os.path.dirname(__file__), "config.json5")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            # json5支持注释,使用json5库或简单替换注释
            content = f.read()
            # 移除单行注释 // ...
            lines = []
            for line in content.split('\n'):
                # 移除//注释(注意不要移除URL中的//)
                if '//' in line and not line.strip().startswith('"'):
                    line = line.split('//')[0]
                lines.append(line)
            content = '\n'.join(lines)
            return json.loads(content)
    except Exception as e:
        print(f"[ERROR] 加载配置文件失败: {e}")
        return None

# 加载配置
config = load_config()
if config is None:
    print("[ERROR] 无法加载配置文件，程序退出")
    sys.exit(1)

# 飞书配置
FEISHU_APP_ID = config.get("feishu", {}).get("app_id", "")
FEISHU_APP_SECRET = config.get("feishu", {}).get("app_secret", "")
FEISHU_API_BASE = config.get("feishu", {}).get("api_base", "https://open.feishu.cn/open-apis")

# ComfyUI配置
comfyui_config = config.get("comfyUI", {})
COMFYUI_API_URL = f"http://{comfyui_config.get('host', '127.0.0.1')}:{comfyui_config.get('port', '8188')}"
comfyui_folder = comfyui_config.get('folder', 'D:\\AI_Graph\\ConfyUI-aki\\ComfyUI-aki-v1')
COMFYUI_INPUT_FOLDER = os.path.join(comfyui_folder, "input")
COMFYUI_OUTPUT_FOLDER = os.path.join(comfyui_folder, "output", "FeiShuBot")

# 创建文件夹
os.makedirs(COMFYUI_INPUT_FOLDER, exist_ok=True)
os.makedirs(COMFYUI_OUTPUT_FOLDER, exist_ok=True)

# 工作流配置 - 从config.json5读取
workflows_config = config.get("workflows", {})
WORKFLOW_CONFIGS = {}

# 直接使用配置中的工作流名称
for name, workflow_data in workflows_config.items():
    WORKFLOW_CONFIGS[name] = {
        "seed_id": str(workflow_data.get("seed_id")),
        "input_image_id": str(workflow_data.get("input_image_id")),
        "output_image_id": str(workflow_data.get("output_image_id")),
        "workflow": workflow_data.get("workflow", ""),
        "points_cost": workflow_data.get("points_cost", 10),
        "remove_iterations": workflow_data.get("remove_iterations", 1)
    }
    # 如果有额外的prompt_node_id,也添加进去
    if "prompt_node_id" in workflow_data:
        WORKFLOW_CONFIGS[name]["prompt_node_id"] = workflow_data["prompt_node_id"]

DEFAULT_WORKFLOW = config.get("default_workflow", "Qwen_remove")

# 文生图配置
TEXT_TO_IMAGE_CONFIG = config.get("text_to_image", {})
if TEXT_TO_IMAGE_CONFIG:
    print(f"[OK] 文生图配置已加载: {TEXT_TO_IMAGE_CONFIG.get('workflow', 'unknown')}")
else:
    print("[WARNING] 文生图配置未找到")

# 导入飞书SDK
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

# DeepSeek API配置
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[WARNING] requests库未安装,自然语言功能不可用")

DEEPSEEK_API_KEY = config.get("deepseek", {}).get("api_key", "")
DEEPSEEK_API_URL = config.get("deepseek", {}).get("api_url", "https://api.deepseek.com/v1/chat/completions")

if not DEEPSEEK_API_KEY:
    print("[WARNING] 未配置DeepSeek API密钥,自然语言功能不可用")


class ComfyUIWorkflow:
    """ComfyUI工作流处理类"""

    def __init__(self, seed_id=65, input_image_id=None, output_image_id=181, workflow='Qwen_remove.json'):
        """
        初始化工作流处理器
        :param input_image_id: 输入图像节点ID,为None时表示不需要输入图像(文生图场景)
        """
        self.seed_id = str(seed_id)
        self.input_image_id = str(input_image_id) if input_image_id is not None else None
        self.output_image_id = str(output_image_id)
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
        """设置随机种子"""
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")
        self.original_workflow[self.seed_id]["inputs"]["seed"] = int(seed_value)

    def set_input_image(self, image_filename):
        """设置输入图像"""
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")
        if self.input_image_id is None:
            print("  警告: 此工作流不需要输入图像")
            return
        self.original_workflow[self.input_image_id]["inputs"]["image"] = image_filename

    def set_output_prefix(self, output_prefix):
        """设置输出文件前缀"""
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")
        self.original_workflow[self.output_image_id]["inputs"]["filename_prefix"] = output_prefix

    def get_workflow(self):
        """获取当前工作流配置"""
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")
        return self.original_workflow

    def create_workflow_copy(self):
        """创建工作流的深拷贝"""
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")
        return json.loads(json.dumps(self.original_workflow))


def generate_random_seed():
    """生成15位随机数种子"""
    return random.randint(10**14, 10**15 - 1)


def check_comfyui_server(max_attempts=3, check_delay=2):
    """检查ComfyUI服务器是否可访问"""
    if not URLLIB_AVAILABLE:
        return False
    for attempt in range(max_attempts):
        try:
            request.urlopen(f"{COMFYUI_API_URL}/system_stats", timeout=3)
            return True
        except urllib_error.URLError:
            if attempt < max_attempts - 1:
                time.sleep(check_delay)
            else:
                return False
    return False


def queue_prompt(prompt_workflow, max_retries=3, retry_delay=2):
    """将prompt workflow发送到ComfyUI服务器并排队执行"""
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req = request.Request(f"{COMFYUI_API_URL}/prompt", data=data)

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


def wait_for_completion(prompt_id, check_interval=5, timeout=120):
    """轮询检查任务完成状态"""
    start_time = time.time()
    check_count = 0
    initial_interval = 15
    initial_phase = True

    while time.time() - start_time < timeout:
        check_count += 1
        current_interval = initial_interval if initial_phase else check_interval

        try:
            req = request.Request(f"{COMFYUI_API_URL}/history/{prompt_id}")
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


def download_feishu_image(client: lark.Client, image_key: str, message_id: str) -> Optional[str]:
    """从飞书下载图片并返回保存路径"""
    try:
        import requests
        import urllib.request
        import json
        
        # 手动获取 tenant_access_token
        token_url = f"{FEISHU_API_BASE}/auth/v3/tenant_access_token/internal"
        token_data = json.dumps({
            "app_id": FEISHU_APP_ID,
            "app_secret": FEISHU_APP_SECRET
        }).encode('utf-8')
        
        token_req = urllib.request.Request(
            token_url,
            data=token_data,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(token_req) as response:
            token_response = json.loads(response.read().decode('utf-8'))
        
        if token_response.get('code') != 0:
            print(f"  获取token失败: {token_response.get('msg')}")
            return None
        
        token = token_response.get('tenant_access_token')
        
        # 使用 "获取消息中的资源文件" 接口
        # API: GET /open-apis/im/v1/messages/{message_id}/resources/{file_key}?type=image
        resource_url = f"{FEISHU_API_BASE}/im/v1/messages/{message_id}/resources/{image_key}?type=image"

        print(f"  正在下载图片: {image_key}")
        print(f"  URL: {resource_url}")

        # 下载图片，使用认证头
        response = requests.get(
            resource_url,
            headers={'Authorization': f'Bearer {token}'},
            timeout=30
        )
        
        print(f"  Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"  下载图片失败: HTTP {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            return None
        
        image_data = response.content
        
        # 保存到临时文件
        temp_filename = f"temp_{int(time.time())}_{image_key[:8]}.jpg"
        temp_path = os.path.join(COMFYUI_INPUT_FOLDER, temp_filename)
        
        with open(temp_path, 'wb') as f:
            f.write(image_data)
        
        print(f"  图片已下载: {temp_path} ({len(image_data)} bytes)")
        return temp_path
    
    except Exception as e:
        print(f"  下载飞书图片失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_image_with_comfyui(image_path: str, workflow_name: str) -> Optional[str]:
    """
    使用ComfyUI处理图像
    :param image_path: 图像文件路径
    :param workflow_name: 工作流名称
    :return: 处理后的图片路径，失败返回None
    """
    if not check_comfyui_server():
        print("  ComfyUI服务器未运行")
        return None

    if workflow_name not in WORKFLOW_CONFIGS:
        print(f"  未知的工作流: {workflow_name}")
        return None

    config = WORKFLOW_CONFIGS[workflow_name]
    remove_iterations = config.get("remove_iterations", 1)

    try:
        # 保存图像到ComfyUI input文件夹
        print(f"  保存图像到 input 文件夹...")
        image_filename = save_image_with_unique_name(image_path, COMFYUI_INPUT_FOLDER)
        image_basename = os.path.splitext(image_filename)[0]

        print(f"  图像文件名: {image_filename}")

        # 初始化工作流处理器
        workflow_handler = ComfyUIWorkflow(
            seed_id=config["seed_id"],
            input_image_id=config["input_image_id"],
            output_image_id=config["output_image_id"],
            workflow=config["workflow"]
        )

        # 加载工作流
        try:
            workflow_handler.load_workflow()
        except FileNotFoundError as e:
            print(f"  {e}")
            return None

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


def process_text_to_image(prompt: str) -> Optional[str]:
    """
    使用ComfyUI进行文生图
    :param prompt: 提示词
    :return: 生成的图片路径，失败返回None
    """
    if not TEXT_TO_IMAGE_CONFIG:
        print("  文生图配置未找到")
        return None

    if not check_comfyui_server():
        print("  ComfyUI服务器未运行")
        return None

    try:
        print(f"  开始文生图: {prompt[:50]}...")

        # 初始化工作流处理器
        workflow_handler = ComfyUIWorkflow(
            seed_id=str(TEXT_TO_IMAGE_CONFIG["seed_id"]),
            input_image_id=None,  # 文生图不需要输入图像
            output_image_id=str(TEXT_TO_IMAGE_CONFIG["output_image_id"]),
            workflow=TEXT_TO_IMAGE_CONFIG["workflow"]
        )

        # 加载工作流
        try:
            workflow_handler.load_workflow()
        except Exception as e:
            print(f"  加载工作流失败: {e}")
            return None

        # 设置提示词
        prompt_node_id = str(TEXT_TO_IMAGE_CONFIG["prompt_node_id"])
        workflow_handler.original_workflow[prompt_node_id]["inputs"]["text"] = prompt

        # 设置随机种子
        seed_value = generate_random_seed()
        workflow_handler.set_seed(seed_value)

        # 设置输出文件前缀（统一使用seed_value作为前缀，方便查找）
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

        # 查找输出文件（使用t2i_{seed_value}作为搜索模式）
        search_pattern = f"t2i_{seed_value}"
        output_file = None

        print(f"  正在搜索输出文件，搜索模式: {search_pattern}")
        print(f"  搜索目录: {COMFYUI_OUTPUT_FOLDER}")

        # 等待一小段时间，确保文件已写入磁盘
        import time
        time.sleep(1)

        # 搜索输出文件
        found_files = []
        for root, dirs, files in os.walk(COMFYUI_OUTPUT_FOLDER):
            for file in files:
                if search_pattern in file:
                    full_path = os.path.join(root, file)
                    found_files.append(full_path)
                    output_file = full_path
                    print(f"  找到匹配文件: {full_path}")
            if output_file:
                break

        if output_file and os.path.exists(output_file):
            print(f"  文生图完成: {output_file}")
            return output_file
        else:
            print(f"  未找到输出文件，搜索模式: {search_pattern}")
            # 打印输出文件夹中的文件用于调试
            print(f"  输出文件夹中的文件:")
            file_count = 0
            for root, dirs, files in os.walk(COMFYUI_OUTPUT_FOLDER):
                for file in files:
                    print(f"    {file}")
                    file_count += 1
                    if file_count > 20:  # 最多显示20个文件
                        print(f"    ... (还有更多文件)")
                        break
                if file_count > 20:
                    break

            # 尝试查找最新的文件（如果搜索模式不匹配）
            if not output_file:
                print(f"  尝试查找最新的图像文件...")
                all_files = []
                for root, dirs, files in os.walk(COMFYUI_OUTPUT_FOLDER):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                            full_path = os.path.join(root, file)
                            mtime = os.path.getmtime(full_path)
                            all_files.append((mtime, full_path))

                # 按修改时间排序，获取最新的文件
                all_files.sort(reverse=True, key=lambda x: x[0])

                if all_files:
                    newest_file = all_files[0][1]
                    # 检查是否是最近创建的（1分钟内）
                    if time.time() - all_files[0][0] < 60:
                        print(f"  找到最近创建的文件: {newest_file}")
                        output_file = newest_file

            return output_file

    except Exception as e:
        print(f"  文生图出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def upload_image(client: lark.Client, image_path: str):
    """
    上传单张图片到飞书,返回image_key
    :param client: 飞书客户端
    :param image_path: 图片路径
    :return: image_key字符串,失败返回None
    """
    try:
        import requests
        from requests_toolbelt import MultipartEncoder

        # 使用HTTP上传图片
        # 需要先获取tenant_access_token
        token_url = f"{FEISHU_API_BASE}/auth/v3/tenant_access_token/internal"
        token_data = json.dumps({
            "app_id": FEISHU_APP_ID,
            "app_secret": FEISHU_APP_SECRET
        }).encode('utf-8')

        import urllib.request
        token_req = urllib.request.Request(
            token_url,
            data=token_data,
            headers={'Content-Type': 'application/json'}
        )

        with urllib.request.urlopen(token_req) as response:
            token_response = json.loads(response.read().decode('utf-8'))

        if token_response.get('code') != 0:
            print(f"  获取token失败: {token_response.get('msg')}")
            return None

        token = token_response.get('tenant_access_token')

        # 上传图片
        upload_url = f"{FEISHU_API_BASE}/im/v1/images"

        with open(image_path, 'rb') as image_file:
            form = {
                'image_type': 'message',
                'image': image_file
            }
            multi_form = MultipartEncoder(form)

            headers = {
                'Authorization': f'Bearer {token}',
            }
            headers['Content-Type'] = multi_form.content_type

            response = requests.post(upload_url, headers=headers, data=multi_form)

        if response.status_code != 200:
            print(f"  上传图片失败: HTTP {response.status_code}")
            print(f"  Response: {response.text}")
            return None

        result = response.json()
        if result.get('code') != 0:
            print(f"  上传图片失败: {result.get('msg')}")
            return None

        image_key = result.get('data', {}).get('image_key')
        if not image_key:
            print(f"  未获取到image_key")
            return None

        print(f"  图片上传成功, image_key: {image_key}")
        return image_key

    except Exception as e:
        print(f"  上传图片异常: {e}")
        import traceback
        traceback.print_exc()
        return None


def send_image_message(client: lark.Client, chat_id: str, image_key: str):
    """
    发送单张图片消息
    :param client: 飞书客户端
    :param chat_id: 聊天ID
    :param image_key: 图片key
    :return: 成功返回True,失败返回False
    """
    try:
        content = json.dumps({
            "image_key": image_key
        })

        request_body = lark.im.v1.CreateMessageRequestBody.builder() \
            .receive_id(chat_id) \
            .content(content) \
            .msg_type("image") \
            .build()

        request = lark.im.v1.CreateMessageRequest.builder() \
            .receive_id_type("chat_id") \
            .request_body(request_body) \
            .build()

        response = client.im.v1.message.create(request)

        if response.code == 0:
            print(f"  图片发送成功")
            return True
        else:
            print(f"  图片发送失败: {response}")
            return False

    except Exception as e:
        print(f"  发送图片消息异常: {e}")
        return False


def upload_and_send_image(client: lark.Client, chat_id: str, image_path: str, caption: str = ""):
    """
    上传图片到飞书并发送（保持向后兼容）
    :param client: 飞书客户端
    :param chat_id: 聊天ID
    :param image_path: 图片路径
    :param caption: 图片说明
    """
    try:
        # 先发送文字说明(如果有)
        if caption:
            caption_content = json.dumps({"text": caption})
            send_message(client, chat_id, caption_content, "text")

        # 上传并发送图片
        image_key = upload_image(client, image_path)
        if image_key:
            return send_image_message(client, chat_id, image_key)
        return False

    except Exception as e:
        print(f"  发送图片异常: {e}")
        return False


def call_deepseek(user_message: str, chat_id: str) -> Optional[dict]:
    """
    调用DeepSeek API进行function calling
    :param user_message: 用户消息
    :param chat_id: 聊天ID（用于返回上下文）
    :return: 返回结构化的指令或None
    """
    if not DEEPSEEK_API_KEY:
        print("  DeepSeek API未配置")
        return None

    if not REQUESTS_AVAILABLE:
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
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }

        print(f"  DeepSeek请求参数: tool_choice=auto, temperature=0.3")

        response = requests.post(
            DEEPSEEK_API_URL,
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


def handle_function_call(client: lark.Client, chat_id: str, function_call: dict):
    """
    处理function calling结果
    :param client: 飞书客户端
    :param chat_id: 聊天ID
    :param function_call: function调用信息
    :return: True表示已处理，False表示需要继续
    """
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
                send_message(client, chat_id, reply_content, "text")
            else:
                reply_content = json.dumps({
                    "text": "❌ 切换失败"
                })
                send_message(client, chat_id, reply_content, "text")
            return True

    elif function_name == "text_to_image":
        prompt = arguments.get("prompt", "")
        if prompt:
            print(f"文生图请求: {prompt[:100]}...")

            # 发送处理中消息
            reply_content = json.dumps({
                "text": f"🎨 正在生成图片...\n\n提示词: {prompt}\n\n请稍候..."
            })
            send_message(client, chat_id, reply_content, "text")

            # 使用ComfyUI生成图片
            output_image_path = process_text_to_image(prompt)

            if output_image_path:
                # 上传并发送生成的图片
                print(f"  正在发送生成的图片...")
                upload_and_send_image(client, chat_id, output_image_path, "✅ 图片生成完成！")
            else:
                reply_content = json.dumps({
                    "text": "❌ 图片生成失败\n\n请检查ComfyUI是否正常运行。"
                })
                send_message(client, chat_id, reply_content, "text")

            return True

    return False


# 创建飞书客户端（用于发送消息）
def create_client():
    """创建飞书客户端"""
    return lark.Client.builder().app_id(FEISHU_APP_ID).app_secret(FEISHU_APP_SECRET).build()


# 发送消息到飞书
def send_message(client: lark.Client, chat_id: str, content: str, msg_type: str = "text"):
    """发送消息到飞书"""
    try:
        # 创建消息请求体
        request_body = lark.im.v1.CreateMessageRequestBody.builder() \
            .receive_id(chat_id) \
            .content(content) \
            .msg_type(msg_type) \
            .build()
        
        # 创建消息请求
        request = lark.im.v1.CreateMessageRequest.builder() \
            .receive_id_type("chat_id") \
            .request_body(request_body) \
            .build()
        
        # 发送消息
        response = client.im.v1.message.create(request)
        
        if response.code == 0:
            print(f"  消息发送成功")
            return response.data
        else:
            print(f"  消息发送失败: {response}")
            return None
    
    except Exception as e:
        print(f"  发送消息异常: {e}")
        import traceback
        traceback.print_exc()
        return None


# 用户当前工作流配置
user_workflows = {}  # {chat_id: workflow_name}

# 消息去重: 防止重复处理同一条消息
processed_messages = set()  # 存储已处理的消息ID
message_timestamps = {}  # 存储消息ID和时间戳，用于清理过期记录

# 任务队列相关
task_queue = []  # [(user_id, task_number), ...]
task_queue_lock = threading.Lock()  # 队列操作锁
task_counter = 0  # 任务计数器
task_counter_lock = threading.Lock()  # 计数器操作锁
user_task_numbers = {}  # {user_id: [task_number, ...]} 记录每个用户的任务序号


def get_user_workflow(chat_id: str) -> str:
    """获取用户当前的工作流设置"""
    result = user_workflows.get(chat_id, DEFAULT_WORKFLOW)
    return result


def set_user_workflow(chat_id: str, workflow_name: str) -> bool:
    """设置用户的工作流"""
    if workflow_name in WORKFLOW_CONFIGS:
        user_workflows[chat_id] = workflow_name
        print(f"  用户 {chat_id} 切换到工作流: {workflow_name}")
        return True
    return False


# ========== 任务队列管理函数 ==========
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


def get_all_queue_info() -> dict:
    """
    获取整个队列信息
    :return: {task_number: (position, user_id), ...}
    """
    with task_queue_lock:
        queue_info = {}
        for i, (uid, tnum) in enumerate(task_queue):
            queue_info[tnum] = (i + 1, uid)
        return queue_info


def format_queue_status(user_id: int) -> str:
    """
    格式化用户的队列状态
    :param user_id: 用户ID
    :return: 格式化的状态字符串
    """
    user_tasks = get_user_tasks(user_id)

    if not user_tasks:
        return "📭 您当前没有在处理的任务"

    # 获取总队列信息
    total_count = len(task_queue)
    queue_info = get_all_queue_info()

    # 构建状态信息
    status_lines = [f"📊 当前队列总任务数: {total_count}"]
    status_lines.append("")

    # 显示用户任务
    for task_num in sorted(user_tasks):
        position, _ = queue_info.get(task_num, (0, 0))
        status_lines.append(f"  任务 #{task_num}: 位置 {position}/{total_count}")

    return "\n".join(status_lines)


# 处理接收到的消息事件
def handle_message_event(data, client: lark.Client):
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
        try:
            chat_id = getattr(message, 'chat_id', '')
            content = getattr(message, 'content', '')
            message_id = getattr(message, 'message_id', '')
            message_type = getattr(message, 'msg_type', '') or getattr(message, 'message_type', '')
            sender = getattr(message, 'sender', None)

            # 发送者信息
            if sender:
                sender_id_obj = getattr(sender, 'sender_id', None)
                sender_id = getattr(sender_id_obj, 'open_id', '') if sender_id_obj else ''
            else:
                sender_id = ''

        except Exception as e:
            print(f"  提取消息属性时出错: {e}")
            return

        print(f"Chat ID: {chat_id}")
        print(f"Message ID: {message_id}")
        print(f"Message Type: {message_type}")
        print(f"Content: {str(content)[:200]}...")

        # 如果message_id为空，使用chat_id+content+time的组合作为唯一标识
        if not message_id:
            current_time = int(time.time() // 10)  # 10秒精度
            message_id = f"{chat_id}_{content[:50]}_{current_time}"
            print(f"  使用生成的message_id: {message_id}")

        # 检查是否已处理过该消息
        if message_id in processed_messages:
            print(f"  [去重] 消息已处理过，跳过")
            return

        # 标记消息为已处理
        processed_messages.add(message_id)
        message_timestamps[message_id] = time.time()

        # 清理过期的消息记录（超过10分钟的）
        current_time = time.time()
        expired_messages = [
            msg_id for msg_id, timestamp in message_timestamps.items()
            if current_time - timestamp > 600
        ]
        for msg_id in expired_messages:
            processed_messages.discard(msg_id)
            message_timestamps.pop(msg_id, None)
        if expired_messages:
            print(f"  [清理] 移除了 {len(expired_messages)} 条过期消息记录")

        # 限制去重集合大小，防止内存泄漏
        if len(processed_messages) > 1000:
            # 按时间戳排序，保留最新的500条
            sorted_messages = sorted(message_timestamps.items(), key=lambda x: x[1], reverse=True)
            to_remove = sorted_messages[500:]
            for msg_id, _ in to_remove:
                processed_messages.discard(msg_id)
                message_timestamps.pop(msg_id, None)
            print(f"  [清理] 限制去重集合大小，保留最新的500条")
        
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
        
        # 处理文本消息
        if message_type == "text":
            try:
                content_json = json.loads(content)
                text = content_json.get("text", "").strip()

                print(f"  收到文本消息: {text}")

                # 处理命令
                if text == "/start":
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
                    send_message(client, chat_id, reply_content, "text")
                    print("  已发送欢迎消息")

                elif text == "/help":
                    help_text = """[使用说明]

📱 基础命令：
/start - 启动机器人
/help - 查看帮助

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
- 帮我生成一张美女图
- 画一个风景
- 创建一个动漫角色
- 生成一张中国网红图

💡 使用方法：
1. 图片处理：选择工作流 → 发送图片 → 等待处理结果
2. 文生图：发送图片描述 → 自动生成
3. 切换模式：发送命令或自然语言"""
                    reply_content = json.dumps({"text": help_text})
                    send_message(client, chat_id, reply_content, "text")
                    print("  已发送帮助消息")
                
                elif text in ["/FaceFix", "/BackgroundRemove", "/Qwen_remove", "/BoobsFix", "/Qwen_edit"]:
                    workflow_name = text.replace("/", "")
                    if set_user_workflow(chat_id, workflow_name):
                        reply_content = json.dumps({
                            "text": f"✅ 已切换到工作流: {workflow_name}\n\n现在发送图片给我，我会使用{workflow_name}进行处理。"
                        })
                        send_message(client, chat_id, reply_content, "text")
                    else:
                        reply_content = json.dumps({
                            "text": f"❌ 未知的工作流: {text}"
                        })
                        send_message(client, chat_id, reply_content, "text")

                elif text == "/queue" or text == "/status":
                    # 查看队列状态
                    queue_status = format_queue_status(chat_id)
                    reply_content = json.dumps({"text": queue_status})
                    send_message(client, chat_id, reply_content, "text")
                    print(f"  已发送队列状态")
                
                else:
                    # 普通文本消息，尝试使用自然语言处理
                    print(f"  尝试使用自然语言处理...")

                    # 为文本消息创建一个临时ID用于去重检查（针对同一chat_id的重复文本）
                    text_request_key = f"{chat_id}_{text}"
                    current_request_time = time.time()

                    # 检查是否在5秒内有相同的文本请求（防止重复处理）
                    if hasattr(handle_message_event, '_text_request_cache'):
                        cache = handle_message_event._text_request_cache
                    else:
                        cache = {}
                        handle_message_event._text_request_cache = cache

                    if text_request_key in cache:
                        last_request_time = cache[text_request_key]
                        if current_request_time - last_request_time < 5:
                            print(f"  [去重] 检测到5秒内的重复文本请求，跳过")
                            return

                    # 更新缓存
                    cache[text_request_key] = current_request_time

                    # 清理过期的缓存条目（超过30秒的）
                    expired_keys = [k for k, v in cache.items() if current_request_time - v > 30]
                    for k in expired_keys:
                        del cache[k]

                    # 调用DeepSeek进行function calling
                    deepseek_result = call_deepseek(text, chat_id)

                    if deepseek_result:
                        result_type = deepseek_result.get("type")

                        if result_type == "function_call":
                            # 处理function调用
                            handle_function_call(client, chat_id, deepseek_result)

                        elif result_type == "text":
                            # 返回文本回复
                            reply_content = json.dumps({
                                "text": deepseek_result.get("content", "")
                            })
                            send_message(client, chat_id, reply_content, "text")
                            print(f"  已回复AI消息")
                        else:
                            # 无法识别的结果,返回默认回复
                            reply_content = json.dumps({
                                "text": f"收到消息: {text}\n\n发送 /help 查看使用说明"
                            })
                            send_message(client, chat_id, reply_content, "text")
                    else:
                        # DeepSeek调用失败,返回默认回复
                        reply_content = json.dumps({
                            "text": f"收到消息: {text}\n\n发送 /help 查看使用说明"
                        })
                        send_message(client, chat_id, reply_content, "text")
                        print(f"  已回复: {text[:30]}...")
            
            except json.JSONDecodeError:
                reply_content = json.dumps({"text": "消息格式错误"})
                send_message(client, chat_id, reply_content, "text")
        
        # 处理图片消息
        elif message_type == "image":
            try:
                content_json = json.loads(content)
                image_key = content_json.get("image_key", "")

                print(f"  收到图片消息，image_key: {image_key}")

                # 生成任务序号
                global task_counter
                with task_counter_lock:
                    task_counter += 1
                    current_task_number = task_counter

                # 记录用户任务
                if chat_id not in user_task_numbers:
                    user_task_numbers[chat_id] = []
                user_task_numbers[chat_id].append(current_task_number)

                # 将任务加入队列
                add_task_to_queue(chat_id, current_task_number)

                # 获取队列信息
                position, waiting, total = get_queue_info(chat_id, current_task_number)

                # 发送处理中消息
                reply_content = json.dumps({
                    "text": f"📸 收到图片！\n\n任务序号: #{current_task_number}\n队列位置: {position}/{total}\n前面等待: {waiting} 个任务\n\n正在使用 {get_user_workflow(chat_id)} 处理...\n请稍候..."
                })
                send_message(client, chat_id, reply_content, "text")

                # 下载图片
                print(f"  正在下载图片...")
                temp_image_path = download_feishu_image(client, image_key, message_id)

                if temp_image_path:
                    # 使用ComfyUI处理图片
                    print(f"  正在处理图片...")
                    output_image_path = process_image_with_comfyui(
                        temp_image_path,
                        get_user_workflow(chat_id)
                    )

                    if output_image_path:
                        # 上传并发送原图和处理后的图片
                        print(f"  正在发送图片...")

                        # 先发送文字说明
                        caption_content = json.dumps({"text": f"✅ 处理完成！\n\n任务序号: #{current_task_number}\n\n📎 原图："})
                        send_message(client, chat_id, caption_content, "text")

                        # 上传并发送原图
                        original_image_key = upload_image(client, temp_image_path)
                        if original_image_key:
                            send_image_message(client, chat_id, original_image_key)

                        # 发送处理后的图片说明
                        processed_caption = json.dumps({"text": "🎨 处理后："})
                        send_message(client, chat_id, processed_caption, "text")

                        # 上传并发送处理后的图片
                        upload_and_send_image(client, chat_id, output_image_path)

                        # 清理临时文件
                        try:
                            os.remove(temp_image_path)
                        except:
                            pass

                        # 任务完成，从队列移除
                        remove_task_from_queue(current_task_number)
                    else:
                        reply_content = json.dumps({
                            "text": "❌ 图片处理失败\n\n请检查ComfyUI是否正常运行。"
                        })
                        send_message(client, chat_id, reply_content, "text")
                        # 处理失败，从队列移除
                        remove_task_from_queue(current_task_number)
                else:
                    reply_content = json.dumps({
                        "text": "❌ 下载图片失败"
                    })
                    send_message(client, chat_id, reply_content, "text")
                    # 下载失败，从队列移除
                    remove_task_from_queue(current_task_number)

            except json.JSONDecodeError:
                reply_content = json.dumps({
                    "text": "❌ 图片消息格式错误"
                })
                send_message(client, chat_id, reply_content, "text")
        
        else:
            print(f"  不支持的消息类型: {message_type}")
        
        print(f"====================================\n")
    
    except Exception as e:
        print(f"  处理消息事件时出错: {e}")
        import traceback
        traceback.print_exc()


# 启动长连接监听
def start_long_connection():
    """使用飞书SDK启动长连接监听"""
    print("=" * 60)
    print("飞书机器人 - 长连接监听模式")
    print("=" * 60)
    print(f"应用ID: {FEISHU_APP_ID}")
    print()
    
    # 创建客户端（用于发送消息）
    client = create_client()
    print(f"[OK] 飞书客户端创建成功")
    
    # 创建事件处理器
    print(f"\n正在创建事件处理器...")
    
    # 使用 EventDispatcherHandlerBuilder 创建事件分发器
    builder = EventDispatcherHandlerBuilder(
        encrypt_key="",  # 长连接模式下可以为空
        verification_token=""  # 长连接模式下可以为空
    )
    
    # 注册消息接收事件处理
    builder.register_p2_im_message_receive_v1(
        lambda event: handle_message_event(event.event, client)
    )
    
    event_dispatcher = builder.build()
    print(f"[OK] 事件处理器创建成功")
    
    # 创建长连接客户端
    print(f"\n正在创建长连接客户端...")
    ws_client = Client(
        app_id=FEISHU_APP_ID,
        app_secret=FEISHU_APP_SECRET,
        log_level=LogLevel.DEBUG,  # 设置日志级别为DEBUG，方便调试
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
        # 启动长连接（这会阻塞）
        ws_client.start()
    
    except KeyboardInterrupt:
        print("\n[INFO] 正在停止长连接...")
        print("[INFO] 长连接已停止")
    
    except Exception as e:
        print(f"\n[ERROR] 长连接运行时出错: {e}")
        import traceback
        traceback.print_exc()


# 主函数
def main():
    """主函数"""
    if not SDK_AVAILABLE:
        print("[ERROR] 飞书SDK未安装")
        print("请运行: pip install lark-oapi")
        return
    
    # 检查ComfyUI是否运行
    if URLLIB_AVAILABLE and check_comfyui_server():
        print(f"[OK] ComfyUI服务正常: {COMFYUI_API_URL}")
    else:
        print(f"[WARNING] ComfyUI服务未启动或无法访问")
        print(f"   请确保ComfyUI运行在: {COMFYUI_API_URL}")
        print(f"   图片处理功能将不可用")
        print()
    
    # 启动长连接监听
    start_long_connection()


if __name__ == "__main__":
    main()
