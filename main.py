"""
飞书 Agent 机器人
使用 ReAct Agent 作为智能问答助手
"""
import os
import re
import sys
import time
import json
import logging
import threading
import signal
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# 日志配置
# ============================================================================

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, f"bot_{time.strftime('%Y%m%d')}.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# 工具函数
# ============================================================================

def strip_markdown(text: str) -> str:
    """去除 Markdown 格式"""
    if not text:
        return text
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = re.sub(r'^[\-\*\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ============================================================================
# 消息去重
# ============================================================================

class MessageDeduplicator:
    """消息去重器，防止同一消息被并发处理或重复处理"""

    def __init__(self, max_processed: int = 1000):
        self._processing = set()
        self._processed = set()
        self._max_processed = max_processed
        self._lock = threading.Lock()

    def try_acquire(self, message_id: str) -> bool:
        """尝试获取消息处理权，返回是否成功"""
        with self._lock:
            if message_id in self._processing:
                logger.info(f"[跳过] 消息正在处理中: {message_id}")
                return False
            if message_id in self._processed:
                logger.info(f"[跳过] 消息已处理过: {message_id}")
                return False
            self._processing.add(message_id)
            return True

    def release(self, message_id: str):
        """标记消息处理完成"""
        with self._lock:
            self._processing.discard(message_id)
            self._processed.add(message_id)
            if len(self._processed) > self._max_processed:
                excess = len(self._processed) - self._max_processed // 2
                for _ in range(excess):
                    self._processed.pop()

    def discard(self, message_id: str):
        """移除处理中标记（不标记为已处理，用于跳过的消息）"""
        with self._lock:
            self._processing.discard(message_id)


# ============================================================================
# 消息解析
# ============================================================================

@dataclass
class ParsedMessage:
    """解析后的飞书消息"""
    chat_id: str
    content: str
    message_id: str
    message_type: str
    sender_id: str
    sender_type: str


def parse_message_event(data) -> Optional[ParsedMessage]:
    """从飞书事件数据中解析消息信息"""
    # 提取 message 对象
    if hasattr(data, 'message'):
        message = data.message
    elif isinstance(data, dict):
        message = data.get("message", {})
    else:
        logger.warning(f"无法解析消息数据类型: {type(data)}")
        return None

    logger.info(f"解析后的消息: {message}")

    # 提取基本字段
    chat_id = getattr(message, 'chat_id', '')
    content = getattr(message, 'content', '')
    message_id = getattr(message, 'message_id', '')
    message_type = getattr(message, 'msg_type', '') or getattr(message, 'message_type', '')

    logger.info(f"chat_id={chat_id}, content={content}, message_id={message_id}, message_type={message_type}")

    # 提取发送者信息（sender 在 data 对象上，不在 message 上）
    sender = getattr(data, 'sender', None) or getattr(message, 'sender', None)
    sender_id = ''
    sender_type = ''
    if sender:
        sender_id_obj = getattr(sender, 'sender_id', None)
        if sender_id_obj:
            sender_id = getattr(sender_id_obj, 'open_id', '') or getattr(sender_id_obj, 'user_id', '')
        sender_type = getattr(sender, 'sender_type', '')

    # 如果 message_type 为空，尝试从 content 推断
    if not message_type and content:
        try:
            content_json = json.loads(content)
            if 'text' in content_json:
                message_type = 'text'
            elif 'image_key' in content_json:
                message_type = 'image'
            logger.info(f"推断的消息类型: {message_type}")
        except Exception:
            pass

    return ParsedMessage(
        chat_id=chat_id,
        content=content,
        message_id=message_id,
        message_type=message_type,
        sender_id=sender_id,
        sender_type=sender_type,
    )


# ============================================================================
# 飞书机器人主类
# ============================================================================

class FeishuBot:
    """飞书 Agent 机器人，整合初始化、消息处理和启动逻辑"""

    CANCEL_KEYWORDS = {"不需要", "不用", "不要", "否", "no", "No", "NO", "取消"}

    def __init__(self):
        self.deduplicator = MessageDeduplicator()
        self.feishu_client = None
        self.agent = None
        self.comfyui_client = None
        self.image_processor = None
        self.ws_client = None

    # ---- 初始化 ----

    def init_all(self):
        """执行所有初始化步骤"""
        self._init_sdk()
        self._init_agent()
        self._init_feishu_client()
        self._init_comfyui()

    def _init_sdk(self):
        """加载飞书 SDK"""
        global lark, Client, EventDispatcherHandlerBuilder
        try:
            import lark_oapi as lark
            from lark_oapi.ws.client import Client
            from lark_oapi.event.dispatcher_handler import EventDispatcherHandlerBuilder
            logger.info("[OK] 飞书SDK已加载")
        except ImportError as e:
            logger.error(f"[ERROR] 未安装 lark-oapi SDK: {e}")
            sys.exit(1)

    def _init_agent(self):
        """初始化 Agent 和工具"""
        from Agent import (
            ReActAgent, HelloAgentsLLM, ToolExecutor,
            search, calculate, get_current_time,
            comfyui_text_to_image, comfyui_check_server, comfyui_edit_image,
            feishu_create_doc, feishu_write_doc, comfyui_context,
        )
        from Agent import comfyui_context  # 保持引用

        logger.info("\n--- 初始化 Agent ---")

        # LLM
        try:
            llm_client = HelloAgentsLLM()
            logger.info(f"[OK] LLM 客户端初始化成功: {llm_client.model}")
        except ValueError as e:
            logger.error(f"[ERROR] LLM 配置缺失: {e}")
            sys.exit(1)

        # 工具
        tool_executor = ToolExecutor()
        tools = [
            ("Search", "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具", search),
            ("Calculator", "一个数学计算器。用于执行复杂的数学计算，支持加减乘除(+、-、*、/)、乘方(^)、括号等运算。输入格式应为数学表达式。", calculate),
            ("GetCurrentTime", "获取当前日期和时间。当需要知道当前时间、日期，或需要判断信息的时效性（如\"今天\"、\"最新\"、\"最近\"等）时，应先调用此工具获取当前时间。输入可选时区偏移，如'+8'表示东八区，默认为东八区(北京时间)。", get_current_time),
            ("TextToImage", "使用ComfyUI进行文生图（文字生成图片）。当用户要求生成图片、画图、创作图像时使用此工具。输入应为图像的详细描述/提示词，如\"一只可爱的猫咪\"、\"夕阳下的海滩\"等。生成的图片将自动发送到聊天中。", comfyui_text_to_image),
            ("CheckComfyUI", "检查ComfyUI服务器是否正在运行。当需要确认图像生成服务是否可用时，应先调用此工具。无需输入参数。", comfyui_check_server),
            ("EditImage", "使用ComfyUI对用户发送的图片进行编辑。当用户发送了图片并要求对图片进行修改/编辑时使用此工具。输入应为编辑提示词，如\"给人物加上墨镜\"、\"把背景换成海滩\"等。注意：只有当用户已发送图片且需要编辑时才调用此工具。", comfyui_edit_image),
            ("CreateDoc", "创建飞书云文档。当用户要求创建文档、记录笔记、撰写报告、写备忘录等场景时使用此工具。输入格式为：标题|正文内容（标题和正文用|分隔，正文可选）。例如：\"会议纪要|今天讨论了项目进度\"或\"学习笔记\"。", feishu_create_doc),
            ("WriteDoc", "向飞书云文档中写入/追加内容。当用户要求往某个已有文档中写入内容、补充笔记、添加段落时使用此工具。输入格式为：文档链接|要写入的内容。例如：\"https://bytedance.larkoffice.com/docx/abc123|这是新增的内容\"。", feishu_write_doc),
        ]
        for name, desc, func in tools:
            tool_executor.registerTool(name, desc, func)

        # Agent
        self.agent = ReActAgent(
            llm_client=llm_client,
            tool_executor=tool_executor,
            max_steps=8,
            max_consecutive_failures=3,
        )
        logger.info("[OK] Agent 初始化完成")
        logger.info("\n--- 可用工具 ---")
        logger.info(tool_executor.getAvailableTools())

        # 保存 comfyui_context 引用
        self._comfyui_context = comfyui_context

    def _init_feishu_client(self):
        """初始化飞书客户端"""
        logger.info("\n--- 初始化飞书客户端 ---")

        from feishu_client import FeishuClient

        app_id = os.getenv("FEISHU_APP_ID", "")
        app_secret = os.getenv("FEISHU_APP_SECRET", "")
        api_base = os.getenv("FEISHU_API_BASE", "https://open.feishu.cn/open-apis")

        if not app_id or not app_secret:
            logger.error("[ERROR] 请在 .env 文件中配置 FEISHU_APP_ID 和 FEISHU_APP_SECRET")
            sys.exit(1)

        self.feishu_client = FeishuClient(
            app_id=app_id,
            app_secret=app_secret,
            api_base=api_base,
        )

        sdk_client = lark.Client.builder() \
            .app_id(app_id) \
            .app_secret(app_secret) \
            .build()
        self.feishu_client.set_client(sdk_client)

        logger.info("[OK] 飞书客户端初始化完成")

    def _init_comfyui(self):
        """初始化 ComfyUI 客户端"""
        logger.info("\n--- 初始化 ComfyUI 客户端 ---")

        try:
            from Comfyui import ComfyUIClient, ImageProcessor
            self.comfyui_client = ComfyUIClient()
            self.image_processor = ImageProcessor(self.comfyui_client)

            self._comfyui_context.set(
                feishu_client=self.feishu_client,
                comfyui_client=self.comfyui_client,
                image_processor=self.image_processor,
            )

            if self.comfyui_client.check_server(max_attempts=1, check_delay=0):
                logger.info("[OK] ComfyUI 服务器已运行")
            else:
                # 内网不通，尝试公网地址
                logger.info("内网地址不可达，尝试通过 ngrok 公网地址连接...")
                public_url = self._get_ngrok_url()
                if public_url:
                    self.comfyui_client.api_url = public_url
                    # ImageProcessor 通过 self.client 引用 ComfyUIClient，无需额外设置
                    if self.comfyui_client.check_server(max_attempts=2, check_delay=2):
                        logger.info(f"[OK] 通过公网地址连接 ComfyUI: {public_url}")
                    else:
                        logger.warning("[警告] 公网地址也不可达，文生图功能暂不可用")
                else:
                    logger.warning("[警告] ComfyUI 服务器未运行且无法获取公网地址，文生图功能暂不可用")
        except Exception as e:
            logger.warning(f"[警告] ComfyUI 初始化失败（文生图功能不可用）: {e}")
            self.comfyui_client = None
            self.image_processor = None

    # 默认 ngrok 公网地址（当本地 API 不可用时使用）
    DEFAULT_NGROK_URL = "https://candi-sporogonial-eliz.ngrok-free.dev"

    @classmethod
    def _get_ngrok_url(cls) -> str:
        """通过 ngrok 本地 API 获取公网地址，失败时返回默认地址"""
        try:
            import urllib.request
            with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=3) as resp:
                data = json.loads(resp.read().decode())
                for tunnel in data.get("tunnels", []):
                    url = tunnel.get("public_url", "")
                    if url.startswith("https://"):
                        return url
        except Exception:
            pass
        return cls.DEFAULT_NGROK_URL

    # ---- 消息处理 ----

    def handle_message_event(self, data):
        """处理接收到的消息事件"""
        try:
            msg = parse_message_event(data)
            if not msg:
                return

            # 消息去重
            if not self.deduplicator.try_acquire(msg.message_id):
                return

            # 过滤机器人自身消息
            if msg.sender_type and msg.sender_type.lower() == 'bot':
                logger.info(f"[跳过] 机器人自身消息: {msg.message_id}")
                self.deduplicator.discard(msg.message_id)
                return

            try:
                # 设置上下文
                self._comfyui_context.chat_id = msg.chat_id
                self._comfyui_context.sender_id = msg.sender_id

                logger.info(f"========== 收到新消息 ==========")
                logger.info(f"消息ID: {msg.message_id}")
                logger.info(f"聊天ID: {msg.chat_id}")
                logger.info(f"发送者: {msg.sender_id}")
                logger.info(f"消息类型: {msg.message_type}")
                logger.info(f"原始内容: {msg.content}")

                # 分发处理
                if msg.message_type == 'image':
                    self._handle_image_message(msg)
                elif msg.message_type == 'text':
                    self._handle_text_message(msg)
                else:
                    logger.info(f"[跳过] 不支持的消息类型: {msg.message_type}")

            finally:
                self.deduplicator.release(msg.message_id)
                self._comfyui_context.chat_id = None
                self._comfyui_context.sender_id = None

        except Exception as e:
            logger.error(f"[ERROR] 处理消息异常: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _handle_image_message(self, msg: ParsedMessage):
        """处理图片消息"""
        try:
            content_json = json.loads(msg.content)
            image_key = content_json.get("image_key", "")
        except Exception:
            image_key = ""

        if not image_key:
            logger.info("[跳过] 无法提取 image_key")
            return

        logger.info(f"收到图片消息, image_key: {image_key}")

        # 如果已有待编辑图片，提示用户
        if self._comfyui_context.pending_image_path:
            self.feishu_client.send_text(
                msg.chat_id,
                "⚠️ 您已发送了一张图片，正在等待您输入编辑提示词。\n\n"
                "请输入提示词（如：给人物加上墨镜）来描述您想要的修改。\n\n"
                "发送\"不需要\"可取消编辑。"
            )
            return

        # 下载图片
        from Comfyui import config as comfyui_config
        temp_image_path = self.feishu_client.download_image(
            image_key, msg.message_id, comfyui_config.input_folder
        )

        if temp_image_path:
            logger.info(f"图片已下载: {temp_image_path}")
            self._comfyui_context.pending_image_path = temp_image_path
            self.feishu_client.send_text(
                msg.chat_id,
                "📸 已收到图片！\n\n请问您是否要对这张图片进行编辑？\n"
                "如果需要编辑，请告诉我您想要进行的修改（如：给人物加上墨镜、把背景换成海滩等）。\n"
                "如果不需要编辑，请回复\"不需要\"。"
            )
        else:
            self.feishu_client.send_text(msg.chat_id, "❌ 下载图片失败，请重新发送。")

    def _handle_text_message(self, msg: ParsedMessage):
        """处理文本消息"""
        # 提取文本内容
        try:
            content_json = json.loads(msg.content)
            user_text = content_json.get("text", "").strip()
        except Exception as e:
            logger.error(f"[跳过] 无法解析消息内容: {e}, content={msg.content}")
            return

        if not user_text:
            logger.info("[跳过] 消息内容为空")
            return

        logger.info(f"用户消息: {user_text}")

        # 检查是否有待编辑的图片
        if self._comfyui_context.pending_image_path:
            self._handle_edit_request(msg.chat_id, user_text)
        else:
            self._handle_normal_message(msg.chat_id, user_text)

    def _handle_edit_request(self, chat_id: str, user_text: str):
        """处理图像编辑请求"""
        # 用户取消编辑
        if user_text in self.CANCEL_KEYWORDS:
            old_path = self._comfyui_context.pending_image_path
            self._comfyui_context.pending_image_path = None
            try:
                os.remove(old_path)
            except Exception:
                pass
            self.feishu_client.send_text(chat_id, "好的，已取消图片编辑。如果需要其他帮助，请随时告诉我！")
            return

        # 执行编辑
        logger.info("--- Agent 正在处理图像编辑请求 ---")
        self.feishu_client.send_text(chat_id, f"🖌️ 收到！正在根据您的需求「{user_text}」编辑图片，请稍候...")

        answer = self._run_agent(
            f"用户已发送了一张图片，图片已保存在服务器上。请直接使用EditImage工具对这张图片进行编辑，"
            f"编辑提示词为：{user_text}。注意：图片已经存在，无需请求用户发送图片。"
        )

        # 发送回复（编辑成功时只发图片，不发文字）
        if answer and answer.strip() == "__EDIT_IMAGE_SUCCESS__":
            logger.info("--- 图像编辑成功，图片已发送 ---")
        elif answer:
            self._send_reply(chat_id, answer)
        else:
            self.feishu_client.send_text(chat_id, "抱歉，图像编辑失败。")

        # 清除待编辑图片状态
        self._comfyui_context.pending_image_path = None

    def _handle_normal_message(self, chat_id: str, user_text: str):
        """处理普通文本消息"""
        logger.info("--- Agent 正在思考... ---")
        answer = self._run_agent(user_text)
        logger.info(f"--- Agent 回答完成, answer={answer[:50] if answer else 'None'}... ---")
        self._send_reply(chat_id, answer)

    def _run_agent(self, prompt: str) -> Optional[str]:
        """运行 Agent 并返回结果"""
        try:
            return self.agent.run(prompt)
        except Exception as e:
            logger.error(f"Agent 执行异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _send_reply(self, chat_id: str, answer: Optional[str]):
        """发送 Agent 回复给用户"""
        if not answer:
            logger.warning("[发送] Agent 返回空，发送默认回复")
            self.feishu_client.send_text(chat_id, "抱歉，我无法回答这个问题。")
            return

        answer = strip_markdown(answer)

        # 飞书消息有长度限制，分段发送
        max_length = 4000
        if len(answer) > max_length:
            for i in range(0, len(answer), max_length):
                segment = answer[i:i + max_length]
                self.feishu_client.send_text(chat_id, segment)
                if i + max_length < len(answer):
                    time.sleep(0.5)
        else:
            self.feishu_client.send_text(chat_id, answer)

        logger.info(f"[发送] 回复已发送: {answer[:100]}...")

    # ---- 启动 ----

    def start(self):
        """启动飞书机器人"""
        logger.info("\n" + "=" * 50)
        logger.info("启动飞书 Agent 机器人")
        logger.info("=" * 50)

        # 创建事件处理器
        logger.info("\n正在创建事件处理器...")
        builder = EventDispatcherHandlerBuilder(
            encrypt_key="",
            verification_token="",
        )
        builder.register_p2_im_message_receive_v1(
            lambda event: self.handle_message_event(event.event)
        )
        event_dispatcher = builder.build()
        logger.info("[OK] 事件处理器创建成功")

        # 创建长连接客户端
        logger.info("\n正在创建长连接客户端...")
        self.ws_client = Client(
            app_id=os.getenv("FEISHU_APP_ID", ""),
            app_secret=os.getenv("FEISHU_APP_SECRET", ""),
            event_handler=event_dispatcher,
            log_level=lark.LogLevel.INFO,
        )
        self.feishu_client.set_ws_client(self.ws_client)
        logger.info("[OK] 长连接客户端创建成功")

        logger.info(f"\n{'=' * 50}")
        logger.info("飞书 Agent 机器人已启动！")
        logger.info("等待接收消息...")
        logger.info(f"{'=' * 50}\n")

        # 在子线程中启动长连接
        stop_event = threading.Event()

        def run_ws():
            try:
                self.ws_client.start()
            except Exception as e:
                logger.error(f"WebSocket 客户端异常: {e}")

        ws_thread = threading.Thread(target=run_ws, daemon=True)
        ws_thread.start()

        # 主线程等待停止信号
        def signal_handler(sig, frame):
            logger.info("\n\n收到停止信号，正在停止机器人...")
            stop_event.set()
            self._stop_ws()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while not stop_event.is_set():
                stop_event.wait(1)
        except KeyboardInterrupt:
            signal_handler(None, None)

    def _stop_ws(self):
        """停止 WebSocket 连接"""
        try:
            if self.ws_client:
                self.ws_client.stop()
        except Exception:
            pass


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    bot = FeishuBot()
    try:
        bot.init_all()
        bot.start()
    except KeyboardInterrupt:
        logger.info("\n\n正在停止机器人...")
        bot._stop_ws()
        logger.info("机器人已停止")
    except Exception as e:
        logger.error(f"[ERROR] 启动失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
