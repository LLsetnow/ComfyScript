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
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# 日志配置
# ============================================================================

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, f"bot_{time.strftime('%Y%m%d')}.log")

# 创建日志目录
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志
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

# 导入飞书 SDK
try:
    import lark_oapi as lark
    from lark_oapi.ws.client import Client
    from lark_oapi.event.dispatcher_handler import EventDispatcherHandlerBuilder
    SDK_AVAILABLE = True
    logger.info("[OK] 飞书SDK已加载")
except ImportError as e:
    SDK_AVAILABLE = False
    logger.error(f"[ERROR] 未安装 lark-oapi SDK: {e}")
    sys.exit(1)

# 导入飞书客户端
from feishu_client import FeishuClient, FeishuMessenger

# 导入 Agent
from Agent import ReActAgent, HelloAgentsLLM, ToolExecutor, search, calculate, get_current_time


# ============================================================================
# 配置
# ============================================================================

FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
FEISHU_API_BASE = os.getenv("FEISHU_API_BASE", "https://open.feishu.cn/open-apis")

if not FEISHU_APP_ID or not FEISHU_APP_SECRET:
    logger.error("[ERROR] 请在 .env 文件中配置 FEISHU_APP_ID 和 FEISHU_APP_SECRET")
    sys.exit(1)


# ============================================================================
# 工具函数
# ============================================================================

def strip_markdown(text: str) -> str:
    """
    去除 Markdown 格式
    """
    if not text:
        return text
    
    # 去除标题符号 (# ## ### 等)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # 去除加粗 (**text** 或 __text__)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    
    # 去除斜体 (*text* 或 _text_)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    
    # 去除行内代码 (`code`)
    text = re.sub(r'`(.+?)`', r'\1', text)
    
    # 去除代码块 (```...```)
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # 去除链接 [text](url) -> text
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    
    # 去除列表符号 (- * + 和数字.)
    text = re.sub(r'^[\-\*\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # 去除引用符号 (>)
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    
    # 去除水平线
    text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
    
    # 清理多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


# 处理中的消息ID，防止同一条消息被并发处理
_processing_messages = set()
_processing_lock = threading.Lock()


# ============================================================================
# 初始化 Agent
# ============================================================================

logger.info("\n--- 初始化 Agent ---")

# 初始化 LLM 客户端
try:
    llm_client = HelloAgentsLLM()
    logger.info(f"[OK] LLM 客户端初始化成功: {llm_client.model}")
except ValueError as e:
    logger.error(f"[ERROR] LLM 配置缺失: {e}")
    logger.error("请在 .env 中配置: LLM_MODEL_ID, LLM_API_KEY, LLM_BASE_URL")
    sys.exit(1)

# 初始化工具执行器
tool_executor = ToolExecutor()

# 注册工具
tool_executor.registerTool(
    "Search",
    "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具",
    search
)
tool_executor.registerTool(
    "Calculator",
    "一个数学计算器。用于执行复杂的数学计算，支持加减乘除(+、-、*、/)、乘方(^)、括号等运算。输入格式应为数学表达式。",
    calculate
)
tool_executor.registerTool(
    "GetCurrentTime",
    "获取当前日期和时间。当需要知道当前时间、日期，或需要判断信息的时效性（如\"今天\"、\"最新\"、\"最近\"等）时，应先调用此工具获取当前时间。输入可选时区偏移，如'+8'表示东八区，默认为东八区(北京时间)。",
    get_current_time
)

# 初始化 ReAct Agent
agent = ReActAgent(
    llm_client=llm_client,
    tool_executor=tool_executor,
    max_steps=8,
    max_consecutive_failures=3
)

logger.info(f"[OK] Agent 初始化完成")
logger.info("\n--- 可用工具 ---")
logger.info(tool_executor.getAvailableTools())


# ============================================================================
# 飞书客户端初始化
# ============================================================================

logger.info("\n--- 初始化飞书客户端 ---")

# 创建飞书客户端
feishu_client = FeishuClient(
    app_id=FEISHU_APP_ID,
    app_secret=FEISHU_APP_SECRET,
    api_base=FEISHU_API_BASE
)

# 创建 lark SDK 客户端
client = lark.Client.builder() \
    .app_id(FEISHU_APP_ID) \
    .app_secret(FEISHU_APP_SECRET) \
    .build()

# 设置客户端
feishu_client.set_client(client)

logger.info("[OK] 飞书客户端初始化完成")


# ============================================================================
# 消息处理
# ============================================================================

def handle_message_event(data):
    """处理接收到的消息事件"""
    try:
        # 解析消息数据
        logger.info(f"收到原始数据: {data}")
        
        if hasattr(data, 'message'):
            message = data.message
        elif isinstance(data, dict):
            message = data.get("message", {})
        else:
            logger.warning(f"无法解析消息数据类型: {type(data)}")
            return

        logger.info(f"解析后的消息: {message}")

        # 提取消息信息
        chat_id = getattr(message, 'chat_id', '')
        content = getattr(message, 'content', '')
        message_id = getattr(message, 'message_id', '')
        message_type = getattr(message, 'msg_type', '') or getattr(message, 'message_type', '')

        logger.info(f"chat_id={chat_id}, content={content}, message_id={message_id}, message_type={message_type}")

        # 获取发送者信息
        sender = getattr(message, 'sender', None)
        sender_id = ''
        sender_type = ''
        if sender:
            sender_id_obj = getattr(sender, 'sender_id', None)
            if sender_id_obj:
                sender_id = getattr(sender_id_obj, 'open_id', '') or getattr(sender_id_obj, 'user_id', '')
            sender_type = getattr(sender, 'sender_type', '')

        # 检查是否正在处理同一消息
        with _processing_lock:
            if message_id in _processing_messages:
                logger.info(f"[跳过] 消息正在处理中: {message_id}")
                return
            _processing_messages.add(message_id)

        # 过滤机器人自身发送的消息，避免循环
        if sender_type and sender_type.lower() == 'bot':
            logger.info(f"[跳过] 机器人自身消息: {message_id}")
            with _processing_lock:
                _processing_messages.discard(message_id)
            return

        try:
            logger.info(f"========== 收到新消息 ==========")
            logger.info(f"消息ID: {message_id}")
            logger.info(f"聊天ID: {chat_id}")
            logger.info(f"发送者: {sender_id}")
            logger.info(f"消息类型: {message_type}")
            logger.info(f"原始内容: {content}")

            # 只处理文本消息
            if message_type != 'text':
                logger.info(f"[跳过] 仅支持文本消息")
                return

            # 提取文本内容
            try:
                content_json = json.loads(content)
                user_text = content_json.get("text", "").strip()
            except Exception as e:
                logger.error(f"[跳过] 无法解析消息内容: {e}, content={content}")
                return

            if not user_text:
                logger.info(f"[跳过] 消息内容为空")
                return

            logger.info(f"用户消息: {user_text}")

            # 使用 Agent 处理消息
            logger.info("--- Agent 正在思考... ---")
            try:
                answer = agent.run(user_text)
            except Exception as e:
                logger.error(f"Agent 执行异常: {e}")
                import traceback
                logger.error(traceback.format_exc())
                answer = None
            
            logger.info(f"--- Agent 回答完成, answer={answer[:50] if answer else 'None'}... ---")

            # 发送回复
            if answer:
                # 去除 Markdown 格式
                answer = strip_markdown(answer)

                # 飞书消息有长度限制，分段发送
                max_length = 4000
                if len(answer) > max_length:
                    # 分段发送
                    for i in range(0, len(answer), max_length):
                        segment = answer[i:i + max_length]
                        feishu_client.send_text(chat_id, segment)
                        if i + max_length < len(answer):
                            time.sleep(0.5)  # 避免发送过快
                else:
                    feishu_client.send_text(chat_id, answer)
                logger.info(f"[发送] 回复已发送: {answer[:100]}...")
            else:
                logger.warning(f"[发送] Agent 返回空，发送默认回复")
                feishu_client.send_text(chat_id, "抱歉，我无法回答这个问题。")

        finally:
            # 处理完成后移除消息ID
            with _processing_lock:
                _processing_messages.discard(message_id)

    except Exception as e:
        logger.error(f"[ERROR] 处理消息异常: {e}")
        import traceback
        logger.error(traceback.format_exc())


# ============================================================================
# 启动长连接
# ============================================================================

logger.info("\n" + "=" * 50)
logger.info("启动飞书 Agent 机器人")
logger.info("=" * 50)

try:
    # 创建事件处理器
    logger.info(f"\n正在创建事件处理器...")

    builder = EventDispatcherHandlerBuilder(
        encrypt_key="",
        verification_token=""
    )

    # 注册消息接收事件
    builder.register_p2_im_message_receive_v1(
        lambda event: handle_message_event(event.event)
    )

    event_dispatcher = builder.build()
    logger.info(f"[OK] 事件处理器创建成功")

    # 创建长连接客户端
    logger.info(f"\n正在创建长连接客户端...")

    ws_client = Client(
        app_id=FEISHU_APP_ID,
        app_secret=FEISHU_APP_SECRET,
        event_handler=event_dispatcher,
        log_level=lark.LogLevel.INFO
    )

    # 设置客户端到飞书客户端
    feishu_client.set_ws_client(ws_client)

    logger.info(f"[OK] 长连接客户端创建成功")
    logger.info(f"\n{'=' * 50}")
    logger.info("飞书 Agent 机器人已启动！")
    logger.info("等待接收消息...")
    logger.info(f"{'=' * 50}\n")

    # 在子线程中启动长连接，主线程监听 Ctrl+C
    import threading
    import signal

    stop_event = threading.Event()

    def run_ws_client():
        try:
            ws_client.start()
        except Exception as e:
            logger.error(f"WebSocket 客户端异常: {e}")

    ws_thread = threading.Thread(target=run_ws_client, daemon=True)
    ws_thread.start()

    # 主线程等待停止信号
    def signal_handler(sig, frame):
        logger.info("\n\n收到停止信号，正在停止机器人...")
        stop_event.set()
        try:
            if feishu_client._ws_client:
                feishu_client._ws_client.stop()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 主线程阻塞等待
    try:
        while not stop_event.is_set():
            stop_event.wait(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

except KeyboardInterrupt:
    logger.info("\n\n正在停止机器人...")
    if feishu_client._ws_client:
        feishu_client._ws_client.stop()
    logger.info("机器人已停止")
except Exception as e:
    logger.error(f"[ERROR] 启动失败: {e}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)
