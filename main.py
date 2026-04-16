"""
飞书 Agent 机器人
使用 ReAct Agent 作为智能问答助手
"""
import os
import sys
import time
import json
import threading
from dotenv import load_dotenv

load_dotenv()

# 导入飞书 SDK
try:
    import lark_oapi as lark
    from lark_oapi.ws.client import Client
    from lark_oapi.event.dispatcher_handler import EventDispatcherHandlerBuilder
    SDK_AVAILABLE = True
    print("[OK] 飞书SDK已加载")
except ImportError as e:
    SDK_AVAILABLE = False
    print(f"[ERROR] 未安装 lark-oapi SDK: {e}")
    sys.exit(1)

# 导入飞书客户端
from feishu_client import FeishuClient, FeishuMessenger

# 导入 Agent
from Agent import ReActAgent, HelloAgentsLLM, ToolExecutor, search, calculate


# ============================================================================
# 配置
# ============================================================================

FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
FEISHU_API_BASE = os.getenv("FEISHU_API_BASE", "https://open.feishu.cn/open-apis")

if not FEISHU_APP_ID or not FEISHU_APP_SECRET:
    print("[ERROR] 请在 .env 文件中配置 FEISHU_APP_ID 和 FEISHU_APP_SECRET")
    sys.exit(1)


# ============================================================================
# 初始化 Agent
# ============================================================================

print("\n--- 初始化 Agent ---")

# 初始化 LLM 客户端
try:
    llm_client = HelloAgentsLLM()
    print(f"[OK] LLM 客户端初始化成功: {llm_client.model}")
except ValueError as e:
    print(f"[ERROR] LLM 配置缺失: {e}")
    print("请在 .env 中配置: LLM_MODEL_ID, LLM_API_KEY, LLM_BASE_URL")
    sys.exit(1)

# 初始化工具执行器
tool_executor = ToolExecutor()

# 注册工具
tool_executor.registerTool(
    "Search",
    "一个网页搜索引擎。搜索时应直接使用用户的原始问题作为关键词，不要添加额外的时间限制（如年份、月份等）。",
    search
)
tool_executor.registerTool(
    "Calculator",
    "一个数学计算器。用于执行复杂的数学计算，支持加减乘除(+、-、*、/)、乘方(^)、括号等运算。输入格式应为数学表达式。",
    calculate
)

# 初始化 ReAct Agent
agent = ReActAgent(
    llm_client=llm_client,
    tool_executor=tool_executor,
    max_steps=5,
    max_consecutive_failures=3
)

print(f"[OK] Agent 初始化完成")
print("\n--- 可用工具 ---")
print(tool_executor.getAvailableTools())


# ============================================================================
# 飞书客户端初始化
# ============================================================================

print("\n--- 初始化飞书客户端 ---")

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

print("[OK] 飞书客户端初始化完成")


# ============================================================================
# 消息处理
# ============================================================================

# 消息去重缓存
processed_messages = {}
deduplicator_lock = threading.Lock()


def is_duplicate(message_id: str) -> bool:
    """检查消息是否重复"""
    if not message_id:
        return False
    
    with deduplicator_lock:
        current_time = time.time()
        
        # 清理 5 分钟前的记录
        expired = [mid for mid, t in processed_messages.items() 
                   if current_time - t > 300]
        for mid in expired:
            del processed_messages[mid]
        
        if message_id in processed_messages:
            return True
        
        processed_messages[message_id] = current_time
        return False


def handle_message_event(data):
    """处理接收到的消息事件"""
    try:
        # 解析消息数据
        if hasattr(data, 'message'):
            message = data.message
        elif isinstance(data, dict):
            message = data.get("message", {})
        else:
            return

        # 提取消息信息
        chat_id = getattr(message, 'chat_id', '')
        content = getattr(message, 'content', '')
        message_id = getattr(message, 'message_id', '')
        message_type = getattr(message, 'msg_type', '') or getattr(message, 'message_type', '')

        # 获取发送者信息
        sender = getattr(message, 'sender', None)
        sender_id = ''
        if sender:
            sender_id_obj = getattr(sender, 'sender_id', None)
            if sender_id_obj:
                sender_id = getattr(sender_id_obj, 'open_id', '') or getattr(sender_id_obj, 'user_id', '')

        # 消息去重
        if is_duplicate(message_id):
            print(f"[去重] 跳过重复消息: {message_id}")
            return

        print(f"\n========== 收到新消息 ==========")
        print(f"消息ID: {message_id}")
        print(f"聊天ID: {chat_id}")
        print(f"发送者: {sender_id}")
        print(f"消息类型: {message_type}")
        print(f"原始内容: {content}")

        # 只处理文本消息
        if message_type != 'text':
            print(f"[跳过] 仅支持文本消息")
            return

        # 提取文本内容
        try:
            content_json = json.loads(content)
            user_text = content_json.get("text", "").strip()
        except:
            print(f"[跳过] 无法解析消息内容")
            return

        if not user_text:
            print(f"[跳过] 消息内容为空")
            return

        print(f"用户消息: {user_text}")

        # 使用 Agent 处理消息
        print("\n--- Agent 正在思考... ---")
        answer = agent.run(user_text)
        print(f"--- Agent 回答完成 ---\n")

        # 发送回复
        if answer:
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
            print(f"[发送] 回复已发送: {answer[:100]}...")
        else:
            feishu_client.send_text(chat_id, "抱歉，我无法回答这个问题。")
            print(f"[发送] 发送默认回复")

    except Exception as e:
        print(f"[ERROR] 处理消息异常: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 启动长连接
# ============================================================================

print("\n" + "=" * 50)
print("启动飞书 Agent 机器人")
print("=" * 50)

try:
    # 创建事件处理器
    print(f"\n正在创建事件处理器...")

    builder = EventDispatcherHandlerBuilder(
        encrypt_key="",
        verification_token=""
    )

    # 注册消息接收事件
    builder.register_p2_im_message_receive_v1(
        lambda event: handle_message_event(event.event)
    )

    event_dispatcher = builder.build()
    print(f"[OK] 事件处理器创建成功")

    # 创建长连接客户端
    print(f"\n正在创建长连接客户端...")

    ws_client = Client(
        app_id=FEISHU_APP_ID,
        app_secret=FEISHU_APP_SECRET,
        event_handler=event_dispatcher,
        log_level=lark.LogLevel.INFO
    )

    # 设置客户端到飞书客户端
    feishu_client.set_ws_client(ws_client)

    print(f"[OK] 长连接客户端创建成功")
    print(f"\n{'=' * 50}")
    print("飞书 Agent 机器人已启动！")
    print("等待接收消息...")
    print(f"{'=' * 50}\n")

    # 启动长连接（阻塞）
    ws_client.start()

except KeyboardInterrupt:
    print("\n\n正在停止机器人...")
    if feishu_client._ws_client:
        feishu_client._ws_client.stop()
    print("机器人已停止")
except Exception as e:
    print(f"[ERROR] 启动失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
