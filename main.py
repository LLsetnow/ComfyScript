import os
import json
import re
import ast
import operator
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
import requests

# 加载 .env 文件中的环境变量
load_dotenv()

# ============================================================
# ReAct 提示词模板（基础版 - Zero-Shot）
# ============================================================
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""

class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    支持工具选择失败的错误追踪和纠错引导。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])

    def listToolNames(self) -> List[str]:
        """返回所有可用工具名称列表"""
        return list(self.tools.keys())

class ErrorRecoveryManager:
    """
    错误恢复管理器。
    追踪工具调用失败模式，并在适当时机提供纠错引导。
    """
    # 错误类型定义
    ERROR_TOOL_NOT_FOUND = "tool_not_found"       # 工具不存在
    ERROR_PARSE_FAILED = "parse_failed"          # Action解析失败
    ERROR_CONSECUTIVE_FAILURES = "consecutive"    # 连续失败
    ERROR_SAME_TOOL_WRONG = "same_tool_wrong"     # 同一工具反复失败

    def __init__(self, max_consecutive_failures: int = 2):
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures = 0
        self.failed_tool_attempts: Dict[str, int] = {}  # 记录每个工具的失败次数
        self.last_failed_tool: str = None
        self.error_history: List[Dict] = []  # 错误历史记录

    def reset(self):
        """重置所有错误状态"""
        self.consecutive_failures = 0
        self.failed_tool_attempts = {}
        self.last_failed_tool = None
        self.error_history = []

    def record_failure(self, error_type: str, tool_name: str = None, details: str = ""):
        """
        记录一次工具调用失败。

        Args:
            error_type: 错误类型
            tool_name: 尝试调用的工具名
            details: 错误详情
        """
        self.consecutive_failures += 1

        if tool_name:
            self.failed_tool_attempts[tool_name] = self.failed_tool_attempts.get(tool_name, 0) + 1
            self.last_failed_tool = tool_name

        self.error_history.append({
            "type": error_type,
            "tool": tool_name,
            "details": details,
            "consecutive": self.consecutive_failures
        })

    def record_success(self):
        """记录一次成功的工具调用"""
        self.consecutive_failures = 0

    def get_guidance(self, available_tools: List[str]) -> str:
        """
        根据错误历史生成纠错引导提示。

        Args:
            available_tools: 可用工具列表

        Returns:
            纠错引导字符串（如果没有错误则返回空字符串）
        """
        if self.consecutive_failures == 0:
            return ""

        # 生成通用的纠错引导
        guidance_parts = []

        # 1. 连续失败警告
        if self.consecutive_failures >= self.max_consecutive_failures:
            guidance_parts.append(
                f"⚠️ 警告: 你已经连续失败了 {self.consecutive_failures} 次。"
            )

            # 2. 检查是否是同一工具反复失败
            if self.last_failed_tool:
                fail_count = self.failed_tool_attempts.get(self.last_failed_tool, 0)
                if fail_count >= 2:
                    guidance_parts.append(
                        f"你多次尝试使用 '{self.last_failed_tool}' 工具但都失败了。"
                        f"这个工具可能不适合当前任务。"
                    )

                # 3. 建议尝试其他工具
                other_tools = [t for t in available_tools if t != self.last_failed_tool]
                if other_tools:
                    guidance_parts.append(
                        f"💡 建议: 考虑使用其他可用工具: {', '.join(other_tools)}"
                    )

            # 4. 提供策略建议
            guidance_parts.append(
                "💡 纠错策略:\n"
                "  - 仔细阅读工具描述，确保选择正确的工具\n"
                "  - 检查工具参数是否符合预期格式\n"
                "  - 如果问题无法用工具解决，尝试用 Finish[你的答案] 直接回答"
            )

        return "\n".join(guidance_parts)

    def should_suggest_give_up(self, max_steps: int, current_step: int) -> bool:
        """
        判断是否应该建议放弃当前策略。

        Args:
            max_steps: 最大步数
            current_step: 当前步数

        Returns:
            是否建议放弃
        """
        return (self.consecutive_failures >= self.max_consecutive_failures and
                current_step >= max_steps - 1)

class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor,
                 max_steps: int = 5, max_consecutive_failures: int = 3):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []
        # 错误恢复管理器
        self.error_manager = ErrorRecoveryManager(max_consecutive_failures=max_consecutive_failures)

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = []  # 每次运行时重置历史记录
        self.error_manager.reset()  # 重置错误状态
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)

            # 1.5 获取纠错引导（如果有）
            guidance = self.error_manager.get_guidance(self.tool_executor.listToolNames())

            # 构建完整提示词
            if guidance:
                full_prompt = f"{REACT_PROMPT_TEMPLATE.format(
                    tools=tools_desc,
                    question=question,
                    history=history_str
                )}\n\n【系统纠错引导】\n{guidance}"
            else:
                full_prompt = REACT_PROMPT_TEMPLATE.format(
                    tools=tools_desc,
                    question=question,
                    history=history_str
                )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": full_prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"🤔 思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                self.error_manager.record_failure(
                    ErrorRecoveryManager.ERROR_PARSE_FAILED,
                    details="无法解析Action"
                )
                continue

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                finish_match = re.match(r"Finish\[(.*)\]", action)
                if finish_match:
                    final_answer = finish_match.group(1)
                    print(f"🎉 最终答案: {final_answer}")
                    self.error_manager.record_success()  # 成功，reset失败计数
                    return final_answer
                else:
                    print(f"⚠️  警告:无法解析Finish指令: {action}")
                    return f"无法解析的Finish指令: {action}"

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                observation = f"错误:无法解析Action格式 '{action}'。请使用格式: 工具名[输入内容]"
                self.history.append(f"Action: {action}")
                self.history.append(f"Observation: {observation}")
                print(f"👀 观察: {observation}")

                # 记录解析失败
                self.error_manager.record_failure(
                    ErrorRecoveryManager.ERROR_PARSE_FAILED,
                    details=action
                )
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")

            # 5. 执行工具并处理错误
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。可用工具: {', '.join(self.tool_executor.listToolNames())}"
                print(f"👀 观察: {observation}")

                # 记录工具不存在错误
                self.error_manager.record_failure(
                    ErrorRecoveryManager.ERROR_TOOL_NOT_FOUND,
                    tool_name=tool_name,
                    details=f"工具 '{tool_name}' 不存在"
                )
            else:
                try:
                    observation = tool_function(tool_input)
                    print(f"👀 观察: {observation}")

                    # 检查工具返回的错误信息
                    if observation and observation.startswith("错误:"):
                        self.error_manager.record_failure(
                            ErrorRecoveryManager.ERROR_SAME_TOOL_WRONG,
                            tool_name=tool_name,
                            details=observation
                        )
                    else:
                        # 工具成功执行
                        self.error_manager.record_success()

                except Exception as e:
                    observation = f"工具执行异常: {str(e)}"
                    print(f"👀 观察: {observation}")
                    self.error_manager.record_failure(
                        ErrorRecoveryManager.ERROR_SAME_TOOL_WRONG,
                        tool_name=tool_name,
                        details=str(e)
                    )

            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

            # 检查是否触发强制 Finish 机制
            if (self.error_manager.consecutive_failures >= self.error_manager.max_consecutive_failures):
                print("\n" + "="*50)
                print(f"⚠️ 检测到连续 {self.error_manager.consecutive_failures} 次工具调用失败")
                print("系统将根据历史观察记录生成答案...")
                print("="*50)

                # 从历史记录中提取最后的有效观察结果
                final_answer = self._extract_answer_from_history()
                if final_answer:
                    print(f"🎉 系统自动Finish: {final_answer}")
                    return final_answer
                else:
                    # 没有有效信息，返回无法回答
                    print(f"🎉 系统自动Finish: 由于工具多次失败，无法获取有效答案")
                    return "由于工具多次失败，无法获取有效答案，请人工介入处理。"

        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _extract_answer_from_history(self) -> str:
        """
        从历史记录中提取最后的有效观察结果作为答案。
        查找包含"计算结果:"、"直接答案:"、"搜索总结:"等成功标记的观察。
        """
        for entry in reversed(self.history):
            if entry.startswith("Observation:"):
                observation = entry.replace("Observation:", "").strip()
                # 跳过错误信息
                if observation.startswith("错误:") or "不存在" in observation:
                    continue
                # 跳过解析错误
                if "无法解析" in observation:
                    continue
                # 跳过系统建议
                if "建议:" in observation or "提示:" in observation:
                    continue
                # 找到有效的观察结果
                if any(keyword in observation for keyword in ["计算结果:", "直接答案:", "搜索总结:", "答案:"]):
                    # 提取答案部分
                    for keyword in ["计算结果:", "直接答案:", "搜索总结:", "答案:"]:
                        if keyword in observation:
                            return observation.split(keyword)[-1].strip()
                # 如果观察本身不是错误，也返回
                if observation and len(observation) > 5:
                    return observation
        return None


def search(query: str) -> str:
    """
    一个基于博查API的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [博查API] 网页搜索: {query}")
    try:
        # 从环境变量获取博查API配置
        api_endpoint = os.getenv("BOC_SEARCH_API_URL", "https://api.bochaai.com/v1/web-search")
        api_key = os.getenv("BOC_SEARCH_API_KEY")
        
        if not api_key or not api_endpoint:
            return "错误: BOC_SEARCH_API_URL 或 BOC_SEARCH_API_KEY 未在 .env 文件中配置。"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            "freshness": "oneYear",
            "summary": True,
            "count": 10
        }
        
        print(f"🌐 连接博查API: {api_endpoint}")
        response = requests.post(
            api_endpoint,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            return f"博查API请求失败，状态码: {response.status_code}, 响应: {response.text[:200]}"
        
        data = response.json()
        
        # 检查API返回状态码 - 博查API使用code=200表示成功
        if "code" in data and data["code"] != 200:
            error_msg = data.get('msg', "未知错误")
            return f"博查API返回错误 (code={data['code']}): {error_msg}"
        
        # 获取实际数据
        response_data = data.get("data", {})
        if not response_data:
            return "博查API返回的数据为空"
        
        # 智能解析:优先寻找最直接的答案
        # 1. 首先检查是否有直接答案（answer字段）
        if "answer" in response_data and response_data["answer"]:
            return f"直接答案:\n{response_data['answer']}"
        
        # 2. 检查是否有总结（summary字段） - 博查API的summary字段通常更完整
        if "summary" in response_data and response_data["summary"]:
            return f"搜索总结:\n{response_data['summary']}"
        
        # 3. 检查博查API的特殊格式：检查第一个结果的summary字段（如果有）
        web_pages = response_data.get("webPages", {})
        if web_pages and "value" in web_pages and web_pages["value"]:
            first_result = web_pages["value"][0] if isinstance(web_pages["value"], list) else {}
            
            # 优先返回更完整的summary字段
            if "summary" in first_result and first_result["summary"]:
                return f"知识摘要:\n{first_result['summary']}"
            
            # 如果没有summary，再返回snippet
            if "snippet" in first_result and first_result["snippet"]:
                return f"知识摘要:\n{first_result['snippet']}"
        
        # 4. 返回有机搜索结果（前三个）
        if "webPages" in response_data and response_data["webPages"]:
            web_pages = response_data["webPages"]
            if "value" in web_pages and web_pages["value"] and isinstance(web_pages["value"], list):
                # 返回前三个有机结果的摘要
                snippets = []
                for i, res in enumerate(web_pages["value"][:3]):
                    title = res.get("name", res.get("title", f"结果 {i+1}"))
                    snippet = res.get("snippet", res.get("description", ""))
                    if title or snippet:
                        snippets.append(f"[{i+1}] {title}\n{snippet}")
                
                if snippets:
                    return "\n\n".join(snippets)
        
        # 5. 尝试其他可能的格式
        for key in ["organic_results", "results", "items"]:
            if key in response_data and response_data[key]:
                items = response_data[key]
                if isinstance(items, list) and items:
                    snippets = []
                    for i, res in enumerate(items[:3]):
                        title = res.get("title", res.get("name", f"结果 {i+1}"))
                        snippet = res.get("snippet", res.get("description", ""))
                        if title or snippet:
                            snippets.append(f"[{i+1}] {title}\n{snippet}")
                    
                    if snippets:
                        return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except requests.exceptions.RequestException as e:
        return f"网络请求错误: {str(e)}"
    except json.JSONDecodeError as e:
        return f"解析API响应失败: {str(e)}"
    except Exception as e:
        return f"搜索时发生错误: {e}"


class SafeCalculator:
    """
    安全计算器类，用于解析和计算数学表达式。
    仅支持基本数学运算，防止恶意代码执行。
    """
    # 支持的运算符映射
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }

    @classmethod
    def _eval_node(cls, node):
        """递归评估 AST 节点"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            op_type = type(node.op)
            if op_type in cls.operators:
                return cls.operators[op_type](left, right)
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        elif isinstance(node, ast.UnaryOp):
            operand = cls._eval_node(node.operand)
            op_type = type(node.op)
            if op_type in cls.operators:
                return cls.operators[op_type](operand)
            raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
        elif isinstance(node, ast.Expression):
            return cls._eval_node(node.body)
        elif isinstance(node, ast.Call):
            # 处理 pow() 函数
            if isinstance(node.func, ast.Name) and node.func.id == 'pow':
                args = [cls._eval_node(arg) for arg in node.args]
                if len(args) == 2:
                    return operator.pow(args[0], args[1])
            raise ValueError(f"不支持的函数调用: {getattr(node.func, 'id', str(node.func))}")
        else:
            raise ValueError(f"不支持的表达式节点: {type(node).__name__}")

    @classmethod
    def calculate(cls, expression: str) -> float:
        """
        安全地计算数学表达式。
        
        Args:
            expression: 数学表达式字符串，如 "(123 + 456) * 789 / 12"
            
        Returns:
            计算结果（浮点数）
        """
        try:
            # 预处理表达式
            expr = expression.strip()
            # 将常见的中文运算符和括号转换为英文
            expr = expr.replace('×', '*').replace('÷', '/').replace('－', '-').replace('＋', '+')
            expr = expr.replace('（', '(').replace('）', ')')  # 中文括号
            # 移除常见文字
            expr = re.sub(r'[=？?]', '', expr)
            
            # 解析为 AST
            tree = ast.parse(expr, mode='eval')
            # 计算并返回结果
            result = cls._eval_node(tree)
            return result
        except ZeroDivisionError:
            raise ValueError("错误: 除数不能为零")
        except Exception as e:
            raise ValueError(f"表达式解析错误: {str(e)}")


def calculate(expression: str) -> str:
    """
    计算器工具函数。
    用于执行复杂的数学计算，支持加减乘除、括号、指数等运算。
    
    使用示例:
    - 输入: "(123 + 456) * 789 / 12"
    - 输出: "计算结果: 37957.5"
    """
    print(f"🧮 正在执行数学计算: {expression}")
    try:
        result = SafeCalculator.calculate(expression)
        # 格式化输出
        if isinstance(result, float):
            # 避免浮点数精度问题
            if result.is_integer():
                return f"计算结果: {int(result)}"
            # 保留合理精度
            formatted = f"{result:.10f}".rstrip('0').rstrip('.')
            return f"计算结果: {formatted}"
        return f"计算结果: {result}"
    except ValueError as e:
        return f"计算错误: {str(e)}"
    except Exception as e:
        return f"未知错误: {str(e)}"


# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)

    # 3. 注册计算器工具
    calc_description = "一个数学计算器。用于执行复杂的数学计算，支持加减乘除(+、-、*、/)、乘方(^)、括号等运算。输入格式应为数学表达式。"
    toolExecutor.registerTool("Calculator", calc_description, calculate)
    
    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())
    
    # 5. 测试 ReActAgent
    print("\n" + "="*60)
    print("🧪 测试 ReActAgent")
    print("="*60)
    
    try:
        # 初始化LLM客户端
        llm_client = HelloAgentsLLM()
        
        # 创建并运行ReActAgent
        agent = ReActAgent(llm_client=llm_client, tool_executor=toolExecutor, max_steps=5)
        
        # 测试1: 数学计算问题
        print("\n📝 测试案例1: 数学计算")
        question1 = "计算 (123 + 456) × 789 / 12 = ? 的结果"
        print(f"问题: {question1}")
        result1 = agent.run(question1)
        if result1:
            print(f"✅ ReActAgent返回结果: {result1[:200]}..." if len(result1) > 200 else f"✅ ReActAgent返回结果: {result1}")
        else:
            print("❌ ReActAgent未能返回结果")
        
        # 测试2: 简单问题测试
        # print("\n📝 测试案例2: 简单搜索问题")
        # question2 = "Python是什么？"
        # print(f"问题: {question2}")
        
        # # 重置agent的历史记录
        # agent.history = []
        
        # result2 = agent.run(question2)
        # if result2:
        #     print(f"✅ ReActAgent返回结果: {result2[:200]}..." if len(result2) > 200 else f"✅ ReActAgent返回结果: {result2}")
        # else:
        #     print("❌ ReActAgent未能返回结果")
        
    except Exception as e:
        print(f"❌ ReActAgent测试失败: {e}")
        print("提示: 请确保.env文件中有正确的LLM配置（API密钥、模型ID、Base URL）")
    