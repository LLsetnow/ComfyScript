# ComfyUI 飞书机器人

基于飞书 SDK WebSocket 长连接实现的图像处理机器人，集成了 ComfyUI 工作流自动化处理能力。

## 功能特性

### 核心功能

- **图像处理自动化**：支持多种 ComfyUI 工作流，一键处理图片
- **长连接实时响应**：使用飞书 SDK WebSocket 长连接，实时接收和处理消息
- **智能任务队列**：多用户并发处理，自动排队管理
- **自然语言交互**：集成 DeepSeek AI，支持自然语言命令控制
- **自动启动 ComfyUI**：检测并自动启动 ComfyUI 服务器

### 工作流支持

| 工作流 | 命令 | 功能说明 |
|--------|------|----------|
| 面部重绘 | `/FaceFix` | 修复面部瑕疵，提升面部质量 |
| 去除背景 | `/BackgroundRemove` | 自动去除图片背景杂物 |
| 图像编辑 | `/Qwen_edit` | 根据文本提示词编辑图片 |

### 图像编辑模式（两步式）

1. 切换到 `/Qwen_edit` 模式
2. 发送待编辑的图片
3. 输入编辑提示词（如：`给人物加上墨镜`）
4. 等待 AI 自动处理并返回结果

## 快速开始

### 环境要求

- Python 3.8+
- ComfyUI（AKI 版本或标准版）
- 飞书企业自建应用

### 安装依赖

```bash
pip install lark-oapi requests deepseek
```

### 配置说明

编辑 `config.json5` 配置文件：

```json5
{
    // 飞书配置
    "feishu": {
        "app_id": "你的应用ID",
        "app_secret": "你的应用密钥"
    },

    // ComfyUI 配置
    "comfyUI": {
        "folder": "D:\\AI_Graph\\ConfyUI-aki\\ComfyUI-aki-v1",
        "python_exe": "D:\\AI_Graph\\ConfyUI-aki\\ComfyUI-aki-v1\\python\\python.exe",
        "main_py": "D:\\AI_Graph\\ConfyUI-aki\\ComfyUI-aki-v1\\main.py",
        "host": "127.0.0.1",
        "port": "8188"
    },

    // DeepSeek API（可选）
    "deepseek": {
        "api_key": "你的API密钥"
    }
}
```

### 启动机器人

**方式一：使用 bat 脚本**
```bash
FeiShuBotV8_启动.bat
```

**方式二：手动启动**
```bash
conda activate sam2Mask
python FeiShuBotComfyuiV8_Refactored.py
```

### ComfyUI 自动启动

如果 ComfyUI 服务器未运行，机器人会自动检测并尝试启动：
- 自动在新终端窗口启动 ComfyUI
- 最多等待 150 秒等待启动完成
- 程序退出时自动停止 ComfyUI（仅限自动启动的进程）

## 使用指南

### 基础命令

| 命令 | 功能 |
|------|------|
| `/start` | 启动机器人，显示欢迎消息 |
| `/help` | 显示详细帮助信息 |
| `/queue` | 查看当前任务队列状态 |
| `/status` | 查看任务队列状态（同 /queue） |
| `/cancel` | 取消当前正在进行的编辑任务 |

### 工作流切换

**方式一：使用命令**
```
/FaceFix          # 切换到面部重绘
/BackgroundRemove # 切换到去除背景
/Qwen_edit        # 切换到图像编辑
```

**方式二：自然语言切换**
```
"切换到面部重绘"
"改成去除背景"
```

### 文生图功能

直接发送描述即可生成图片：
```
帮我生成一张美女图
画一个风景
创建一个动漫角色
生成一张中国网红图
```

## 项目架构

```
FeiShuBotComfyuiV8_Refactored.py
├── 配置管理模块 (ConfigManager)
├── 消息去重管理器 (MessageDeduplicator)
├── 任务队列管理 (TaskQueue)
├── ComfyUI 工作流处理器 (ComfyUIWorkflow)
├── 图像处理器 (ImageProcessor)
├── 飞书 API 交互 (FeishuAPI)
├── 消息发送器 (FeishuMessenger)
├── 自然语言处理器 (NLProcessor - DeepSeek)
└── 主消息处理器 (MessageHandler)
```

### 核心模块

- **MessageDeduplicator**：防止重复处理同一条消息
- **TaskQueue**：管理多用户任务队列，支持查看排队位置
- **ComfyUIWorkflow**：封装工作流加载和参数设置
- **ImageProcessor**：统一处理各种图像操作
- **MessageHandler**：核心消息处理逻辑，支持命令和自然语言

## 文件说明

| 文件 | 说明 |
|------|------|
| `FeiShuBotComfyuiV8_Refactored.py` | 飞书机器人主程序（V8 重构版） |
| `TeleBotComfyuiV3.py` | Telegram 机器人版本 |
| `config.json5` | 配置文件（含敏感信息，请勿上传） |
| `*.json` | ComfyUI 工作流文件 |

## 工作流配置

工作流配置在 `config.json5` 中：

```json5
"workflows": {
    "FaceFix": {
        "seed_id": 9,
        "input_image_id": 27,
        "output_image_id": 72,
        "workflow": "FaceFix.json",
        "points_cost": 1
    },
    "Qwen_edit": {
        "seed_id": 65,
        "input_image_id": 41,
        "output_image_id": 181,
        "workflow": "Qwen_edit.json",
        "prompt_node_id": 68,
        "points_cost": 2
    }
}
```

- `seed_id`：随机种子节点 ID
- `input_image_id`：输入图像节点 ID
- `output_image_id`：输出图像节点 ID
- `prompt_node_id`：提示词节点 ID（仅图像编辑需要）
- `points_cost`：积分消耗

## 注意事项

1. **配置文件安全**：`config.json5` 包含敏感信息，已加入 `.gitignore`
2. **ComfyUI 端口**：确保 ComfyUI 运行在配置的端口（默认 8188）
3. **飞书应用配置**：需要在飞书开放平台开启「使用长连接接收事件」
4. **API 配额**：注意 DeepSeek API 的调用配额限制

## License

MIT License
