# 自动化脚本

## 工作流脚本

### 前提准备
- 打开开发者模式
- 将对应工作流导出为 API 格式

### 脚本配置
- **Comfyui自动化脚本.py**
  - 运行前提前打开 ComfyUI
  - 配置输入图像路径
    - 默认输入文件夹为：`D:\桌面中转\input`
  - 配置输出图像文件名
    - 默认输出文件夹为：`D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1\output\AutoOutput`
  - 默认每处理一张图等待 45 秒
  - 节点标号修改
    - 根据节点右上角标志确定编号；或者看工作流 json 文件
    - 第一个是加载图像节点，第二个是保存图像节点

### 脚本运行
1. 默认处理 `D:\桌面中转\input` 内的图像
2. 将一张或多张图像拖至该文件

## 脚本打包
使用 `build_exe.py` 打包 `Comfyui单图处理.py` 与 `Qwen 单图编辑.json` 合成 `dist\Comfyui单图处理.exe`
只需将待处理图像拖至该 exe 文件即可运行

## 功能说明

### Comfyui单图处理.py
- **工作流**：`Qwen 单图编辑.json`
- **模型**：`qwen_image_edit_2511_fp8mixed`, `z_image_turbo_bf16`
- **解释**：对图像先用 qwen 进行 remove，再用 ZIT 进行 fix

### QwenRemove&fixV1.py
- **工作流**：`Qwen_remove.json`, `remove_fix.json`
- **模型**：`qwnImageEdit_v16Fp8Scaled`, `z_image_turbo_bf16`
- **解释**：将所有图像 remove 完毕，再进行 fix

### QwenRemove&fixV2.py
- **工作流**：`Qwen_remove.json`, `remove_fix.json`
- **模型**：`qwnImageEdit_v16Fp8Scaled`, `z_image_turbo_bf16`
- **解释**：在 v1 的基础上会查询任务进度，等待上一个工作流执行完毕，再提交下一个任务

### QwenRemoveV1.py
- **工作流**：`Qwen_remove.json`
- **模型**：`qwnImageEdit_v16Fp8Scaled`
- **解释**：将所有图像 remove，不进行 fix

### QwenRemoveV2.py
- **工作流**：`Qwen_remove.json`
- **模型**：`qwnImageEdit_v16Fp8Scaled`
- **解释**：在 v1 的基础上会查询任务进度，等待上一个工作流执行完毕，再提交下一个任务

### QwenRemoveParaV1.py
- **工作流**：`Qwen_remove.json`
- **模型**：`qwnImageEdit_v16Fp8Scaled`
- **解释**：在 `QwenRemoveV2.py` 的基础上，改为参数调用形式
- **使用方法**：`python QwenRemoveParaV1.py image.jpg`

### QwenRemoveParaV2.py
- **工作流**：`Qwen_remove.json`, `remove_fix.json`
- **模型**：`qwnImageEdit_v16Fp8Scaled`, `z_image_turbo_bf16`
- **解释**：参数调用版本，支持 remove 和 fix 两个步骤

### QwenClassify&RemoveV1.py
- **工作流**：`ImageClassify.json`, `Qwen_remove.json`, `boobs_fix.json`
- **模型**：`qwnImageEdit_v16Fp8Scaled`, `moodyPornMix_zitV7.safetensors`
- **解释**：先将图像分类，分为 boobs、normal。对于 boobs 图像执行 `boobs_fix.json`；对于 normal 图像执行 `Qwen_remove.json`

### Video2Image4Remove.py
- **功能**：将视频转换为图像帧，支持批量处理
- **解释**：
  - 重命名视频为 15 位随机 ID
  - 调整分辨率为 720x1080
  - 设置帧率为 16fps
  - 保留音频
  - 提取视频帧到输入文件夹
- **使用方法**：将视频文件拖至脚本或运行 `python Video2Image4Remove.py video.mp4`

### QwenRemoveV1ForEXE.py
- **工作流**：`Qwen_remove.json`
- **模型**：`qwnImageEdit_v16Fp8Scaled`
- **解释**：用于打包成 exe 的版本，支持拖拽处理

### QwenRemoveV1ForSingle.py
- **工作流**：`Qwen_remove.json`
- **模型**：`qwnImageEdit_v16Fp8Scaled`
- **解释**：单图处理版本，优化处理流程

## Telegram Bot 脚本

### TeleBot.py
- **功能**：Telegram 机器人基础版本
- **解释**：分离 Telegram 机器人控制和工作流提交部分
- **使用方法**：`python TeleBot.py`
- **说明**：会调用对应的工作流文件，如 `QwenRemoveParaV1.py`

### QwenRemove_TeleBot.py
- **功能**：集成 Telegram 机器人
- **解释**：
  - 在 `QwenRemoveV2.py` 的基础上，调用 Telegram 机器人
  - 监听 `AUTHORIZED_USER_IDS` 列表内的 id 对应的用户消息
  - 如果消息为图像，则执行 `QwenRemoveV2.py` 并返回结果

### TeleBotComfyui.py
- **功能**：Telegram 机器人完整版本
- **解释**：集成工作流处理，支持多种图像处理模式

### TeleBotComfyuiV2.py
- **功能**：Telegram 机器人 V2 版本
- **新功能**：
  - 支持用户积分系统
  - 支持密钥兑换
  - 支持管理员权限
  - 支持多工作流切换
  - 支持任务队列管理

### TeleBotComfyuiV2 copy.py
- **说明**：TeleBotComfyuiV2.py 的备份副本

### TeleBotComfyuiV3.py
- **功能**：Telegram 机器人 V3 版本
- **新功能**：
  - 集成 DeepSeek AI 助手，支持自然语言对话
  - 支持文生图功能
  - 支持图像编辑功能
  - 支持批量图片处理（3秒内收集多张图片后统一处理）
  - 支持任务查询命令 `/task`
  - 改进的用户界面和提示信息
  - 多种工作流模式：面部重绘、去除背景杂物、服装移除、胸部重绘、图像编辑
- **使用方法**：`python TeleBotComfyuiV3.py`

## 工具脚本

### build_exe.py
- **功能**：将 Python 脚本打包为 exe 文件
- **使用方法**：`python build_exe.py`
- **说明**：使用 PyInstaller 打包，需要提前安装 `pyinstaller`

### generate_keys.py
- **功能**：生成用户密钥
- **输出**：生成 10 个 16 位密钥，格式为 JSON
- **使用方法**：`python generate_keys.py`

## 相关链接
