"""
飞书客户端模块
统一管理所有飞书相关功能：消息收发、图片上传下载、长连接管理
"""
import json
import time
import os
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# 飞书客户端 - 统一管理类
# ============================================================================

class FeishuClient:
    """
    飞书客户端 - 统一管理所有飞书相关功能
    包含：消息收发、图片上传下载、长连接管理
    """
    
    def __init__(self, app_id: str, app_secret: str, api_base: str = "https://open.feishu.cn/open-apis"):
        self.app_id = app_id
        self.app_secret = app_secret
        self.api_base = api_base
        self._client = None
        self._ws_client = None
        self._token_cache = {"token": None, "expires_at": 0}
    
    # ==================== 客户端设置 ====================
    
    def set_client(self, client):
        """设置飞书SDK客户端"""
        self._client = client
    
    def set_ws_client(self, ws_client):
        """设置WebSocket客户端"""
        self._ws_client = ws_client
    
    # ==================== Token管理 ====================
    
    def _get_tenant_access_token(self) -> Optional[str]:
        """获取tenant_access_token（带缓存）"""
        current_time = time.time()
        
        # 检查缓存是否有效
        if (self._token_cache["token"] and 
            self._token_cache["expires_at"] > current_time + 60):
            return self._token_cache["token"]
        
        try:
            import requests
        except ImportError:
            print("[FeishuClient] requests库未安装")
            return None
        
        token_url = f"{self.api_base}/auth/v3/tenant_access_token/internal"
        token_data = json.dumps({
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }).encode('utf-8')
        
        try:
            from urllib import request
            token_req = request.Request(
                token_url,
                data=token_data,
                headers={'Content-Type': 'application/json'}
            )
            
            with request.urlopen(token_req, timeout=10) as response:
                token_response = json.loads(response.read().decode('utf-8'))
            
            if token_response.get('code') != 0:
                print(f"[FeishuClient] 获取token失败: {token_response.get('msg')}")
                return None
            
            # 缓存token
            self._token_cache["token"] = token_response.get('tenant_access_token')
            # 飞书token有效期2小时
            self._token_cache["expires_at"] = current_time + 7200
            
            return self._token_cache["token"]
            
        except Exception as e:
            print(f"[FeishuClient] 获取token异常: {e}")
            return None
    
    # ==================== 消息发送 ====================
    
    def send_text(self, chat_id: str, text: str) -> bool:
        """
        发送文本消息
        
        Args:
            chat_id: 聊天ID
            text: 文本内容
        
        Returns:
            bool: 发送是否成功
        """
        content = json.dumps({"text": text})
        return self.send_message(chat_id, content, "text")
    
    def send_message(self, chat_id: str, content: str, msg_type: str = "text") -> bool:
        """
        发送消息到飞书
        
        Args:
            chat_id: 聊天ID
            content: 消息内容（JSON字符串格式）
            msg_type: 消息类型 (text/image/post)
        
        Returns:
            bool: 发送是否成功
        """
        if not self._client:
            print("[FeishuClient] 客户端未初始化")
            return False
        
        try:
            import lark_oapi as lark
            request_body = lark.im.v1.CreateMessageRequestBody.builder() \
                .receive_id(chat_id) \
                .content(content) \
                .msg_type(msg_type) \
                .build()
            
            request = lark.im.v1.CreateMessageRequest.builder() \
                .receive_id_type("chat_id") \
                .request_body(request_body) \
                .build()
            
            response = self._client.im.v1.message.create(request)
            
            if response.code == 0:
                print(f"[FeishuClient] 消息发送成功")
                return True
            else:
                print(f"[FeishuClient] 消息发送失败: {response}")
                return False
                
        except Exception as e:
            print(f"[FeishuClient] 发送消息异常: {e}")
            return False
    
    def send_image(self, chat_id: str, image_key: str) -> bool:
        """
        发送图片消息
        
        Args:
            chat_id: 聊天ID
            image_key: 飞书图片key
        
        Returns:
            bool: 发送是否成功
        """
        content = json.dumps({"image_key": image_key})
        return self.send_message(chat_id, content, "image")
    
    # ==================== 图片上传下载 ====================
    
    def upload_image(self, image_path: str) -> Optional[str]:
        """
        上传图片到飞书，返回image_key
        
        Args:
            image_path: 图片文件路径
        
        Returns:
            str: image_key，失败返回None
        """
        try:
            import requests
            from requests_toolbelt import MultipartEncoder
        except ImportError:
            print("[FeishuClient] requests或requests_toolbelt库未安装")
            return None
        
        token = self._get_tenant_access_token()
        if not token:
            return None
        
        upload_url = f"{self.api_base}/im/v1/images"
        
        try:
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
                
                response = requests.post(upload_url, headers=headers, data=multi_form, timeout=30)
            
            if response.status_code != 200:
                print(f"[FeishuClient] 上传图片失败: HTTP {response.status_code}")
                return None
            
            result = response.json()
            if result.get('code') != 0:
                print(f"[FeishuClient] 上传图片失败: {result.get('msg')}")
                return None
            
            image_key = result.get('data', {}).get('image_key')
            if not image_key:
                print(f"[FeishuClient] 未获取到image_key")
                return None
            
            print(f"[FeishuClient] 图片上传成功, image_key: {image_key}")
            return image_key
            
        except Exception as e:
            print(f"[FeishuClient] 上传图片异常: {e}")
            return None
    
    def download_image(self, image_key: str, message_id: str, save_folder: str) -> Optional[str]:
        """
        从飞书下载图片
        
        Args:
            image_key: 飞书图片key
            message_id: 消息ID
            save_folder: 保存文件夹路径
        
        Returns:
            str: 保存的文件路径，失败返回None
        """
        try:
            import requests
        except ImportError:
            print("[FeishuClient] requests库未安装")
            return None
        
        token = self._get_tenant_access_token()
        if not token:
            return None
        
        resource_url = f"{self.api_base}/im/v1/messages/{message_id}/resources/{image_key}?type=image"
        
        print(f"[FeishuClient] 正在下载图片: {image_key}")
        
        try:
            response = requests.get(
                resource_url,
                headers={'Authorization': f'Bearer {token}'},
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"[FeishuClient] 下载图片失败: HTTP {response.status_code}")
                return None
            
            image_data = response.content
            
            # 保存到临时文件
            temp_filename = f"temp_{int(time.time())}_{image_key[:8]}.jpg"
            os.makedirs(save_folder, exist_ok=True)
            temp_path = os.path.join(save_folder, temp_filename)
            
            with open(temp_path, 'wb') as f:
                f.write(image_data)
            
            print(f"[FeishuClient] 图片已下载: {temp_path} ({len(image_data)} bytes)")
            return temp_path
            
        except Exception as e:
            print(f"[FeishuClient] 下载图片异常: {e}")
            return None
    
    def send_image_with_caption(self, chat_id: str, image_path: str, caption: str = "") -> bool:
        """
        上传并发送图片（带文字说明）
        
        Args:
            chat_id: 聊天ID
            image_path: 图片路径
            caption: 图片说明文字
        
        Returns:
            bool: 发送是否成功
        """
        try:
            # 先发送文字说明(如果有)
            if caption:
                self.send_text(chat_id, caption)
            
            # 上传并发送图片
            image_key = self.upload_image(image_path)
            if image_key:
                return self.send_image(chat_id, image_key)
            return False
            
        except Exception as e:
            print(f"[FeishuClient] 发送图片异常: {e}")
            return False
    
    # ==================== 消息解析 ====================
    
    @staticmethod
    def parse_message(data) -> Optional[Dict]:
        """
        解析消息事件数据
        
        Args:
            data: 原始消息数据
        
        Returns:
            Dict: 解析后的消息信息，包含 chat_id, content, message_id, message_type
        """
        try:
            # 解析消息数据
            if hasattr(data, 'message'):
                message = data.message
            elif isinstance(data, dict):
                message = data.get("message", {})
            else:
                return None
            
            # 提取消息信息
            result = {
                "chat_id": getattr(message, 'chat_id', ''),
                "content": getattr(message, 'content', ''),
                "message_id": getattr(message, 'message_id', ''),
                "message_type": getattr(message, 'msg_type', '') or getattr(message, 'message_type', '')
            }
            
            # 如果 message_type 为空，尝试从 content 推断
            if not result["message_type"] and result["content"]:
                try:
                    content_json = json.loads(result["content"])
                    if 'text' in content_json:
                        result["message_type"] = 'text'
                    elif 'image_key' in content_json:
                        result["message_type"] = 'image'
                except:
                    pass
            
            return result
            
        except Exception as e:
            print(f"[FeishuClient] 解析消息异常: {e}")
            return None
    
    @staticmethod
    def extract_text(content: str) -> Optional[str]:
        """
        从消息content中提取文本
        
        Args:
            content: 消息content字段
        
        Returns:
            str: 提取的文本内容
        """
        try:
            content_json = json.loads(content)
            return content_json.get("text", "").strip()
        except:
            return None
    
    @staticmethod
    def extract_image_key(content: str) -> Optional[str]:
        """
        从消息content中提取image_key
        
        Args:
            content: 消息content字段
        
        Returns:
            str: 图片key
        """
        try:
            content_json = json.loads(content)
            return content_json.get("image_key", "")
        except:
            return None

# ============================================================================
# 飞书消息发送模块（保留向后兼容）
# ============================================================================

class FeishuMessenger:
    """飞书消息发送器（兼容旧代码）"""

    def __init__(self, client, save_folder: str = None):
        self.client = client
        self._save_folder = save_folder or os.path.join(os.path.dirname(__file__), "temp")
        # 创建统一的FeishuClient（从 .env 读取配置）
        self._feishu_client = FeishuClient(
            os.getenv("FEISHU_APP_ID", ""),
            os.getenv("FEISHU_APP_SECRET", ""),
            os.getenv("FEISHU_API_BASE", "https://open.feishu.cn/open-apis")
        )
        self._feishu_client.set_client(client)

    def send_message(self, chat_id: str, content: str,
                    msg_type: str = "text") -> Optional[Dict]:
        """发送消息到飞书"""
        if msg_type == "text":
            success = self._feishu_client.send_text(chat_id, json.loads(content).get("text", ""))
        else:
            success = self._feishu_client.send_message(chat_id, content, msg_type)
        return {"success": success} if success else None

    def send_image_message(self, chat_id: str, image_key: str) -> bool:
        """发送单张图片消息"""
        return self._feishu_client.send_image(chat_id, image_key)

    def upload_and_send_image(self, chat_id: str, image_path: str,
                             caption: str = "") -> bool:
        """上传图片到飞书并发送"""
        return self._feishu_client.send_image_with_caption(chat_id, image_path, caption)

# ============================================================================
# 飞书API交互模块（保留向后兼容，内部使用FeishuClient）
# ============================================================================

class FeishuAPI:
    """飞书API交互类（兼容旧代码）"""

    @staticmethod
    def get_tenant_access_token() -> Optional[str]:
        """获取tenant_access_token"""
        client = FeishuClient(
            os.getenv("FEISHU_APP_ID", ""),
            os.getenv("FEISHU_APP_SECRET", ""),
            os.getenv("FEISHU_API_BASE", "https://open.feishu.cn/open-apis")
        )
        return client._get_tenant_access_token()

    @staticmethod
    def download_image(image_key: str, message_id: str, save_folder: str = None) -> Optional[str]:
        """从飞书下载图片并返回保存路径"""
        if save_folder is None:
            save_folder = os.getenv("COMFYUI_INPUT_FOLDER", "/tmp")
        client = FeishuClient(
            os.getenv("FEISHU_APP_ID", ""),
            os.getenv("FEISHU_APP_SECRET", ""),
            os.getenv("FEISHU_API_BASE", "https://open.feishu.cn/open-apis")
        )
        return client.download_image(image_key, message_id, save_folder)

    @staticmethod
    def upload_image(image_path: str) -> Optional[str]:
        """上传单张图片到飞书,返回image_key"""
        client = FeishuClient(
            os.getenv("FEISHU_APP_ID", ""),
            os.getenv("FEISHU_APP_SECRET", ""),
            os.getenv("FEISHU_API_BASE", "https://open.feishu.cn/open-apis")
        )
        return client.upload_image(image_path)
