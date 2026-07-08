"""对话标题和快捷提示词生成服务"""
import os
from typing import Optional, List
from utils.logger import logger
from utils.llm_client import get_openai_client


class TitleGenerator:
    """使用小模型生成对话标题和快捷提示词"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        初始化标题生成器
        
        Args:
            model_name: 模型名称，默认从 LLM_MODEL 获取
        """
        self._client = None
        self._model_name = model_name
        self.timeout = 30.0
        logger.info(f"标题生成器初始化 - 模型: {self.model_name}")

    @property
    def client(self):
        return get_openai_client()

    @property
    def model_name(self):
        if self._model_name is None or isinstance(self._model_name, str) and self._model_name == os.getenv("LLM_MODEL", "mimo-v2.5"):
            self._model_name = self._model_name or os.getenv("LLM_MODEL", "mimo-v2.5")
        return self._model_name
    
    def generate_conversation_title(self, messages: List[dict]) -> str:
        """
        根据对话消息生成对话标题
        
        Args:
            messages: 对话消息列表
        
        Returns:
            生成的标题，失败则返回默认标题
        """
        try:
            # 提取前几条消息作为上下文（最多3轮对话）
            context_messages = []
            user_count = 0
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "").strip()
                if role in ["user", "assistant"] and content:
                    context_messages.append(f"{'用户' if role == 'user' else '助手'}: {content}")
                    if role == "user":
                        user_count += 1
                        if user_count >= 3:
                            break
            
            if not context_messages:
                return "新对话"
            
            context_text = "\n".join(context_messages)
            system_prompt = "你是一个对话标题生成器。请根据对话内容生成简洁的标题（不超过20个字，不要包含标点符号）。只返回标题，不要包含其他内容。"
            user_prompt = f"对话内容：\n{context_text}\n\n请生成标题："
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=30,
                timeout=self.timeout
            )
            
            title = (response.choices[0].message.content or "").strip()
            title = title.strip('"').strip("'").strip("。").strip(".").strip()
            
            if len(title) > 30:
                title = title[:30]
            
            if not title or len(title) < 2:
                first_user_msg = None
                for msg in messages:
                    if msg.get("role") == "user":
                        first_user_msg = msg.get("content", "").strip()
                        break
                title = first_user_msg[:30] + ("..." if first_user_msg and len(first_user_msg) > 30 else "") if first_user_msg else "新对话"
            
            logger.info(f"生成对话标题成功: {title}")
            return title
            
        except Exception as e:
            logger.warning(f"生成对话标题失败: {str(e)}，使用默认标题")
            try:
                for msg in messages:
                    if msg.get("role") == "user":
                        first_msg = msg.get("content", "").strip()
                        if first_msg:
                            return first_msg[:30] + ("..." if len(first_msg) > 30 else "")
            except:
                pass
            return "新对话"


# 全局实例
title_generator = TitleGenerator()
