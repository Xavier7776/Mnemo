"""统一 OpenAI 兼容客户端（支持 Mimo / DeepSeek / OpenAI 等），惰性单例"""
import os
from typing import Optional
from openai import OpenAI, AsyncOpenAI

_client: Optional[OpenAI] = None
_async_client: Optional[AsyncOpenAI] = None


def get_openai_client() -> OpenAI:
    """
    获取全局 OpenAI 兼容客户端（惰性初始化）

    - 读取 OPENAI_API_KEY / OPENAI_BASE_URL 环境变量
    - 兼容旧的 OLLAMA_BASE_URL（无 OPENAI_BASE_URL 时回退）
    - 只在首次调用时创建实例，避免 import 顺序问题
    """
    global _client
    if _client is not None:
        return _client

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY", "")

    # 兼容 old config: 只有 OLLAMA_BASE_URL 没有 OPENAI_BASE_URL
    if not base_url and os.getenv("OLLAMA_BASE_URL"):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

    base_url = base_url or "https://api.openai.com/v1"

    _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def get_async_openai_client() -> AsyncOpenAI:
    """
    获取全局 AsyncOpenAI 兼容客户端（惰性初始化）

    用于流式生成等需要高并发的场景，避免同步客户端阻塞事件循环。
    与 get_openai_client 共享相同的 base_url / api_key 配置。
    """
    global _async_client
    if _async_client is not None:
        return _async_client

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY", "")

    if not base_url and os.getenv("OLLAMA_BASE_URL"):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

    base_url = base_url or "https://api.openai.com/v1"

    _async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return _async_client
