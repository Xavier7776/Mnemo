"""记忆系统数据模型 - Core Memory / Recall Memory / Archival Memory

对应 Letta 的三层记忆：
- Core Memory：常驻注入 system prompt 的记忆块（persona / human）
- Recall Memory：对话历史 + 摘要（存 conversations 集合的 summary 字段）
- Archival Memory：长期归档、按需语义检索（存 Qdrant）

scope 字段统一抽象记忆归属，避免写死 assistant_id：
- scope_type="global",      scope_id="default"      —— 默认：全局共享记忆
- scope_type="assistant",   scope_id=<assistant_id> —— 同一助手共享
- scope_type="conversation",scope_id=<conversation_id> —— 会话级隔离
"""
from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field


class CoreMemoryBlock(BaseModel):
    """单个核心记忆块，对应 Letta 里的一个 memory block"""
    value: str = ""
    limit: int = 2000  # 字符数上限，超过时 append/replace 应报错而不是静默截断


class CoreMemory(BaseModel):
    """一个 scope 下的核心记忆，常驻注入系统提示词"""
    scope_type: str  # "global" | "assistant" | "conversation"
    scope_id: str
    blocks: Dict[str, CoreMemoryBlock] = Field(default_factory=lambda: {
        "persona": CoreMemoryBlock(
            value="你是 Xavier 的个人科研/开发助手，专注时间序列预测、Agent框架与全栈开发。"
        ),
        "human": CoreMemoryBlock(value=""),
    })
    updated_at: Optional[datetime] = None


class ArchivalMemoryItem(BaseModel):
    """归档记忆的一条记录，存 Qdrant，payload 走这个结构"""
    scope_type: str
    scope_id: str
    content: str
    source: str = "manual"  # manual | auto_summary | recall_migration
    created_at: datetime
    conversation_id: Optional[str] = None
