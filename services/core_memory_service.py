"""Core Memory 服务：常驻记忆块的读取与编辑

存储：MongoDB collection `agent_core_memory`
特性：Core Memory 永远注入 system prompt，不需要模型主动检索——
      这是它和 Archival Memory 的本质区别。

工具暴露给模型：
- core_memory_append(label, content)  追加
- core_memory_replace(label, old, new) 替换
"""
from datetime import datetime
from typing import Dict, Optional

from database.mongodb import mongodb
from models.memory import CoreMemory, CoreMemoryBlock
from utils.logger import logger

COLLECTION = "agent_core_memory"


class CoreMemoryService:
    """Core Memory 服务：常驻记忆块的读取与编辑"""

    async def get(self, scope_type: str, scope_id: str) -> CoreMemory:
        """读取一个 scope 下的核心记忆，不存在则用默认值初始化"""
        col = mongodb.get_collection(COLLECTION)
        doc = await col.find_one({"scope_type": scope_type, "scope_id": scope_id})
        if not doc:
            mem = CoreMemory(scope_type=scope_type, scope_id=scope_id)
            await self._save(mem)
            return mem
        blocks = {k: CoreMemoryBlock(**v) for k, v in doc.get("blocks", {}).items()}
        return CoreMemory(
            scope_type=scope_type,
            scope_id=scope_id,
            blocks=blocks,
            updated_at=doc.get("updated_at"),
        )

    async def _save(self, mem: CoreMemory):
        col = mongodb.get_collection(COLLECTION)
        await col.update_one(
            {"scope_type": mem.scope_type, "scope_id": mem.scope_id},
            {
                "$set": {
                    "blocks": {k: v.model_dump() for k, v in mem.blocks.items()},
                    "updated_at": datetime.utcnow(),
                }
            },
            upsert=True,
        )

    async def append(
        self,
        scope_type: str,
        scope_id: str,
        label: str,
        content: str,
    ) -> Dict:
        """
        向指定记忆块追加内容

        Args:
            label: 记忆块名称，如 persona / human
            content: 要追加的文本
        """
        if not content or not content.strip():
            return {"success": False, "error": "content 不能为空"}

        try:
            mem = await self.get(scope_type, scope_id)
            #第1个参数 label：要查找的键（比如 "persona" 或 "human"）
            #第2个参数 CoreMemoryBlock()：如果键不存在时的默认返回值(空的)
            block = mem.blocks.get(label, CoreMemoryBlock())
            new_value = (block.value + "\n" + content).strip() if block.value else content.strip()
            if len(new_value) > block.limit:
                return {
                    "success": False,
                    "error": f"超出 {label} 记忆块上限({block.limit}字符)，请先精简",
                }
            block.value = new_value
            mem.blocks[label] = block
            await self._save(mem)
            logger.info(
                f"CoreMemory append 成功 - scope={scope_type}/{scope_id}, "
                f"label={label}, len={len(new_value)}"
            )
            return {"success": True, "label": label, "value": block.value}
        except Exception as e:
            logger.error(f"CoreMemory append 失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def replace(
        self,
        scope_type: str,
        scope_id: str,
        label: str,
        old_content: str,
        new_content: str,
    ) -> Dict:
        """
        替换指定记忆块中的旧内容为新内容（首次匹配替换）

        Args:
            label: 记忆块名称
            old_content: 要被替换的精确文本片段
            new_content: 替换后的新文本
        """
        if not old_content:
            return {"success": False, "error": "old_content 不能为空"}

        try:
            mem = await self.get(scope_type, scope_id)
            block = mem.blocks.get(label, CoreMemoryBlock())
            if old_content not in block.value:
                return {"success": False, "error": f"在 {label} 记忆块中未找到要替换的内容"}
            #1是最大替换次数
            new_value = block.value.replace(old_content, new_content, 1)
            if len(new_value) > block.limit:
                return {
                    "success": False,
                    "error": f"替换后超出 {label} 记忆块上限({block.limit}字符)",
                }
            block.value = new_value
            mem.blocks[label] = block
            await self._save(mem)
            logger.info(
                f"CoreMemory replace 成功 - scope={scope_type}/{scope_id}, label={label}"
            )
            return {"success": True, "label": label, "value": block.value}
        except Exception as e:
            logger.error(f"CoreMemory replace 失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def render_for_prompt(self, scope_type: str, scope_id: str) -> str:
        """渲染成可以直接塞进 system prompt 的文本块"""
        try:
            mem = await self.get(scope_type, scope_id)
            lines = ["## 核心记忆（长期有效，除非你主动修改）"]
            for label, block in mem.blocks.items():
                if block.value.strip():
                    lines.append(f"### {label}\n{block.value.strip()}")
            return "\n\n".join(lines) if len(lines) > 1 else ""
        except Exception as e:
            logger.warning(f"渲染 Core Memory 失败，跳过: {e}")
            return ""


core_memory_service = CoreMemoryService()
