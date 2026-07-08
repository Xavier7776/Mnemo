"""对话摘要器：超过阈值时把旧消息压缩成摘要

对应 Letta 的 Recall Memory + Summarizer：
- 超过 TRIGGER_MESSAGE_COUNT 条消息时触发
- 旧消息（保留最近 KEEP_RECENT 条）压缩成 summary 字段
- 原始对话归档到 Archival Memory，可追溯
- 避免硬截断造成的信息丢失

注意：LLMService.generate 是 async generator，即使 stream=False 也会 yield 一次完整结果。
"""
from typing import List, Dict
from database.mongodb import mongodb
from services.llm_service import LLMService
from services.archival_memory_service import archival_memory_service
from utils.logger import logger

TRIGGER_MESSAGE_COUNT = 30  # 超过 30 条消息触发摘要
KEEP_RECENT = 8             # 摘要后仍保留最近 8 条原文


class MemorySummarizer:
    """对话摘要器：超长对话自动压缩成摘要 + 归档"""

    async def maybe_summarize(
        self,
        conversation_id: str,
        scope_type: str = "global",
        scope_id: str = "default",
    ):
        """
        检查对话是否需要摘要化，若需要则执行。

        幂等：消息数未达阈值时直接 return，不会重复摘要。
        建议在 routers/chat.py 的 add_message 后用 BackgroundTasks 触发，不阻塞主流程。
        """
        try:
            col = mongodb.get_collection("conversations")
            doc = await col.find_one({"_id": conversation_id})
            if not doc:
                return

            messages: List[Dict] = doc.get("messages", [])
            if len(messages) <= TRIGGER_MESSAGE_COUNT:
                return

            #-8 表示倒数第 8 个元素。
            to_summarize = messages[:-KEEP_RECENT]
            existing_summary = doc.get("summary", "")

            transcript = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in to_summarize
                if m.get("content")
            )
            if not transcript.strip():
                return

            prompt = (
                f"已有摘要：\n{existing_summary or '（无）'}\n\n"
                f"请把下面的新增对话内容，合并进已有摘要，输出一段更新后的摘要。"
                f"只保留事实、结论、用户偏好和未解决的问题，不要逐句复述：\n\n{transcript}"
            )

            # LLMService.generate 是 async generator（即使 stream=False 也会 yield 一次）
            llm = LLMService()
            new_summary = ""
            async for chunk in llm.generate(prompt=prompt, stream=False):
                new_summary += chunk
            new_summary = new_summary.strip()

            if not new_summary:
                logger.warning(f"会话 {conversation_id} 摘要生成结果为空，跳过摘要化")
                return

            # 旧消息归档到 Archival Memory，保留可追溯性
            await archival_memory_service.insert(
                scope_type=scope_type,
                scope_id=scope_id,
                content=transcript,
                source="auto_summary",
                conversation_id=conversation_id,
            )

            await col.update_one(
                {"_id": conversation_id},
                {
                    "$set": {
                        "summary": new_summary,
                        "messages": messages[-KEEP_RECENT:],  # 原文只留最近一小段
                    }
                },
            )
            logger.info(
                f"会话 {conversation_id} 已摘要化，压缩 {len(to_summarize)} 条消息，"
                f"摘要长度 {len(new_summary)}"
            )
        except Exception as e:
            logger.error(f"会话 {conversation_id} 摘要化失败: {e}", exc_info=True)


memory_summarizer = MemorySummarizer()
