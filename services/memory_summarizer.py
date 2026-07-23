"""对话压缩器：5 段式精简压缩 + 基于 token 剩余的触发条件

v6.2 改造（2026-07-21）：
1. 9 段 → 5 段：压缩比从 114.76% 降为预期 < 60%
2. 新增 Key Facts 段：显式保留数字/API 合同/决策/错误日志等硬细节
   （9 段式对这三类细节保真度损失 37.5%，详见 tests/compression_fidelity_results.md）
3. 合并重复段落：Errors+ProblemSolving → Key Facts；Pending+Current+Next → Status

v6.1 改造（2026-07-20）：
1. 大工具结果落盘：由 tool_result_store 在 llm_service 中处理（>2000 字符存磁盘）
2. ~~清理「可重新获取」的工具结果~~ → 暂不实施（未来可基于"长时间未调用"触发清理）
3. 触发条件改为基于 token 剩余：当 context_window - messages_tokens < CONTEXT_WINDOW_RESERVE 时触发

5 段式压缩结构（v6.2）：
    1. Request                  — 用户要什么，一句话
    2. ⭐⭐⭐ Key Facts          — 硬细节原文保留（数字/配置/API/决策/错误/专名）
    3. Files                    — 涉及的文件路径 + 关键代码位置
    4. ⭐⭐⭐ User Messages      — 按顺序列出用户发言原文（不概括！）
    5. Status                   — 当前进度 + 待办 + 下一步

对应 Letta 的 Recall Memory + Summarizer：
- 当上下文剩余空间 < CONTEXT_WINDOW_RESERVE 时触发
- 旧消息（保留最近 KEEP_RECENT 条原文）压缩成 compressed_summary 字段
- 原始对话归档到 Archival Memory，可追溯
- MongoDB messages 数组不裁剪（保留全部原始消息，便于审计和 conversation_search）
"""
from typing import List, Dict, Optional
from database.mongodb import mongodb
from services.llm_service import LLMService
from services.archival_memory_service import archival_memory_service
from utils.logger import logger
from utils.token_utils import estimate_tokens

# 触发压缩的 token 剩余阈值：当 context_window - messages_tokens < 此值时触发
CONTEXT_WINDOW_RESERVE = 50_000

# 压缩后仍保留最近 KEEP_RECENT 条原文注入 prompt
KEEP_RECENT = 8

# 兜底 context window（未识别模型时使用，取主流大窗口模型的常见值）
DEFAULT_CONTEXT_WINDOW_FOR_COMPRESS = 128_000

# 可重新获取的工具结果模式（暂不启用清理，保留定义以便未来基于空闲时间触发）
# 未来方案：长时间（1 小时）模型没有调用这些工具时，才删除其结果
REGENERATABLE_TOOL_PATTERNS = (
    "read", "bash", "grep", "glob", "ls",
    "web_search", "websearch", "fetch",
    "edit", "write", "create", "delete",
    "mcp__filesystem__read", "mcp__filesystem__list", "mcp__filesystem__get",
    "mcp__filesystem__search", "mcp__filesystem__glob",
    "mcp__firecrawl", "mcp__webfetch",
    "rag_retrieve", "archival_memory_search", "conversation_search",
)


def _is_regeneratable_tool(tool_name: str) -> bool:
    """判断工具结果是否可重新获取（暂不启用清理，保留供未来使用）"""
    if not tool_name:
        return False
    name_lower = tool_name.lower()
    for pattern in REGENERATABLE_TOOL_PATTERNS:
        if pattern in name_lower:
            return True
    return False


def _build_compact_prompt(transcript: str, existing_summary: str) -> str:
    """构造精简 5 段式压缩的 LLM 提示词

    v6.2（2026-07-21）：从 9 段精简到 5 段
    - 原因：9 段式压缩后长度 > 原文（压缩比 114.76%），且对 api_contract/decision_history/error_log
      三类硬细节完全失真（保真度损失 37.5%）
    - 改进：合并重复段落，新增 Key Facts 段显式保留硬细节，强制压缩比 < 60%
    - 详见 tests/compression_fidelity_results.md
    """
    return f"""压缩以下对话为 5 段摘要。目标：压缩比 < 60%（摘要长度必须小于原文的 60%）。

已有摘要（合并进来）：
{existing_summary or "（无）"}

待压缩对话（{len(transcript)} 字符）：
{transcript}

输出 5 段，无内容的段写"（无）"。禁止复述原文，禁止添加原文没有的信息：

## 1. Request
用户要什么。一句话。

## 2. Key Facts
只保留与"用户提问/决策/错误"直接相关的硬细节，每项一行简短：
- 数字/配置值（端口号、阈值、QPS、延迟、版本号）
- API 合同（端点路径、限流策略）
- 决策结果（投票数、最终选型）
- 错误（错误消息、持续时长、根因）
- 专有名词（人名、项目名、库名）
不要列出所有配置，只保留用户明确提到或决策相关的。

## 3. Files
涉及的文件路径 + 函数名/行号。代码段不复制。

## 4. User Messages
按顺序列出每条用户发言要点（省略问候，省略客套）。每条一行。

## 5. Status
当前进度 + 待办 + 下一步。每项一行，简短。
"""


def _build_9section_prompt(transcript: str, existing_summary: str) -> str:
    """[已废弃] 9 段式压缩 prompt，保留供回滚

    v6.2 起改用 _build_compact_prompt（5 段式），原因见该函数 docstring。
    """
    return f"""你是对话压缩器。请把下面的对话内容压缩成结构化的 9 段式摘要。

已有摘要（如有，请合并进来）：
{existing_summary or "（无）"}

待压缩的对话内容：
{transcript}

请严格按以下 9 段格式输出，不要遗漏任何一段，不要添加额外内容：

## 1. Primary Request and Intent
用户最初要什么，方向不能偏。一句话概括用户的核心诉求。

## 2. Key Technical Concepts
技术栈、技术决策不丢。列出涉及的关键技术概念、框架、库、算法。

## 3. Files and Code Sections
碰了哪些文件，代码范围不丢。列出涉及的文件路径和关键代码段（函数名/类名/行号）。

## 4. Errors and fixes
踩了什么坑、怎么修的。列出遇到的错误及修复方式。

## 5. Problem Solving
解决了哪些具体问题。列出已解决的问题及方案。

## 6. All user messages ⭐⭐⭐
枚举所有用户发言——不概括！用户中途改需求、提约束、说放弃的信号全在这。
按顺序列出用户每一条发言的原文（可以省略纯问候，但需求/约束/反馈必须完整保留）。

## 7. Pending Tasks
还有哪些活没干完。列出未完成的任务。

## 8. Current Work ⭐⭐⭐
最细颗粒度当前进度——不是"在调试"，是"在调试 login.ts 第 42 行 token 刷新"。
描述当前正在做的具体工作（文件/函数/行号级别）。

## 9. Optional Next Step
建议下一步。给出明确的下一步行动建议。

输出要求：
- 严格按上面的 9 段格式，每段用 `## N. 标题` 开头
- 第 6 段和第 8 段最重要，必须尽可能详细，不能概括
- 如果某段没有内容，写"（无）"
- 不要输出任何额外说明
"""


def _estimate_messages_tokens(messages: List[Dict]) -> int:
    """估算 messages 数组的总 token 数

    只统计 content 字段，其他字段（sources/evidence 等）不计入。
    """
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif content is not None:
            # 非字符串 content（如 dict/list），转 str 后估算
            total += estimate_tokens(str(content))
    return total


def _get_context_window(model_name: Optional[str]) -> int:
    """根据模型名查询 context window，未识别时用兜底值"""
    if not model_name:
        return DEFAULT_CONTEXT_WINDOW_FOR_COMPRESS
    try:
        from services.mcp_client_service import get_model_context_window
        return get_model_context_window(model_name)
    except Exception:
        return DEFAULT_CONTEXT_WINDOW_FOR_COMPRESS


class MemorySummarizer:
    """对话压缩器：5 段式精简压缩 + 基于 token 剩余触发"""

    async def maybe_summarize(
        self,
        conversation_id: str,
        scope_type: str = "global",
        scope_id: str = "default",
        model_name: Optional[str] = None,
    ):
        """检查对话是否需要压缩，若需要则执行 5 段式压缩

        触发条件（v6.1）：当 context_window - messages_tokens < CONTEXT_WINDOW_RESERVE 时触发。
        model_name 用于推断 context_window；若为 None，则从 conversation 文档的
        last_model_name 字段读取；都没有则用 DEFAULT_CONTEXT_WINDOW_FOR_COMPRESS。

        幂等：未达 token 阈值时直接 return，不会重复压缩。
        建议在 routers/chat.py 的 add_message 后用 BackgroundTasks 触发，不阻塞主流程。

        v6 改造：
        - 不再裁剪 MongoDB messages 数组（保留全部原始消息）
        - 新增 compressed_summary 字段存压缩结果
        - 旧字段 summary 保留向后兼容（写入压缩结果的前 500 字符预览）
        - 旧消息归档到 Archival Memory

        v6.1 改造：
        - 触发条件从"消息数 > 30"改为"token 剩余 < 50k"
        - 移除"可重新获取工具结果清理"（暂不实施，未来基于空闲时间触发）

        v6.2 改造：
        - 9 段式 → 5 段式精简（Key Facts 段显式保留硬细节，解决 api_contract/decision_history/error_log 失真）
        """
        try:
            col = mongodb.get_collection("conversations")
            doc = await col.find_one({"_id": conversation_id})
            if not doc:
                return

            messages: List[Dict] = doc.get("messages", [])
            if not messages:
                return

            # v6.1: 基于 token 剩余判断是否触发压缩
            # model_name 优先用参数传入的，否则从 conversation 文档读 last_model_name
            effective_model = model_name or doc.get("last_model_name")
            context_window = _get_context_window(effective_model)
            total_tokens = _estimate_messages_tokens(messages)
            remaining = context_window - total_tokens

            # 预留阈值：当 context_window - messages_tokens < reserve 时触发压缩
            # 保护：对于小窗口模型（context_window <= CONTEXT_WINDOW_RESERVE），
            # 50k 预留永远不可能满足，此时降级为预留 context_window 的 25%（至少 4k）
            if context_window > CONTEXT_WINDOW_RESERVE:
                effective_reserve = CONTEXT_WINDOW_RESERVE
            else:
                effective_reserve = max(context_window // 4, 4_000)

            if remaining >= effective_reserve:
                # 剩余空间充足，不需要压缩
                return

            logger.info(
                f"会话 {conversation_id} 触发压缩：tokens={total_tokens}, "
                f"context_window={context_window}, remaining={remaining}, "
                f"reserve={effective_reserve}, model={effective_model or 'unknown'}"
            )

            # v6: 不裁剪 messages 数组，用 last_summarized_count 记录已压缩到的位置
            # 只压缩上次压缩之后的新消息（增量压缩），避免重复压缩
            last_summarized_count = doc.get("last_summarized_count", 0)
            # 只处理"上次压缩点之后 + 排除最近 KEEP_RECENT 条"的消息
            if len(messages) <= KEEP_RECENT:
                return  # 消息太少，没有可压缩的内容
            to_summarize = messages[last_summarized_count:-KEEP_RECENT]
            if not to_summarize:
                return  # 没有新消息需要压缩

            existing_summary = doc.get("compressed_summary", "") or doc.get("summary", "")

            # v6.1: 暂不清理可重新获取的工具结果（未来基于空闲时间触发）
            # 直接用原始 messages 拼接转录文本
            transcript = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in to_summarize
                if m.get("content")
            )
            if not transcript.strip():
                return

            # v6.2: 5 段式精简压缩（替代 9 段式）
            prompt = _build_compact_prompt(transcript, existing_summary)
            llm = LLMService()
            new_summary = ""
            async for chunk in llm.generate(prompt=prompt, stream=False):
                new_summary += chunk
            new_summary = new_summary.strip()

            if not new_summary:
                logger.warning(f"会话 {conversation_id} 5 段式压缩结果为空，跳过")
                return

            # 旧消息归档到 Archival Memory，可追溯
            await archival_memory_service.insert(
                scope_type=scope_type,
                scope_id=scope_id,
                content=transcript,
                source="auto_summary",
                conversation_id=conversation_id,
            )

            # v6: 不再裁剪 messages 数组，保留全部原始消息
            # 新增 compressed_summary 字段存 9 段式压缩结果
            # summary 字段保留向后兼容（写入压缩结果的前 500 字符预览）
            # last_summarized_count 记录已压缩到的位置（下次从这里继续增量压缩）
            new_summarized_count = len(messages) - KEEP_RECENT
            await col.update_one(
                {"_id": conversation_id},
                {
                    "$set": {
                        "compressed_summary": new_summary,
                        "summary": new_summary[:500] + ("..." if len(new_summary) > 500 else ""),
                        "last_summarized_count": new_summarized_count,
                        "last_summarized_at": __import__("datetime").datetime.utcnow().isoformat(),
                    }
                },
            )
            logger.info(
                f"会话 {conversation_id} 已 5 段式压缩，增量压缩 {len(to_summarize)} 条消息"
                f"（{last_summarized_count} → {new_summarized_count}），"
                f"压缩前 tokens={total_tokens}, 压缩结果长度 {len(new_summary)}, "
                f"messages 数组保留全部 {len(messages)} 条"
            )
        except Exception as e:
            logger.error(f"会话 {conversation_id} 5 段式压缩失败: {e}", exc_info=True)


memory_summarizer = MemorySummarizer()
