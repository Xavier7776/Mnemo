"""通用高级RAG助手Agent - 封装对话流程（混合检索 + 知识图谱 + 重排 + LLM生成）"""
import re
import json
from typing import Dict, Any, Optional, AsyncGenerator
from agents.base.base_agent import BaseAgent
from services.rag_service import rag_service
from utils.logger import logger
from utils.citation import validate_citations

# 模块级预编译正则：用于检测流式输出中的工具调用事件标记 \x1eTOOL_CALL:...\x1e
tool_call_re = re.compile(r'\x1eTOOL_CALL:(.*?)\x1e', re.DOTALL)
# 思考链标记 \x1eTHINKING:{json}\x1e
thinking_re = re.compile(r'\x1eTHINKING:(.*?)\x1e', re.DOTALL)


class GeneralAssistantAgent(BaseAgent):
    """通用高级RAG助手Agent - 处理通用领域的问答任务"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化Agent
        
        Args:
            model_name: 如果提供，则使用指定模型；否则根据问题自动选择
            base_url: Ollama服务地址
        """
        # 如果提供了model_name，使用它；否则在execute时动态选择
        self.fixed_model = model_name
        super().__init__(model_name=None, base_url=base_url)
    
    def get_default_model(self) -> str:
        """获取默认模型名称"""
        import os
        return os.getenv("LLM_MODEL", "mimo-v2.5")
    
    def get_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个通用的高级知识助手，基于提供的上下文信息回答问题。

你的职责：
1. 准确回答用户的问题，优先使用检索到的上下文（包括文本文档和知识图谱）。
2. 如果上下文包含知识图谱信息（实体、关系），请在回答中明确指出这些关联。
3. 如果上下文信息不足，请明确说明，并基于你的通用知识进行补充，但需区分来源。
4. 回答结构清晰，逻辑严密，使用Markdown格式。

回答要求：
- **引用来源**：尽可能引用上下文中的具体信息。
- **知识融合**：将文本文档中的细节与知识图谱中的结构化关系结合起来。
- **客观真实**：严禁编造检索结果中不存在的事实（避免幻觉）。
- **格式规范**：公式使用LaTeX格式，代码使用代码块。
"""
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行通用助手任务
        
        Args:
            task: 用户问题
            context: 上下文信息，包含：
                - assistant_id: 助手ID
                - knowledge_space_ids: 知识空间ID列表
                - document_id: 文档ID
                - enable_rag: 是否启用RAG检索（默认True）
                - conversation_history: 对话历史
            stream: 是否流式输出
        
        Yields:
            包含结果和元数据的字典
        """
        # 提取上下文信息
        assistant_id = context.get("assistant_id") if context else None
        knowledge_space_ids = context.get("knowledge_space_ids") if context else None
        document_id = context.get("document_id") if context else None
        enable_rag = context.get("enable_rag", True) if context else True
        conversation_history = context.get("conversation_history") if context else None
        generation_config = context.get("generation_config") if context else None
        embedding_model = generation_config.get("embedding_model") if generation_config else None
        import os
        # 0. 选择模型（固定模型 > 环境变量 > 默认）
        selected_model = self.fixed_model or os.getenv("LLM_MODEL", "mimo-v2.5")
        logger.info(f"GeneralAssistantAgent: 使用模型: {selected_model}")

        # 1. RAG 上下文准备
        rag_context = ""
        sources = []
        recommended_resources = []
        evidence = []
        query_plan = {}
        rag_trace = {}

        # 构造 Agentic RAG 工具执行上下文
        # 这个字典在整个 Agent 执行周期内保持状态，供 rag_retrieve 工具读取/更新
        tool_execution_context = None
        if enable_rag:
            tool_execution_context = {
                "document_id": document_id,
                "assistant_id": assistant_id,
                "knowledge_space_ids": knowledge_space_ids,
                "embedding_model": embedding_model,
                "query_plan": None,  # 前置 Plan 会被注入到这里，供 rag_retrieve 工具复用
                "retrieval_context": {
                    "seen_chunk_ids": set(),       # 跨轮次去重
                    "total_retrieval_count": 0,    # 已检索次数
                    "max_retrievals": 5,           # 安全阀：最多检索 5 次
                    "observations": [],            # 历次观察列表
                    "collected_evidence": [],      # 收集的验证后证据（供 Agent 层回收）
                    "reflections": [],             # 历次反思结果
                },
            }
            logger.info(f"GeneralAssistantAgent: Agentic RAG 模式（Plan-Act-Observe-Reflect），Agent 将自主决定是否调用 rag_retrieve 工具")

        # —— 前置检索参数规划（不是 reasoning）：生成 final_k/prefetch_k/fusion_strategy 等结构化参数 ——
        # 职责区分：
        #   - 这里的 build_plan：产出 LLM reasoning 不会生成的结构化检索参数
        #   - PAOR 循环里的 P（LLM reasoning）：LLM 实时决策要不要调工具、调什么参数
        # 两者不冲突：前置规划产出参数，LLM reasoning 决定是否使用
        #
        # 规划模式由环境变量 AGENT_PLANNER_MODE 控制（默认 auto）：
        #   - auto：智能选择。短查询/明确意图用规则引擎（毫秒级），长查询/复杂结构用 LLM（更准）
        #   - rules：强制规则引擎，毫秒级
        #   - llm：强制 LLM 判断，最准但每次增加 ~1-3s 延迟
        plan = None
        pre_retrieval_hint = ""  # 给 LLM 的规划提示（注入 context）
        if enable_rag:
            try:
                from services.query_planner import query_planner
                from services.runtime_config import get_runtime_config

                runtime_cfg = await get_runtime_config()
                runtime_modules = runtime_cfg.get("modules") or {}
                runtime_params = runtime_cfg.get("params") or {}

                # 读环境变量决定规划模式（默认 auto：智能选择）
                agent_planner_mode = os.getenv("AGENT_PLANNER_MODE", "auto").lower()

                plan = await query_planner.build_plan(
                    query=task,
                    runtime_modules=runtime_modules,
                    runtime_params=runtime_params,
                    filters={
                        "document_id": document_id,
                        "assistant_id": assistant_id,
                        "knowledge_space_ids": knowledge_space_ids or [],
                    },
                    planner_mode=agent_planner_mode,
                )
                query_plan = plan.model_dump()
                # 注入到 tool_execution_context，让 rag_retrieve 工具复用 plan 的参数
                if tool_execution_context is not None:
                    tool_execution_context["query_plan"] = plan
                logger.info(
                    f"GeneralAssistantAgent: 前置参数规划完成 "
                    f"(mode={agent_planner_mode}, intent={plan.intent}, source={plan.planner_source}, "
                    f"final_k={plan.final_k}, prefetch_k={plan.prefetch_k}, "
                    f"fusion={plan.fusion_strategy})"
                )

                # —— 条件预检索：复杂查询（compare/summary/clause/verification）才预检索 ——
                # 简单查询（general）让 LLM 自主决定是否调 rag_retrieve，避免增加首轮延迟
                if plan.intent != "general":
                    try:
                        from services.rag_service import rag_service
                        pre_result = await rag_service.retrieve_context(
                            query=task,
                            document_id=document_id,
                            assistant_id=assistant_id,
                            knowledge_space_ids=knowledge_space_ids,
                            embedding_model=embedding_model,
                            strategy="auto",
                            plan=plan,  # 复用上面的 plan，避免重复规划
                        )
                        # 把预检索结果作为初始 context（让 LLM 首轮就能看到证据）
                        pre_context = pre_result.get("context", "")
                        if pre_context:
                            rag_context = pre_context
                        # 把预检索 evidence 注入 collected_evidence，供 Agent 阶段三回收
                        pre_evidence = pre_result.get("evidence", [])
                        retrieval_ctx = tool_execution_context["retrieval_context"] if tool_execution_context else None
                        if retrieval_ctx is not None and pre_evidence:
                            for ev in pre_evidence:
                                cid = ev.get("chunk_id", "")
                                if cid and cid in retrieval_ctx["seen_chunk_ids"]:
                                    continue
                                if cid:
                                    retrieval_ctx["seen_chunk_ids"].add(cid)
                                retrieval_ctx["collected_evidence"].append({
                                    "chunk_id": cid,
                                    "content": ev.get("text", ""),
                                    "document_title": ev.get("document_title", ""),
                                    "document_id": ev.get("document_id", ""),
                                    "score": ev.get("score", 0.0),
                                    "retrieval_type": "pre_retrieval",
                                    "verified": True,  # 预检索结果默认可信
                                    "relevance_score": ev.get("score", 0.0),
                                    "retrieved_at_round": 0,  # 标记为预检索
                                    "section_path": ev.get("section_path", []),
                                })
                            # 预检索不计入 total_retrieval_count（那是 LLM 工具调用次数）
                            logger.info(
                                f"GeneralAssistantAgent: 预检索完成 (intent={plan.intent}), "
                                f"注入 {len(pre_evidence)} 条初始证据, "
                                f"context 长度 {len(pre_context)} 字符"
                            )
                        # 构建给 LLM 的规划提示
                        pre_retrieval_hint = (
                            f"【检索参数】意图={plan.intent}, final_k={plan.final_k}, "
                            f"已预检索 {len(pre_evidence)} 条证据。如需补充，调用 rag_retrieve 工具换关键词重试。"
                        )
                    except Exception as e:
                        logger.warning(f"GeneralAssistantAgent: 预检索失败，降级到无预检索模式: {e}")
                else:
                    # general 意图：不预检索，LLM reasoning 自主决定
                    pre_retrieval_hint = (
                        f"【检索参数】意图={plan.intent}, final_k={plan.final_k}。"
                        f"如需检索，调用 rag_retrieve 工具。"
                    )
            except Exception as e:
                logger.warning(f"GeneralAssistantAgent: 前置参数规划失败，降级到无 Plan 的 PAOR: {e}")
                plan = None

        # 2. 使用 LLM 生成回复
        try:
            full_response = ""
            evidence_instruction = ""
            if rag_context:
                evidence_instruction = (
                    "请优先依据以下证据回答，并在关键事实后使用 [S1]、[S2] 这类证据编号。"
                    "如果资料中找不到支持信息，请明确说明“资料中未找到”。\n\n"
                )
            # 把前置 Plan 的提示拼到 context 前面，让 LLM 首轮就知道规划意图
            # 如果有预检索证据：evidence_instruction + rag_context（证据）+ pre_retrieval_hint（规划提示）
            # 如果只有 Plan 无预检索：pre_retrieval_hint（规划提示）
            # 如果 Plan 失败：context=None（降级到当前行为）
            context_for_llm = None
            if rag_context:
                context_for_llm = evidence_instruction + rag_context
                if pre_retrieval_hint:
                    context_for_llm += "\n\n" + pre_retrieval_hint
            elif pre_retrieval_hint:
                context_for_llm = pre_retrieval_hint
            # LLMService.generate 会自动构建包含 context 的 prompt
            buffer_chunk = ""
            async for chunk in self.llm_service.generate(
                prompt=task,
                context=context_for_llm,
                stream=stream,
                document_id=document_id,
                # document_info=document_info, # 可以根据需要获取并传入
                # knowledge_base_status=knowledge_base_status,
                assistant_id=assistant_id,
                conversation_history=conversation_history,
                tool_execution_context=tool_execution_context,
            ):
                buffer_chunk += chunk

                # 检测事件标记（THINKING / TOOL_CALL，可能跨 chunk，所以用 buffer 累积）
                while True:
                    m_t = thinking_re.search(buffer_chunk)
                    m_c = tool_call_re.search(buffer_chunk)
                    # 取最早出现的标记
                    if m_t and (not m_c or m_t.start() < m_c.start()):
                        # THINKING 标记
                        before = buffer_chunk[:m_t.start()]
                        if before and stream:
                            full_response += before
                            yield {
                                "type": "chunk",
                                "content": before,
                                "agent_type": "general_assistant",
                                "sources": [],
                                "recommended_resources": []
                            }
                        try:
                            thinking_data = json.loads(m_t.group(1))
                            yield {
                                "type": "thinking",
                                "content": thinking_data.get("content", ""),
                                "agent_type": "general_assistant",
                            }
                        except Exception as e:
                            logger.warning(f"解析思考链事件失败: {e}")
                        buffer_chunk = buffer_chunk[m_t.end():]
                        continue
                    if m_c:
                        # TOOL_CALL 标记
                        before = buffer_chunk[:m_c.start()]
                        if before and stream:
                            full_response += before
                            yield {
                                "type": "chunk",
                                "content": before,
                                "agent_type": "general_assistant",
                                "sources": [],
                                "recommended_resources": []
                            }
                        try:
                            tool_call_data = json.loads(m_c.group(1))
                            yield {
                                "type": "tool_call",
                                "round": tool_call_data.get("round", 1),
                                "tools": tool_call_data.get("tools", []),
                                "agent_type": "general_assistant",
                            }
                        except Exception as e:
                            logger.warning(f"解析工具调用事件失败: {e}")
                        buffer_chunk = buffer_chunk[m_c.end():]
                        continue
                    break

                if stream and buffer_chunk and '\x1eTOOL_CALL:' not in buffer_chunk and '\x1eTHINKING:' not in buffer_chunk:
                    # buffer 中没有未闭合的标记，全部输出
                    to_emit = buffer_chunk
                    buffer_chunk = ""
                    full_response += to_emit
                    yield {
                        "type": "chunk",
                        "content": to_emit,
                        "agent_type": "general_assistant",
                        "sources": [],
                        "recommended_resources": []
                    }

            # 循环结束后，buffer 中剩余的正常文本输出
            if stream and buffer_chunk:
                full_response += buffer_chunk
                yield {
                    "type": "chunk",
                    "content": buffer_chunk,
                    "agent_type": "general_assistant",
                    "sources": [],
                    "recommended_resources": []
                }
            
            if not stream or full_response:
                # —— 阶段三：从 tool_execution_context 回收 evidence/sources ——
                if tool_execution_context and not evidence:
                    retrieval_ctx = tool_execution_context.get("retrieval_context") or {}
                    collected = retrieval_ctx.get("collected_evidence", [])
                    observations = retrieval_ctx.get("observations", [])
                    # 去重回收（按 chunk_id + 内容前缀双重去重）
                    # 阶段四优化：先按 score 降序排序，保留高分数的 chunk
                    collected_sorted = sorted(collected, key=lambda x: x.get("score", 0.0), reverse=True)
                    seen_ids = set()
                    seen_content_prefixes = set()  # 内容前缀去重（同一文档中相似内容只保留最高分）
                    dedup_count = 0
                    for item in collected_sorted:
                        cid = item.get("chunk_id", "")
                        content = item.get("content", "")
                        content_prefix = content[:100].strip() if content else ""
                        # chunk_id 去重
                        if cid and cid in seen_ids:
                            continue
                        # 内容前缀去重（同一文档中内容高度相似的 chunk 只保留分数最高的）
                        dedup_key = f"{item.get('document_title', '')}::{content_prefix}"
                        if content_prefix and dedup_key in seen_content_prefixes:
                            dedup_count += 1
                            continue
                        if cid:
                            seen_ids.add(cid)
                        if content_prefix:
                            seen_content_prefixes.add(dedup_key)
                        evidence.append({
                            "id": cid,
                            "text": content,
                            "chunk_id": cid,
                            "document_title": item.get("document_title", ""),
                            "score": item.get("score", 0.0),
                            "retrieval_type": "agentic_rag",
                            "verified": item.get("verified", False),
                            "relevance_score": item.get("relevance_score"),
                            "retrieved_at_round": item.get("retrieved_at_round"),
                            "section_path": item.get("section_path", []),
                        })
                    # 构建 sources（每文档取最高分）
                    doc_best = {}
                    for item in collected:
                        doc_title = item.get("document_title", "未知文档")
                        score = item.get("score", 0.0)
                        if doc_title not in doc_best or score > doc_best[doc_title].get("score", 0):
                            doc_best[doc_title] = {
                                "chunk_id": item.get("chunk_id", ""),
                                "document_title": doc_title,
                                "score": score,
                                "retrieval_type": "agentic_rag",
                            }
                    sources = list(doc_best.values())
                    # 构建反思 trace
                    if observations:
                        rag_trace = {
                            "observations": observations,
                            "total_retrievals": retrieval_ctx.get("total_retrieval_count", 0),
                            "total_evidence": len(evidence),
                            "verified_evidence": sum(1 for e in evidence if e.get("verified")),
                        }
                    logger.info(f"GeneralAssistantAgent: 阶段三 Evidence 回收 - evidence={len(evidence)}, sources={len(sources)}, 内容去重={dedup_count}条")

                citation_warnings = validate_citations(full_response, evidence) if evidence else []
                yield {
                    "type": "complete",
                    "content": full_response,
                    "agent_type": "general_assistant",
                    "sources": sources,
                    "evidence": evidence,
                    "query_plan": query_plan,
                    "trace": rag_trace,
                    "citation_warnings": citation_warnings,
                    "recommended_resources": recommended_resources,
                    "confidence": 0.9 # 高阶RAG通常置信度较高
                }
        
        except Exception as e:
            logger.error(f"GeneralAssistantAgent: 生成回复失败: {e}", exc_info=True)
            yield {
                "type": "error",
                "content": f"生成回复时出错: {str(e)}",
                "agent_type": "general_assistant",
                "sources": sources,
                "evidence": evidence,
                "query_plan": query_plan,
                "trace": rag_trace,
                "recommended_resources": recommended_resources
            }
