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
        
        # 1. RAG检索（如果启用）
        rag_context = ""
        sources = []
        recommended_resources = []
        evidence = []
        query_plan = {}
        rag_trace = {}
        
        if enable_rag:
            try:
                logger.info(f"GeneralAssistantAgent: 开始高阶RAG检索 (混合检索+重排) - 问题: {task[:50]}...")
                # rag_service 内部封装了 vector search, keyword search, graph search, 和 reranking
                retrieval_result = await rag_service.retrieve_context(
                    query=task,
                    document_id=document_id,
                    assistant_id=assistant_id,
                    knowledge_space_ids=knowledge_space_ids,
                    embedding_model=embedding_model,
                )
                
                rag_context = retrieval_result.get("context", "")
                sources = retrieval_result.get("sources", [])
                evidence = retrieval_result.get("evidence", [])
                query_plan = retrieval_result.get("query_plan", {})
                rag_trace = retrieval_result.get("trace", {})
                recommended_resources = retrieval_result.get("recommended_resources", [])
                
                logger.info(f"GeneralAssistantAgent: RAG检索完成 - 上下文长度: {len(rag_context)}, 来源数: {len(sources)}")
            except Exception as e:
                logger.warning(f"GeneralAssistantAgent: RAG检索失败: {e}")
                # RAG检索失败不影响继续生成回复
        
        # 2. 使用 LLM 生成回复
        try:
            full_response = ""
            evidence_instruction = ""
            if rag_context:
                evidence_instruction = (
                    "请优先依据以下证据回答，并在关键事实后使用 [S1]、[S2] 这类证据编号。"
                    "如果资料中找不到支持信息，请明确说明“资料中未找到”。\n\n"
                )
            # LLMService.generate 会自动构建包含 context 的 prompt
            buffer_chunk = ""
            async for chunk in self.llm_service.generate(
                prompt=task,
                context=(evidence_instruction + rag_context) if rag_context else None,
                stream=stream,
                document_id=document_id,
                # document_info=document_info, # 可以根据需要获取并传入
                # knowledge_base_status=knowledge_base_status,
                assistant_id=assistant_id,
                conversation_history=conversation_history
            ):
                buffer_chunk += chunk

                # 检测工具调用事件标记（可能跨 chunk，所以用 buffer 累积）
                while True:
                    m = tool_call_re.search(buffer_chunk)
                    if not m:
                        break
                    # 标记前的正常文本先输出
                    before = buffer_chunk[:m.start()]
                    if before and stream:
                        full_response += before
                        yield {
                            "type": "chunk",
                            "content": before,
                            "agent_type": "general_assistant",
                            "sources": [],
                            "recommended_resources": []
                        }
                    # 解析并产出工具调用事件
                    try:
                        tool_call_data = json.loads(m.group(1))
                        yield {
                            "type": "tool_call",
                            "round": tool_call_data.get("round", 1),
                            "tools": tool_call_data.get("tools", []),
                            "agent_type": "general_assistant",
                        }
                    except Exception as e:
                        logger.warning(f"解析工具调用事件失败: {e}")
                    buffer_chunk = buffer_chunk[m.end():]

                if stream and buffer_chunk and '\x1eTOOL_CALL:' not in buffer_chunk:
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
