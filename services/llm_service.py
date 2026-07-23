"""OpenAI兼容模型调用服务（支持Mimo/DeepSeek/OpenAI等）"""
import os
import json
import time
from typing import AsyncGenerator, Optional, Dict, Any, List, Tuple
from openai import APIError, APITimeoutError, APIConnectionError
from utils.llm_client import get_openai_client, get_async_openai_client
from utils.logger import logger


class LLMService:
    """模型调用服务（OpenAI兼容格式，支持Mimo/DeepSeek/OpenAI等）"""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self._base_url = base_url
        self._model_name = model_name
        self._api_key = api_key
        self._timeout = None
        self._base_prompt_cache = None
        self._base_prompt_cache_time = 0
        self._adapter_cache = None

        logger.info(f"模型服务初始化 - 模型: {self.model_name}")

    @property
    def client(self):
        return get_openai_client()

    @property
    def async_client(self):
        """异步 OpenAI 客户端，用于流式生成等需要高并发的场景"""
        return get_async_openai_client()

    @property
    def model_name(self):
        if self._model_name is None:
            self._model_name = os.getenv("LLM_MODEL", "mimo-v2.5")
        return self._model_name

    @property
    def timeout(self):
        if self._timeout is None:
            self._timeout = float(os.getenv("LLM_TIMEOUT", "600.0"))
        return self._timeout

    @property
    def _adapter(self):
        """Function Calling 统一适配器（惰性初始化，基于 base_url/model 自动探测厂商）"""
        if self._adapter_cache is None:
            from services.tool_call_adapter import ToolCallAdapter
            base_url = self._base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or ""
            self._adapter_cache = ToolCallAdapter.from_env(base_url=base_url, model_name=self.model_name)
        return self._adapter_cache

    async def list_models(self) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        try:
            models = await self.async_client.models.list()
            return [
                {"name": m.id, "owned_by": m.owned_by}
                for m in models.data
            ]
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []

    async def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        stream: bool = False,
        document_id: Optional[str] = None,
        document_info: Optional[Dict[str, Any]] = None,
        knowledge_base_status: Optional[Dict[str, Any]] = None,
        assistant_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        tool_execution_context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """生成回复（流式或非流式）

        Args:
            tool_execution_context: Agent 工具执行上下文，用于向 rag_retrieve 等工具
                注入 document_id/assistant_id/knowledge_space_ids/embedding_model 和
                跨轮次去重状态（retrieval_context.seen_chunk_ids / total_retrieval_count）。
                仅在 Agentic RAG 模式下由 GeneralAssistantAgent 传入。
        """
        messages, tool_context_added = await self._build_messages(
            prompt, context,
            document_id=document_id,
            document_info=document_info,
            knowledge_base_status=knowledge_base_status,
            assistant_id=assistant_id,
            conversation_history=conversation_history
        )

        if stream:
            async for chunk in self._generate_stream(
                messages,
                assistant_id=assistant_id,
                tool_execution_context=tool_execution_context,
            ):
                yield chunk
        else:
            response = await self._generate_once(messages)
            yield response
    
    async def _build_messages(
        self,
        prompt: str,
        context: Optional[str] = None,
        document_id: Optional[str] = None,
        document_info: Optional[Dict[str, Any]] = None,
        knowledge_base_status: Optional[Dict[str, Any]] = None,
        assistant_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[Dict[str, str]], bool]:
        """构建 OpenAI messages 格式"""
        # ——— 1. System instruction ———
        assistant_prompt = None
        if assistant_id:
            try:
                from database.mongodb import mongodb
                collection = mongodb.get_collection("course_assistants")
                assistant_doc = await collection.find_one({"_id": assistant_id})
                if assistant_doc:
                    assistant_prompt = assistant_doc.get("system_prompt")
            except Exception as e:
                logger.warning(f"获取助手系统提示词失败: {str(e)}")
        
        from services.prompt_chain import prompt_chain

        # base_prompt 缓存（5 分钟 TTL）——仅当不需要自定义 prompt 时启用
        base_prompt_cache = None
        if assistant_id is None:
            now_ts = time.time()
            if self._base_prompt_cache is not None and (now_ts - self._base_prompt_cache_time) < 300:
                base_prompt_cache = self._base_prompt_cache
            else:
                try:
                    base_prompt_cache = await prompt_chain.get_base_prompt()
                    self._base_prompt_cache = base_prompt_cache
                    self._base_prompt_cache_time = now_ts
                except Exception as e:
                    logger.warning(f"获取 base_prompt 失败，降级为 None: {e}")
                    base_prompt_cache = None

        system_instruction = await prompt_chain.build_prompt_chain(
            base_prompt=base_prompt_cache,
            assistant_prompt=assistant_prompt
        )
        if not system_instruction:
            system_instruction = await prompt_chain.get_base_prompt()
        
        # ——— 2. Build system message parts ———
        system_parts = [system_instruction]
        
        # 知识库状态
        if knowledge_base_status:
            total_docs = knowledge_base_status.get("total", 0)
            completed_docs = knowledge_base_status.get("completed", 0)
            processing_docs = knowledge_base_status.get("processing", 0)
            failed_docs = knowledge_base_status.get("failed", 0)
            
            kb_text = (
                f"\n知识库当前状态：\n"
                f"- 文档总数：{total_docs}\n"
                f"- 已处理完成：{completed_docs}\n"
                f"- 处理中：{processing_docs}\n"
                f"- 处理失败：{failed_docs}\n"
                f"- 生成模型：{self.model_name}"
            )
            
            doc_list = knowledge_base_status.get("documents", [])
            if doc_list:
                sorted_docs = sorted(doc_list, key=lambda x: x.get("created_at", ""), reverse=True)
                kb_text += "\n- 文档列表（按时间排序）："
                for doc in sorted_docs[:10]:
                    kb_text += f"\n  • {doc.get('title', '未命名')} ({doc.get('status', 'unknown')})"
                if len(sorted_docs) > 10:
                    kb_text += f"\n  ... 还有 {len(sorted_docs) - 10} 个文档"
            system_parts.append(kb_text)
        
        # 文档信息
        if document_info:
            doc_title = document_info.get("title") or f"文档_{str(document_info.get('document_id', 'unknown'))[:8]}"
            doc_info_text = (
                f"\n当前查询的文档信息：\n"
                f"- 文档标题：{doc_title}\n"
                f"- 文档类型：{document_info.get('file_type', 'unknown')}\n"
                f"- 处理状态：{document_info.get('status', 'unknown')}\n"
                f"- 文本块数量：{document_info.get('total_chunks', 0)}\n"
                f"- 向量数量：{document_info.get('total_vectors', 0)}"
            )
            metadata = document_info.get("metadata", {})
            if metadata and metadata.get("author"):
                doc_info_text += f"\n- 作者：{metadata['author']}"
            system_parts.append(doc_info_text)
        
        # RAG 上下文
        if context:
            system_parts.append(
                f"\n【核心指令】\n"
                f"请基于以下「检索知识」回答问题。\n"
                f"1. 你的回答必须严格基于提供的检索知识。\n"
                f"2. 如果检索知识中包含「知识图谱上下文」（Knowledge Graph Context），请重点关注其中的实体关系。\n"
                f"3. 如果检索知识不足以回答问题，请明确告知，禁止编造事实。\n"
                f"4. 引用知识时，请尽量自然融入回答中。\n\n"
                f"【检索知识】\n{context}"
            )

        # MCP compact 模式：工具数 > 阈值时注入工具摘要，告知 LLM 已按需加载相关工具
        try:
            from services.mcp_client_service import mcp_client_manager
            if mcp_client_manager.is_enabled and mcp_client_manager.should_use_compact_mode(self.model_name):
                compact_summary = mcp_client_manager.get_compact_tool_summary()
                if compact_summary:
                    system_parts.append(
                        f"\n【MCP 工具说明（按需加载模式）】\n"
                        f"当前 MCP 工具数量较多（> 100），已启用按需加载模式。\n\n"
                        f"## 工作方式\n"
                        f"- 系统**已根据用户问题自动检索**最相关的 MCP 工具，并注入了它们的完整 schema。\n"
                        f"- 你**直接调用**已注入的 mcp__{{server}}__{{tool}} 工具即可，**无需先调 mcp_list_tools**。\n"
                        f"- 只有当你需要的工具**不在已注入列表中**时，才调 `mcp_list_tools(server_name)` 查询其他工具。\n\n"
                        f"## 决策流程\n"
                        f"1. 查看当前 tools 列表，看是否有匹配用户需求的 mcp__ 工具。\n"
                        f"2. 如果有：直接调用（按 schema 传参）。\n"
                        f"3. 如果没有但你确信存在该 MCP 工具：调 mcp_list_tools(server_name) 查询，拿到工具名后调用。\n"
                        f"4. 如果不需要 MCP 工具（纯对话/知识问答）：直接回答，不要调任何 MCP 工具。\n\n"
                        f"## 全部 MCP 工具摘要（供你判断是否需要调 mcp_list_tools）\n"
                        f"{compact_summary}"
                    )
        except Exception as e:
            logger.debug(f"注入 MCP compact 摘要失败（非关键路径）: {e}")

        system_content = "\n".join(system_parts)
        messages = [{"role": "system", "content": system_content}]
        
        # ——— 3. Conversation history ———
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-20:] if len(conversation_history) > 20 else conversation_history
            for msg in recent_history:
                role = msg.get("role", "user")
                content = (msg.get("content", "") or "").strip()
                if not content:
                    continue
                if role in ("user", "assistant", "system"):
                    messages.append({"role": role, "content": content})
        
        # ——— 4. Build user message ———
        user_message = prompt
        
        # 处理引用内容
        if "[引用内容]" in prompt and "[/引用内容]" in prompt:
            import re
            match = re.search(r'\[引用内容\](.*?)\[/引用内容\]', prompt, re.DOTALL)
            if match:
                quoted = match.group(1).strip()
                user_q = prompt.split("[/引用内容]")[-1].strip()
                if not user_q:
                    user_q = "请针对引用的内容进行回答或解释。"
                user_message = f"用户引用的内容：\n{quoted}\n\n用户问题：{user_q}"
        
        messages.append({"role": "user", "content": user_message})
        
        # ——— 5. Process tool calls ———
        # 原生 Function Calling 模式下，工具调用在 _generate_stream 中通过 LLM 原生 tools 参数处理，
        # 不再需要在 messages 预处理阶段解析 <function_calls> XML。
        tool_context_added = False

        # ——— 6. 保存完整上下文到文件（便于调试和查看 prompt）———
        try:
            import os
            import json
            from datetime import datetime
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "prompts")
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_file = os.path.join(log_dir, f"prompt_{ts}.json")
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "assistant_id": assistant_id,
                "message_count": len(messages),
                "messages": messages,
            }
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Prompt 上下文已保存到: {log_file}")
        except Exception as e:
            logger.warning(f"保存 prompt 上下文失败（不影响主流程）: {e}")

        return messages, tool_context_added
    
    async def _generate_stream(
        self,
        messages: List[Dict[str, str]],
        assistant_id: Optional[str] = None,
        max_tool_rounds: int = 20,
        tool_execution_context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """流式生成 + 原生 Function Calling 工具调用循环

        使用 LLM 原生 tools 参数触发工具调用，通过 ToolCallAdapter 统一适配不同厂商的
        Function Calling 差异，StreamAggregator 聚合流式工具调用增量。

        Args:
            tool_execution_context: Agentic RAG 工具执行上下文，用于向 rag_retrieve
                工具注入检索范围参数和跨轮次去重状态。
        """
        from services.ai_tools import ai_tools
        from services.tool_call_adapter import (
            CanonicalToolCall,
            CanonicalToolResult,
            ToolSchemaConverter,
            serialize_tool_result,
        )

        adapter = self._adapter

        # v6 阈值模式：工具数 > 100 时走 compact 模式（按需加载），否则全量注入
        from services.mcp_client_service import mcp_client_manager
        use_compact_mode = (
            mcp_client_manager.is_enabled
            and mcp_client_manager.should_use_compact_mode(self.model_name)
        )

        if use_compact_mode:
            # compact 模式：按 user query 检索相关 MCP 工具，注入 top_k 个 schema
            # 流程：用户问题 → ToolIndex 检索 → top_k 个工具 schema → 注入 tools_payload
            user_query = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_query = (msg.get("content") or "").strip()
                    break

            dynamic_schemas = ai_tools.get_dynamic_tools_schema(
                query=user_query,
                top_k=5,
                include_mcp_list_tools=True,
            )
            canonical_tools = ToolSchemaConverter.from_ai_tools_schema(dynamic_schemas)
            tools_payload = adapter.convert_tools(canonical_tools)
            logger.info(
                f"v6 compact 模式按需加载: query={user_query[:50]!r}, "
                f"tools_payload 数量={len(tools_payload)}"
            )
        else:
            # 全量模式：注入所有工具 schema
            canonical_tools = ToolSchemaConverter.from_ai_tools_schema(ai_tools.get_tools_schema())
            tools_payload = adapter.convert_tools(canonical_tools)

        tool_choice = adapter.convert_tool_choice("auto")

        current_messages = list(messages)

        for round_idx in range(max_tool_rounds + 1):
            try:
                logger.debug(
                    f"流式生成 - 第{round_idx + 1}轮, 模型: {self.model_name}, "
                    f"messages: {len(current_messages)} 条"
                )

                # 使用 AsyncOpenAI 客户端，原生 tools 参数触发 Function Calling
                stream = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=current_messages,
                    stream=True,
                    tools=tools_payload,
                    tool_choice=tool_choice,
                    timeout=self.timeout,
                )

                aggregator = adapter.create_stream_aggregator()
                assistant_text = ""
                canonical_calls: List[CanonicalToolCall] = []
                chunk_count = 0

                async for chunk in stream:
                    events = aggregator.process_chunk(chunk)
                    thinking_emitted = False
                    for event in events:
                        etype = event["type"]
                        if etype == "text":
                            chunk_count += 1
                            text = event["content"]
                            assistant_text += text
                            yield text
                        elif etype == "thinking":
                            thinking_emitted = True
                            yield "\x1eTHINKING:" + json.dumps(
                                {"content": event["content"]}, ensure_ascii=False
                            ) + "\x1e"
                        elif etype == "tool_call":
                            canonical_calls.append(event["tool_call"])
                        elif etype == "finish":
                            # 流结束标记（tool_calls 已在 finish_reason=tool_calls 时聚合完毕）
                            pass

                    # 兼容：部分厂商将 reasoning_content 放在 delta.model_extra 中，
                    # 适配器的 StreamAggregator 仅检查直接属性，这里补充捕获，避免思考链丢失
                    if not thinking_emitted and chunk.choices:
                        delta = chunk.choices[0].delta
                        reasoning = None
                        if hasattr(delta, "model_extra"):
                            reasoning = (delta.model_extra or {}).get("reasoning_content")
                        if reasoning:
                            yield "\x1eTHINKING:" + json.dumps(
                                {"content": reasoning}, ensure_ascii=False
                            ) + "\x1e"

                logger.info(
                    f"流式生成 - 第{round_idx + 1}轮完成, {chunk_count} 块, "
                    f"文本长度 {len(assistant_text)}, 工具调用: {len(canonical_calls)} 个"
                )

            except APITimeoutError:
                logger.error(f"API 请求超时 (timeout={self.timeout}s)")
                raise
            except APIConnectionError as e:
                logger.error(f"API 连接错误: {e}")
                raise
            except APIError as e:
                logger.error(f"API 错误: status={e.status_code}, message={e.message}")
                raise
            except Exception as e:
                logger.error(f"流式生成错误: {str(e)}", exc_info=True)
                raise

            # 无工具调用，或已达最大轮次：结束流
            if not canonical_calls or round_idx >= max_tool_rounds:
                return

            # —— 执行工具调用 ——
            tool_results = []  # [{"tool", "params", "result", "tool_call_id"}]
            canonical_results: List[CanonicalToolResult] = []

            for tc in canonical_calls:
                tool_name = tc.name
                params = dict(tc.arguments or {})

                if not tool_name or tool_name not in ai_tools.functions:
                    logger.warning(f"未知工具函数: '{tool_name}'，跳过")
                    err = {"success": False, "error": f"未知工具: {tool_name}"}
                    tool_results.append({"tool": tool_name, "params": params, "result": err, "tool_call_id": tc.id})
                    canonical_results.append(CanonicalToolResult(
                        tool_call_id=tc.id,
                        content=serialize_tool_result(err),
                        is_error=True,
                    ))
                    continue

                # assistant_id 自动注入：基于 schema 检查是否需要注入
                tool_schema = ai_tools.tools.get(tool_name, {})
                tool_params_schema = tool_schema.get("parameters", {}).get("properties", {})
                if "assistant_id" in tool_params_schema and "assistant_id" not in params and assistant_id:
                    params["assistant_id"] = assistant_id

                try:
                    # === Agentic RAG: rag_retrieve 工具特殊处理 ===
                    # 注入 tool_execution_context 中的检索范围参数和去重状态
                    if tool_name == "rag_retrieve" and tool_execution_context is not None:
                        retrieval_ctx = tool_execution_context.get("retrieval_context") or {}
                        max_retrievals = retrieval_ctx.get("max_retrievals", 5)
                        current_count = retrieval_ctx.get("total_retrieval_count", 0)
                        if current_count >= max_retrievals:
                            result = {
                                "error": f"已达到最大检索次数限制（{max_retrievals}次），请基于已有信息回答",
                                "max_retrievals_reached": True,
                                "query": params.get("query", ""),
                                "chunks": [],
                                "total_found": 0,
                            }
                            logger.warning(f"rag_retrieve 达到最大检索次数限制: {max_retrievals}")
                        else:
                            # 从 tool_execution_context 读取前置 plan（对话路径注入），让工具复用 plan 的 rewritten_queries/final_k 等
                            tool_plan = tool_execution_context.get("query_plan")
                            result = await ai_tools.rag_retrieve_with_context(
                                query=params.get("query", ""),
                                strategy=params.get("strategy", "auto"),
                                top_k=params.get("top_k", 5),
                                min_score=params.get("min_score", 0.3),
                                document_id=tool_execution_context.get("document_id"),
                                assistant_id=tool_execution_context.get("assistant_id") or assistant_id,
                                knowledge_space_ids=tool_execution_context.get("knowledge_space_ids"),
                                embedding_model=tool_execution_context.get("embedding_model"),
                                exclude_chunk_ids=retrieval_ctx.get("seen_chunk_ids", set()),
                                plan=tool_plan,
                            )
                            # 更新去重状态
                            retrieval_ctx["total_retrieval_count"] = current_count + 1
                            for chunk in result.get("chunks", []):
                                cid = chunk.get("chunk_id")
                                if cid:
                                    retrieval_ctx.setdefault("seen_chunk_ids", set()).add(cid)
                            logger.info(
                                f"rag_retrieve 调用成功 (第{current_count + 1}次), "
                                f"返回 {result.get('total_found', 0)} 个片段"
                            )
                    else:
                        result = await ai_tools.async_call_tool(tool_name, params if params else None)
                        logger.info(f"工具调用成功: {tool_name}")

                    tool_results.append({"tool": tool_name, "params": params, "result": result, "tool_call_id": tc.id})
                    canonical_results.append(CanonicalToolResult(
                        tool_call_id=tc.id,
                        content=serialize_tool_result(result),
                        is_error=isinstance(result, dict) and result.get("success") is False,
                    ))
                except Exception as e:
                    logger.error(f"工具调用 {tool_name} 失败: {str(e)}", exc_info=True)
                    err = {"success": False, "error": str(e)}
                    tool_results.append({"tool": tool_name, "params": params, "result": err, "tool_call_id": tc.id})
                    canonical_results.append(CanonicalToolResult(
                        tool_call_id=tc.id,
                        content=serialize_tool_result(err),
                        is_error=True,
                    ))

            if not tool_results:
                return

            # 向前端发送工具调用事件（ASCII 记录分隔符 \x1e 包裹，agent.execute 会解析）
            tool_call_event = {
                "round": round_idx + 1,
                "tools": [
                    {
                        "name": tr["tool"],
                        "params": tr["params"],
                        "success": not (isinstance(tr["result"], dict) and tr["result"].get("success") is False),
                        "result": tr["result"],
                    }
                    for tr in tool_results
                ],
            }
            yield "\x1eTOOL_CALL:" + json.dumps(tool_call_event, ensure_ascii=False) + "\x1e"

            # v6 工具结果落盘：大的结果存磁盘，messages 里只放引用（节省 token）
            # SSE 事件已发送完整结果给前端，这里只替换注入到 current_messages 的内容
            try:
                from services.tool_result_store import tool_result_store
                for i, tr in enumerate(tool_results):
                    if tool_result_store.should_store(tr["result"]):
                        file_path = tool_result_store.store(
                            tool_name=tr["tool"],
                            arguments=tr.get("params") or {},
                            result=tr["result"],
                        )
                        if file_path:
                            # 替换 tool_results 中的结果为引用文本（用于 results_text 拼接）
                            ref_text = tool_result_store.make_reference(tr["tool"], tr["result"], file_path)
                            tool_results[i]["result"] = ref_text
                            # 替换 canonical_results 中的 content（用于 tool_result_messages 构造）
                            if i < len(canonical_results):
                                canonical_results[i].content = ref_text
                            logger.debug(f"工具结果已落盘替换: tool={tr['tool']}, file={os.path.basename(file_path)}")
            except Exception as e:
                logger.warning(f"工具结果落盘处理失败（非关键路径，使用原始结果）: {e}")

            # —— 阶段三：Observe（观察）+ Reflect（反思）——
            # 对 rag_retrieve 工具结果做证据验证和反思，决定是否继续检索
            reflection_result = None
            if tool_execution_context is not None:
                reflection_result = await self._observe_and_reflect(
                    tool_results, tool_execution_context, round_idx + 1
                )

            # 用适配器构造厂商特定的工具结果消息（替代原来的文本拼接）
            tool_result_messages = adapter.build_tool_result_messages(
                assistant_content=assistant_text,
                tool_calls=canonical_calls,
                tool_results=canonical_results,
            )
            current_messages.extend(tool_result_messages)

            # —— 基于反思结果动态生成下一轮提示 ——
            results_text = "\n\n工具函数调用结果：\n"
            for tr in tool_results:
                results_text += f"\n调用 {tr['tool']} 的结果：\n{json.dumps(tr['result'], ensure_ascii=False, indent=2)}\n"

            if reflection_result and not reflection_result.get("sufficient", True):
                # 证据不足，引导 Agent 继续检索
                gaps_text = "；".join(reflection_result.get("gaps", [])) or "证据不够充分"
                next_query_hint = ""
                if reflection_result.get("next_query") and reflection_result.get("next_query") != reflection_result.get("original_query", ""):
                    next_query_hint = f"\n建议尝试用不同的查询关键词再次检索，例如：'{reflection_result['next_query']}'"
                current_messages.append({
                    "role": "user",
                    "content": (
                        f"工具调用结果如下：{results_text}\n\n"
                        f"⚠️ 检索反思：{gaps_text}。{next_query_hint}\n"
                        f"如果认为需要更多信息，请调用 rag_retrieve 工具再次检索（换用不同关键词或策略）。"
                        f"如果认为现有信息已足够回答，请直接给出最终回答。"
                    )
                })
                logger.info(f"阶段三反思: 证据不足({gaps_text})，引导继续检索 (round={round_idx+1})")
            else:
                # 证据充分或无反思（非 RAG 工具）
                # 检查是否有失败的工具调用，如果有则允许 LLM 修正参数重试
                failed_tools = [tr for tr in tool_results if isinstance(tr.get("result"), dict) and not tr["result"].get("success", True)]
                if failed_tools and round_idx < max_tool_rounds:
                    failed_text = "\n".join(
                        f"工具 {tr['tool']} 失败: {tr['result'].get('error', '未知错误')}"
                        for tr in failed_tools
                    )
                    current_messages.append({
                        "role": "user",
                        "content": (
                            f"工具调用结果如下：{results_text}\n\n"
                            f"⚠️ 以下工具调用失败：\n{failed_text}\n\n"
                            f"请检查失败原因并修正参数后重新调用工具。"
                            f"常见问题：数组类型参数应直接写 JSON 数组（如 [\"web\"]）而非字符串；"
                            f"如果参数格式没问题，可以换一种方式调用或基于已有信息回答。"
                        )
                    })
                    logger.info(f"工具调用失败，引导 LLM 修正参数重试 (round={round_idx+1})")
                else:
                    # 无失败或已到最大轮次，直接回答
                    current_messages.append({
                        "role": "user",
                        "content": (
                            f"工具调用结果如下：{results_text}\n\n"
                            f"请基于以上工具返回的实时数据，用自然语言回答用户的问题。"
                            f"直接给出最终回答。"
                        )
                    })

    async def _observe_and_reflect(
        self,
        tool_results: List[Dict[str, Any]],
        tool_execution_context: Dict[str, Any],
        round_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """阶段三：Observe + Reflect

        1. 从 rag_retrieve 结果中提取证据（Observe）
        2. 调用 EvidenceVerifier 验证证据相关性
        3. 调用 Reflector 判断是否充分，决定下一步

        Returns:
            Reflection 字典（sufficient, gaps, next_action, next_query, ...）
            若无 rag_retrieve 调用则返回 None
        """
        retrieval_ctx = tool_execution_context.get("retrieval_context") or {}

        # 只对 rag_retrieve 工具做反思
        rag_results = [tr for tr in tool_results if tr.get("tool") == "rag_retrieve"]
        if not rag_results:
            return None

        try:
            from services.evidence_verifier import evidence_verifier, reflector

            # —— Observe：收集本轮证据 ——
            all_chunks = []
            original_query = ""
            for tr in rag_results:
                result = tr.get("result") or {}
                if result.get("error"):
                    continue
                chunks = result.get("chunks", [])
                query = result.get("query", "")
                if not original_query:
                    original_query = query
                all_chunks.extend(chunks)

            if not all_chunks:
                # 检索无结果，证据不足
                obs_list = retrieval_ctx.setdefault("observations", [])
                obs_list.append({
                    "round": round_idx,
                    "query": original_query,
                    "evidence_count": 0,
                    "top_score": 0.0,
                })
                return {
                    "sufficient": False,
                    "gaps": ["检索无结果"],
                    "next_action": "retrieve_more",
                    "next_query": original_query,
                    "reason": "检索返回 0 条结果",
                    "verified_count": 0,
                    "source": "rules",
                    "original_query": original_query,
                }

            # —— Verify：证据验证 ——
            verified_chunks = await evidence_verifier.verify(original_query, all_chunks)
            verified_count = sum(1 for c in verified_chunks if c.get("verified"))
            top_score = max((c.get("score", 0) for c in all_chunks), default=0.0)

            # 收集验证后的证据到 tool_execution_context（供 Agent 层回收）
            collected_evidence = retrieval_ctx.setdefault("collected_evidence", [])
            for chunk in verified_chunks:
                collected_evidence.append({
                    **chunk,
                    "retrieved_at_round": round_idx,
                    "query": original_query,
                })

            # 记录观察
            obs_list = retrieval_ctx.setdefault("observations", [])
            obs_list.append({
                "round": round_idx,
                "query": original_query,
                "evidence_count": len(all_chunks),
                "top_score": top_score,
                "verified_count": verified_count,
            })

            # —— Reflect：反思充分性 ——
            # 传入 verified_chunks 用于覆盖度分析（增强 rules 模式）
            verified_chunks_for_reflect = [c for c in collected_evidence if c.get("verified")]
            reflection = await reflector.reflect(
                query=original_query,
                observations=obs_list,
                verified_count=len({c.get("chunk_id") for c in verified_chunks_for_reflect}),
                total_retrieval_count=retrieval_ctx.get("total_retrieval_count", 0),
                max_retrievals=retrieval_ctx.get("max_retrievals", 5),
                verified_chunks=verified_chunks_for_reflect,
            )
            reflection["original_query"] = original_query

            logger.info(
                f"阶段三反思 (round={round_idx}): sufficient={reflection.get('sufficient')}, "
                f"verified={reflection.get('verified_count')}, "
                f"next_action={reflection.get('next_action')}, "
                f"source={reflection.get('source')}"
            )
            return reflection

        except Exception as e:
            logger.warning(f"阶段三 Observe+Reflect 异常: {e}，跳过反思")
            return None

    async def _generate_once(self, messages: List[Dict[str, str]]) -> str:
        """非流式生成"""
        try:
            logger.debug(f"开始非流式生成 - 模型: {self.model_name}, messages: {len(messages)} 条")

            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False,
                timeout=self.timeout
            )
            
            content = response.choices[0].message.content or ""
            logger.debug(f"非流式生成完成 - 回复长度: {len(content)}")
            return content
            
        except APITimeoutError:
            logger.error(f"API 请求超时 (timeout={self.timeout}s)")
            raise
        except APIConnectionError as e:
            logger.error(f"API 连接错误: {e}")
            raise
        except APIError as e:
            logger.error(f"API 错误: status={e.status_code}, message={e.message}")
            raise
        except Exception as e:
            logger.error(f"非流式生成错误: {str(e)}", exc_info=True)
            raise


# 全局模型服务实例
llm_service = LLMService()
