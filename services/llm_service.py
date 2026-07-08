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
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """生成回复（流式或非流式）"""
        messages, tool_context_added = await self._build_messages(
            prompt, context,
            document_id=document_id,
            document_info=document_info,
            knowledge_base_status=knowledge_base_status,
            assistant_id=assistant_id,
            conversation_history=conversation_history
        )
        
        if stream:
            async for chunk in self._generate_stream(messages, assistant_id=assistant_id):
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
        tool_context_added = await self._process_tool_calls_in_messages(messages, assistant_id=assistant_id)

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
    
    async def _process_tool_calls_in_messages(
        self,
        messages: List[Dict[str, str]],
        assistant_id: Optional[str] = None
    ) -> bool:
        """处理 messages 中的工具函数调用"""
        import re
        from services.ai_tools import ai_tools
        
        last_user_msg = messages[-1]["content"] if messages else ""
        if not last_user_msg:
            return False
        
        pattern = r'<function_calls>\s*<invoke\s+name="([^"]+)">\s*(.*?)\s*</invoke>\s*</function_calls>'
        matches = list(re.finditer(pattern, last_user_msg, re.DOTALL))
        if not matches:
            return False
        
        tool_results = []
        for match in matches:
            tool_name = match.group(1).strip()
            params_text = match.group(2)
            
            if not tool_name or tool_name in [
                "工具函数名称", "function_name", "tool_name",
                "函数名称", "实际的工具函数名称"
            ]:
                logger.warning(f"检测到占位符工具函数名称: '{tool_name}'，跳过调用")
                continue
            
            if tool_name not in ai_tools.functions:
                logger.warning(f"未知的工具函数: '{tool_name}'")
                continue
            
            # 解析参数
            params = {}
            param_pattern = r'<parameter\s+name="([^"]+)">([^<]+)</parameter>'
            for pm in re.finditer(param_pattern, params_text):
                pname = pm.group(1)
                pvalue = pm.group(2).strip()
                if pvalue.isdigit():
                    params[pname] = int(pvalue)
                elif '.' in pvalue and pvalue.replace('.', '').isdigit():
                    params[pname] = float(pvalue)
                elif pvalue.lower() in ('true', 'false'):
                    params[pname] = pvalue.lower() == 'true'
                else:
                    params[pname] = pvalue
            
            # 自动注入 assistant_id
            tool_schema = ai_tools.tools.get(tool_name, {})
            tool_params_schema = tool_schema.get("parameters", {}).get("properties", {})
            if "assistant_id" in tool_params_schema and "assistant_id" not in params and assistant_id:
                params["assistant_id"] = assistant_id
            
            try:
                result = await ai_tools.async_call_tool(tool_name, params if params else None)
                tool_results.append({"tool": tool_name, "result": result})
                logger.info(f"成功调用工具函数: {tool_name}")
            except Exception as e:
                logger.error(f"调用工具函数 {tool_name} 失败: {str(e)}")
                tool_results.append({"tool": tool_name, "result": {"success": False, "error": str(e)}})
        
        if tool_results:
            results_text = "\n\n工具函数调用结果：\n"
            for tr in tool_results:
                results_text += f"\n调用 {tr['tool']} 的结果：\n{json.dumps(tr['result'], ensure_ascii=False, indent=2)}\n"
            messages[-1]["content"] += results_text
            return True
        
        return False
    
    async def _generate_stream(
        self,
        messages: List[Dict[str, str]],
        assistant_id: Optional[str] = None,
        max_tool_rounds: int = 4,
    ) -> AsyncGenerator[str, None]:
        """流式生成 + 工具调用循环"""
        import re
        from services.ai_tools import ai_tools

        pattern = r'<function_calls>\s*<invoke\s+name="([^"]+)">\s*(.*?)\s*</invoke>\s*</function_calls>'
        tool_call_start = '<function_calls>'
        param_pattern = r'<parameter\s+name="([^"]+)">([^<]+)</parameter>'
        placeholder_names = {"工具函数名称", "function_name", "tool_name", "函数名称", "实际的工具函数名称"}

        current_messages = list(messages)

        for round_idx in range(max_tool_rounds + 1):
            try:
                logger.debug(f"流式生成 - 第{round_idx + 1}轮, 模型: {self.model_name}, messages: {len(current_messages)} 条")

                # 使用 AsyncOpenAI 客户端，避免同步迭代阻塞事件循环
                stream = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=current_messages,
                    stream=True,
                    timeout=self.timeout
                )

                full_response = ""
                tool_call_detected = False
                chunk_count = 0

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            chunk_count += 1
                            full_response += delta.content

                            if tool_call_detected:
                                # 已检测到工具调用，不再输出（等待完整标签后处理）
                                continue
                            if tool_call_start in full_response:
                                tool_call_detected = True
                                # 先输出标签之前的正常文本
                                idx = full_response.find(tool_call_start)
                                before_text = full_response[:idx]
                                if before_text:
                                    yield before_text
                                continue
                            yield delta.content

                logger.info(f"流式生成 - 第{round_idx + 1}轮完成, {chunk_count} 块, 长度 {len(full_response)}, 工具调用: {tool_call_detected}")

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

            if not tool_call_detected or round_idx >= max_tool_rounds:
                if tool_call_detected and full_response:
                    cleaned = re.sub(pattern, '', full_response).strip()
                    if cleaned:
                        yield cleaned
                return

            matches = list(re.finditer(pattern, full_response, re.DOTALL))

            tool_results = []
            for match in matches:
                tool_name = match.group(1).strip()
                params_text = match.group(2)

                if not tool_name or tool_name in placeholder_names:
                    logger.warning(f"占位符工具名: '{tool_name}'，跳过")
                    continue
                if tool_name not in ai_tools.functions:
                    logger.warning(f"未知工具函数: '{tool_name}'")
                    continue

                params = {}
                for pm in re.finditer(param_pattern, params_text):
                    pname = pm.group(1)
                    pvalue = pm.group(2).strip()
                    if pvalue.isdigit():
                        params[pname] = int(pvalue)
                    elif '.' in pvalue and pvalue.replace('.', '').isdigit():
                        params[pname] = float(pvalue)
                    elif pvalue.lower() in ('true', 'false'):
                        params[pname] = pvalue.lower() == 'true'
                    else:
                        params[pname] = pvalue

                tool_schema = ai_tools.tools.get(tool_name, {})
                tool_params_schema = tool_schema.get("parameters", {}).get("properties", {})
                if "assistant_id" in tool_params_schema and "assistant_id" not in params and assistant_id:
                    params["assistant_id"] = assistant_id

                try:
                    result = await ai_tools.async_call_tool(tool_name, params if params else None)
                    tool_results.append({"tool": tool_name, "params": params, "result": result})
                    logger.info(f"工具调用成功: {tool_name}")
                except Exception as e:
                    logger.error(f"工具调用 {tool_name} 失败: {str(e)}", exc_info=True)
                    tool_results.append({"tool": tool_name, "params": params, "result": {"success": False, "error": str(e)}})

            if not tool_results:
                cleaned = re.sub(pattern, '', full_response).strip()
                if cleaned:
                    yield cleaned
                return

            # 向前端发送工具调用事件（用 ASCII 记录分隔符 \x1e 包裹，agent.execute 会解析并转换为 tool_call 事件）
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

            results_text = "\n\n工具函数调用结果：\n"
            for tr in tool_results:
                results_text += f"\n调用 {tr['tool']} 的结果：\n{json.dumps(tr['result'], ensure_ascii=False, indent=2)}\n"

            current_messages.append({"role": "assistant", "content": full_response})
            current_messages.append({
                "role": "user",
                "content": (
                    f"工具调用结果如下：{results_text}\n\n"
                    f"请基于以上工具返回的实时数据，用自然语言回答用户的问题。"
                    f"不要再输出 <function_calls> 格式，直接给出最终回答。"
                )
            })

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
