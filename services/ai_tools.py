"""AI智能体工具函数库 - 允许AI调用这些函数获取基础数据"""
from typing import Dict, Any, List, Optional, Callable
from utils.logger import logger
import requests
import os
import json
import asyncio
from database.mongodb import mongodb


class AITools:
    """AI工具函数库"""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.functions: Dict[str, Callable] = {}
        self._async_tools = None
        self._register_tools()
    
    def _register_tools(self):
        """注册所有可用的工具函数"""
        # 工具1: 获取当前使用的模型信息
        self.register_tool(
            name="get_available_ollama_models",
            description="获取当前使用的模型信息。当用户询问可用模型、模型列表、有哪些模型、当前用什么模型等问题时调用此函数。",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            function=self._get_available_ollama_models
        )
        
        # 工具2: 获取知识库文档列表
        self.register_tool(
            name="get_knowledge_base_documents",
            description="获取知识库中的文档列表。当用户询问知识库有哪些文档、文档列表、文档数量、知识库现在有什么文档等问题时，必须调用此函数来实时获取最新信息。",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "返回的文档数量限制，默认10",
                        "default": 10
                    },
                    "assistant_id": {
                        "type": "string",
                        "description": "助手ID（可选），如果提供则获取该助手知识库的文档列表"
                    }
                },
                "required": []
            },
            function=self._get_knowledge_base_documents
        )
        
        # 工具3: 获取系统信息
        self.register_tool(
            name="get_system_info",
            description="获取系统基本信息，包括当前使用的模型、向量化模型、知识库状态等。当用户询问系统信息、当前配置、用了什么模型、知识库情况等问题时，必须调用此函数来实时获取最新信息。",
            parameters={
                "type": "object",
                "properties": {
                    "assistant_id": {
                        "type": "string",
                        "description": "助手ID（可选），如果提供则获取该助手的特定配置"
                    }
                },
                "required": []
            },
            function=self._get_system_info
        )
        
        # 工具4: 获取知识库详细统计
        self.register_tool(
            name="get_knowledge_base_stats",
            description="获取知识库的详细统计信息，包括文档数量、向量数量、各状态文档统计等。当用户询问知识库状态、知识库信息、知识库统计等问题时，必须调用此函数来实时获取最新信息。",
            parameters={
                "type": "object",
                "properties": {
                    "assistant_id": {
                        "type": "string",
                        "description": "助手ID（可选），如果提供则获取该助手知识库的统计信息"
                    }
                },
                "required": []
            },
            function=self._get_knowledge_base_stats
        )

        # —— Letta 记忆系统工具（Core Memory / Archival Memory / Recall 检索）——
        # 这 5 个工具均为 async 实现，已在 async_call_tool 的 async_tools 字典里映射
        from services.core_memory_service import core_memory_service
        from services.archival_memory_service import archival_memory_service

        # 工具5: core_memory_append
        self.register_tool(
            name="core_memory_append",
            description=(
                "向核心记忆追加内容。用于记录用户明确要求你长期记住的信息（如偏好、身份、长期目标）。"
                "核心记忆会常驻在系统提示词中，不需要每次检索。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "记忆块名称，如 persona（你的人设）/ human（用户画像）",
                    },
                    "content": {"type": "string", "description": "要追加的内容"},
                },
                "required": ["label", "content"],
            },
            function=lambda label, content: core_memory_service.append(
                "global", "default", label, content
            ),
        )

        # 工具6: core_memory_replace
        self.register_tool(
            name="core_memory_replace",
            description=(
                "替换核心记忆中的过时内容。用于修正之前记错或已经变化的信息。"
                "old_content 必须是核心记忆中已存在的精确文本片段。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "记忆块名称"},
                    "old_content": {"type": "string", "description": "要被替换的精确文本"},
                    "new_content": {"type": "string", "description": "替换后的新文本"},
                },
                "required": ["label", "old_content", "new_content"],
            },
            function=lambda label, old_content, new_content: core_memory_service.replace(
                "global", "default", label, old_content, new_content
            ),
        )

        # 工具7: archival_memory_insert
        self.register_tool(
            name="archival_memory_insert",
            description=(
                "把一段值得长期保存但不需要一直占用上下文的信息归档。"
                "之后可以用 archival_memory_search 检索回来。"
                "适用于：用户提到的项目背景、技术决策、长期任务状态等。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "要归档的文本"},
                },
                "required": ["content"],
            },
            function=lambda content: archival_memory_service.insert(
                "global", "default", content
            ),
        )

        # 工具8: archival_memory_search
        self.register_tool(
            name="archival_memory_search",
            description=(
                "语义检索归档记忆。当你怀疑之前聊过某个话题但当前上下文里没有时调用。"
                "返回最相关的若干条归档记录。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问题"},
                    "top_k": {"type": "integer", "default": 5, "description": "返回结果数"},
                },
                "required": ["query"],
            },
            function=lambda query, top_k=5: archival_memory_service.search(
                "global", "default", query, top_k
            ),
        )

        # 工具9: conversation_search（Recall Memory 文本检索）
        self.register_tool(
            name="conversation_search",
            description=(
                '在历史对话中按关键词检索。'
                '当用户问"我们之前聊过什么关于X的"或需要回溯某次对话时调用。'
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词"},
                    "limit": {"type": "integer", "default": 5, "description": "返回结果数"},
                },
                "required": ["query"],
            },
            function=self._conversation_search,
        )

        # 工具10: rag_retrieve（Agentic RAG 自适应检索工具）
        # 该工具让 Agent 在生成过程中主动决定是否检索、检索什么、用什么策略
        # 真正的 async 实现是 rag_retrieve_with_context，由 llm_service 直接调用（不走 async_call_tool）
        self.register_tool(
            name="rag_retrieve",
            description=(
                "从知识库检索与查询相关的文档片段。当用户问题涉及具体事实、技术细节、"
                "或需要引用文档内容时调用此工具。可指定检索策略：\n"
                "- 'auto'（默认）：自动选择向量+BM25+图谱混合检索\n"
                "- 'vector'：仅向量语义检索，适合模糊/概念性查询\n"
                "- 'keyword'：仅关键词检索，适合专有名词/代码/ID\n"
                "- 'graph'：仅图谱检索，适合关系/路径查询\n"
                "可多次调用以迭代补充上下文，系统会自动去重。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询语句。应聚焦于具体信息需求，避免复述整个用户问题。"
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["auto", "vector", "keyword", "graph"],
                        "default": "auto",
                        "description": "检索策略"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                        "description": "返回的文档片段数量"
                    },
                    "min_score": {
                        "type": "number",
                        "default": 0.3,
                        "description": "相关性阈值，低于此分数的结果将被过滤"
                    },
                },
                "required": ["query"]
            },
            function=self._rag_retrieve_sync_placeholder,
        )
    
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Callable
    ):
        """
        注册一个工具函数
        
        Args:
            name: 工具名称
            description: 工具描述
            parameters: 工具参数定义（JSON Schema格式）
            function: 工具函数实现
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters
        }
        self.functions[name] = function
        logger.debug(f"注册AI工具: {name}")
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        获取所有工具的JSON Schema定义

        Returns:
            工具列表（OpenAI Function Calling格式）
        """
        return list(self.tools.values())

    def get_dynamic_tools_schema(
        self,
        query: str,
        top_k: int = 5,
        include_mcp_list_tools: bool = True,
    ) -> List[Dict[str, Any]]:
        """v6 compact 模式按需加载：返回内置工具 + 与 query 相关的 MCP 工具 schema

        compact 模式下替代 get_tools_schema()，避免把所有 MCP 工具 schema 都塞进 prompt。
        流程：
            1. 取所有内置工具 schema（self.tools 中非 mcp__ 前缀的，包括 mcp_list_tools）
            2. 调 mcp_client_manager.get_relevant_tool_schemas(query, top_k) 取相关 MCP 工具
            3. 可选加上 mcp_list_tools 元工具（兜底，让 LLM 能查询其他工具）

        Args:
            query: 用户查询
            top_k: 检索的 MCP 工具数量（建议 3~10）
            include_mcp_list_tools: 是否包含 mcp_list_tools 元工具作为兜底

        Returns:
            工具 schema 列表（OpenAI Function Calling 格式）
        """
        schemas: List[Dict[str, Any]] = []

        # 1. 内置工具（非 mcp__ 前缀的工具）
        for tool_schema in self.tools.values():
            name = tool_schema.get("name", "")
            if not name.startswith("mcp__") and name != "mcp_list_tools":
                schemas.append(tool_schema)

        # 2. 按 query 检索相关 MCP 工具
        try:
            from services.mcp_client_service import mcp_client_manager
            if mcp_client_manager.is_enabled and query and query.strip():
                relevant_mcp_tools = mcp_client_manager.get_relevant_tool_schemas(query, top_k=top_k)
                schemas.extend(relevant_mcp_tools)
        except Exception as e:
            logger.warning(f"get_dynamic_tools_schema: 检索 MCP 工具失败，仅返回内置工具: {e}")

        # 3. 兜底元工具：mcp_list_tools（让 LLM 在检索结果不准时能主动查询）
        if include_mcp_list_tools:
            list_tools_schema = self.tools.get("mcp_list_tools")
            if list_tools_schema:
                schemas.append(list_tools_schema)

        mcp_count = len([s for s in schemas if s.get("name", "").startswith("mcp__")])
        builtin_count = len(schemas) - mcp_count
        logger.info(
            f"get_dynamic_tools_schema: query={query[:50]!r}, "
            f"内置={builtin_count}, MCP相关={mcp_count}, 总计={len(schemas)}"
        )
        return schemas
    
    def _filter_tool_arguments(self, name: str, arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """只保留该工具 schema 中声明的参数，忽略模型误传的占位符（如 参数名、参数值）

        MCP 工具（name 以 mcp__ 开头）不过滤，直接透传参数：
        - MCP 工具 schema 不在 self.tools 中（按需加载模式），由 MCP Server 自己校验
        - 避免参数被误清空导致 MCP 工具调用失败
        """
        if not arguments:
            return {}
        # MCP 工具直接透传参数（schema 由 MCP Server 校验）
        if name.startswith("mcp__"):
            return arguments
        schema = self.tools.get(name, {}).get("parameters", {})
        allowed = schema.get("properties", {})
        if not allowed:
            return {}
        return {k: v for k, v in arguments.items() if k in allowed}

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """
        同步调用指定的工具函数（注意：在已运行事件循环的线程中调用含 MongoDB 的工具会报错，请使用 async_call_tool）
        
        Args:
            name: 工具名称
            arguments: 工具参数
        
        Returns:
            工具函数的返回值
        """
        if name not in self.functions:
            raise ValueError(f"未知的工具函数: {name}")
        filtered = self._filter_tool_arguments(name, arguments)
        try:
            func = self.functions[name]
            if filtered:
                return func(**filtered)
            return func()
        except Exception as e:
            logger.error(f"调用工具函数 {name} 失败: {str(e)}", exc_info=True)
            raise

    def _get_async_tools(self):
        """懒加载并复用 async_tools 字典，避免每次调用都重建"""
        if self._async_tools is None:
            from services.core_memory_service import core_memory_service
            from services.archival_memory_service import archival_memory_service

            self._async_tools = {
                "get_knowledge_base_documents": self._aget_knowledge_base_documents,
                "get_system_info": self._aget_system_info,
                "get_knowledge_base_stats": self._aget_knowledge_base_stats,
                # Letta 记忆系统工具
                "core_memory_append": lambda **kw: core_memory_service.append(
                    "global", "default", kw.get("label", ""), kw.get("content", "")
                ),
                "core_memory_replace": lambda **kw: core_memory_service.replace(
                    "global", "default",
                    kw.get("label", ""),
                    kw.get("old_content", ""),
                    kw.get("new_content", ""),
                ),
                "archival_memory_insert": lambda **kw: archival_memory_service.insert(
                    "global", "default", kw.get("content", "")
                ),
                "archival_memory_search": lambda **kw: archival_memory_service.search(
                    "global", "default", kw.get("query", ""), kw.get("top_k", 5)
                ),
                "conversation_search": self._conversation_search,
            }
        return self._async_tools

    async def async_call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """
        异步调用指定的工具函数（在已有事件循环的上下文中使用，避免 asyncio.run 与 Motor 跨 loop 问题）

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具函数的返回值
        """
        if name not in self.functions:
            raise ValueError(f"未知的工具函数: {name}")
        filtered = self._filter_tool_arguments(name, arguments)

        # 使用 MongoDB 的异步工具：直接 await 异步实现
        # 注意：core_memory_* / archival_memory_* 的 function 是 lambda 包裹的 async 调用，
        # 调用 lambda 会返回一个 coroutine，await 后得到真实结果
        async_tools = self._get_async_tools()
        if name in async_tools:
            try:
                func = async_tools[name]
                if filtered:
                    return await func(**filtered)
                return await func()
            except Exception as e:
                logger.error(f"调用工具函数 {name} 失败: {str(e)}", exc_info=True)
                raise
        
        # 纯同步工具（如 get_available_ollama_models）在线程池中执行，避免阻塞事件循环
        try:
            loop = asyncio.get_event_loop()
            func = self.functions[name]
            if filtered:
                return await loop.run_in_executor(None, lambda: func(**filtered))
            return await loop.run_in_executor(None, func)
        except Exception as e:
            logger.error(f"调用工具函数 {name} 失败: {str(e)}", exc_info=True)
            raise

    def register_mcp_tools(self, manager, force_compact: Optional[bool] = None) -> int:
        """将 MCP Server 的工具动态注册为 ai_tools

        v6 恢复阈值模式：
        - 工具数 ≤ TOOL_COUNT_THRESHOLD(100)：全量注册（每个 MCP 工具进 self.tools）
        - 工具数 > 阈值：compact 模式，只注册 wrapper + mcp_list_tools 元工具
          LLM 通过 ToolIndex 按需加载相关工具 schema

        Args:
            manager: MCPClientManager 实例
            force_compact: 强制指定 compact 模式
                - True：compact 模式（只注册 wrapper + mcp_list_tools 元工具）
                - False：全量模式（注册所有 MCP 工具到 self.tools）
                - None：根据 manager.should_use_compact_mode() 自动判断

        Returns:
            注册的工具数量
        """
        if not manager.is_enabled:
            logger.info("MCP 未启用或未初始化，跳过 MCP 工具注册")
            return 0

        # 清理旧注册的 MCP 工具（热更新场景）
        self._cleanup_mcp_tools()

        # 判断是否启用 compact 模式
        use_compact = force_compact if force_compact is not None else manager.should_use_compact_mode()

        if use_compact:
            # compact 模式：只注册 wrapper + mcp_list_tools 元工具
            registered = self._register_mcp_compact_mode(manager)
            logger.info(
                f"MCP compact 模式启用（共 {manager.total_tool_count} 个工具 > 阈值 {30}），"
                f"LLM 通过 ToolIndex 按需加载相关工具 schema"
            )
            return registered

        # 全量模式：注册所有 MCP 工具到 self.tools
        tool_map = manager.get_tool_map()
        all_tools = manager.get_all_tools()
        registered = 0

        for tool_schema in all_tools:
            tool_name = tool_schema["name"]
            server_name, original_name = tool_map[tool_name]

            # 为每个 MCP 工具生成 async wrapper
            wrapper = self._make_mcp_tool_wrapper(manager, server_name, original_name)

            self.register_tool(
                name=tool_name,
                description=tool_schema["description"],
                parameters=tool_schema["parameters"],
                function=wrapper,
            )
            # 注册到 async_tools 字典，确保 async_call_tool 走 await 分支
            self._get_async_tools()
            self._async_tools[tool_name] = wrapper
            registered += 1

        logger.info(f"MCP 工具注册完成（全量模式）: {registered} 个工具")
        return registered

    def _register_mcp_compact_mode(self, manager) -> int:
        """compact 模式注册：所有 wrapper + mcp_list_tools 元工具

        v6 设计：
        1. 所有 MCP 工具的 wrapper 注册到 async_tools + self.functions（保证可执行）
           但不注册到 self.tools（不进 tools schema，避免 prompt 膨胀）
        2. 注册 mcp_list_tools 元工具到 self.tools（让 LLM 能主动查询未检索到的工具）
        3. tools_payload 由 get_dynamic_tools_schema(query) 实时检索注入相关工具 schema

        关键设计：
        - wrapper 全量注册保证 LLM 调用任意检索出来的 MCP 工具都能执行
        - tools schema 按需取出，避免 prompt 膨胀
        - mcp_list_tools 作为兜底，当 ToolIndex 检索不准时 LLM 可主动查询
        """
        # 1. 注册所有 MCP 工具的 wrapper 到 async_tools（不进 tools schema）
        tool_map = manager.get_tool_map()
        all_tools = manager.get_all_tools()
        self._get_async_tools()  # 触发懒加载

        for tool_schema in all_tools:
            tool_name = tool_schema["name"]
            server_name, original_name = tool_map[tool_name]
            wrapper = self._make_mcp_tool_wrapper(manager, server_name, original_name)
            # 只注册到 async_tools，不注册到 self.tools（不进 tools_payload）
            self._async_tools[tool_name] = wrapper
            # 同时注册到 self.functions（async_call_tool 走 self.functions 校验）
            self.functions[tool_name] = wrapper

        # 2. 注册 mcp_list_tools 元工具（作为兜底，让 LLM 能主动查询未检索到的工具）
        async def mcp_list_tools(server_name: Optional[str] = None, **_):
            """元工具：列出 MCP 工具详情"""
            result = await manager.list_tools_detail(server_name)
            return result

        self.register_tool(
            name="mcp_list_tools",
            description=(
                "列出可用的 MCP 工具详情（兜底用）。当系统已按需注入工具 schema 但你仍需要"
                "查询其他未注入的工具时调用。可选传入 server_name 只查某个 Server 的工具。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "server_name": {
                        "type": "string",
                        "description": "可选，指定 Server 名称（如 filesystem/firecrawl-mcp/time）。不传则返回所有 Server 的工具。",
                    },
                },
                "required": [],
            },
            function=mcp_list_tools,
        )
        self._async_tools["mcp_list_tools"] = mcp_list_tools
        return 1

    def _cleanup_mcp_tools(self) -> None:
        """清理已注册的 MCP 工具 wrapper（热更新时调用）"""
        # 清理 self.functions
        mcp_names = [name for name in self.functions if name.startswith("mcp__")]
        for name in mcp_names:
            self.functions.pop(name, None)

        # 清理 self.tools（全量模式注册的 MCP 工具 + mcp_list_tools 元工具）
        mcp_names_in_tools = [name for name in self.tools if name.startswith("mcp__") or name == "mcp_list_tools"]
        for name in mcp_names_in_tools:
            self.tools.pop(name, None)

        # 清理 async_tools
        if self._async_tools:
            mcp_names_in_async = [
                name for name in self._async_tools
                if name.startswith("mcp__") or name == "mcp_list_tools"
            ]
            for name in mcp_names_in_async:
                self._async_tools.pop(name, None)

    @staticmethod
    def _make_mcp_tool_wrapper(manager, server_name: str, tool_name: str):
        """为 MCP 工具生成 async wrapper

        Args:
            manager: MCPClientManager 实例
            server_name: MCP Server 名称
            tool_name: 工具名称（不含前缀）

        Returns:
            async 可调用对象
        """
        async def wrapper(**kwargs):
            result = await manager.call_tool(server_name, tool_name, kwargs)
            # 返回格式与内置工具保持一致
            if result.get("success"):
                return {"success": True, "result": result.get("result")}
            else:
                return {"success": False, "error": result.get("error", "未知错误")}
        return wrapper

    def _rag_retrieve_sync_placeholder(self, **kwargs):
        """rag_retrieve 的同步占位：真正的 async 实现是 rag_retrieve_with_context，
        由 llm_service._generate_stream 直接调用，不经过 async_call_tool。
        若被误调用（如通过 call_tool），返回提示信息。"""
        return {
            "error": "rag_retrieve 必须通过 Agent 工具调用循环异步调用，不支持同步调用",
            "hint": "请确保通过 llm_service._generate_stream 中的工具调用循环触发"
        }

    async def rag_retrieve_with_context(
        self,
        query: str,
        strategy: str = "auto",
        top_k: int = 5,
        min_score: float = 0.3,
        document_id: Optional[str] = None,
        assistant_id: Optional[str] = None,
        knowledge_space_ids: Optional[list] = None,
        embedding_model: Optional[str] = None,
        exclude_chunk_ids: Optional[set] = None,
        plan: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        rag_retrieve 工具的真正 async 实现，由 llm_service 直接调用。
        调用 rag_service.retrieve_context 执行检索，返回精简结构给 Agent。

        Args:
            query: 检索查询
            strategy: 检索策略（auto/vector/keyword/graph）
            top_k: 返回片段数
            min_score: 相关性阈值
            document_id: 文档ID过滤（系统注入）
            assistant_id: 助手ID（系统注入）
            knowledge_space_ids: 知识空间ID列表（系统注入）
            embedding_model: 向量模型名（系统注入）
            exclude_chunk_ids: 需排除的chunk_id集合（系统注入，跨轮次去重）
            plan: 可选的 QueryPlan 对象，若提供则复用（避免重复规划）
        """
        from services.rag_service import rag_service

        try:
            result = await rag_service.retrieve_context(
                query=query,
                document_id=document_id,
                assistant_id=assistant_id,
                knowledge_space_ids=knowledge_space_ids,
                embedding_model=embedding_model,
                strategy=strategy,
                top_k=top_k,
                min_score=min_score,
                exclude_chunk_ids=exclude_chunk_ids,
                plan=plan,
            )

            evidence = result.get("evidence", [])
            # 返回给 Agent 的精简结构（控制 token 预算）
            # 阶段三：新增 document_id 字段（供 Agent 层 sources 回收）
            return {
                "query": query,
                "strategy": strategy,
                "chunks": [
                    {
                        "content": item.get("text", "")[:800],
                        "score": round(item.get("score", 0), 3),
                        "document_title": item.get("document_title", ""),
                        "document_id": item.get("document_id", ""),
                        "chunk_id": item.get("chunk_id", ""),
                        "section_path": item.get("section_path", []),
                    }
                    for item in evidence
                ],
                "total_found": len(evidence),
                "context": result.get("context", "")[:2000],
            }
        except Exception as e:
            logger.error(f"rag_retrieve_with_context 失败: {str(e)}", exc_info=True)
            return {
                "error": f"检索失败: {str(e)}",
                "query": query,
                "chunks": [],
                "total_found": 0,
            }

    def _get_available_ollama_models(self) -> Dict[str, Any]:
        """获取当前使用的模型信息"""
        try:
            llm_model = os.getenv("LLM_MODEL", "mimo-v2.5")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
            
            models_info = [
                {
                    "name": llm_model,
                    "provider": base_url,
                    "type": "chat (推理)"
                },
                {
                    "name": embedding_model,
                    "type": "embedding (本地 sentence-transformers)"
                }
            ]
            
            logger.info(f"获取模型列表成功 - LLM: {llm_model}, Embedding: {embedding_model}")
            
            return {
                "success": True,
                "models": models_info,
                "count": len(models_info),
                "message": f"当前使用 LLM 模型: {llm_model}，向量模型: {embedding_model}"
            }
        except Exception as e:
            logger.error(f"获取模型信息失败: {str(e)}", exc_info=True)
            return {
                "success": False,
                "models": [],
                "count": 0,
                "error": f"获取模型信息时发生错误: {str(e)}"
            }
    
    async def _aget_knowledge_base_documents(self, limit: int = 10, assistant_id: Optional[str] = None) -> Dict[str, Any]:
        """异步：获取知识库中的文档列表（供 async_call_tool 在事件循环内调用）"""
        try:
            collection = mongodb.get_collection("documents")
            query = {}
            if assistant_id:
                query["assistant_id"] = assistant_id
            cursor = collection.find(query).sort("created_at", -1).limit(limit)
            documents = []
            async for doc in cursor:
                documents.append({
                    "id": str(doc["_id"]),
                    "title": doc.get("title", "未命名文档"),
                    "file_type": doc.get("file_type", ""),
                    "status": doc.get("status", ""),
                    "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else "",
                    "file_size": doc.get("file_size", 0)
                })
            total = await collection.count_documents(query)
            logger.info(f"获取知识库文档列表成功 - 助手ID: {assistant_id or '全部'}, 总数: {total}, 返回: {len(documents)}")
            result = {
                "success": True,
                "documents": documents,
                "total": total,
                "returned": len(documents),
                "message": f"知识库共有 {total} 个文档，返回了最新的 {len(documents)} 个（实时数据）"
            }
            if assistant_id:
                result["assistant_id"] = assistant_id
                result["message"] = f"助手知识库共有 {total} 个文档，返回了最新的 {len(documents)} 个（实时数据）"
            return result
        except Exception as e:
            logger.error(f"获取知识库文档列表失败: {str(e)}", exc_info=True)
            return {
                "success": False,
                "documents": [],
                "total": 0,
                "returned": 0,
                "error": f"获取文档列表时发生错误: {str(e)}"
            }

    def _get_knowledge_base_documents(self, limit: int = 10, assistant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取知识库中的文档列表（实时获取）。在已有事件循环中请使用 async_call_tool + _aget_knowledge_base_documents。
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                # 已在异步上下文中，不应在此同步方法内再起 asyncio.run，由调用方使用 async_call_tool
                raise RuntimeError(
                    "在已运行的事件循环中不能在此同步方法内执行 MongoDB 异步操作，请使用 ai_tools.async_call_tool(name, arguments)"
                )
            return asyncio.run(self._aget_knowledge_base_documents(limit=limit, assistant_id=assistant_id))
        except Exception as e:
            logger.error(f"获取知识库文档列表失败: {str(e)}", exc_info=True)
            return {"success": False, "documents": [], "total": 0, "returned": 0, "error": str(e)}
    
    async def _aget_system_info(self, assistant_id: Optional[str] = None) -> Dict[str, Any]:
        """异步：获取系统基本信息（供 async_call_tool 在事件循环内调用）"""
        try:
            default_generation_model = os.getenv("LLM_MODEL", "mimo-v2.5")
            from embedding.embedding_service import embedding_service
            default_embedding_model = embedding_service.model_name if hasattr(embedding_service, 'model_name') else "未知"
            actual_generation_model = default_generation_model
            actual_embedding_model = default_embedding_model
            assistant_name = None

            if assistant_id:
                try:
                    collection = mongodb.get_collection("course_assistants")
                    assistant_doc = await collection.find_one({"_id": assistant_id})
                    if assistant_doc:
                        assistant_name = assistant_doc.get("name", "")
                        inference_model = assistant_doc.get("inference_model")
                        embedding_model = assistant_doc.get("embedding_model")
                        if inference_model:
                            actual_generation_model = inference_model
                        if embedding_model:
                            actual_embedding_model = embedding_model
                except Exception as e:
                    logger.warning(f"获取助手配置失败: {str(e)}")

            collection = mongodb.get_collection("documents")
            query = {}
            if assistant_id:
                query["assistant_id"] = assistant_id
            total_docs = await collection.count_documents(query)
            completed_docs = await collection.count_documents({**query, "status": "completed"})
            processing_docs = await collection.count_documents({**query, "status": "processing"})
            failed_docs = await collection.count_documents({**query, "status": "failed"})
            kb_stats = {
                "total_documents": total_docs,
                "completed": completed_docs,
                "processing": processing_docs,
                "failed": failed_docs
            }

            logger.info(f"获取系统信息成功 - 助手ID: {assistant_id or '默认'}, 推理模型: {actual_generation_model}, 向量化模型: {actual_embedding_model}")
            result = {
                "success": True,
                "generation_model": actual_generation_model,
                "embedding_model": actual_embedding_model,
                "knowledge_base": kb_stats,
                "message": "系统信息获取成功（实时数据）"
            }
            if assistant_id and assistant_name:
                result["assistant_id"] = assistant_id
                result["assistant_name"] = assistant_name
                result["message"] = f"助手 '{assistant_name}' 的系统信息获取成功（实时数据）"
            return result
        except Exception as e:
            logger.error(f"获取系统信息失败: {str(e)}", exc_info=True)
            return {"success": False, "error": f"获取系统信息时发生错误: {str(e)}"}

    def _get_system_info(self, assistant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取系统基本信息（实时获取）。在已有事件循环中请使用 async_call_tool。
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                raise RuntimeError(
                    "在已运行的事件循环中不能在此同步方法内执行 MongoDB 异步操作，请使用 ai_tools.async_call_tool(name, arguments)"
                )
            return asyncio.run(self._aget_system_info(assistant_id=assistant_id))
        except Exception as e:
            logger.error(f"获取系统信息失败: {str(e)}", exc_info=True)
            return {"success": False, "error": f"获取系统信息时发生错误: {str(e)}"}
    
    async def _aget_knowledge_base_stats(self, assistant_id: Optional[str] = None) -> Dict[str, Any]:
        """异步：获取知识库详细统计（供 async_call_tool 在事件循环内调用）"""
        try:
            collection = mongodb.get_collection("documents")
            chunks_collection = mongodb.get_collection("chunks")
            query = {}
            if assistant_id:
                query["assistant_id"] = assistant_id

            total_docs = await collection.count_documents(query)
            completed_docs = await collection.count_documents({**query, "status": "completed"})
            processing_docs = await collection.count_documents({**query, "status": "processing"})
            failed_docs = await collection.count_documents({**query, "status": "failed"})

            cursor = collection.find(query).sort("created_at", -1).limit(10)
            recent_docs = []
            async for doc in cursor:
                recent_docs.append({
                    "id": str(doc["_id"]),
                    "title": doc.get("title", "未命名文档"),
                    "file_type": doc.get("file_type", ""),
                    "status": doc.get("status", ""),
                    "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else "",
                    "file_size": doc.get("file_size", 0)
                })

            chunk_query = {}
            if assistant_id:
                doc_ids = []
                async for doc in collection.find(query, {"_id": 1}):
                    doc_ids.append(str(doc["_id"]))
                chunk_query["document_id"] = {"$in": doc_ids} if doc_ids else {"$in": []}
            total_chunks = await chunks_collection.count_documents(chunk_query) if chunk_query.get("document_id") else await chunks_collection.count_documents({})

            total_vectors = 0
            try:
                from database.qdrant_client import get_qdrant_client
                if assistant_id:
                    assistant_collection = mongodb.get_collection("course_assistants")
                    assistant_doc = await assistant_collection.find_one({"_id": assistant_id})
                    if assistant_doc:
                        collection_name = assistant_doc.get("collection_name", "mnemo_knowledge")
                        qdrant = get_qdrant_client(collection_name)
                        info = await asyncio.to_thread(qdrant.get_collection_info)
                        total_vectors = info.get("points_count", 0)
                else:
                    qdrant = get_qdrant_client("mnemo_knowledge")
                    info = await asyncio.to_thread(qdrant.get_collection_info)
                    total_vectors = info.get("points_count", 0)
            except Exception as e:
                logger.warning(f"获取向量统计失败: {str(e)}")

            stats = {
                "total_documents": total_docs,
                "completed": completed_docs,
                "processing": processing_docs,
                "failed": failed_docs,
                "total_chunks": total_chunks,
                "total_vectors": total_vectors,
                "recent_documents": recent_docs
            }
            logger.info(f"获取知识库统计成功 - 助手ID: {assistant_id or '全部'}, 文档数: {stats['total_documents']}, 向量数: {stats['total_vectors']}")
            result = {
                "success": True,
                **stats,
                "message": f"知识库统计信息获取成功（实时数据）- 共有 {stats['total_documents']} 个文档，{stats['total_chunks']} 个文本块，{stats['total_vectors']} 个向量"
            }
            if assistant_id:
                result["assistant_id"] = assistant_id
                result["message"] = f"助手知识库统计信息获取成功（实时数据）- 共有 {stats['total_documents']} 个文档，{stats['total_chunks']} 个文本块，{stats['total_vectors']} 个向量"
            return result
        except Exception as e:
            logger.error(f"获取知识库统计失败: {str(e)}", exc_info=True)
            return {"success": False, "error": f"获取知识库统计时发生错误: {str(e)}"}

    def _get_knowledge_base_stats(self, assistant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取知识库的详细统计信息（实时获取）。在已有事件循环中请使用 async_call_tool。
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                raise RuntimeError(
                    "在已运行的事件循环中不能在此同步方法内执行 MongoDB 异步操作，请使用 ai_tools.async_call_tool(name, arguments)"
                )
            return asyncio.run(self._aget_knowledge_base_stats(assistant_id=assistant_id))
        except Exception as e:
            logger.error(f"获取知识库统计失败: {str(e)}", exc_info=True)
            return {"success": False, "error": f"获取知识库统计时发生错误: {str(e)}"}

    async def _conversation_search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Recall Memory 文本检索：在历史对话中按关键词搜索。

        使用 MongoDB $text 全文检索（首次调用时自动在 messages.content 上建文本索引）。
        返回命中的对话列表（含 conversation_id / title / 命中片段）。
        """
        if not query or not query.strip():
            return {"success": False, "error": "query 不能为空"}

        try:
            col = mongodb.get_collection("conversations")

            # 首次调用时确保文本索引存在（幂等，已存在会静默失败）
            try:
                await col.create_index([("messages.content", "text")], name="messages_content_text")
            except Exception as ie:
                # 索引已存在或权限不足都忽略，后续 $text 查询会暴露真实问题
                logger.debug(f"创建文本索引跳过: {ie}")

            cursor = col.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}, "title": 1, "messages": 1},
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)

            hits = []
            async for doc in cursor:
                # 找出命中查询的具体消息片段
                matched_snippets = []
                for msg in doc.get("messages", []):
                    content = msg.get("content", "") or ""
                    if query.lower() in content.lower():
                        # 截取查询词前后 80 字符的上下文
                        idx = content.lower().find(query.lower())
                        start = max(0, idx - 40)
                        end = min(len(content), idx + len(query) + 40)
                        snippet = ("..." if start > 0 else "") + content[start:end] + ("..." if end < len(content) else "")
                        matched_snippets.append({
                            "role": msg.get("role"),
                            "snippet": snippet,
                        })
                        if len(matched_snippets) >= 3:
                            break
                hits.append({
                    "conversation_id": str(doc["_id"]),
                    "title": doc.get("title", "未命名对话"),
                    "score": doc.get("score", 0.0),
                    "matched_snippets": matched_snippets,
                })

            logger.info(f"conversation_search 成功 - query='{query[:30]}', hits={len(hits)}")
            return {
                "success": True,
                "query": query,
                "results": hits,
                "count": len(hits),
                "message": f"在历史对话中检索到 {len(hits)} 条相关结果",
            }
        except Exception as e:
            logger.error(f"conversation_search 失败: {e}", exc_info=True)
            return {"success": False, "error": f"检索历史对话失败: {str(e)}"}


# 全局AI工具实例
ai_tools = AITools()

