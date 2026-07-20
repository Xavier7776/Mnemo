"""MCP 工具索引 — 基于 embedding 的按需工具检索

设计目的：
    当 MCP 工具数量过多（>几十个）时，把所有工具 schema 注入 LLM prompt 会浪费大量 token，
    且 LLM 容易在众多工具中选错。Tool Index 通过语义检索，只把与用户 query 最相关的
    N 个工具 schema 注入 prompt，实现"按需加载"。

工作流程：
    1. 启动时：收集所有 MCP 工具的 name + description，用 embedding_service 编码成向量
    2. 请求时：用同一 embedding 模型编码用户 query，与工具向量做余弦相似度匹配
    3. 取 top_k 个最相关的工具，返回其完整 schema 给 LLM

性能：
    - 工具数量通常 <1000，向量维度 1024，内存占用 <4MB
    - 检索是纯内存点积，<1ms
    - embedding 编码 query 一次约 10~50ms（取决于模型）

兜底策略：
    - 如果 embedding_service 不可用，降级为关键词匹配（difflib）
    - 如果检索结果不足 top_k，不补齐（宁缺毋滥，避免注入不相关工具）
"""
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from utils.logger import logger


class ToolIndex:
    """MCP 工具语义索引（基于 embedding + 余弦相似度）"""

    def __init__(self):
        # tool_full_name → embedding vector
        self._vectors: Dict[str, List[float]] = {}
        # tool_full_name → tool schema（含 name/description/parameters）
        self._schemas: Dict[str, Dict[str, Any]] = {}
        # tool_full_name → (server_name, original_tool_name)
        self._tool_meta: Dict[str, Tuple[str, str]] = {}
        # 是否已构建
        self._built = False
        # embedding 服务实例（懒加载）
        self._embedding_service = None
        # 降级标志：embedding 不可用时用关键词匹配
        self._use_keyword_fallback = False

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def size(self) -> int:
        return len(self._schemas)

    def _get_embedding_service(self):
        """懒加载 embedding 服务"""
        if self._embedding_service is None:
            try:
                from embedding.embedding_service import embedding_service
                # 触发模型加载（如果还没加载）
                _ = embedding_service.dimension
                self._embedding_service = embedding_service
                logger.info(f"ToolIndex 使用 embedding 服务: dim={embedding_service.dimension}")
            except Exception as e:
                logger.warning(
                    f"ToolIndex 加载 embedding 服务失败，降级为关键词匹配: {e}"
                )
                self._use_keyword_fallback = True
        return self._embedding_service

    def build(self, tools_by_server: Dict[str, List[Any]]) -> None:
        """构建工具索引

        Args:
            tools_by_server: server_name → List[MCP Tool 对象]
                每个 Tool 对象需要有 .name 和 .description 属性
        """
        self._vectors.clear()
        self._schemas.clear()
        self._tool_meta.clear()

        if not tools_by_server:
            logger.info("ToolIndex 构建：无工具可索引")
            self._built = True
            return

        # 收集所有工具的 (full_name, description, meta)
        items: List[Tuple[str, str, Tuple[str, str]]] = []
        for server_name, tools in tools_by_server.items():
            for tool in tools:
                original_name = tool.name
                full_name = f"mcp__{server_name}__{original_name}"
                description = (tool.description or "").strip() or original_name
                items.append((full_name, description, (server_name, original_name)))

        if not items:
            self._built = True
            return

        # 尝试用 embedding 编码所有 description
        embedding_service = self._get_embedding_service()
        if embedding_service is not None and not self._use_keyword_fallback:
            try:
                descriptions = [desc for _, desc, _ in items]
                # 批量编码（is_query=False，因为是建索引）
                vectors = embedding_service.encode(descriptions, is_query=False)
                for (full_name, description, meta), vec in zip(items, vectors):
                    self._vectors[full_name] = vec
                    self._tool_meta[full_name] = meta
                    # schema 留空，按需从 manager 获取（避免重复存储）
                    self._schemas[full_name] = {"name": full_name, "description": description}
                logger.info(
                    f"ToolIndex 构建完成（embedding 模式）: {len(self._vectors)} 个工具已索引"
                )
            except Exception as e:
                logger.warning(
                    f"ToolIndex embedding 编码失败，降级为关键词匹配: {e}"
                )
                self._use_keyword_fallback = True

        # 关键词匹配降级路径
        if self._use_keyword_fallback:
            import difflib
            for full_name, description, meta in items:
                self._tool_meta[full_name] = meta
                self._schemas[full_name] = {"name": full_name, "description": description}
            logger.info(
                f"ToolIndex 构建完成（关键词匹配降级模式）: {len(self._schemas)} 个工具已索引"
            )

        self._built = True

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """检索与 query 最相关的 top_k 个工具名

        Args:
            query: 用户查询（通常是 user prompt）
            top_k: 返回的工具数量

        Returns:
            工具全名列表（如 ["mcp__filesystem__read_file", ...]），按相关性降序
        """
        if not self._built or not self._schemas:
            return []

        if not query or not query.strip():
            return []

        query = query.strip()

        # embedding 检索路径
        if not self._use_keyword_fallback and self._vectors:
            embedding_service = self._get_embedding_service()
            if embedding_service is not None:
                try:
                    query_vec = embedding_service.encode_single(query, is_query=True)
                    # 余弦相似度（向量已归一化，点积即 cosine）
                    scored = []
                    for tool_name, tool_vec in self._vectors.items():
                        score = sum(a * b for a, b in zip(query_vec, tool_vec))
                        scored.append((tool_name, score))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    return [name for name, _ in scored[:top_k]]
                except Exception as e:
                    logger.warning(f"ToolIndex embedding 检索失败，降级关键词匹配: {e}")
                    self._use_keyword_fallback = True

        # 关键词匹配降级路径
        import difflib
        # 把工具名也作为匹配文本的一部分（用户可能直接说工具名）
        candidates: List[Tuple[str, float]] = []
        for tool_name, schema in self._schemas.items():
            description = schema.get("description", "")
            short_name = tool_name.split("__")[-1] if "__" in tool_name else tool_name
            # 工具名 split 成关键词（如 get_current_time → get, current, time）
            name_tokens = set(short_name.lower().replace("-", "_").split("_"))
            # 整体匹配文本（工具名 + description）
            full_text = f"{short_name} {description}".lower()
            # 用 SequenceMatcher 计算相似度
            desc_score = difflib.SequenceMatcher(None, query.lower(), description.lower()).ratio()
            name_score = difflib.SequenceMatcher(None, query.lower(), short_name.lower()).ratio()
            full_score = difflib.SequenceMatcher(None, query.lower(), full_text).ratio()

            # 关键词命中：英文按空格分词，中文按 2-gram
            query_lower = query.lower()
            query_words = set(query_lower.split())
            # 中文 2-gram（处理"现在几点了"这种没分词的情况）
            query_bigrams = set()
            for i in range(len(query_lower) - 1):
                if not query_lower[i].isspace() and not query_lower[i+1].isspace():
                    query_bigrams.add(query_lower[i:i+2])
            # description 的 bigrams
            desc_bigrams = set()
            desc_lower = description.lower()
            for i in range(len(desc_lower) - 1):
                if not desc_lower[i].isspace() and not desc_lower[i+1].isspace():
                    desc_bigrams.add(desc_lower[i:i+2])
            # 工具名的 bigrams
            name_bigrams = set()
            name_lower = short_name.lower()
            for i in range(len(name_lower) - 1):
                if not name_lower[i].isspace() and not name_lower[i+1].isspace():
                    name_bigrams.add(name_lower[i:i+2])

            # 英文关键词重叠
            desc_words = set(description.lower().replace("/", " ").replace("_", " ").split())
            keyword_overlap = len(query_words & desc_words) / max(len(query_words), 1)
            keyword_name_overlap = len(query_words & name_tokens) / max(len(query_words), 1)
            # 中文 bigram 重叠（更宽松的匹配）
            bigram_overlap_desc = len(query_bigrams & desc_bigrams) / max(len(query_bigrams), 1)
            bigram_overlap_name = len(query_bigrams & name_bigrams) / max(len(query_bigrams), 1)

            # 综合分数：取多种匹配方式的最大值 + 关键词/bigram 加分
            base_score = max(desc_score, name_score * 0.8, full_score * 0.9)
            keyword_bonus = max(keyword_overlap, keyword_name_overlap) * 0.4
            bigram_bonus = max(bigram_overlap_desc, bigram_overlap_name) * 0.5
            final_score = base_score + keyword_bonus + bigram_bonus
            candidates.append((tool_name, final_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        # 关键词模式下过滤掉分数过低的（<0.15）
        return [name for name, score in candidates[:top_k] if score > 0.15]

    def get_tool_meta(self, tool_full_name: str) -> Optional[Tuple[str, str]]:
        """获取工具的元信息（server_name, original_name）"""
        return self._tool_meta.get(tool_full_name)

    def clear(self) -> None:
        """清空索引"""
        self._vectors.clear()
        self._schemas.clear()
        self._tool_meta.clear()
        self._built = False


# 全局单例
mcp_tool_index = ToolIndex()
