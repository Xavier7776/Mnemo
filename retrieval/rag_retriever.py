
"""RAG检索服务"""
from typing import List, Dict, Any, Optional
import asyncio
import os
import math
import re
from database.mongodb import ChunkRepository, mongodb_client
from database.qdrant_client import qdrant_client
from database.neo4j_client import neo4j_client
from embedding.embedding_service import embedding_service
from retrieval.fusion import merge_results_rrf
from services.knowledge_extraction_service import knowledge_extraction_service
from utils.logger import logger
from utils.token_utils import truncate_to_tokens

def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")

class RAGRetriever:
    """RAG检索器（混合检索：向量检索 + 关键词检索 + 图谱检索 + 重排）"""
    
    def __init__(
        self,
        final_k: int = 5,
        score_threshold: float = 0.5,
        prefetch_k: Optional[int] = None,
        enable_reranker: Optional[bool] = None,
        reranker_model: Optional[str] = None,
        reranker_device: Optional[str] = None,
        reranker_max_tokens: int = 512,
        fusion_strategy: str = "rrf",
    ):
        """
        初始化RAG检索器
        
        Args:
            final_k: 最终返回的检索结果数量（用于拼上下文）
            score_threshold: 相似度阈值
            prefetch_k: 向量检索候选池大小（用于重排/动态裁剪），默认按 final_k 放大
            enable_reranker: 是否启用重排（默认读取环境变量 ENABLE_RERANKER）
            reranker_model: CrossEncoder 模型名（默认读取环境变量 RERANKER_MODEL）
            reranker_device: cpu/cuda（默认读取环境变量 RERANKER_DEVICE）
            reranker_max_tokens: 送入 CrossEncoder 的文本最大 token（近似预算）
        """
        self.final_k = final_k
        self.prefetch_k = prefetch_k or max(50, final_k * 10)
        self.score_threshold = score_threshold
        self.chunk_repo = ChunkRepository(mongodb_client)
        #实例化后的crossEncoder
        self._reranker = None
        self.enable_reranker = _env_flag("ENABLE_RERANKER", "0") if enable_reranker is None else bool(enable_reranker)
        self.reranker_model = reranker_model or os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        self.reranker_device = reranker_device or os.getenv("RERANKER_DEVICE", "cuda")
        self.reranker_max_tokens = reranker_max_tokens
        self.fusion_strategy = (fusion_strategy or "rrf").lower()

    def _get_reranker(self):
        """延迟加载 CrossEncoder，避免导入阶段崩溃影响服务启动。"""
        if not self.enable_reranker:
            return None
        #已经加载过了就直接返回
        if self._reranker is not None:
            return self._reranker
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._reranker = CrossEncoder(self.reranker_model, device=self.reranker_device)
            logger.info(f"重排模型加载成功: {self.reranker_model} ({self.reranker_device})")
            return self._reranker
        except Exception as e:
            # 失败自动降级，避免反复尝试
            self.enable_reranker = False
            logger.warning(f"重排模型加载失败，已自动禁用重排: {e}")
            self._reranker = None
            return None

    async def retrieve_async(
        self,
        query: str,
        document_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        query_variants: Optional[List[str]] = None,
        graph_enabled: Optional[bool] = None,
        strategy: str = "auto",
        exclude_chunk_ids: Optional[set] = None,
        enable_query_rewrite: bool = False,
        enable_hyde: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        异步检索相关文档块 (High-level RAG)

        Args:
            query: 查询文本
            document_id: 可选的文档ID过滤
            collection_name: 可选的集合名称（用于多助手支持）
            embedding_model: 可选的向量模型名称
            query_variants: 同一个问题的多个问法,一般由LLM自动生成
            strategy: 检索策略
                - "auto"/"hybrid": 三路混合（向量+BM25+图谱）+ RRF + CrossEncoder（默认）
                - "vector": 仅向量语义检索
                - "keyword": 仅 BM25 关键词检索
                - "graph": 仅图谱检索
            exclude_chunk_ids: 需要排除的 chunk_id 集合（用于 Agent 跨轮次去重）
            enable_query_rewrite: Phase 1 优化 - 启用 LLM 同义改写（生成 3 个变体并行检索）
            enable_hyde: Phase 1 优化 - 启用 HyDE（LLM 生成假想答案，用答案做向量检索）
        Returns:
            检索结果列表，包含文本、相似度分数、元数据等
        """
        # 归一化 strategy，决定启用哪些检索路径
        strategy = (strategy or "auto").lower()
        if strategy == "hybrid":
            strategy = "auto"
        use_vector = strategy in ("auto", "vector")
        use_keyword = strategy in ("auto", "keyword")
        use_graph = strategy in ("auto", "graph")

        # 运行时开关：决定是否启用图谱检索/重排等高耗模块
        # 注意：MongoDB 未连接时不能兜底为 True，否则会意外加载 reranker 模型导致 OOM
        try:
            from services.runtime_config import get_runtime_config

            runtime_cfg = await get_runtime_config()
            modules = runtime_cfg.get("modules") or {}
            # 只有 MongoDB 显式配置了 rerank_enabled 时才覆盖
            if "rerank_enabled" in modules:
                self.enable_reranker = bool(modules.get("rerank_enabled"))
        except Exception:
            modules = {}

        if graph_enabled is None:
            neo4j_enabled = os.getenv("NEO4J_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
            graph_enabled = neo4j_enabled and bool(modules.get("kg_retrieve_enabled", True)) and use_graph

        # Phase 1 优化：LLM Query 改写（生成同义变体）
        if enable_query_rewrite and query_variants is None:
            try:
                query_variants = await self._rewrite_query_llm(query)
            except Exception as e:
                logger.warning(f"LLM Query 改写失败，使用原始 query: {e}")
                query_variants = None

        # Phase 1 优化：HyDE（生成假想答案用于向量检索）
        hyde_doc = None
        if enable_hyde and use_vector:
            try:
                hyde_doc = await self._generate_hyde(query)
            except Exception as e:
                logger.warning(f"HyDE 生成失败，跳过: {e}")
                hyde_doc = None

        # 1. 并行执行多种检索策略（按 strategy 选择性启用）
        queries = [q for q in (query_variants or [query]) if q and q.strip()]
        if not queries:
            queries = [query]

        tasks = []
        task_names = []

        if use_vector:
            vector_tasks = [
                self._vector_search(q, document_id, collection_name, embedding_model)
                for q in queries
            ]
            # Phase 1 优化：HyDE 假想答案额外做一路向量检索
            if hyde_doc:
                vector_tasks.append(
                    self._vector_search(hyde_doc, document_id, collection_name, embedding_model)
                )
            tasks.append(asyncio.gather(*vector_tasks))
            task_names.append("vector")

        if use_keyword:
            keyword_tasks = [
                self._keyword_search(q, document_id)
                for q in queries
            ]
            tasks.append(asyncio.gather(*keyword_tasks))
            task_names.append("keyword")

        if use_graph and graph_enabled:
            tasks.append(self._graph_search(query, document_id))
            task_names.append("graph")

        # 兜底：若所有路径都被禁用，退化为向量检索
        if not tasks:
            vector_tasks = [
                self._vector_search(q, document_id, collection_name, embedding_model)
                for q in queries
            ]
            tasks.append(asyncio.gather(*vector_tasks))
            task_names.append("vector")

        # return_exceptions=True 作为双保险：三路检索内部已 try/except 吞异常返回 []，
        # 但若漏网异常冒泡到这里，return_exceptions 能保证单路失败不拖垮其他两路。
        # 详见 docs/failure-modes.md
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # 按 task_names 顺序解析各路结果，缺失的路或异常路用空列表占位
        vector_groups: List[List[Dict[str, Any]]] = []
        keyword_groups: List[List[Dict[str, Any]]] = []
        graph_results: List[Dict[str, Any]] = []

        for name, res in zip(task_names, results_list):
            # return_exceptions=True 下 res 可能是 Exception 实例
            if isinstance(res, Exception):
                logger.error(
                    f"检索路径 '{name}' 未捕获异常（已降级为空结果）: {res}",
                    exc_info=res,
                )
                res = [] if name != "graph" else []
                if name == "vector":
                    vector_groups = []
                elif name == "keyword":
                    keyword_groups = []
                elif name == "graph":
                    graph_results = []
                continue
            if name == "vector":
                vector_groups = res
            elif name == "keyword":
                keyword_groups = res
            elif name == "graph":
                graph_results = res

        vector_results = self._flatten_ranked_groups(vector_groups)
        keyword_results = self._flatten_ranked_groups(keyword_groups)

        # 2. 混合检索结果（合并和初步去重）
        merged_results = self._merge_results(vector_results, keyword_results, graph_results)

        # 2.1 RRF 融合后相对阈值过滤
        # 注意：RRF 分数范围（~0.001-0.05）与 score_threshold（默认 0.7，用于向量检索）完全不匹配，
        # 不能直接复用 self.score_threshold。这里用相对阈值：低于 top1 * ratio 的剔除，
        # 避免低分噪声（如 0.003 分的 chunk）进入下游 reranker / LLM 上下文 / 前端 evidence 展示。
        # ratio 通过环境变量 RRF_FILTER_RATIO 控制，默认 0.1（与 fusion.py 中 keyword/graph 的 ratio 一致）。
        if merged_results:
            top_score = float(merged_results[0].get("score", 0.0) or 0.0)
            if top_score > 0:
                rrf_filter_ratio = float(os.getenv("RRF_FILTER_RATIO", "0.1"))
                threshold = top_score * rrf_filter_ratio
                before_count = len(merged_results)
                merged_results = [
                    r for r in merged_results
                    if float(r.get("score", 0.0) or 0.0) >= threshold
                ]
                if before_count != len(merged_results):
                    logger.debug(
                        f"RRF 融合后相对阈值过滤: {before_count} -> {len(merged_results)} "
                        f"(top1={top_score:.4f}, threshold={threshold:.4f}, ratio={rrf_filter_ratio})"
                    )

        # 2.5 排除已检索过的 chunk（Agent 跨轮次去重）
        if exclude_chunk_ids:
            before_count = len(merged_results)
            merged_results = [
                r for r in merged_results
                if r.get("payload", {}).get("chunk_id") not in exclude_chunk_ids
                and r.get("id") not in exclude_chunk_ids
            ]
            if before_count != len(merged_results):
                logger.debug(f"exclude_chunk_ids 过滤: {before_count} -> {len(merged_results)}")

        # 3. 重排 (Rerank)
        reranker = self._get_reranker()
        if reranker and merged_results:
            # Phase 4 优化：reranker 之前先 RRF 截取 top-N
            # 原逻辑：reranker 对全部 merged_results 打分（prefetch_k=200 时几百个候选，极慢）
            # 新逻辑：混合检索+RRF 融合后截取 top-N（默认 20），只对这 N 个候选 reranker
            rerank_candidate_k = int(os.getenv("RERANK_CANDIDATE_K", "20"))
            candidates_for_rerank = merged_results[:rerank_candidate_k]
            logger.debug(f"reranker 候选集: {len(merged_results)} -> {len(candidates_for_rerank)}")
            #把分数修改,改为更精确的cross-encoder计算出来的
            reranked_results = await self._rerank(query, candidates_for_rerank, reranker=reranker)
            # 在线动态裁剪 k：基于重排分数分布自适应（兼顾 recall/precision）
            #这个很好,判断gap大不大也就是后面的和前面的关联度大不大,如果不大那就只取前面几个的
            #如果差距很小那就都取几个交给LLM去判断
            k = self._dynamic_k_from_scores(reranked_results, default_k=self.final_k)
            return reranked_results[:k]
        
        # 4. 如果没有重排，直接返回按合并分数排序的结果
        return merged_results[: self.final_k]

    def _dynamic_k_from_scores(self, results: List[Dict[str, Any]], default_k: int) -> int:
        """
        在线动态调 k（仅在 reranker 启用时生效）。
        - 区分度高（top1 与 topN 差距大）：减小 k 提升 precision
        - 区分度低（分数接近）：增大 k 保留 recall
        """
        if not results:
            return int(default_k)
        scores = [float(r.get("score", 0.0) or 0.0) for r in results]
        k_min = int(os.getenv("DYNK_MIN", "8"))
        k_max = int(os.getenv("DYNK_MAX", str(max(default_k, 24))))

        # 默认 k
        k = int(default_k)

        # 仅在有足够候选时判断
        if len(scores) >= max(10, default_k):
            s1 = scores[0]
            s10 = scores[min(9, len(scores) - 1)]
            gap = s1 - s10

            # gap 大：强相关集中
            if gap >= float(os.getenv("DYNK_GAP_HIGH", "2.0")):
                k = max(k_min, min(k, 12))
            # gap 小：区分度差，需要更多证据
            elif gap <= float(os.getenv("DYNK_GAP_LOW", "0.6")):
                k = min(k_max, max(k, 24))

        return max(k_min, min(k_max, k))

    # ==================== Phase 1: LLM Query 改写 + HyDE ====================

    async def _rewrite_query_llm(self, query: str) -> List[str]:
        """Phase 1 优化：LLM 同义改写 query，生成 3 个变体

        策略：
        1. 调用 LLM 生成 3 个同义改写（不同表达方式、不同术语）
        2. 加上原始 query，共 4 个变体并行检索
        3. RRF 融合后取最优

        适用场景：用户 query 口语化、术语不全、与文档表达差距大
        """
        from utils.llm_client import get_async_openai_client

        model = os.getenv("LLM_MODEL", "mimo-v2.5").strip()
        timeout = float(os.getenv("QUERY_REWRITE_TIMEOUT", "30.0"))

        prompt = f"""你是一个检索查询改写器。把下面的用户查询改写成 3 个不同表达方式的同义查询，用于提升检索召回率。

要求：
1. 保留核心语义，改变表达方式
2. 第 1 个：用更正式/学术的表达
3. 第 2 个：用更口语化的表达
4. 第 3 个：补充相关术语或同义词
5. 每个改写不超过 50 字
6. 只输出改写后的查询，每行一个，不要编号不要解释

用户查询：{query}

改写："""

        client = get_async_openai_client()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
            timeout=timeout,
        )
        text = response.choices[0].message.content or ""
        variants = [line.strip() for line in text.split("\n") if line.strip()][:3]
        # 加上原始 query
        variants = [query] + variants
        logger.info(f"[Phase1] Query 改写: {variants}")
        return variants

    async def _generate_hyde(self, query: str) -> Optional[str]:
        """Phase 1 优化：HyDE - LLM 生成假想答案，用答案做向量检索

        原理：用户 query 通常短、模糊，与文档表达差距大。
        让 LLM 先生成一个"假想答案"，答案的语义分布更接近文档，向量检索召回更高。

        适用场景：query 短、模糊、口语化；文档长、正式、术语密集
        """
        from utils.llm_client import get_async_openai_client

        model = os.getenv("LLM_MODEL", "mimo-v2.5").strip()
        timeout = float(os.getenv("HYDE_TIMEOUT", "30.0"))

        prompt = f"""请简要回答以下问题（即使你不确定，也要给出一个 plausible 的答案，200 字以内）：

问题：{query}

回答："""

        client = get_async_openai_client()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
            timeout=timeout,
        )
        hyde_doc = (response.choices[0].message.content or "").strip()
        if len(hyde_doc) < 10:
            return None
        logger.info(f"[Phase1] HyDE 生成 ({len(hyde_doc)} 字): {hyde_doc[:100]}...")
        return hyde_doc

    # # groups 是多个结果列表组成的列表
    # groups = [
    #     # 变体1 "BM25检索算法" 的结果
    #     [
    #         {"id": "c1", "score": 0.92, "payload": {"chunk_id": "c1", "text": "..."}},
    #         {"id": "c2", "score": 0.85, "payload": {"chunk_id": "c2", "text": "..."}},
    #         {"id": "c3", "score": 0.71, "payload": {"chunk_id": "c3", "text": "..."}},
    #     ],
    #     # 变体2 "关键词匹配排序方法" 的结果
    #     [
    #         {"id": "c2", "score": 0.95, "payload": {"chunk_id": "c2", "text": "..."}},  # c2又出现了
    #         {"id": "c4", "score": 0.88, "payload": {"chunk_id": "c4", "text": "..."}},
    #         {"id": "c1", "score": 0.60, "payload": {"chunk_id": "c1", "text": "..."}},  # c1又出现了
    #     ],
    # ]
    def _flatten_ranked_groups(self, groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Merge query-variant result groups while preserving the best rank per chunk."""
        merged: Dict[str, Dict[str, Any]] = {}
        for group in groups or []:
            for rank, item in enumerate(group or [], start=1):
                payload = item.get("payload") or {}
                key = payload.get("chunk_id") or item.get("id")
                if not key:
                    continue
                current = merged.get(str(key))
                score = float(item.get("score", 0.0) or 0.0)
                if current is None or score > float(current.get("score", 0.0) or 0.0):
                    copied = dict(item)
                    copied["_variant_rank"] = rank
                    merged[str(key)] = copied
        return sorted(merged.values(), key=lambda x: x.get("score", 0.0), reverse=True)

    async def _vector_search(self, query: str, document_id: Optional[str], collection_name: Optional[str], embedding_model: Optional[str] = None) -> List[Dict[str, Any]]:
        """向量检索"""
        try:
            def _search_sync() -> List[Dict[str, Any]]:
                query_vector = embedding_service.encode_single(query, model_name=embedding_model, is_query=True)

                filter_conditions = None
                if document_id:
                    filter_conditions = {"document_id": document_id}

                from database.qdrant_client import get_qdrant_client
                client = get_qdrant_client(collection_name) if collection_name else qdrant_client
                return client.search(
                    query_vector=query_vector,
                    limit=self.prefetch_k,
                    score_threshold=self.score_threshold,
                    filter_conditions=filter_conditions,
                    query_text=query
                )

            results = await asyncio.to_thread(_search_sync)
            return results
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    async def _keyword_search(self, query: str, document_id: Optional[str]) -> List[Dict[str, Any]]:
        """关键词检索 — 优先 Redis 倒排索引（停用词过滤），fallback 到 MongoDB 全表扫描"""
        # 查询端必须过滤停用词，否则 "是/用/什么/的" 等高频虚词会淹没 BM25 结果
        # 之前的 BUG：用了 tokenize（索引端版本，不过滤停用词），导致 keyword recall@5 从 0.3 跌到 0.007
        from utils.tokenizer import tokenize_for_query
        query_terms = tokenize_for_query(query)
        if not query_terms:
            return []

        # —— 优先尝试 Redis 倒排索引 ——
        # BUG 修复：原代码 fallback 是静默的，导致 keyword 策略实际走 MongoDB 全表扫描
        # 时延迟从 ~33ms 退化到 ~780ms 也不打日志，无法定位问题。
        # 现在所有 fallback 路径都打 WARNING 日志，便于排查 RediSearch 是否真正生效。
        try:
            from utils.redis_client import search_bm25, is_available
            if is_available():
                id_score_pairs = await asyncio.to_thread(
                    search_bm25, query_terms, self.prefetch_k, document_id
                )
                if id_score_pairs:
                    chunks = await asyncio.to_thread(self._fetch_chunks_by_ids, id_score_pairs)
                    if chunks:
                        logger.debug(f"Redis BM25 命中: {len(chunks)} 个 chunk (query='{query[:30]}')")
                        return chunks
                    logger.warning(
                        f"Redis BM25 返回 {len(id_score_pairs)} 个 id 但 MongoDB 回查 0 chunk，"
                        f"fallback 到 MongoDB 全表扫描 (query='{query[:30]}')"
                    )
                else:
                    logger.warning(
                        f"Redis BM25 返回空（RediSearch 可用但无匹配或索引为空），"
                        f"fallback 到 MongoDB 全表扫描 (query='{query[:30]}')"
                    )
            else:
                logger.warning(
                    f"Redis/RediSearch 不可用（is_available=False），"
                    f"fallback 到 MongoDB 全表扫描 (query='{query[:30]}')"
                )
        except Exception as e:
            logger.warning(f"Redis BM25 检索异常，fallback 到 MongoDB: {e}")

        # —— Fallback: 原有 MongoDB 全表扫描逻辑 ——
        return await self._keyword_search_mongo(query, document_id)

    def _fetch_chunks_by_ids(self, id_score_pairs: List[tuple]) -> List[Dict[str, Any]]:
        """根据 Redis 返回的 (chunk_id, score) 列表，从 MongoDB 回查 chunk 完整信息"""
        try:
            from bson import ObjectId
            chunk_ids = [cid for cid, _ in id_score_pairs]
            score_map = {cid: score for cid, score in id_score_pairs}

            # 批量查询 MongoDB
            object_ids = []
            for cid in chunk_ids:
                try:
                    object_ids.append(ObjectId(cid))
                except Exception:
                    continue
            if not object_ids:
                return []

            cursor = self.chunk_repo.collection.find({"_id": {"$in": object_ids}})
            results = []
            for chunk in cursor:
                cid = str(chunk["_id"])
                score = score_map.get(cid, 0.0)
                results.append({
                    "id": cid,
                    "score": score,
                    "payload": {
                        "chunk_id": cid,
                        "document_id": chunk.get("document_id"),
                        "text": chunk.get("text"),
                        "chunk_index": chunk.get("chunk_index"),
                        "metadata": chunk.get("metadata", {}),
                    },
                })
            # 按 Redis 计算的 BM25 分数排序
            results.sort(key=lambda x: x["score"], reverse=True)
            return results
        except Exception as e:
            logger.error(f"回查 chunk 失败: {e}")
            return []

    async def _keyword_search_mongo(self, query: str, document_id: Optional[str]) -> List[Dict[str, Any]]:
        """MongoDB 全表扫描 BM25（fallback 逻辑，原 _keyword_search 的实现）"""
        try:
            def _keyword_search_sync() -> List[Dict[str, Any]]:
                chunks = []
                if document_id:
                    chunks = self.chunk_repo.get_chunks_by_document(document_id)
                else:
                    chunks = self._candidate_chunks_for_keyword(query, limit=int(os.getenv("BM25_CANDIDATE_LIMIT", "1200")))

                if not chunks:
                    return []

                query_terms = self._tokenize(query)
                if not query_terms:
                    return []

                # 先一次性分词所有 chunk，后续复用（避免重复分词）
                doc_freq: Dict[str, int] = {}
                tokenized_chunks = []
                total_terms_len = 0
                for chunk in chunks:
                    terms = self._tokenize(chunk.get("text", ""))
                    tokenized_chunks.append((chunk, terms))
                    total_terms_len += len(terms)
                    for term in set(terms):
                        doc_freq[term] = doc_freq.get(term, 0) + 1
                #每个chunk的字符加起来最后除以个数算出平均长度
                avgdl = total_terms_len / max(len(chunks), 1)

                k1 = 1.5
                b = 0.75
                results = []
                total_docs = len(chunks)
                for chunk, terms in tokenized_chunks:
                    if not terms:
                        continue
                    term_counts: Dict[str, int] = {}
                    for term in terms:
                        term_counts[term] = term_counts.get(term, 0) + 1
                    dl = len(terms)
                    score = 0.0
                    for term in query_terms:
                        #tf是当前chunk出现的次数,df是全局的
                        tf = term_counts.get(term, 0)
                        if tf <= 0:
                            continue
                        df = doc_freq.get(term, 0)
                        idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                        denom = tf + k1 * (1 - b + b * dl / max(avgdl, 1))
                        score += idf * (tf * (k1 + 1)) / denom
                    if score > 0:
                        results.append({
                            "id": str(chunk.get("_id")),
                            "score": score,
                            "payload": {
                                "chunk_id": str(chunk.get("_id")),
                                "document_id": chunk.get("document_id"),
                                "text": chunk.get("text"),
                                "chunk_index": chunk.get("chunk_index"),
                                "metadata": chunk.get("metadata", {})
                            }
                        })
                return sorted(results, key=lambda x: x["score"], reverse=True)[: self.prefetch_k]

            return await asyncio.to_thread(_keyword_search_sync)
        except Exception as e:
            logger.error(f"关键词检索(MongoDB fallback)失败: {e}")
            return []

    def _tokenize(self, text: str) -> List[str]:
        # Phase 3 优化：统一走 utils/tokenizer.py，支持驼峰切分 + 专有名词词典
        # 查询端用 tokenize_for_query（过滤停用词），避免高频虚词稀释 BM25 关键词权重
        # 修复：之前用 tokenize（不过滤停用词），导致 keyword recall@5 从 0.3 跌到 0.007
        from utils.tokenizer import tokenize_for_query
        return tokenize_for_query(text)

    def _tokenize_weighted(self, text: str):
        """Phase 3 优化：查询端加权分词（英文 token 权重更高）

        返回 [(token, weight), ...]，供 BM25 加权查询使用。
        """
        from utils.tokenizer import tokenize_for_query_weighted
        return tokenize_for_query_weighted(text)

    def _candidate_chunks_for_keyword(self, query: str, limit: int = 1200) -> List[Dict[str, Any]]:
        terms = self._tokenize(query)[:8]
        if not terms:
            return []
        try:
            #都是mongoDB语法,在text中模糊匹配term,i是忽略大小写
            regexes = [{"text": {"$regex": re.escape(term), "$options": "i"}} for term in terms]
            cursor = self.chunk_repo.collection.find({"$or": regexes}).limit(limit)
            return [{**chunk, "_id": str(chunk["_id"])} for chunk in cursor]
        except Exception as e:
            logger.warning(f"关键词候选块查询失败: {e}")
            return []

    async def _graph_search(self, query: str, document_id: Optional[str]) -> List[Dict[str, Any]]:
        """图谱检索

        打分策略（v2 修复：原版所有结果都是固定 0.75 分无区分度；v1 打分重写后短节点名
        反向包含匹配刷分，"Ne"/"R"/"am" 等短节点被任意包含该子串的实体命中累加到高分）：
        - 精确匹配（toLower(n.name) == entity）：1.0 分
        - 节点名包含 query 实体（如 "lung cancer" contains "cancer"）：0.8 分
        - query 实体包含节点名（如 "smoking cessation" contains "smoking"）：0.6 分
          ※ 反向包含要求节点名 ≥4 字符，避免 "ne" in "homocysteine" 等短子串噪声
        - fallback（Cypher CONTAINS 命中但分数分支未匹配）：0.3 分
        - 单实体对同一 chunk 只取最高分（不累加），跨实体再累加
          ※ 避免 "Ne" 节点 6 条边都指向同一 chunk 累加到 3.6 分压过真实 1.0 分精确匹配
        """
        try:
            # 1. 提取查询实体
            entities = await knowledge_extraction_service.extract_entities(query)
            if not entities:
                return []

            # 过滤过短实体（单字母如 "R" 会匹配大量噪声）
            # 保留 2 字符的缩写（如 "UK"、"PrP" 去掉后变小写 "uk"、"prp"）
            # 但纯单字母实体跳过
            filtered_entities = []
            for ent in entities:
                ent_lower = ent.lower().strip()
                if len(ent_lower) < 2:
                    continue
                # 单字母 + 数字组合（如 "R" "Ne"）太短容易误匹配，要求至少 2 个字母
                alpha_chars = [c for c in ent_lower if c.isalpha()]
                if len(alpha_chars) < 2:
                    continue
                filtered_entities.append((ent, ent_lower))

            if not filtered_entities:
                return []

            if neo4j_client.driver is None:
                neo4j_client.connect()

            # chunk_id -> {entity_lower: max_score} 单实体取 max，跨实体再 sum
            # 避免 "Ne" 节点 6 条边都指向同一 chunk 单实体内累加到 3.6 分
            chunk_entity_scores: Dict[str, Dict[str, float]] = {}
            chunk_relations: Dict[str, List[str]] = {}
            chunk_doc_ids: Dict[str, set] = {}

            if neo4j_client.driver:
                for entity, entity_lower in filtered_entities:
                    # LIMIT 50：原 LIMIT 10 会截断掉精确匹配边（Ne 等高频节点边先返回）
                    cypher = (
                        f"MATCH (n)-[r]-(m) "
                        f"WHERE (toLower(n.name) CONTAINS $name OR $name CONTAINS toLower(n.name)) "
                        f"AND r.source_doc = $doc_id "
                        f"RETURN n.name as head, type(r) as relation, m.name as tail, "
                        f"r.source_doc as doc_id, r.source_chunk as chunk_id LIMIT 50"
                    )
                    records = await asyncio.to_thread(
                        neo4j_client.execute_query, cypher,
                        {"name": entity_lower, "doc_id": document_id or ""}
                    )

                    for record in records:
                        head = (record.get('head') or '').lower()
                        tail = (record.get('tail') or '').lower()
                        relation = record.get('relation')
                        chunk_id = record.get('chunk_id')
                        doc_id = record.get('doc_id')

                        if not chunk_id:
                            continue

                        # 计算匹配分数
                        score = 0.0
                        # head 精确匹配
                        if head == entity_lower:
                            score = 1.0
                        # tail 精确匹配
                        elif tail == entity_lower:
                            score = 1.0
                        # head 包含 entity（如 "lung cancer" contains "cancer"）—— 正向包含
                        elif entity_lower in head:
                            score = 0.8
                        # tail 包含 entity
                        elif entity_lower in tail:
                            score = 0.8
                        # entity 包含 head（如 "smoking cessation" contains "smoking"）—— 反向包含
                        # ※ 要求节点名 ≥4 字符，避免 "ne" in "homocysteine" 等短子串噪声
                        elif len(head) >= 4 and head in entity_lower:
                            score = 0.6
                        elif len(tail) >= 4 and tail in entity_lower:
                            score = 0.6
                        else:
                            score = 0.3

                        cid = str(chunk_id)
                        # 单实体对同一 chunk 取 max（不累加），避免高频节点多边刷分
                        ent_map = chunk_entity_scores.setdefault(cid, {})
                        if score > ent_map.get(entity_lower, 0.0):
                            ent_map[entity_lower] = score
                        if cid not in chunk_relations:
                            chunk_relations[cid] = []
                            chunk_doc_ids[cid] = set()
                        head_raw = record.get('head')
                        tail_raw = record.get('tail')
                        if head_raw and relation and tail_raw:
                            chunk_relations[cid].append(f"{head_raw} {relation} {tail_raw}")
                        if doc_id:
                            chunk_doc_ids[cid].add(doc_id)

            # 跨实体 sum：每个 chunk 的总分 = 各实体对该 chunk 的最高分之和
            chunk_scores: Dict[str, float] = {
                cid: sum(ent_map.values())
                for cid, ent_map in chunk_entity_scores.items()
            }

            if not chunk_scores:
                return []

            # 过滤 document_id 不匹配的 chunk
            valid_chunk_ids = []
            for cid, doc_ids in chunk_doc_ids.items():
                if document_id and document_id not in doc_ids:
                    continue
                valid_chunk_ids.append(cid)

            if not valid_chunk_ids:
                return []

            # 批量查询 chunk 文本
            def _batch_get_chunks(ids=valid_chunk_ids):
                from bson import ObjectId, errors as bson_errors
                oids = []
                non_oid_ids = []
                for cid in ids:
                    try:
                        oids.append(ObjectId(cid))
                    except (bson_errors.InvalidId, TypeError):
                        non_oid_ids.append(cid)

                query_filter = None
                if oids and not non_oid_ids:
                    query_filter = {"_id": {"$in": oids}}
                elif oids:
                    query_filter = {"$or": [
                        {"_id": {"$in": oids}},
                        {"metadata.source_chunk": {"$in": non_oid_ids}},
                    ]}
                else:
                    scifact_ids = []
                    for cid in non_oid_ids:
                        if cid.startswith("scifact_"):
                            scifact_ids.append(cid[len("scifact_"):])
                    or_clauses = []
                    if non_oid_ids:
                        or_clauses.append({"metadata.source_chunk": {"$in": non_oid_ids}})
                    if scifact_ids:
                        or_clauses.append({"metadata.scifact_id": {"$in": scifact_ids}})
                    if or_clauses:
                        query_filter = {"$or": or_clauses}

                if not query_filter:
                    return []
                docs = self.chunk_repo.collection.find(query_filter)
                return [{**d, "_id": str(d["_id"])} for d in docs]

            fetched_chunks = await asyncio.to_thread(_batch_get_chunks)

            # 用 chunk 的 source_chunk 或 _id 匹配回 chunk_scores
            # fetched_chunks 可能包含 parent 和 child，需要映射到 source_chunk key
            results = []
            for chunk in fetched_chunks:
                if document_id and chunk.get("document_id") != document_id:
                    continue
                meta = (chunk.get("metadata") or {}).copy()
                # 尝试匹配 chunk_scores 的 key
                cid_str = str(chunk.get("_id"))
                source_chunk = meta.get("source_chunk", "")
                scifact_id = meta.get("scifact_id", "")
                # 匹配策略：优先 source_chunk，其次 scifact_id 拼接，最后 _id
                score = 0.0
                relations = []
                matched_key = None
                for key in [source_chunk, f"scifact_{scifact_id}", cid_str]:
                    if key in chunk_scores:
                        score = max(score, chunk_scores[key])
                        relations = chunk_relations.get(key, [])
                        matched_key = key
                        break

                if score <= 0:
                    continue

                meta["graph_relations"] = relations
                results.append({
                    "id": cid_str,
                    "score": score,
                    "payload": {
                        "chunk_id": cid_str,
                        "document_id": chunk.get("document_id"),
                        "text": chunk.get("text"),
                        "chunk_index": chunk.get("chunk_index"),
                        "metadata": meta,
                        "retrieval_type": "graph",
                        "entities": [e[0] for e in filtered_entities],
                    }
                })

            # 按分数降序排序
            results.sort(key=lambda x: x["score"], reverse=True)
            return results
        except Exception as e:
            logger.error(f"图谱检索失败: {e}")
            return []

    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """合并多种检索结果（RRF 融合）"""
        return self._merge_results_rrf(vector_results, keyword_results, graph_results)

    def _merge_results_rrf(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion across vector, BM25, and graph retrieval.

        权重通过环境变量配置，便于消融实验：
        - RRF_WEIGHT_VECTOR（默认 1.0，主路）
        - RRF_WEIGHT_KEYWORD（默认 0.1，辅助；SciFact 消融实验确定，0.3 会拖累 nDCG）
        - RRF_WEIGHT_GRAPH（默认 0.0，图谱为空时无贡献；抽取后需重新消融）

        SciFact 300 题消融实验结论（2026-07-22）：
        - kw=0.1: nDCG@10=0.6577（最佳，优于纯向量 0.6528）
        - kw=0.3: nDCG@10=0.6410（旧默认，反而低于纯向量，BM25 噪声稀释）
        - kw=1.0: nDCG@10=0.6255（等权最差，keyword 弱时不能等权）
        - vector 权重越大 nDCG 越高（0.5→1.5: 0.6416→0.6532）
        """
        w_vector = float(os.getenv("RRF_WEIGHT_VECTOR", "1.0"))
        w_keyword = float(os.getenv("RRF_WEIGHT_KEYWORD", "0.1"))
        w_graph = float(os.getenv("RRF_WEIGHT_GRAPH", "0.0"))
        lists = [
            ("vector", vector_results, w_vector),
            ("keyword", keyword_results, w_keyword),
            ("graph", graph_results, w_graph),
        ]
        return merge_results_rrf(lists)

    async def _rerank(self, query: str, results: List[Dict[str, Any]], reranker) -> List[Dict[str, Any]]:
        """使用 Cross-Encoder 重排（predict 用 asyncio.to_thread 包装，避免阻塞事件循环）"""
        if not reranker or not results:
            return results

        try:
            # 准备 pairs [query, doc_text]
            pairs = []
            for res in results:
                text = res["payload"].get("text", "")
                # 控制送入 CrossEncoder 的 token 预算，避免长 chunk 造成延迟/崩溃
                #二分裁剪
                text = truncate_to_tokens(text, self.reranker_max_tokens)
                pairs.append([query, text])

            # 预测分数（同步 CPU 推理，用 to_thread 避免阻塞事件循环）
            scores = await asyncio.to_thread(reranker.predict, pairs)

            # 更新分数并排序
            for i, score in enumerate(scores):
                results[i]["score"] = float(score)
                # 归一化分数? BGE reranker 输出 logits，可能需要 sigmoid，但直接排序即可

            results.sort(key=lambda x: x["score"], reverse=True)
            return results
        except Exception as e:
            logger.error(f"重排失败: {e}")
            return results
