
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
        try:
            from services.runtime_config import get_runtime_config

            runtime_cfg = await get_runtime_config()
            modules = runtime_cfg.get("modules") or {}
            if not bool(modules.get("rerank_enabled", True)):
                self.enable_reranker = False
        except Exception:
            modules = {}

        if graph_enabled is None:
            # 同时检查 runtime_config 和 NEO4J_ENABLED 环境变量
            # NEO4J_ENABLED=false 时直接禁用图谱检索，避免每题浪费 ~18s 调用空图谱
            neo4j_enabled = os.getenv("NEO4J_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
            graph_enabled = neo4j_enabled and bool(modules.get("kg_retrieve_enabled", True)) and use_graph

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

        results_list = await asyncio.gather(*tasks)

        # 按 task_names 顺序解析各路结果，缺失的路用空列表占位
        vector_groups: List[List[Dict[str, Any]]] = []
        keyword_groups: List[List[Dict[str, Any]]] = []
        graph_results: List[Dict[str, Any]] = []

        for name, res in zip(task_names, results_list):
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
            #把分数修改,改为更精确的cross-encoder计算出来的
            reranked_results = await self._rerank(query, merged_results, reranker=reranker)
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
        """关键词检索 — 优先 Redis 倒排索引，fallback 到 MongoDB 全表扫描（阶段二改造）"""
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # —— 优先尝试 Redis 倒排索引 ——
        try:
            from utils.redis_client import search_bm25, is_available
            if is_available():
                id_score_pairs = await asyncio.to_thread(
                    search_bm25, query_terms, self.prefetch_k, document_id
                )
                if id_score_pairs:
                    # Redis 返回 (chunk_id, score) 列表，从 MongoDB 回查 chunk text
                    chunks = await asyncio.to_thread(self._fetch_chunks_by_ids, id_score_pairs)
                    if chunks:
                        logger.debug(f"Redis BM25 命中: {len(chunks)} 个 chunk (query='{query[:30]}')")
                        return chunks
                # Redis 无结果，继续 fallback 到 MongoDB
        except Exception as e:
            logger.warning(f"Redis BM25 检索失败，fallback 到 MongoDB: {e}")

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
        clean = (text or "").lower()
        try:
            import jieba  # type: ignore
            #"信息检索技术" → ["信息", "检索", "技术"]
            tokens = [t.strip() for t in jieba.cut(clean) if t.strip()]
        except Exception:
            tokens = re.findall(r"[\w\u4e00-\u9fff]+", clean)
        return [t for t in tokens if len(t) > 1 or re.match(r"[\u4e00-\u9fff]", t)]

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
        """图谱检索"""
        try:
            # 1. 提取查询实体
            entities = await knowledge_extraction_service.extract_entities(query)
            if not entities:
                return []
            
            results = []
            if neo4j_client.driver is None:
                neo4j_client.connect()
                
            if neo4j_client.driver:
                for entity in entities:
                    cypher = (
                        f"MATCH (n {{name: $name}})-[r]->(m) "
                        f"RETURN n.name as head, type(r) as relation, m.name as tail, r.source_doc as doc_id, r.source_chunk as chunk_id LIMIT 10"
                    )
                    records = await asyncio.to_thread(neo4j_client.execute_query, cypher, {"name": entity})
                    
                    if records:
                        text_parts = []
                        chunk_ids = set()
                        doc_ids = set()
                        
                        for record in records:
                            head = record.get('head')
                            relation = record.get('relation')
                            tail = record.get('tail')
                            if head and relation and tail:
                                text_parts.append(f"{head} {relation} {tail}")
                            
                            if record.get('chunk_id'):
                                chunk_ids.add(record.get('chunk_id'))
                            if record.get('doc_id'):
                                doc_ids.add(record.get('doc_id'))
                        
                        if document_id and document_id not in doc_ids:
                            continue

                        # 批量查询所有 chunk_ids（用 asyncio.to_thread 包装同步 MongoDB 调用）
                        chunk_ids_list = [str(cid) for cid in chunk_ids]
                        if chunk_ids_list:
                            def _batch_get_chunks(ids=chunk_ids_list):
                                from bson import ObjectId
                                try:
                                    oids = [ObjectId(cid) for cid in ids]
                                except Exception:
                                    return []
                                docs = self.chunk_repo.collection.find({"_id": {"$in": oids}})
                                return [{**d, "_id": str(d["_id"])} for d in docs]

                            fetched_chunks = await asyncio.to_thread(_batch_get_chunks)
                            for chunk in fetched_chunks:
                                if document_id and chunk.get("document_id") != document_id:
                                    continue
                                meta = (chunk.get("metadata") or {}).copy()
                                meta["graph_relations"] = text_parts
                                results.append({
                                    "id": str(chunk.get("_id")),
                                    "score": 0.75,
                                    "payload": {
                                        "chunk_id": str(chunk.get("_id")),
                                        "document_id": chunk.get("document_id"),
                                        "text": chunk.get("text"),
                                        "chunk_index": chunk.get("chunk_index"),
                                        "metadata": meta,
                                        "retrieval_type": "graph",
                                        "entities": entities,
                                    }
                                })

                        if text_parts and not chunk_ids:
                            combined_text = "Knowledge Graph Context:\n" + "\n".join(text_parts)
                            results.append({
                                "id": f"graph_{entity}",
                                "score": 0.35,
                                "payload": {
                                    "text": combined_text,
                                    "retrieval_type": "graph",
                                    "entities": entities,
                                    "metadata": {"graph_relations": text_parts},
                                }
                            })
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

        权重调整：vector=1.0（主路），keyword=0.3（辅助），graph=0.2（辅助）
        原 keyword=0.8 权重过高，在语义改写场景下 BM25 高分噪声会稀释 vector 正确排名。
        降低后 keyword 只在 vector 也认可的 chunk 上加分，不会单独把噪声推到前面。
        """
        lists = [
            ("vector", vector_results, 1.0),
            ("keyword", keyword_results, 0.3),
            ("graph", graph_results, 0.2),
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
