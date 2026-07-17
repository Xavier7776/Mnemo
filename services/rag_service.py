"""RAG服务核心逻辑"""
from typing import Dict, Any, Optional
import asyncio
import time
from utils.logger import logger
from utils.token_utils import estimate_tokens, truncate_to_tokens
from models.rag import EvidenceItem
from utils.citation import format_evidence_context


class RAGService:
    """RAG服务封装（通过HTTP调用知识库服务）"""

    async def retrieve_context(
        self,
        query: str,
        document_id: Optional[str] = None,
        assistant_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        conversation_id: Optional[str] = None,
        knowledge_space_ids: Optional[list[str]] = None,
        embedding_model: Optional[str] = None,
        strategy: str = "auto",
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        exclude_chunk_ids: Optional[set] = None,
    ) -> Dict[str, Any]:
        """
        检索相关上下文（并行检索文档和资源）

        Args:
            query: 用户查询
            document_id: 可选的文档ID过滤
            assistant_id: 可选的助手ID（用于获取集合名称）
            collection_name: 可选的集合名称（如果提供则直接使用）
            conversation_id: 可选的对话ID（如果提供，会同时检索对话专用向量空间）
            embedding_model: 可选的向量模型名称
            strategy: 检索策略（auto/vector/keyword/graph），默认 auto 走完整混合流水线
            top_k: 覆盖 query_planner 的 final_k（若提供）
            min_score: 覆盖默认的 score_threshold（若提供）
            exclude_chunk_ids: 需排除的 chunk_id 集合（Agent 跨轮次去重）

        Returns:
            包含上下文、来源信息和推荐资源的字典
        """
        from database.mongodb import mongodb
        trace: Dict[str, Any] = {"query": query, "started_at": int(time.time() * 1000)}
        # 运行时开关：决定是否启用图谱检索/重排等高耗模块
        try:
            from services.runtime_config import get_runtime_config

            runtime_cfg = await get_runtime_config()
            modules = runtime_cfg.get("modules") or {}
            params = runtime_cfg.get("params") or {}
            rerank_enabled = bool(modules.get("rerank_enabled", True))
        except Exception:
            modules = {}
            params = {}
            rerank_enabled = True
        # query_planner（阶段二：LLM 驱动 + 规则 fallback）
        from services.query_planner import query_planner
        plan = await query_planner.build_plan(
            query,
            runtime_modules=modules,
            runtime_params=params,
            filters={
                "document_id": document_id,
                "assistant_id": assistant_id,
                "collection_name": collection_name,
                "knowledge_space_ids": knowledge_space_ids or [],
            },
        )
        #塞进日志
        trace["query_plan"] = plan.model_dump()
        # 解析需要检索的集合列表（知识空间优先）
        collection_names: list[str] = []
        if knowledge_space_ids:  # 如果用户指定了知识空间
            try:
                from bson import ObjectId  # MongoDB 的 ID 类型转换
                spaces = mongodb.get_collection("knowledge_spaces")  # 查 knowledge_spaces 表
                for sid in knowledge_space_ids:  # 遍历每个知识空间 ID
                    try:
                        doc = await spaces.find_one({"_id": ObjectId(sid)})  # 按 ID 查记录
                        if doc and doc.get("collection_name"):  # 如果有 collection_name 字段
                            collection_names.append(doc["collection_name"])  # 记下来
                    except Exception:  # 单个 ID 查失败就跳过
                        continue
            except Exception as e:
                logger.warning(f"获取知识空间集合名称失败: {str(e)}")

        # 兼容：如果没有知识空间选择，则按旧 assistant_id/collection_name
        if not collection_names:
            if assistant_id and not collection_name:
                try:
                    from bson import ObjectId
                    collection = mongodb.get_collection("course_assistants")
                    assistant_doc = await collection.find_one({"_id": ObjectId(assistant_id)})
                    if assistant_doc:
                        collection_name = assistant_doc.get("collection_name")
                except Exception as e:
                    logger.warning(f"获取助手集合名称失败: {str(e)}")
            collection_names = [collection_name or "default_knowledge"]
        
        # 并行检索文档和资源
        # 阶段二：通过 RetrievalContextManager 编排（支持查询分解）
        from retrieval.rag_retriever import RAGRetriever
        from services.retrieval_context_manager import retrieval_context_manager
        # 工具调用时允许覆盖 final_k / score_threshold
        effective_final_k = top_k if top_k is not None else plan.final_k
        effective_score_threshold = min_score if min_score is not None else 0.7
        doc_retriever = RAGRetriever(
            final_k=effective_final_k,
            prefetch_k=plan.prefetch_k,
            score_threshold=effective_score_threshold,
            enable_reranker=rerank_enabled,
            fusion_strategy=plan.fusion_strategy,
        )

        # 通过 RetrievalContextManager 编排检索（单查询或查询分解）
        results = await retrieval_context_manager.orchestrate(
            retriever=doc_retriever,
            query=query,
            plan=plan,
            collection_names=collection_names,
            document_id=document_id,
            embedding_model=embedding_model,
            strategy=strategy,
            exclude_chunk_ids=exclude_chunk_ids,
        )
        trace["retrieval"] = {
            "collection_count": len(collection_names),
            "result_count": len(results),
            "fusion_strategy": plan.fusion_strategy,
            "rewritten_queries": plan.rewritten_queries,
            "sub_queries": plan.sub_queries,
            "planner_source": plan.planner_source,
        }
        logger.info(f"知识空间检索完成 - 集合数: {len(collection_names)}, 结果数: {len(results)}")
        logger.info(f"RAG检索完成 - 文档结果: {len(results)} 个")
        
        # 构建上下文和来源（包含文档信息）
        evidence_items: list[EvidenceItem] = []
        sources = []

        # 邻居扩展：对命中 chunk 拉取前后窗口补齐定义/条件/例外
        # 注意：图谱结果没有 chunk_index，不参与邻居扩展
        from database.mongodb import ChunkRepository, mongodb_client
        chunk_repo = ChunkRepository(mongodb_client)
        neighbor_window = int((0 or 1))
        seen_chunk_ids = set()
        expanded_evidence: list[EvidenceItem] = []
        
        # 获取所有涉及的文档ID和文件ID（对话附件兼容）
        document_ids = set()
        file_ids = set()
        for result in results:
            doc_id = result["payload"].get("document_id")
            if doc_id:
                document_ids.add(doc_id)
            # 对话附件使用 file_id
            file_id = result["payload"].get("file_id")
            if file_id:
                file_ids.add(file_id)
        
        # 批量查询文档信息（用 $in 一次查询，替代串行 for 循环）
        document_info_map = {}
        if document_ids:
            try:
                from database.mongodb import mongodb_client, DocumentRepository
                # 确保 MongoDB 客户端已连接
                if mongodb_client.db is None:
                    mongodb_client.connect()
                doc_repo = DocumentRepository(mongodb_client)

                # 用 $in 批量查询所有文档
                try:
                    from bson import ObjectId
                    doc_ids_obj = []
                    for did in document_ids:
                        try:
                            doc_ids_obj.append(ObjectId(did))
                        except Exception:
                            pass

                    if doc_ids_obj:
                        # 在线程池中执行同步 MongoDB 查询（asyncio 已在模块顶部导入）
                        def _batch_get_docs():
                            return list(doc_repo.collection.find({"_id": {"$in": doc_ids_obj}}))
                        docs = await asyncio.to_thread(_batch_get_docs)

                        for doc in docs:
                            did = str(doc["_id"])
                            doc_title = doc.get("title") or doc.get("file_path", "").split("/")[-1] or f"文档_{did[:8]}"
                            document_info_map[did] = {
                                "title": doc_title,
                                "file_type": doc.get("file_type", ""),
                                "status": doc.get("status", "")
                            }

                    # 对未查到的文档用默认值兜底
                    for did in document_ids:
                        if did not in document_info_map:
                            document_info_map[did] = {
                                "title": f"文档_{did[:8]}",
                                "file_type": "",
                                "status": ""
                            }
                except Exception as e:
                    logger.warning(f"批量查询文档信息失败: {str(e)}")
                    # 兜底：所有文档用默认值
                    for did in document_ids:
                        if did not in document_info_map:
                            document_info_map[did] = {
                                "title": f"文档_{did[:8]}",
                                "file_type": "",
                                "status": ""
                            }
            except Exception as e:
                logger.warning(f"文档信息查询初始化失败: {str(e)}")
        
        # 用于去重相同文档：每个文档只保留最高分的chunk
        document_sources_map = {}  # {document_id: source_info}

        # —— 邻居扩展预取：先收集所有需要扩展的 chunk，并发查询，避免串行 N 次 DB 往返 ——
        neighbor_prefetch_map = {}  # {chunk_id: neighbors_list}
        neighbor_prefetch_keys = []  # [(chunk_id, doc_id, chunk_index, score, document_title)]
        for result in results:
            text = result["payload"].get("text", "")
            if not text:
                continue
            chunk_id = result["payload"].get("chunk_id")
            doc_id = result["payload"].get("document_id")
            chunk_index = result["payload"].get("chunk_index")
            if chunk_id and doc_id is not None and chunk_index is not None and isinstance(chunk_index, int):
                if chunk_id not in neighbor_prefetch_map:
                    neighbor_prefetch_keys.append((chunk_id, doc_id, chunk_index))
                    neighbor_prefetch_map[chunk_id] = None  # 占位

        if neighbor_prefetch_keys:
            def _get_neighbors(doc_id, chunk_index):
                return chunk_repo.get_neighbor_chunks(doc_id, chunk_index, window=neighbor_window)
            # 并发查询所有邻居（asyncio 已在模块顶部导入）
            neighbor_tasks = [
                asyncio.to_thread(_get_neighbors, did, cidx)
                for _, did, cidx in neighbor_prefetch_keys
            ]
            neighbor_results_list = await asyncio.gather(*neighbor_tasks, return_exceptions=True)
            for (cid, _, _), nb_result in zip(neighbor_prefetch_keys, neighbor_results_list):
                if isinstance(nb_result, Exception):
                    neighbor_prefetch_map[cid] = []
                else:
                    neighbor_prefetch_map[cid] = nb_result

        for result in results:
            text = result["payload"].get("text", "")
            if text:
                chunk_id = result["payload"].get("chunk_id")
                doc_id = result["payload"].get("document_id")
                chunk_index = result["payload"].get("chunk_index")
                metadata = result["payload"].get("metadata", {}) or {}
                file_id = result["payload"].get("file_id")
                conversation_id = result["payload"].get("conversation_id")
                score = result.get("score", 0) or result.get("combined_score", 0)
                doc_info = document_info_map.get(doc_id, {}) if doc_id else {}
                document_title = (
                    result["payload"].get("filename")
                    or doc_info.get("title")
                    or (f"文档_{doc_id[:8]}" if doc_id else None)
                    or "Knowledge Graph"
                )
                section_path = metadata.get("section_path") or []
                if isinstance(section_path, str):
                    section_path = [section_path]
                elif not isinstance(section_path, list):
                    section_path = []
                page = metadata.get("page") or metadata.get("page_number")
                try:
                    page = int(page) if page is not None else None
                except Exception:
                    page = None
                evidence_items.append(EvidenceItem(
                    id=f"S{len(evidence_items) + 1}",
                    text=text,
                    document_id=doc_id,
                    file_id=file_id,
                    conversation_id=conversation_id,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index if isinstance(chunk_index, int) else None,
                    document_title=document_title,
                    section_path=[str(s) for s in section_path],
                    page=page,
                    score=float(score or 0.0),
                    retrieval_type=result.get("retrieval_type") or result["payload"].get("retrieval_type") or metadata.get("retrieval_type") or "vector",
                    metadata=metadata,
                ))

                # 邻居扩展（仅对普通文档 chunk 生效，结果已在上方并发预取）
                if chunk_id and doc_id is not None and chunk_index is not None and isinstance(chunk_index, int):
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        neighbors = neighbor_prefetch_map.get(chunk_id) or []
                        for nb in neighbors:
                            nb_id = nb.get("_id")
                            if nb_id and nb_id not in seen_chunk_ids:
                                seen_chunk_ids.add(nb_id)
                                nb_meta = nb.get("metadata") or {}
                                nb_section_path = nb_meta.get("section_path") or []
                                if isinstance(nb_section_path, str):
                                    nb_section_path = [nb_section_path]
                                elif not isinstance(nb_section_path, list):
                                    nb_section_path = []
                                expanded_evidence.append(EvidenceItem(
                                    id=f"S{len(evidence_items) + len(expanded_evidence) + 1}",
                                    text=nb.get("text", ""),
                                    document_id=doc_id,
                                    chunk_id=nb_id,
                                    chunk_index=nb.get("chunk_index") if isinstance(nb.get("chunk_index"), int) else None,
                                    document_title=document_title,
                                    section_path=[str(s) for s in nb_section_path],
                                    score=float(score or 0.0),
                                    retrieval_type="neighbor",
                                    metadata=nb_meta,
                                ))
                
                # 判断是文档还是对话附件
                if file_id and conversation_id:
                    # 对话附件
                    filename = result["payload"].get("filename", f"附件_{file_id[:8]}")
                    source_key = f"conversation_{conversation_id}_{file_id}"
                    source_info = {
                        "chunk_id": result["payload"].get("chunk_id"),
                        "evidence_id": evidence_items[-1].id if evidence_items else None,
                        "file_id": file_id,
                        "conversation_id": conversation_id,
                        "score": score,
                        "retrieval_type": result.get("retrieval_type") or result["payload"].get("retrieval_type", "vector"),
                        "document_title": filename,
                        "file_type": result["payload"].get("metadata", {}).get("file_type", ""),
                        "is_conversation_attachment": True
                    }
                else:
                    # 普通文档
                    doc_title = doc_info.get("title") or (f"文档_{doc_id[:8]}" if doc_id else document_title)
                    source_key = doc_id or result.get("id") or (evidence_items[-1].id if evidence_items else "unknown")
                    source_info = {
                        "chunk_id": result["payload"].get("chunk_id"),
                        "evidence_id": evidence_items[-1].id if evidence_items else None,
                        "document_id": doc_id,
                        "score": score,
                        "retrieval_type": result.get("retrieval_type") or result["payload"].get("retrieval_type", "vector"),
                        "document_title": doc_title,
                        "file_type": doc_info.get("file_type", ""),
                        "status": doc_info.get("status", ""),
                        "is_conversation_attachment": False
                    }
                
                # 如果该来源还没有记录，或者当前chunk的分数更高，则更新
                if source_key not in document_sources_map or score > document_sources_map[source_key]["score"]:
                    document_sources_map[source_key] = source_info
        
        # 将去重后的来源转换为列表，并按分数排序
        sources = list(document_sources_map.values())
        sources.sort(key=lambda x: x["score"], reverse=True)

        # 拼接上下文：证据块 + 邻居补齐，并控制总 token 预算
        evidence_items.extend([e for e in expanded_evidence if e.text and e.text.strip()])
        # 重新编号，确保邻居扩展后编号连续
        for idx, item in enumerate(evidence_items, start=1):
            item.id = f"S{idx}"
        max_context_tokens = int(plan.context_budget or 30_000)
        joined = format_evidence_context(evidence_items)
        if estimate_tokens(joined) > max_context_tokens:
            joined = truncate_to_tokens(joined, max_context_tokens)
        context = joined
        trace["context"] = {
            "evidence_count": len(evidence_items),
            "context_tokens_estimate": estimate_tokens(context),
            "context_budget": max_context_tokens,
        }
        trace["finished_at"] = int(time.time() * 1000)
        
        return {
            "context": context,
            "sources": sources,
            "evidence": [item.model_dump() for item in evidence_items],
            "query_plan": plan.model_dump(),
            "trace": trace,
            "recommended_resources": []
        }
    
# 全局RAG服务实例
rag_service = RAGService()
