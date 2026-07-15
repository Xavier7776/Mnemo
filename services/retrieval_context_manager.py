"""RetrievalContextManager — 查询分解编排层（阶段二）

职责：
  1. 如果 QueryPlan.sub_queries 为空，走原流程（单查询多集合并行检索）
  2. 如果 sub_queries 非空，对每个子查询独立检索，跨子查询去重，合并证据
  3. 结果充分性检查（简化版反思）：检查每个子查询是否有足够证据

设计原则：
  - 保持 retrieve_async 的单查询语义不变
  - 跨子查询去重复用阶段一的 exclude_chunk_ids 机制
  - 子查询间并行检索（而非串行），提高性能
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Set

from utils.logger import logger


class RetrievalContextManager:
    """检索编排器 — 管理查询分解、跨子查询去重、证据合并"""

    def __init__(self):
        self.sufficiency_min_results = 2
        self.sufficiency_min_score = 0.1

    async def orchestrate(
        self,
        retriever,
        query: str,
        plan,
        collection_names: List[str],
        document_id: Optional[str] = None,
        embedding_model: Optional[str] = None,
        strategy: str = "auto",
        exclude_chunk_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """编排检索流程

        Args:
            retriever: RAGRetriever 实例
            query: 原始查询
            plan: QueryPlan（含 sub_queries）
            collection_names: 要检索的集合名列表
            document_id: 文档 ID 过滤
            embedding_model: 向量模型名
            strategy: 检索策略
            exclude_chunk_ids: 需排除的 chunk_id 集合（Agent 跨轮次去重）

        Returns:
            合并后的检索结果列表
        """
        if not plan.sub_queries:
            # 无子查询：走原流程（单查询多集合并行）
            return await self._single_query_retrieve(
                retriever, query, plan, collection_names,
                document_id, embedding_model, strategy, exclude_chunk_ids,
            )
        else:
            # 有子查询：查询分解检索
            return await self._decomposed_retrieve(
                retriever, query, plan, collection_names,
                document_id, embedding_model, strategy, exclude_chunk_ids,
            )

    async def _single_query_retrieve(
        self,
        retriever,
        query: str,
        plan,
        collection_names: List[str],
        document_id: Optional[str],
        embedding_model: Optional[str],
        strategy: str,
        exclude_chunk_ids: Optional[Set[str]],
    ) -> List[Dict[str, Any]]:
        """单查询检索（原流程，保持向后兼容）"""
        doc_tasks = [
            retriever.retrieve_async(
                query,
                document_id,
                cn,
                embedding_model=embedding_model,
                query_variants=plan.rewritten_queries,
                graph_enabled=plan.need_graph,
                strategy=strategy,
                exclude_chunk_ids=exclude_chunk_ids,
            )
            for cn in collection_names
        ]
        results_list = await asyncio.gather(*doc_tasks) if doc_tasks else [[]]
        results = []
        for part in results_list:
            results.extend(part or [])
        logger.info(f"RetrievalContextManager: 单查询检索完成 - 集合数: {len(collection_names)}, 结果数: {len(results)}")
        return results

    async def _decomposed_retrieve(
        self,
        retriever,
        query: str,
        plan,
        collection_names: List[str],
        document_id: Optional[str],
        embedding_model: Optional[str],
        strategy: str,
        exclude_chunk_ids: Optional[Set[str]],
    ) -> List[Dict[str, Any]]:
        """查询分解检索 — 对每个子查询独立检索，跨子查询去重

        策略：
          1. 子查询间串行执行（因为需要跨子查询去重，后续子查询要排除前面的结果）
          2. 每个子查询内部多集合并行（与原流程一致）
          3. 累积 seen_chunk_ids，传给后续子查询的 exclude_chunk_ids
          4. 合并所有子查询的结果，按 chunk_id 去重保留最高分
        """
        accumulated_seen: Set[str] = set(exclude_chunk_ids or [])
        all_results: Dict[str, Dict[str, Any]] = {}  # chunk_id → result（保留最高分）
        sub_query_stats: List[Dict[str, Any]] = []

        for i, sub_query in enumerate(plan.sub_queries):
            logger.info(f"RetrievalContextManager: 子查询 {i+1}/{len(plan.sub_queries)} - \"{sub_query[:50]}...\"")

            # 对每个子查询，多集合并行检索
            doc_tasks = [
                retriever.retrieve_async(
                    sub_query,
                    document_id,
                    cn,
                    embedding_model=embedding_model,
                    query_variants=plan.rewritten_queries,  # 改写变体仍用于单子查询内部多路召回
                    graph_enabled=plan.need_graph,
                    strategy=strategy,
                    exclude_chunk_ids=accumulated_seen,  # 跨子查询去重
                )
                for cn in collection_names
            ]
            results_list = await asyncio.gather(*doc_tasks) if doc_tasks else [[]]

            # 合并当前子查询的结果
            sub_results = []
            for part in results_list:
                sub_results.extend(part or [])

            # 更新去重集合 + 合并到全局结果
            for r in sub_results:
                cid = r.get("payload", {}).get("chunk_id") or r.get("id")
                if cid:
                    accumulated_seen.add(cid)
                    # 按 chunk_id 去重，保留最高分
                    existing = all_results.get(cid)
                    if existing is None or float(r.get("score", 0)) > float(existing.get("score", 0)):
                        all_results[cid] = r

            # 结果充分性检查（简化版反思）
            is_sufficient = self._check_sufficiency(sub_query, sub_results)
            sub_query_stats.append({
                "sub_query": sub_query,
                "result_count": len(sub_results),
                "sufficient": is_sufficient,
            })

            logger.info(f"RetrievalContextManager: 子查询 {i+1} 完成 - 结果数: {len(sub_results)}, 充分: {is_sufficient}")

        # 按分数排序
        merged = sorted(all_results.values(), key=lambda x: float(x.get("score", 0)), reverse=True)

        # 记录整体统计
        insufficient = [s for s in sub_query_stats if not s["sufficient"]]
        if insufficient:
            logger.warning(
                f"RetrievalContextManager: {len(insufficient)}/{len(plan.sub_queries)} 个子查询证据不足: "
                + "; ".join([f"\"{s['sub_query'][:30]}...\"({s['result_count']})" for s in insufficient])
            )

        logger.info(
            f"RetrievalContextManager: 查询分解检索完成 - "
            f"子查询数: {len(plan.sub_queries)}, 合并结果数: {len(merged)}, "
            f"总去重chunk数: {len(accumulated_seen)}"
        )

        return merged

    def _check_sufficiency(self, sub_query: str, results: List[Dict[str, Any]]) -> bool:
        """检查子查询的证据充分性（简化版反思）

        判断标准：
          - 结果数量 >= sufficiency_min_results
          - Top1 分数 >= sufficiency_min_score

        Returns:
            True 充分，False 不足
        """
        if len(results) < self.sufficiency_min_results:
            return False
        if results:
            top_score = float(results[0].get("score", 0))
            if top_score < self.sufficiency_min_score:
                return False
        return True


# 全局实例
retrieval_context_manager = RetrievalContextManager()
