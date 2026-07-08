"""Archival Memory 服务：长期归档、按需语义检索

存储：Qdrant collection `agent_archival_memory`
复用：database.qdrant_client.get_qdrant_client + embedding.embedding_service

注意 Qdrant 客户端真实签名（核对自 database/qdrant_client.py）：
- insert_vectors(vectors, payloads, ids=None, max_retries=3, retry_delay=1.0)
- search(query_vector, limit=5, score_threshold=None, filter_conditions=None, query_text=None)
  ↑ 关键：参数名是复数 `filter_conditions`，返回 [{"id","score","payload"}]
"""
import uuid
from datetime import datetime
from typing import List, Dict, Optional

from database.qdrant_client import get_qdrant_client
from embedding.embedding_service import embedding_service
from utils.logger import logger

COLLECTION = "agent_archival_memory"


class ArchivalMemoryService:
    """Archival Memory 服务：写入归档 + 语义召回"""

    def _client(self):
        """获取（或创建）Qdrant 客户端实例"""
        return get_qdrant_client(COLLECTION)

    def _ensure_collection(self):
        """确保 Qdrant collection 存在，不存在则创建（向量维度跟随 embedding_service）"""
        client = self._client()
        try:
            client.create_collection(vector_size=embedding_service.dimension)
        except Exception:
            pass  # 已存在或其他非致命错误，insert 时会再报

    async def insert(
        self,
        scope_type: str,
        scope_id: str,
        content: str,
        source: str = "manual",
        conversation_id: Optional[str] = None,
    ) -> Dict:
        """
        向归档记忆插入一条记录

        Args:
            scope_type / scope_id: 记忆归属维度
            content: 要归档的文本
            source: manual | auto_summary | recall_migration
            conversation_id: 来源会话（可选）
        """
        if not content or not content.strip():
            return {"success": False, "error": "content 不能为空"}

        try:
            vector = embedding_service.encode_single(content)
            if not vector:
                return {"success": False, "error": "向量化失败，返回空向量"}

            # 确保 collection 存在
            self._ensure_collection()

            point_id = str(uuid.uuid4())
            payload = {
                "scope_type": scope_type,
                "scope_id": scope_id,
                "content": content,
                "source": source,
                "conversation_id": conversation_id,
                "created_at": datetime.utcnow().isoformat(),
            }

            # 真实签名：insert_vectors(vectors, payloads, ids=None, ...)
            self._client().insert_vectors(
                vectors=[vector],
                payloads=[payload],
                ids=[point_id],
            )
            logger.info(
                f"Archival insert 成功 - scope={scope_type}/{scope_id}, "
                f"source={source}, len={len(content)}"
            )
            return {"success": True, "id": point_id}
        except Exception as e:
            logger.error(f"Archival insert 失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def search(
        self,
        scope_type: str,
        scope_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        语义检索归档记忆

        Returns:
            [{"content", "score", "created_at", "id"}, ...]
        """
        if not query or not query.strip():
            return []

        try:
            vector = embedding_service.encode_single(query, is_query=True)
            if not vector:
                return []

            # 确保 collection 存在
            self._ensure_collection()

            # 真实签名：search(query_vector, limit, filter_conditions=None, ...)
            results = self._client().search(
                query_vector=vector,
                limit=top_k,
                filter_conditions={
                    "scope_type": scope_type,
                    "scope_id": scope_id,
                },
            )

            return [
                {
                    "id": r.get("id"),
                    "content": r["payload"].get("content", ""),
                    "score": r.get("score", 0.0),
                    "created_at": r["payload"].get("created_at"),
                    "source": r["payload"].get("source"),
                    "conversation_id": r["payload"].get("conversation_id"),
                }
                for r in results
                if r.get("payload")
            ]
        except Exception as e:
            logger.error(f"Archival search 失败: {e}", exc_info=True)
            return []


archival_memory_service = ArchivalMemoryService()
