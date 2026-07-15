"""Redis 倒排索引服务 —— BM25 关键词检索加速

数据结构：
  rag:invindex:{token}    → Hash {chunk_id: tf}      倒排表（token → chunk 词频映射）
  rag:chunk_tokens:{cid}  → Set {token1, ...}        正排表（chunk → token 集合，删除时清理用）
  rag:chunk_len:{cid}     → String (int)             chunk 的 token 数（BM25 的 dl）
  rag:chunk_doc:{cid}     → String                   chunk 的 document_id（按文档过滤）
  rag:stats               → Hash {total_docs, total_terms_len, avgdl}  全局统计

查询流程：
  1. jieba 分词 query
  2. 对每个 query token，HGETALL 倒排表 → 候选 chunk_id + tf
  3. 从 Redis 读取每个候选 chunk 的 dl
  4. 用全局统计计算 IDF
  5. Okapi BM25 打分
  6. 返回 top-K chunk_id 列表
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import logger

# —— Redis 客户端惰性单例 ——
_redis_client = None
_redis_unavailable = False  # 标记 Redis 不可用，避免重复警告


def get_redis_client():
    """获取全局 Redis 客户端（惰性初始化，不可用时只警告一次）"""
    global _redis_client, _redis_unavailable
    if _redis_client is not None:
        return _redis_client
    if _redis_unavailable:
        return None
    try:
        import redis
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        db = int(os.getenv("REDIS_DB", "0"))
        password = os.getenv("REDIS_PASSWORD", "")
        _redis_client = redis.Redis(
            host=host, port=port, db=db,
            password=password if password else None,
            decode_responses=True,  # 自动解码为 str
            socket_timeout=5.0,
            socket_connect_timeout=3.0,
            retry_on_timeout=True,
        )
        _redis_client.ping()
        logger.info(f"Redis 连接成功: {host}:{port}/{db}")
        return _redis_client
    except ImportError:
        logger.warning("redis 库未安装，BM25 倒排索引不可用，将回退到 MongoDB 全表扫描")
        _redis_unavailable = True
        return None
    except Exception as e:
        logger.warning(f"Redis 连接失败，BM25 倒排索引不可用: {e}")
        _redis_unavailable = True
        return None


def _tokenize(text: str) -> List[str]:
    """jieba 分词（与 rag_retriever._tokenize 保持一致）"""
    clean = (text or "").lower()
    try:
        import jieba
        tokens = [t.strip() for t in jieba.cut(clean) if t.strip()]
    except Exception:
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", clean)
    return [t for t in tokens if len(t) > 1 or re.match(r"[\u4e00-\u9fff]", t)]


# —— 倒排索引操作 ——

def index_chunk(chunk_id: str, text: str, document_id: str) -> bool:
    """为单个 chunk 构建倒排索引

    Args:
        chunk_id: chunk 的唯一 ID（MongoDB ObjectId 的字符串形式）
        text: chunk 原始文本
        document_id: 所属文档 ID

    Returns:
        True 成功，False 失败（Redis 不可用）
    """
    r = get_redis_client()
    if r is None:
        return False
    try:
        tokens = _tokenize(text)
        if not tokens:
            return True

        # 统计词频
        term_counts: Dict[str, int] = {}
        for t in tokens:
            term_counts[t] = term_counts.get(t, 0) + 1

        pipe = r.pipeline()

        # 写倒排表：rag:invindex:{token} → {chunk_id: tf}
        for token, tf in term_counts.items():
            pipe.hset(f"rag:invindex:{token}", chunk_id, tf)

        # 写正排表：rag:chunk_tokens:{chunk_id} → Set{token1, ...}（删除时清理用）
        pipe.sadd(f"rag:chunk_tokens:{chunk_id}", *term_counts.keys())

        # 写 chunk 元数据
        pipe.set(f"rag:chunk_len:{chunk_id}", len(tokens))
        pipe.set(f"rag:chunk_doc:{chunk_id}", document_id or "")

        # 更新全局统计
        pipe.hincrby("rag:stats", "total_docs", 1)
        pipe.hincrby("rag:stats", "total_terms_len", len(tokens))

        pipe.execute()
        return True
    except Exception as e:
        logger.error(f"倒排索引构建失败 chunk={chunk_id}: {e}")
        return False


def remove_chunk(chunk_id: str) -> bool:
    """删除单个 chunk 的倒排索引（文档删除时调用）

    Args:
        chunk_id: chunk 的唯一 ID

    Returns:
        True 成功，False 失败
    """
    r = get_redis_client()
    if r is None:
        return False
    try:
        # 获取该 chunk 的所有 token
        tokens = r.smembers(f"rag:chunk_tokens:{chunk_id}")
        if not tokens:
            # 可能已经被删除了
            return True

        pipe = r.pipeline()

        # 从倒排表中删除该 chunk
        for token in tokens:
            pipe.hdel(f"rag:invindex:{token}", chunk_id)
            # 如果倒排表空了，删除整个 key
            # 注意：这里不删除空 key，HDEL 后 Redis 会自动清理空 Hash

        # 删除正排表和元数据
        pipe.delete(f"rag:chunk_tokens:{chunk_id}")
        pipe.delete(f"rag:chunk_len:{chunk_id}")
        pipe.delete(f"rag:chunk_doc:{chunk_id}")

        # 更新全局统计
        old_len = r.get(f"rag:chunk_len:{chunk_id}")
        pipe.hincrby("rag:stats", "total_docs", -1)
        if old_len:
            pipe.hincrby("rag:stats", "total_terms_len", -int(old_len))

        pipe.execute()
        return True
    except Exception as e:
        logger.error(f"倒排索引删除失败 chunk={chunk_id}: {e}")
        return False


def search_bm25(
    query_terms: List[str],
    top_k: int = 200,
    document_id: Optional[str] = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[str, float]]:
    """用倒排索引计算 BM25 分数

    Args:
        query_terms: 已分词的 query token 列表
        top_k: 返回的前 K 个结果
        document_id: 可选，按文档过滤
        k1, b: BM25 参数

    Returns:
        [(chunk_id, score), ...] 按 score 降序
    """
    r = get_redis_client()
    if r is None:
        return []

    if not query_terms:
        return []

    try:
        # 1. 收集候选 chunk 和每个 token 的倒排表
        # candidate_tf: {chunk_id: {token: tf}}
        candidate_chunks: Dict[str, Dict[str, int]] = {}
        # doc_freq: {token: df}  该 token 在多少个 chunk 中出现
        doc_freq: Dict[str, int] = {}

        for token in query_terms:
            # HGETALL 获取该 token 的倒排表 {chunk_id: tf}
            invlist = r.hgetall(f"rag:invindex:{token}")
            if not invlist:
                continue
            df = len(invlist)
            doc_freq[token] = df
            for cid, tf_str in invlist.items():
                candidate_chunks.setdefault(cid, {})[token] = int(tf_str)

        if not candidate_chunks:
            return []

        # 2. 获取全局统计
        stats = r.hgetall("rag:stats")
        total_docs = int(stats.get("total_docs", 1))
        total_terms_len = int(stats.get("total_terms_len", 0))
        avgdl = total_terms_len / max(total_docs, 1)

        # 3. 如果需要按文档过滤，批量获取候选 chunk 的 document_id
        if document_id:
            pipe = r.pipeline()
            chunk_ids = list(candidate_chunks.keys())
            for cid in chunk_ids:
                pipe.get(f"rag:chunk_doc:{cid}")
            doc_ids = pipe.execute()
            # 过滤掉不属于该文档的 chunk
            filtered = {}
            for cid, did in zip(chunk_ids, doc_ids):
                if did == document_id:
                    filtered[cid] = candidate_chunks[cid]
            candidate_chunks = filtered

        if not candidate_chunks:
            return []

        # 4. 批量获取候选 chunk 的长度（dl）
        pipe = r.pipeline()
        chunk_ids = list(candidate_chunks.keys())
        for cid in chunk_ids:
            pipe.get(f"rag:chunk_len:{cid}")
        chunk_lens = pipe.execute()

        # 5. 计算 BM25 分数
        results: List[Tuple[str, float]] = []
        for cid, dl_str in zip(chunk_ids, chunk_lens):
            dl = int(dl_str) if dl_str else 0
            term_counts = candidate_chunks[cid]
            score = 0.0
            for token in query_terms:
                tf = term_counts.get(token, 0)
                if tf <= 0:
                    continue
                df = doc_freq.get(token, 0)
                if df <= 0:
                    continue
                # Okapi BM25 (Lucene 变体 IDF，避免负值)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1 - b + b * dl / max(avgdl, 1))
                score += idf * (tf * (k1 + 1)) / denom
            if score > 0:
                results.append((cid, score))

        # 6. 排序并返回 top-K
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    except Exception as e:
        logger.error(f"Redis BM25 检索失败: {e}")
        return []


def get_stats() -> Dict[str, Any]:
    """获取倒排索引全局统计"""
    r = get_redis_client()
    if r is None:
        return {"available": False}
    try:
        stats = r.hgetall("rag:stats")
        return {
            "available": True,
            "total_docs": int(stats.get("total_docs", 0)),
            "total_terms_len": int(stats.get("total_terms_len", 0)),
            "avgdl": int(stats.get("total_terms_len", 0)) / max(int(stats.get("total_docs", 1)), 1),
        }
    except Exception:
        return {"available": False}


def is_available() -> bool:
    """检查 Redis 倒排索引是否可用"""
    return get_redis_client() is not None


def clear_all() -> bool:
    """清空所有倒排索引（仅用于全量重建）"""
    r = get_redis_client()
    if r is None:
        return False
    try:
        # 获取所有 rag: 前缀的 key
        keys = list(r.scan_iter(match="rag:*"))
        if keys:
            r.delete(*keys)
        logger.info(f"已清空 {len(keys)} 个倒排索引 key")
        return True
    except Exception as e:
        logger.error(f"清空倒排索引失败: {e}")
        return False
