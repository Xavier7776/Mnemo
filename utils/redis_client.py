"""Redis 倒排索引服务 —— 基于 RediSearch 的 BM25 关键词检索

使用 Redis Stack 的 RediSearch 模块实现工业级全文检索：
- 自动倒排索引维护（无需手写倒排表/正排表）
- 内置 BM25 打分（SCORER BM25）
- 内置停用词过滤（STOPWORDS）
- TAG 字段高效过滤（按 document_id 过滤）
- 自动文档频率统计和 IDF 计算

数据结构：
  RediSearch 索引: rag_chunk_idx
  文档存储: rag:chunk:{chunk_id} -> Hash {text, document_id, chunk_id}
  RediSearch 自动维护倒排索引和统计信息

接口与旧版手写倒排索引完全兼容，调用方无需修改。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import logger

# —— Redis 客户端惰性单例 ——
_redis_client = None
_redis_unavailable = False
_redisearch_available = False
_index_ready = False

INDEX_NAME = "rag_chunk_idx"
PREFIX = "rag:chunk:"


def get_redis_client():
    """获取全局 Redis 客户端（惰性初始化）"""
    global _redis_client, _redis_unavailable, _redisearch_available
    if _redis_client is not None:
        return _redis_client
    if _redis_unavailable:
        return None
    try:
        import os
        import redis
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        db = int(os.getenv("REDIS_DB", "0"))
        password = os.getenv("REDIS_PASSWORD", "")
        _redis_client = redis.Redis(
            host=host, port=port, db=db,
            password=password if password else None,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=3.0,
            retry_on_timeout=True,
        )
        _redis_client.ping()
        logger.info(f"Redis 连接成功: {host}:{port}/{db}")

        try:
            modules = _redis_client.module_list()
            _redisearch_available = any(m.get("name") == "search" for m in modules)
            if _redisearch_available:
                logger.info("RediSearch 模块可用，使用工业级全文检索")
            else:
                logger.warning("RediSearch 模块不可用，需要 Redis Stack 镜像")
        except Exception as me:
            logger.warning(f"RediSearch 模块检测失败: {me}")
            _redisearch_available = False

        return _redis_client
    except ImportError:
        logger.warning("redis 库未安装，BM25 倒排索引不可用")
        _redis_unavailable = True
        return None
    except Exception as e:
        logger.warning(f"Redis 连接失败: {e}")
        _redis_unavailable = True
        return None


def _tokenize(text: str) -> List[str]:
    """jieba 分词"""
    clean = (text or "").lower()
    try:
        import jieba
        tokens = [t.strip() for t in jieba.cut(clean) if t.strip()]
    except Exception:
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", clean)
    return [t for t in tokens if len(t) > 1 or re.match(r"[\u4e00-\u9fff]", t)]


def _ensure_index():
    """确保 RediSearch 索引已创建（幂等）"""
    global _index_ready
    if _index_ready:
        return True
    r = get_redis_client()
    if r is None or not _redisearch_available:
        return False
    try:
        r.execute_command(
            "FT.CREATE", INDEX_NAME,
            "ON", "HASH",
            "PREFIX", "1", PREFIX,
            "SCHEMA",
            "text", "TEXT",
            "document_id", "TAG",
            "chunk_id", "TAG",
        )
        logger.info(f"RediSearch 索引创建成功: {INDEX_NAME}")
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info(f"RediSearch 索引已存在: {INDEX_NAME}")
        else:
            logger.warning(f"RediSearch 索引创建失败: {e}")
            return False
    _index_ready = True
    return True


def index_chunk(chunk_id: str, text: str, document_id: str) -> bool:
    """为单个 chunk 构建倒排索引"""
    r = get_redis_client()
    if r is None or not _redisearch_available:
        return False
    if not _ensure_index():
        return False
    try:
        tokens = _tokenize(text)
        tokenized_text = " ".join(tokens) if tokens else ""
        key = f"{PREFIX}{chunk_id}"
        r.hset(key, mapping={
            "text": tokenized_text,
            "document_id": document_id or "",
            "chunk_id": chunk_id,
        })
        return True
    except Exception as e:
        logger.error(f"RediSearch 索引构建失败 chunk={chunk_id}: {e}")
        return False


def remove_chunk(chunk_id: str) -> bool:
    """删除单个 chunk 的倒排索引"""
    r = get_redis_client()
    if r is None:
        return False
    try:
        key = f"{PREFIX}{chunk_id}"
        r.delete(key)
        return True
    except Exception as e:
        logger.error(f"RediSearch 索引删除失败 chunk={chunk_id}: {e}")
        return False


def search_bm25(
    query_terms: List[str],
    top_k: int = 200,
    document_id: Optional[str] = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[str, float]]:
    """用 RediSearch BM25 检索

    Returns: [(chunk_id, score), ...] 按 score 降序
    """
    r = get_redis_client()
    if r is None or not _redisearch_available:
        return []
    if not _ensure_index():
        return []
    if not query_terms:
        return []

    try:
        # 构造查询字符串
        # 清洗 token：过滤掉包含 RediSearch 特殊字符的 token
        # RediSearch 特殊字符: ,.<>{}[]"':;!@#$%^&*()-+=~  等
        import re as _re
        clean_terms = []
        for t in query_terms:
            # 只保留字母、数字、中文
            if _re.match(r"^[\w\u4e00-\u9fff]+$", t):
                clean_terms.append(t)
        if not clean_terms:
            return []
        # OR 语义，BM25 会自动给匹配更多词的文档更高分
        text_query = "|".join(clean_terms)

        if document_id:
            # TAG 过滤语法：@document_id:{value}
            # 用 DIALECT 2 语法
            query_str = f"@text:({text_query}) @document_id:{{{document_id}}}"
        else:
            query_str = f"@text:({text_query})"

        # FT.SEARCH，使用 WITHSCORES + SCORER BM25
        # redis-py 新版返回字典格式：{'results': [...], 'total_results': N}
        result = r.execute_command(
            "FT.SEARCH", INDEX_NAME,
            query_str,
            "WITHSCORES",
            "SCORER", "BM25",
            "LIMIT", 0, top_k,
        )

        # 适配两种返回格式（字典或列表）
        if isinstance(result, dict):
            results_list = result.get("results", [])
            total = result.get("total_results", 0)
        elif isinstance(result, list):
            total = result[0] if result else 0
            results_list = []
            items = result[1:] if len(result) > 1 else []
            i = 0
            while i + 1 < len(items):
                doc_key = items[i]
                score_str = items[i + 1]
                cid = doc_key.replace(PREFIX, "") if doc_key.startswith(PREFIX) else doc_key
                try:
                    score = float(score_str)
                except (ValueError, TypeError):
                    score = 0.0
                results_list.append({"id": doc_key, "score": score})
                i += 3
        else:
            return []

        if not results_list:
            return []

        # 解析结果
        results: List[Tuple[str, float]] = []
        for item in results_list:
            doc_key = item.get("id", "")
            score = item.get("score", 0.0)
            cid = doc_key.replace(PREFIX, "") if doc_key.startswith(PREFIX) else doc_key
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = 0.0
            results.append((cid, score))

        return results

    except Exception as e:
        logger.error(f"RediSearch BM25 检索失败: {e}")
        return []


def get_stats() -> Dict[str, Any]:
    """获取索引统计信息"""
    r = get_redis_client()
    if r is None or not _redisearch_available:
        return {"available": False}
    try:
        info = r.execute_command("FT.INFO", INDEX_NAME)
        # redis-py 新版返回字典，旧版返回列表
        if isinstance(info, dict):
            num_docs = info.get("num_docs", 0)
            num_terms = info.get("num_terms", 0)
        else:
            info_dict = {}
            for i in range(0, len(info), 2):
                info_dict[info[i]] = info[i + 1]
            num_docs = info_dict.get("num_docs", 0)
            num_terms = info_dict.get("num_terms", 0)

        return {
            "available": True,
            "engine": "RediSearch",
            "index_name": INDEX_NAME,
            "total_docs": num_docs,
            "total_terms": num_terms,
        }
    except Exception as e:
        logger.warning(f"获取 RediSearch 统计失败: {e}")
        return {"available": False}


def is_available() -> bool:
    """检查 RediSearch 是否可用"""
    if get_redis_client() is None:
        return False
    return _redisearch_available


def clear_all() -> bool:
    """清空所有索引数据"""
    r = get_redis_client()
    if r is None:
        return False
    try:
        if _redisearch_available:
            try:
                r.execute_command("FT.DROPINDEX", INDEX_NAME)
                logger.info(f"RediSearch 索引已删除: {INDEX_NAME}")
            except Exception:
                pass

        keys = list(r.scan_iter(match=f"{PREFIX}*"))
        if keys:
            r.delete(*keys)
        logger.info(f"已清空 {len(keys)} 个 chunk 文档")

        global _index_ready
        _index_ready = False
        return True
    except Exception as e:
        logger.error(f"清空 RediSearch 索引失败: {e}")
        return False
