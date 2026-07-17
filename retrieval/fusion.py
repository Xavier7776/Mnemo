"""Pure retrieval result fusion helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple


def _filter_low_quality(results: List[Dict[str, Any]], modality: str) -> List[Dict[str, Any]]:
    """融合前过滤低质量结果，避免噪声稀释正确排名。

    策略：
    1. 排名截断：每路只保留 top-N（避免长尾噪声参与 RRF）
       - vector: top-30（向量分数区分度高，多保留）
       - keyword: top-20（BM25 分数扁平，长尾噪声多）
       - graph: top-10（图谱结果少）
    2. 相对分数过滤：在截断后的集合中，剔除分数低于 top1 * ratio 的结果
       - vector: ratio=0.3（向量分数区分度高）
       - keyword: ratio=0.1（BM25 分数扁平，ratio 低一些）
       - graph: ratio=0.1
    3. 兜底：至少保留前 5 名
    """
    if not results:
        return results

    # 步骤 1：排名截断
    rank_cutoff_map = {"vector": 30, "keyword": 20, "graph": 10}
    rank_cutoff = rank_cutoff_map.get(modality, 20)
    truncated = results[:rank_cutoff]

    if len(truncated) < 5:
        return truncated

    # 步骤 2：相对分数过滤
    ratio_map = {"vector": 0.3, "keyword": 0.1, "graph": 0.1}
    ratio = ratio_map.get(modality, 0.2)

    top_score = 0.0
    for r in truncated:
        s = float(r.get("score", 0.0) or 0.0)
        if s > top_score:
            top_score = s

    if top_score <= 0:
        return truncated

    threshold = top_score * ratio
    filtered = [r for r in truncated if float(r.get("score", 0.0) or 0.0) >= threshold]
    # 兜底：至少保留前 5 名
    if len(filtered) < 5:
        return truncated[:5]
    return filtered


def merge_results_rrf(
    ranked_lists: List[Tuple[str, List[Dict[str, Any]], float]],
    rrf_k: float | None = None,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion for heterogeneous retrieval result lists.

    优化：融合前对每路结果做相对分数阈值过滤，剔除低质量噪声，
    避免 keyword/graph 的噪声结果通过 RRF 稀释 vector 的正确排名。
    RRF_K 从 60 降到 30：增大排名差距对分数的影响，让 top 结果更容易浮上来。
    """
    k = float(rrf_k if rrf_k is not None else os.getenv("RRF_K", "30"))
    result_dict: Dict[str, Dict[str, Any]] = {}

    for modality, results, weight in ranked_lists:
        # 融合前过滤低质量结果
        filtered_results = _filter_low_quality(results or [], modality)
        for rank, res in enumerate(filtered_results, start=1):
            payload = res.get("payload") or {}
            key = str(payload.get("chunk_id") or res.get("id"))
            if not key:
                continue
            if key not in result_dict:
                copied = dict(res)
                copied["payload"] = dict(payload)
                copied["score"] = 0.0
                copied["combined_score"] = 0.0
                copied["retrieval_types"] = []
                copied["raw_scores"] = {}
                result_dict[key] = copied

            item = result_dict[key]
            item["score"] = float(item.get("score", 0.0) or 0.0) + weight / (k + rank)
            item["combined_score"] = item["score"]
            if modality not in item["retrieval_types"]:
                item["retrieval_types"].append(modality)
            item["payload"]["retrieval_type"] = "hybrid" if len(item["retrieval_types"]) > 1 else modality
            item["payload"]["retrieval_types"] = item["retrieval_types"]
            item["raw_scores"][modality] = res.get("score", 0.0)

    merged = list(result_dict.values())
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return merged
