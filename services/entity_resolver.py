"""实体对齐与异常过滤

解决 LLM 抽取的实体在 Neo4j 里产生重复节点的问题：
- "Transformer" vs "Transformer 模型" vs "transformer" → 同一个节点
- "  Transformer  "（前后空格）→ 同一个节点
- 超长实体名（>50 字）→ 异常，丢弃
- 空实体名 → 异常，丢弃

策略：
1. 规范化：去首尾空格、合并中间多空格、统一全半角
2. 长度检查：1-50 字符之间为合法
3. 模糊对齐：查 Neo4j 已有实体，相似度 >=0.9 时复用已有节点名
"""
import re
from typing import Optional, Tuple, List, Dict
from difflib import SequenceMatcher


# 实体名长度限制
MIN_ENTITY_NAME_LEN = 1
MAX_ENTITY_NAME_LEN = 50

# 模糊对齐阈值：相似度 >= 此值时复用已有实体名
ENTITY_FUZZY_THRESHOLD = 0.9


def normalize_entity_name(raw: Optional[str]) -> Tuple[str, bool]:
    """规范化实体名：返回 (规范化后的名称, 是否合法)

    规范化步骤：
    1. 去首尾空格
    2. 合并中间多个连续空格为单个空格
    3. 全角空格 → 半角空格
    4. 长度检查（1-50 字符）

    Returns:
        (normalized, is_valid): is_valid=False 表示非法实体名，应丢弃
    """
    if not raw or not isinstance(raw, str):
        return "", False

    # 全角空格 → 半角
    name = raw.replace("\u3000", " ")
    # 去首尾空格
    name = name.strip()
    # 合并中间连续空格
    name = re.sub(r"\s+", " ", name)

    # 长度检查
    if len(name) < MIN_ENTITY_NAME_LEN or len(name) > MAX_ENTITY_NAME_LEN:
        return name, False

    return name, True


def is_valid_entity_name(name: str) -> bool:
    """检查实体名是否合法（不做规范化，只判断）"""
    if not name or not isinstance(name, str):
        return False
    if len(name) < MIN_ENTITY_NAME_LEN or len(name) > MAX_ENTITY_NAME_LEN:
        return False
    return True


def find_existing_entity(
    name: str,
    existing_entities: List[str],
    threshold: float = ENTITY_FUZZY_THRESHOLD,
) -> Optional[str]:
    """在已有实体列表中找最接近的实体名（模糊对齐）

    如果找到相似度 >= threshold 的已有实体，返回已有实体名（复用节点）
    否则返回 None（创建新节点）

    Args:
        name: 待对齐的实体名
        existing_entities: 已有实体名列表（从 Neo4j 查询）
        threshold: 相似度阈值，默认 0.9

    Returns:
        已有实体名（复用）或 None（新建）
    """
    if not name or not existing_entities:
        return None

    name_lower = name.lower()

    # 策略 1：大小写不敏感精确匹配
    for existing in existing_entities:
        if existing.lower() == name_lower:
            return existing

    # 策略 2：包含关系（"Transformer 模型" 包含 "Transformer" 或反之）
    for existing in existing_entities:
        e_lower = existing.lower()
        if len(name_lower) >= 3 and len(e_lower) >= 3:
            # 较短的实体名是较长的子串时，复用较长的（信息更完整）
            if name_lower in e_lower and len(e_lower) <= len(name_lower) + 5:
                return existing
            if e_lower in name_lower and len(name_lower) <= len(e_lower) + 5:
                return existing

    # 策略 3：相似度匹配
    best_match = None
    best_score = 0.0
    for existing in existing_entities:
        score = SequenceMatcher(None, name_lower, existing.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = existing

    if best_score >= threshold:
        return best_match

    return None


def deduplicate_triplets(triplets: List[Dict]) -> List[Dict]:
    """三元组去重：同一对实体 + 同一关系只保留一条

    LLM 对不同 chunk 独立抽取，可能产出重复三元组。
    去重策略：以 (head, relation, tail) 三元组为 key 去重，
    相同 key 的多条记录合并 source_chunk 列表。
    """
    if not triplets:
        return []

    seen: Dict[Tuple[str, str, str], Dict] = {}
    for t in triplets:
        head = str(t.get("head", "")).strip().lower()
        tail = str(t.get("tail", "")).strip().lower()
        relation = str(t.get("relation", "")).strip().upper()
        if not head or not tail or not relation:
            continue
        key = (head, relation, tail)
        if key not in seen:
            seen[key] = dict(t)
        else:
            # 合并 source_chunk（保留多个来源）
            existing = seen[key]
            existing_chunks = existing.get("_source_chunks", [])
            new_chunk = t.get("_source_chunk")
            if new_chunk and new_chunk not in existing_chunks:
                existing_chunks.append(new_chunk)
            existing["_source_chunks"] = existing_chunks

    return list(seen.values())
