"""知识图谱 Ontology 注册表

预定义实体类型和关系类型，约束 LLM 输出，避免：
- 同一语义产出不同关系名（"定义于"→DEFINED_IN vs "described in"→DESCRIBED_IN）
- 同一实体产出不同 label（Transformer 标成 Concept vs Technology）
- 关系类型完全失控（LLM 每轮独立产出不同名称）

设计：
- 实体类型：6 大类 + Other 兜底，与原 prompt 一致
- 关系类型：分 4 大类（结构关系/语义关系/动作关系/属性关系），共 20+ 预定义关系
- 别名表：LLM 常见输出 → 预定义关系/类型的映射
- 模糊匹配：edit distance <=2 或包含关系时自动对齐
"""
import re
from typing import Optional, Tuple, List, Dict
from difflib import SequenceMatcher


# ==================== 预定义实体类型 ====================
ENTITY_TYPES: Dict[str, List[str]] = {
    "Concept": ["概念", "理论", "思想", "原理", "conception", "theory", "idea"],
    "Technology": ["技术", "工具", "框架", "平台", "系统", "tech", "tool", "framework", "platform"],
    "Person": ["人", "人物", "作者", "开发者", "person", "people", "author"],
    "Organization": ["组织", "机构", "公司", "团队", "org", "organization", "company", "team"],
    "Location": ["地点", "位置", "国家", "城市", "location", "place", "country", "city"],
    "Event": ["事件", "活动", "会议", "发布", "event", "activity", "conference"],
    "Other": ["其他", "other", "unknown"],
}


# ==================== 预定义关系类型 ====================
# 关系分类：(预定义关系名, 中文别名列表, 英文别名列表)
RELATION_TYPES: List[Tuple[str, List[str], List[str]]] = [
    # —— 结构关系 ——
    ("DEFINED_IN", ["定义于", "定义在", "被定义", "定义为"], ["defined_in", "defined_at", "is_defined_in", "definition_of"]),
    ("DESCRIBED_IN", ["描述于", "描述在", "记载于", "提及于", "提到", "出现于"], ["described_in", "mentioned_in", "appears_in", "referenced_in"]),
    ("PART_OF", ["属于", "组成部分", "一部分", "包含于", "隶属于"], ["part_of", "belongs_to", "subset_of", "contained_in"]),
    ("HAS_PART", ["包含", "包括", "由...组成", "组成部分有"], ["has_part", "contains", "consists_of", "includes"]),
    ("INSTANCE_OF", ["实例", "实例化", "是...的实例", "属于类型"], ["instance_of", "is_a", "type_of"]),

    # —— 语义关系 ——
    ("RELATED_TO", ["相关", "关联", "有关", "联系"], ["related_to", "associated_with", "connected_to"]),
    ("SIMILAR_TO", ["相似", "类似", "类似于是"], ["similar_to", "analogous_to"]),
    ("OPPOSITE_OF", ["相反", "对立", "反义"], ["opposite_of", "contradicts", "antonym_of"]),
    ("DEPENDS_ON", ["依赖", "依赖于", "需要", "基于"], ["depends_on", "requires", "based_on", "relies_on"]),
    ("DERIVED_FROM", ["源自", "派生自", "来源于", "来自"], ["derived_from", "originates_from", "comes_from"]),
    ("EQUIVALENT_TO", ["等价", "等同", "相同", "等于"], ["equivalent_to", "same_as", "equals"]),

    # —— 动作关系 ——
    ("CREATED_BY", ["创建者", "由...创建", "作者", "发明者"], ["created_by", "authored_by", "invented_by", "developed_by"]),
    ("USED_BY", ["被使用", "被...使用", "应用于"], ["used_by", "applied_in", "utilized_by"]),
    ("MANAGES", ["管理", "负责", "控制"], ["manages", "controls", "responsible_for"]),
    ("PRODUCES", ["产出", "生成", "产生"], ["produces", "generates", "creates", "outputs"]),
    ("TRIGGERS", ["触发", "引起", "导致"], ["triggers", "causes", "leads_to"]),

    # —— 属性关系 ——
    ("HAS_PROPERTY", ["有属性", "属性是", "特征是"], ["has_property", "property_of", "characteristic_of"]),
    ("LOCATED_IN", ["位于", "在...地方"], ["located_in", "situated_in"]),
    ("OCCURRED_AT", ["发生于", "发生在", "时间在"], ["occurred_at", "happened_at", "took_place_at"]),
    ("HAS_ROLE", ["角色是", "担任", "作为"], ["has_role", "acts_as", "serves_as"]),
]


class OntologyRegistry:
    """Ontology 注册表：实体类型 + 关系类型规范化

    主要功能：
    1. normalize_entity_type: LLM 输出的实体类型 → 预定义类型
    2. normalize_relation: LLM 输出的关系名 → 预定义关系
    3. fuzzy_match: 模糊匹配（处理拼写差异、中英混合）
    """

    def __init__(self):
        # 构建别名 → 标准类型的反查表
        self._entity_alias_map: Dict[str, str] = {}
        for canonical, aliases in ENTITY_TYPES.items():
            self._entity_alias_map[canonical.lower()] = canonical
            for alias in aliases:
                self._entity_alias_map[alias.lower()] = canonical

        self._relation_alias_map: Dict[str, str] = {}
        self._canonical_relations: List[str] = []
        for canonical, cn_aliases, en_aliases in RELATION_TYPES:
            self._canonical_relations.append(canonical)
            self._relation_alias_map[canonical.lower()] = canonical
            for alias in cn_aliases:
                self._relation_alias_map[alias.lower()] = canonical
            for alias in en_aliases:
                self._relation_alias_map[alias.lower()] = canonical

    def normalize_entity_type(self, raw: Optional[str]) -> str:
        """规范化实体类型：LLM 输出 → 预定义类型

        策略：
        1. 精确匹配别名表（大小写不敏感）
        2. 模糊匹配（相似度 >= 0.8）
        3. 兜底返回 Other
        """
        if not raw or not raw.strip():
            return "Other"

        key = raw.strip().lower()

        # 策略 1：精确匹配
        if key in self._entity_alias_map:
            return self._entity_alias_map[key]

        # 策略 2：模糊匹配（处理 "concept." "concepts" "技术类" 等变体）
        best_match, best_score = self._fuzzy_match(key, list(self._entity_alias_map.keys()))
        if best_score >= 0.8:
            return self._entity_alias_map[best_match]

        # 策略 3：兜底
        return "Other"

    def normalize_relation(self, raw: Optional[str]) -> str:
        """规范化关系名：LLM 输出 → 预定义关系

        策略：
        1. 清洗（去除标点、空格）
        2. 精确匹配别名表
        3. 模糊匹配（相似度 >= 0.85，关系名要求更高）
        4. 兜底：RELATED_TO（最通用的关系）
        """
        if not raw or not raw.strip():
            return "RELATED_TO"

        # 清洗：去除非字母数字和中文字符
        cleaned = re.sub(r"[^\w\u4e00-\u9fa5]", "_", raw.strip())
        key = cleaned.lower().strip("_")

        # 策略 1：精确匹配
        if key in self._relation_alias_map:
            return self._relation_alias_map[key]

        # 策略 2：模糊匹配
        best_match, best_score = self._fuzzy_match(key, list(self._relation_alias_map.keys()))
        if best_score >= 0.85:
            return self._relation_alias_map[best_match]

        # 策略 3：兜底为最通用的关系
        return "RELATED_TO"

    def is_valid_entity_type(self, entity_type: str) -> bool:
        """检查是否为预定义实体类型"""
        return entity_type in ENTITY_TYPES

    def is_valid_relation(self, relation: str) -> bool:
        """检查是否为预定义关系类型"""
        return relation in self._canonical_relations

    def get_all_entity_types(self) -> List[str]:
        """获取所有预定义实体类型"""
        return list(ENTITY_TYPES.keys())

    def get_all_relations(self) -> List[str]:
        """获取所有预定义关系类型"""
        return list(self._canonical_relations)

    @staticmethod
    def _fuzzy_match(query: str, candidates: List[str], threshold: float = 0.8) -> Tuple[Optional[str], float]:
        """模糊匹配：返回最佳匹配和相似度

        使用 difflib.SequenceMatcher，相似度 = 2*M/T（M=匹配字符数，T=总字符数）
        """
        if not query or not candidates:
            return None, 0.0

        best_match = None
        best_score = 0.0
        for c in candidates:
            score = SequenceMatcher(None, query, c).ratio()
            if score > best_score:
                best_score = score
                best_match = c
        return best_match, best_score


# 全局单例
ontology_registry = OntologyRegistry()
