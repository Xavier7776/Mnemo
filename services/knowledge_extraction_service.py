"""知识抽取与图谱构建服务"""
import json
import re
import asyncio
import os
import time
from typing import List, Dict, Any
from utils.llm_client import get_openai_client
from database.neo4j_client import neo4j_client
from utils.logger import logger
from services.ontology_registry import ontology_registry
from services.entity_resolver import (
    normalize_entity_name,
    is_valid_entity_name,
    find_existing_entity,
    deduplicate_triplets,
)


class KnowledgeExtractionService:
    """知识抽取与图谱构建服务"""

    def __init__(self):
        # 不在此处创建 OpenAI 客户端——main.py 加载 .env 后才可用
        self._client = None
        self._model_name = None
        self._neo4j_disabled_until_ts = 0.0
        # 受控的抽取 prompt：明确告诉 LLM 实体类型和关系类型
        self.extraction_prompt_template = """
你是一个知识图谱专家。请从以下文本中提取"实体-关系-实体"三元组。
请严格按照 JSON 格式返回结果，不要包含任何其他解释性文字。
返回格式示例：
[
  {{ "head": "实体1", "head_type": "类型1", "relation": "关系", "tail": "实体2", "tail_type": "类型2" }},
  ...
]

重要约束：
- 最多提取 8 个最核心的三元组，不要贪多（避免输出被截断）
- 只提取明确表述的关系，不要推断或猜测

【实体类型】必须是以下 7 种之一（不允许自由发挥）：
- Concept: 概念、理论、思想、原理
- Technology: 技术、工具、框架、平台、系统
- Person: 人物、作者、开发者
- Organization: 组织、机构、公司、团队
- Location: 地点、国家、城市
- Event: 事件、活动、会议
- Other: 以上都不属于

【关系类型】尽量使用以下预定义关系（中英文均可）：
- 结构关系: 定义于/定义在(DEFINED_IN), 描述于/提及于(DESCRIBED_IN), 属于/组成部分(PART_OF), 包含/包括(HAS_PART), 实例(INSTANCE_OF)
- 语义关系: 相关(RELATED_TO), 相似(SIMILAR_TO), 相反(OPPOSITE_OF), 依赖(DEPENDS_ON), 源自(DERIVED_FROM), 等价(EQUIVALENT_TO)
- 动作关系: 由...创建(CREATED_BY), 被使用(USED_BY), 管理(MANAGES), 产出(PRODUCES), 触发(TRIGGERS)
- 属性关系: 有属性(HAS_PROPERTY), 位于(LOCATED_IN), 发生于(OCCURRED_AT), 担任(HAS_ROLE)

如果以上都不合适，可以输出自定义关系，但必须简洁（<10 字符）。

实体名要求：
- 长度 1-50 字符
- 不要包含句子片段或整句描述
- 不要使用代词（他/她/它/这个/那个）

文本内容：
{text}
"""

    @property
    def client(self):
        return get_openai_client()

    @property
    def model_name(self):
        if self._model_name is None:
            self._model_name = os.getenv("LLM_MODEL", "mimo-v2.5")
        return self._model_name

    async def extract_triplets(self, text: str) -> List[Dict[str, Any]]:
        """使用 LLM 提取三元组"""
        prompt = self.extraction_prompt_template.format(text=text)

        def _sync_call_llm() -> str:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个知识图谱专家，只返回 JSON 格式的三元组列表。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4096,
                timeout=120
            )
            return response.choices[0].message.content or ""

        try:
            content = await asyncio.to_thread(_sync_call_llm)
            return self._parse_json(content)
        except Exception as e:
            logger.error(f"知识抽取失败: {e}")
            return []

    def _parse_json(self, content: str) -> List[Dict[str, Any]]:
        """
        解析 LLM 返回的 JSON 字符串。

        LLM 输出经常前后带自然语言解释、用非标准 json 代码块、或者返回
        "无实体" 这类纯文本。这里依次尝试多种策略，尽量把 JSON 抠出来。

        特殊处理：reasoning model（如 mimo-v2.5）可能因 max_tokens 不足导致
        JSON 数组被截断（有 `[` 无 `]`，最后一个对象不完整）。此时逐个提取
        已完整的 `{...}` 对象， salvage 部分结果。
        """
        if not content or not content.strip():
            return []

        parsed = None

        # 策略 1：直接 json.loads
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            pass

        # 策略 2：从 ```json ... ``` 或 ``` ... ``` 代码块提取
        if parsed is None:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                except Exception:
                    pass

        # 策略 3：正则抠首个 [ ... ] 或 { ... }（贪婪到最外层括号匹配）
        if parsed is None:
            try:
                # 优先匹配数组（三元组/实体列表常用数组）
                arr_match = re.search(r'\[[\s\S]*\]', content)
                if arr_match:
                    parsed = json.loads(arr_match.group(0))
                else:
                    obj_match = re.search(r'\{[\s\S]*\}', content)
                    if obj_match:
                        parsed = json.loads(obj_match.group(0))
            except Exception:
                pass

        # 策略 4：截断 JSON 数组 salvage（reasoning model 输出被 max_tokens 截断）
        # 场景：LLM 输出了 `[ {完整对象}, {完整对象}, {截断对象` 但没有闭合 `]`
        # 处理：逐个提取完整的 {...} 对象，跳过最后一个不完整的
        if parsed is None:
            salvaged = self._salvage_truncated_json_array(content)
            if salvaged:
                logger.info(f"截断 JSON salvage 成功: 提取 {len(salvaged)} 个完整对象")
                parsed = salvaged

        if parsed is None:
            logger.warning(f"无法解析 JSON: {content[:100]}...")
            return []

        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        logger.warning(f"JSON 解析结果不是列表或字典: {type(parsed)}")
        return []

    def _salvage_truncated_json_array(self, content: str) -> List[Any]:
        """从截断的 JSON 数组中提取已完整的元素。

        处理 reasoning model 输出被 max_tokens 截断的情况，支持两种格式：
        1. 对象数组: '[\n  {"head": "a", ...},\n  {"head": "b", ...},\n  {"head": "c"（截断）
           输出: [{"head": "a", ...}, {"head": "b", ...}]
        2. 字符串数组: '["entity1", "entity2", "enti（截断）
           输出: ["entity1", "entity2"]

        策略：逐字符扫描，用引号/括号深度判断元素边界。
        """
        # 找到第一个 [
        start = content.find('[')
        if start == -1:
            return []

        # 尝试逐个提取完整的 JSON 元素（对象或字符串）
        elements = []
        i = start + 1
        n = len(content)

        while i < n:
            # 跳过空白和逗号
            while i < n and content[i] in ' \t\n\r,':
                i += 1
            if i >= n:
                break

            ch = content[i]
            if ch == ']':
                # 数组正常结束
                break
            elif ch == '{':
                # 对象元素：找到匹配的 }
                depth = 0
                obj_start = i
                while i < n:
                    if content[i] == '{':
                        depth += 1
                    elif content[i] == '}':
                        depth -= 1
                        if depth == 0:
                            obj_str = content[obj_start:i + 1]
                            try:
                                obj = json.loads(obj_str)
                                if isinstance(obj, dict):
                                    elements.append(obj)
                            except json.JSONDecodeError:
                                pass
                            i += 1
                            break
                    i += 1
                # 如果 depth > 0 到结尾，说明对象被截断
                if depth > 0:
                    break
            elif ch == '"':
                # 字符串元素：找到匹配的结束引号（处理转义）
                str_start = i
                i += 1  # 跳过开引号
                while i < n:
                    if content[i] == '\\':
                        i += 2  # 跳过转义字符
                        continue
                    if content[i] == '"':
                        # 找到结束引号
                        str_str = content[str_start:i + 1]
                        try:
                            s = json.loads(str_str)
                            if isinstance(s, str):
                                elements.append(s)
                        except json.JSONDecodeError:
                            pass
                        i += 1
                        break
                    i += 1
                # 如果到结尾还没找到结束引号，说明字符串被截断
                else:
                    break
            else:
                # 数字、true/false/null 等（实体提取不会用到，跳过）
                i += 1

        return elements

    async def extract_entities(self, query: str) -> List[str]:
        """从查询中提取实体（用 asyncio.to_thread 包装同步调用，避免阻塞事件循环）"""
        prompt = f"""请从以下查询中提取关键实体（人名、地名、组织、概念、技术术语等）。
只返回实体列表，JSON 格式：["实体1", "实体2"]。
重要约束：
- 最多提取 5 个最核心的实体，不要贪多（避免输出被截断）
- 实体名要简洁（1-5个词），不要包含句子片段
- 不要包含任何解释

查询：{query}"""

        try:
            def _sync_extract() -> str:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个实体提取器，只返回 JSON 列表。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2048,
                    timeout=60
                )
                return response.choices[0].message.content or ""

            content = await asyncio.to_thread(_sync_extract)
            entities = self._parse_json(content)
            if isinstance(entities, list):
                return [str(e) for e in entities if isinstance(e, (str, int, float))]
            return []
        except Exception as e:
            logger.error(f"实体提取失败: {e}")
            return []

    async def build_graph(self, text: str, metadata: Dict[str, Any] = None):
        """构建知识图谱：抽取并存入 Neo4j

        修复三个核心问题：
        1. Ontology 约束：预定义实体类型和关系类型，LLM 输出经 ontology_registry 规范化
        2. 实体对齐：查询 Neo4j 已有实体，相似度 >=0.9 时复用节点名，避免重复节点
        3. 去重：同一 (head, relation, tail) 三元组合并 source_chunk 列表，避免重复边
        """
        neo4j_enabled = os.getenv("NEO4J_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
        if not neo4j_enabled:
            return

        now = time.time()
        if now < self._neo4j_disabled_until_ts:
            return

        if not neo4j_client.driver:
            neo4j_client.connect()
            if not neo4j_client.driver:
                self._neo4j_disabled_until_ts = now + 300
                logger.warning("Neo4j 未连接，已跳过图谱构建（将于 5 分钟后重试）")
                return

        triplets = await self.extract_triplets(text)
        if not triplets:
            return

        doc_id = (metadata.get("document_id") or metadata.get("source_doc")) if metadata else None
        chunk_id = (metadata.get("chunk_id") or metadata.get("source_chunk")) if metadata else None

        # ===== 后处理 1：规范化三元组 + 异常过滤 =====
        cleaned_triplets = []
        stats = {"total": len(triplets), "invalid_name": 0, "normalized": 0}
        for triplet in triplets:
            try:
                head_raw = triplet.get("head")
                tail_raw = triplet.get("tail")
                relation_raw = triplet.get("relation")
                head_type_raw = triplet.get("head_type", "Concept")
                tail_type_raw = triplet.get("tail_type", "Concept")

                if not head_raw or not tail_raw or not relation_raw:
                    continue

                # 实体名规范化 + 合法性检查
                head, head_valid = normalize_entity_name(head_raw)
                tail, tail_valid = normalize_entity_name(tail_raw)
                if not head_valid or not tail_valid:
                    stats["invalid_name"] += 1
                    continue

                # 实体类型 + 关系名规范化（Ontology 约束）
                head_type = ontology_registry.normalize_entity_type(head_type_raw)
                tail_type = ontology_registry.normalize_entity_type(tail_type_raw)
                relation = ontology_registry.normalize_relation(relation_raw)

                if head != head_raw or tail != tail_raw or relation != relation_raw:
                    stats["normalized"] += 1

                cleaned_triplets.append({
                    "head": head,
                    "head_type": head_type,
                    "relation": relation,
                    "tail": tail,
                    "tail_type": tail_type,
                    "_source_chunk": chunk_id,
                })
            except Exception as e:
                logger.error(f"三元组规范化失败 (triplet: {triplet}): {e}")

        # ===== 后处理 2：去重（同一对实体 + 同一关系只保留一条）=====
        cleaned_triplets = deduplicate_triplets(cleaned_triplets)

        if not cleaned_triplets:
            logger.info(f"图谱抽取完成但无合法三元组（total={stats['total']}, invalid={stats['invalid_name']}）")
            return

        # ===== 后处理 3：实体对齐（查 Neo4j 已有实体，复用节点名）=====
        existing_entities = await self._fetch_existing_entities(cleaned_triplets)

        aligned_count = 0
        for triplet in cleaned_triplets:
            head_aligned = find_existing_entity(triplet["head"], existing_entities)
            tail_aligned = find_existing_entity(triplet["tail"], existing_entities)
            if head_aligned and head_aligned != triplet["head"]:
                triplet["head"] = head_aligned
                aligned_count += 1
            if tail_aligned and tail_aligned != triplet["tail"]:
                triplet["tail"] = tail_aligned
                aligned_count += 1

        # ===== 写入 Neo4j =====
        write_count = 0
        for triplet in cleaned_triplets:
            try:
                head = triplet["head"]
                head_type = triplet["head_type"]
                tail = triplet["tail"]
                tail_type = triplet["tail_type"]
                relation = triplet["relation"]
                source_chunks = triplet.get("_source_chunks", [chunk_id] if chunk_id else [])

                await asyncio.to_thread(neo4j_client.create_entity, head_type, {"name": head})
                await asyncio.to_thread(neo4j_client.create_entity, tail_type, {"name": tail})

                rel_props = {}
                if doc_id:
                    rel_props["source_doc"] = doc_id
                # 合并多个 source_chunk（去重后的三元组可能来自多个 chunk）
                if source_chunks:
                    rel_props["source_chunks"] = source_chunks
                    rel_props["source_chunk"] = source_chunks[0]  # 保留第一个用于兼容旧查询

                await asyncio.to_thread(
                    neo4j_client.create_relationship,
                    head, head_type, tail, tail_type,
                    relation,  # 已经是规范化后的预定义关系名
                    rel_props,
                )
                write_count += 1
            except Exception as e:
                logger.error(f"图谱构建错误 (triplet: {triplet}): {e}")

        logger.info(
            f"图谱构建完成: 抽取 {stats['total']} → 合法 {len(cleaned_triplets)} → 写入 {write_count} "
            f"(规范化 {stats['normalized']}, 非法丢弃 {stats['invalid_name']}, 实体对齐 {aligned_count})"
        )
        return cleaned_triplets if write_count > 0 else []

    async def _fetch_existing_entities(self, triplets: List[Dict[str, Any]]) -> List[str]:
        """查询 Neo4j 中已有的实体名（用于实体对齐）

        策略：对每个新实体名，查询以首字符开头的已有实体（限制 200 条），避免全表扫描
        """
        if not triplets or not neo4j_client.driver:
            return []

        # 收集所有新实体名
        new_names = set()
        for t in triplets:
            if t.get("head"):
                new_names.add(t["head"])
            if t.get("tail"):
                new_names.add(t["tail"])

        if not new_names:
            return []

        existing: List[str] = []
        try:
            # 查询所有已有实体名（限制 500 条，避免大图查询过慢）
            query = "MATCH (n) WHERE n.name IS NOT NULL RETURN DISTINCT n.name AS name LIMIT 500"
            results = await asyncio.to_thread(neo4j_client.execute_query, query)
            if results:
                for r in results:
                    name = r.get("name")
                    if name and isinstance(name, str):
                        existing.append(name)
        except Exception as e:
            logger.warning(f"查询已有实体失败，跳过实体对齐: {e}")
            return []

        return existing

    def _normalize_relation(self, relation: str) -> str:
        """规范化关系名称（已迁移到 ontology_registry.normalize_relation，保留向后兼容）

        新代码请使用 ontology_registry.normalize_relation()，本方法仅作 fallback。
        """
        # 优先使用 ontology_registry
        normalized = ontology_registry.normalize_relation(relation)
        if normalized != "RELATED_TO":
            return normalized
        # fallback：原来的清洗逻辑
        clean = re.sub(r'[^\w]', '_', relation or "")
        return clean.upper() if clean else "RELATED_TO"


knowledge_extraction_service = KnowledgeExtractionService()
