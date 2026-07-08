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


class KnowledgeExtractionService:
    """知识抽取与图谱构建服务"""
    
    def __init__(self):
        # 不在此处创建 OpenAI 客户端——main.py 加载 .env 后才可用
        self._client = None
        self._model_name = None
        self._neo4j_disabled_until_ts = 0.0
        self.extraction_prompt_template = """
你是一个知识图谱专家。请从以下文本中提取"实体-关系-实体"三元组。
请严格按照 JSON 格式返回结果，不要包含任何其他解释性文字。
返回格式示例：
[
  {{ "head": "实体1", "head_type": "类型1", "relation": "关系", "tail": "实体2", "tail_type": "类型2" }},
  ...
]

实体类型可以是：Concept(概念), Technology(技术), Person(人物), Organization(组织), Location(地点), Event(事件), Other(其他)。
关系应当简洁明了。

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
                max_tokens=2000,
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

        if parsed is None:
            logger.warning(f"无法解析 JSON: {content[:100]}...")
            return []

        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        logger.warning(f"JSON 解析结果不是列表或字典: {type(parsed)}")
        return []

    async def extract_entities(self, query: str) -> List[str]:
        """从查询中提取实体（用 asyncio.to_thread 包装同步调用，避免阻塞事件循环）"""
        prompt = f"""请从以下查询中提取关键实体（人名、地名、组织、概念、技术术语等）。
只返回实体列表，JSON 格式：["实体1", "实体2"]。
不要包含任何解释。

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
                    max_tokens=500,
                    timeout=30
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
        """构建知识图谱：抽取并存入 Neo4j"""
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

        doc_id = metadata.get("document_id") if metadata else None
        chunk_id = metadata.get("chunk_id") if metadata else None

        for triplet in triplets:
            try:
                head = triplet.get("head")
                head_type = triplet.get("head_type", "Concept")
                tail = triplet.get("tail")
                tail_type = triplet.get("tail_type", "Concept")
                relation = triplet.get("relation")

                if not head or not tail or not relation:
                    continue

                await asyncio.to_thread(neo4j_client.create_entity, head_type, {"name": head})
                await asyncio.to_thread(neo4j_client.create_entity, tail_type, {"name": tail})

                rel_props = {}
                if doc_id:
                    rel_props["source_doc"] = doc_id
                if chunk_id:
                    rel_props["source_chunk"] = chunk_id
                
                await asyncio.to_thread(
                    neo4j_client.create_relationship,
                    head, head_type, tail, tail_type,
                    self._normalize_relation(relation),
                    rel_props,
                )
            except Exception as e:
                logger.error(f"图谱构建错误 (triplet: {triplet}): {e}")

    def _normalize_relation(self, relation: str) -> str:
        """规范化关系名称"""
        clean = re.sub(r'[^\w]', '_', relation)
        return clean.upper()


knowledge_extraction_service = KnowledgeExtractionService()
