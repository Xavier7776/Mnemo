"""阶段三：证据验证器 + 反思器

EvidenceVerifier：判断检索到的 chunk 是否真正与用户查询相关（LLM 驱动 + 规则 fallback）
Reflector：基于观察结果判断证据是否充分，决定下一步动作（继续检索 / 直接回答）
"""
import os
import json
import asyncio
from typing import List, Dict, Any, Optional

from utils.logger import logger


class EvidenceVerifier:
    """证据验证器：判断检索结果与查询的语义相关性。

    LLM 驱动模式：调用 LLM 对每个 chunk 做相关性判断
    规则 fallback：基于分数阈值
    """

    def __init__(self):
        self._model = None
        self._mode = (os.getenv("VERIFIER_MODE") or "rules").strip().lower()
        self._min_score = float(os.getenv("VERIFIER_MIN_SCORE", "0.15"))
        self._timeout = float(os.getenv("VERIFIER_TIMEOUT", "15"))

    @property
    def model(self) -> str:
        if self._model is None:
            model = (os.getenv("VERIFIER_MODEL") or "").strip()
            if not model:
                model = (os.getenv("LLM_MODEL") or "").strip() or "mimo-v2.5"
            self._model = model
        return self._model

    async def verify(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """验证 chunk 列表与查询的相关性。

        Args:
            query: 用户查询
            chunks: 检索到的 chunk 列表（含 chunk_id, content, score）

        Returns:
            每个chunk附加 verified 和 relevance_score 字段
        """
        if not chunks:
            return []

        if self._mode == "rules":
            return self._verify_rules(query, chunks)

        # LLM 模式
        try:
            return await asyncio.wait_for(
                self._verify_llm(query, chunks),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"EvidenceVerifier: LLM 验证超时({self._timeout}s)，fallback 到规则")
        except Exception as e:
            logger.warning(f"EvidenceVerifier: LLM 验证失败: {e}，fallback 到规则")
        return self._verify_rules(query, chunks)

    def _verify_rules(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """规则验证：基于分数阈值"""
        results = []
        for chunk in chunks:
            score = float(chunk.get("score", 0.0) or 0.0)
            verified = score >= self._min_score
            results.append({
                **chunk,
                "verified": verified,
                "relevance_score": min(score, 1.0),
            })
        return results

    async def _verify_llm(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LLM 验证：批量判断 chunk 与查询的相关性"""
        from utils.llm_client import get_async_openai_client

        # 构建 chunk 摘要（控制 token）
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            content = (chunk.get("content") or chunk.get("text") or "")[:300]
            chunk_summaries.append(f"[{i}] score={chunk.get('score', 0):.3f}\n{content}")

        prompt = f"""判断以下文档片段是否与用户查询相关。

用户查询：{query}

文档片段：
{chr(10).join(chunk_summaries)}

对每个片段输出 JSON 数组，格式：
[{{"index": 0, "relevant": true, "score": 0.9}}, ...]

只输出 JSON，不要其他内容。"""

        client = get_async_openai_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
            timeout=max(self._timeout - 1.0, 3.0),
        )

        text = response.choices[0].message.content or ""
        judgments = self._parse_verification_json(text, len(chunks))

        results = []
        for i, chunk in enumerate(chunks):
            j = judgments.get(i, {})
            results.append({
                **chunk,
                "verified": bool(j.get("relevant", True)),
                "relevance_score": float(j.get("score", chunk.get("score", 0.5))),
            })
        return results

    def _parse_verification_json(self, text: str, expected_count: int) -> Dict[int, Dict]:
        """解析 LLM 验证结果"""
        try:
            # 尝试提取 JSON 数组
            import re
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                arr = json.loads(json_match.group())
                return {item.get("index", i): item for i, item in enumerate(arr)}
        except Exception:
            pass
        # 解析失败，默认全部相关
        return {i: {"relevant": True, "score": 0.5} for i in range(expected_count)}


class Reflector:
    """反思器：基于观察结果判断证据充分性，决定下一步动作。

    LLM 驱动模式：调用 LLM 判断充分性、识别缺口、生成下一步查询
    规则 fallback：基于验证后的证据数量和分数
    """

    def __init__(self):
        self._model = None
        self._mode = (os.getenv("REFLECTOR_MODE") or "rules").strip().lower()
        self._timeout = float(os.getenv("REFLECTOR_TIMEOUT", "15"))
        self._min_verified = int(os.getenv("REFLECTOR_MIN_VERIFIED", "2"))
        self._min_top_score = float(os.getenv("REFLECTOR_MIN_TOP_SCORE", "0.3"))

    @property
    def model(self) -> str:
        if self._model is None:
            model = (os.getenv("REFLECTOR_MODEL") or "").strip()
            if not model:
                model = (os.getenv("LLM_MODEL") or "").strip() or "mimo-v2.5"
            self._model = model
        return self._model

    async def reflect(
        self,
        query: str,
        observations: List[Dict[str, Any]],
        verified_count: int,
        total_retrieval_count: int,
        max_retrievals: int,
    ) -> Dict[str, Any]:
        """反思：判断证据是否充分，决定下一步。

        Args:
            query: 原始用户查询
            observations: 历次观察列表（含 round, evidence_count, top_score, summary）
            verified_count: 经验证相关的证据总数
            total_retrieval_count: 已检索次数
            max_retrievals: 最大检索次数限制

        Returns:
            Reflection 字典（sufficient, gaps, next_action, next_query, reason, verified_count, source）
        """
        # 达到最大检索次数，强制回答
        if total_retrieval_count >= max_retrievals:
            return {
                "sufficient": True,
                "gaps": [],
                "next_action": "answer",
                "next_query": "",
                "reason": f"已达最大检索次数({max_retrievals})，基于已有证据回答",
                "verified_count": verified_count,
                "source": "rules",
            }

        if self._mode == "rules":
            return self._reflect_rules(query, observations, verified_count, total_retrieval_count)

        # LLM 模式
        try:
            return await asyncio.wait_for(
                self._reflect_llm(query, observations, verified_count, total_retrieval_count, max_retrievals),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Reflector: LLM 反思超时({self._timeout}s)，fallback 到规则")
        except Exception as e:
            logger.warning(f"Reflector: LLM 反思失败: {e}，fallback 到规则")
        return self._reflect_rules(query, observations, verified_count, total_retrieval_count)

    def _reflect_rules(
        self,
        query: str,
        observations: List[Dict[str, Any]],
        verified_count: int,
        total_retrieval_count: int,
    ) -> Dict[str, Any]:
        """规则反思：基于验证证据数量和分数"""
        # 计算累计 top_score
        top_score = max((obs.get("top_score", 0) for obs in observations), default=0.0)

        if verified_count >= self._min_verified and top_score >= self._min_top_score:
            return {
                "sufficient": True,
                "gaps": [],
                "next_action": "answer",
                "next_query": "",
                "reason": f"证据充分（verified={verified_count}, top_score={top_score:.3f}）",
                "verified_count": verified_count,
                "source": "rules",
            }

        # 证据不足
        gaps = []
        if verified_count < self._min_verified:
            gaps.append(f"验证相关证据不足（{verified_count}<{self._min_verified}）")
        if top_score < self._min_top_score:
            gaps.append(f"top_score 过低（{top_score:.3f}<{self._min_top_score}）")

        # 阶段四优化：rules 模式查询改写（基于检索轮次建议不同策略）
        if total_retrieval_count >= 2:
            next_query = f"换用 keyword 策略检索：{query}"
        else:
            next_query = f"提取核心概念重新检索：{query}"

        return {
            "sufficient": False,
            "gaps": gaps,
            "next_action": "retrieve_more",
            "next_query": next_query,
            "reason": "; ".join(gaps),
            "verified_count": verified_count,
            "source": "rules",
        }

    async def _reflect_llm(
        self,
        query: str,
        observations: List[Dict[str, Any]],
        verified_count: int,
        total_retrieval_count: int,
        max_retrievals: int,
    ) -> Dict[str, Any]:
        """LLM 反思：判断充分性、识别缺口、生成下一步查询"""
        from utils.llm_client import get_async_openai_client

        obs_text = "\n".join([
            f"  第{o.get('round', i+1)}轮: 查询='{o.get('query', '')}', 证据数={o.get('evidence_count', 0)}, top_score={o.get('top_score', 0):.3f}"
            for i, o in enumerate(observations)
        ])

        prompt = f"""你是检索反思器。基于以下检索观察，判断证据是否充分。

用户原始查询：{query}

检索观察（共{total_retrieval_count}轮）：
{obs_text}

经验证相关的证据总数：{verified_count}
剩余检索次数：{max_retrievals - total_retrieval_count}

输出 JSON：
{{
  "sufficient": false,
  "gaps": ["证据缺口1", "证据缺口2"],
  "next_action": "retrieve_more",
  "next_query": "改写后的查询（若 next_action=answer 则留空）",
  "reason": "判断理由"
}}

规则：
- sufficient=true 表示证据足以回答，next_action 设为 "answer"
- sufficient=false 且仍有检索次数时，next_action 设为 "retrieve_more" 或 "refine_query"
- next_query 应是改写后的、更有针对性的查询
只输出 JSON。"""

        client = get_async_openai_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
            timeout=max(self._timeout - 1.0, 3.0),
        )

        text = response.choices[0].message.content or ""
        result = self._parse_reflection_json(text)
        result["verified_count"] = verified_count
        result["source"] = "llm"
        return result

    def _parse_reflection_json(self, text: str) -> Dict[str, Any]:
        """解析 LLM 反思结果"""
        import re
        try:
            # 尝试提取 JSON 对象
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "sufficient": bool(data.get("sufficient", False)),
                    "gaps": data.get("gaps", []),
                    "next_action": data.get("next_action", "answer"),
                    "next_query": data.get("next_query", ""),
                    "reason": data.get("reason", ""),
                }
        except Exception:
            pass
        # 解析失败，默认回答
        return {
            "sufficient": True,
            "gaps": [],
            "next_action": "answer",
            "next_query": "",
            "reason": "LLM 反思解析失败，默认回答",
        }


# 全局单例
evidence_verifier = EvidenceVerifier()
reflector = Reflector()
