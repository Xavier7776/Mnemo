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
    规则 fallback：基于分数阈值 + query 覆盖度 + 长度惩罚的多关卡验证
    """

    def __init__(self):
        self._model = None
        self._mode = (os.getenv("VERIFIER_MODE") or "rules").strip().lower()
        self._min_score = float(os.getenv("VERIFIER_MIN_SCORE", "0.15"))
        self._timeout = float(os.getenv("VERIFIER_TIMEOUT", "15"))
        # 增强规则参数
        self._min_coverage = float(os.getenv("VERIFIER_MIN_COVERAGE", "0.3"))  # chunk 命中 query 关键词的最低比例
        self._min_chunk_len = int(os.getenv("VERIFIER_MIN_CHUNK_LEN", "50"))   # chunk 文本最低字符数
        self._boost_top_score = float(os.getenv("VERIFIER_BOOST_TOP_SCORE", "0.5"))  # 分数高于此值时放宽覆盖度
        # CrossEncoder 验证（可选，复用已加载的 reranker）
        self._cross_encoder = None  # 由 attach_cross_encoder() 注入
        self._ce_high = float(os.getenv("VERIFIER_CE_HIGH", "0.7"))   # CE 分数 >此值 → 强 verified
        self._ce_low = float(os.getenv("VERIFIER_CE_LOW", "0.1"))     # CE 分数 <此值 → 强拒绝

    def attach_cross_encoder(self, reranker) -> None:
        """注入已加载的 CrossEncoder（如 BAAI/bge-reranker-base），复用避免重复加载模型

        Agent 启动时如果 RAGRetriever 已加载 reranker，调用此方法注入，
        EvidenceVerifier 会在 rules 验证后用 CrossEncoder 做分层确认。
        如果不注入，跳过 CrossEncoder 验证，不影响原有逻辑。
        """
        self._cross_encoder = reranker
        if reranker is not None:
            logger.info("EvidenceVerifier: 已注入 CrossEncoder，启用分层验证")

    def detach_cross_encoder(self) -> None:
        """卸载 CrossEncoder（如 reranker 被禁用）"""
        self._cross_encoder = None

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

        分层验证策略（如果 CrossEncoder 可用）：
        1. 先用规则/LLM 做基础验证
        2. 如果 CrossEncoder 已注入，对验证后的 chunk 做二次分层确认：
           - CE 分数 > CE_HIGH(0.7) → 强 verified（覆盖规则判断）
           - CE 分数 < CE_LOW(0.1) → 强拒绝（覆盖规则判断）
           - 中间区间 → 保留规则/LLM 的判断

        Args:
            query: 用户查询
            chunks: 检索到的 chunk 列表（含 chunk_id, content, score）

        Returns:
            每个chunk附加 verified 和 relevance_score 字段
        """
        if not chunks:
            return []

        # 第一步：基础验证（rules 或 LLM）
        if self._mode == "rules":
            base_results = self._verify_rules(query, chunks)
        else:
            try:
                base_results = await asyncio.wait_for(
                    self._verify_llm(query, chunks),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f"EvidenceVerifier: LLM 验证超时({self._timeout}s)，fallback 到规则")
                base_results = self._verify_rules(query, chunks)
            except Exception as e:
                logger.warning(f"EvidenceVerifier: LLM 验证失败: {e}，fallback 到规则")
                base_results = self._verify_rules(query, chunks)

        # 第二步：CrossEncoder 分层确认（如果已注入）
        if self._cross_encoder is not None:
            base_results = await self._verify_with_cross_encoder(query, base_results)

        return base_results

    async def _verify_with_cross_encoder(
        self,
        query: str,
        base_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """CrossEncoder 分层确认：对规则/LLM 的验证结果做二次校准

        策略：
        - CE > CE_HIGH(0.7) → 强 verified（即使规则拒绝也覆盖为 verified）
        - CE < CE_LOW(0.1) → 强拒绝（即使规则通过也覆盖为 not verified）
        - 中间区间 → 保留基础验证的判断
        - CrossEncoder 调用失败 → 保留基础验证的判断（降级）

        这样做的价值：
        1. 高分 chunk 不用走 LLM（省延迟），CE 直接确认
        2. 低分 chunk 不用走 LLM（省延迟），CE 直接拒绝
        3. 只有中间区间才需要 LLM 精细判断
        4. CE 是专门的 query-doc 相关性模型，比规则和 LLM 都更可靠
        """
        try:
            import asyncio as _asyncio
            from utils.token_utils import truncate_to_tokens

            # 构建 pairs [query, doc_text]
            pairs = []
            max_tokens = int(os.getenv("VERIFIER_CE_MAX_TOKENS", "512"))
            for r in base_results:
                text = (r.get("content") or r.get("text") or "") or ""
                text = truncate_to_tokens(text, max_tokens)
                pairs.append([query, text])

            # 同步推理用 to_thread 包装（避免阻塞事件循环）
            ce_scores = await _asyncio.to_thread(self._cross_encoder.predict, pairs)

            # 应用分层策略
            for i, ce_score in enumerate(ce_scores):
                ce_score = float(ce_score)
                detail = base_results[i].get("verification_detail", {})
                detail["cross_encoder_score"] = ce_score

                if ce_score >= self._ce_high:
                    # 强 verified：CE 高分，覆盖规则判断
                    detail["ce_override"] = "force_verified"
                    base_results[i]["verified"] = True
                    base_results[i]["relevance_score"] = max(
                        float(base_results[i].get("relevance_score", 0.0)),
                        min(ce_score, 1.0),
                    )
                elif ce_score <= self._ce_low:
                    # 强拒绝：CE 低分，覆盖规则判断
                    detail["ce_override"] = "force_rejected"
                    base_results[i]["verified"] = False
                    base_results[i]["relevance_score"] = ce_score * 0.2
                else:
                    # 中间区间：保留基础验证的判断
                    detail["ce_override"] = "no_override"

                base_results[i]["verification_detail"] = detail

            return base_results
        except Exception as e:
            logger.warning(f"EvidenceVerifier: CrossEncoder 验证失败，保留基础验证结果: {e}")
            return base_results

    def _verify_rules(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """增强规则验证：4 道关卡

        关卡 1：分数阈值（VERIFIER_MIN_SCORE，默认 0.15）
            检索分太低的直接砍，不配当证据
        关卡 2：query 覆盖度（VERIFIER_MIN_COVERAGE，默认 0.3）
            chunk 必须命中 query 至少 30% 的关键词，避免只命中 1 个高频词就过线
            高分 chunk（>= VERIFIER_BOOST_TOP_SCORE）放宽覆盖度要求，因为向量检索的语义匹配可能不体现在字面重叠
        关卡 3：长度惩罚（VERIFIER_MIN_CHUNK_LEN，默认 50）
            过短的 chunk（纯标题、导航文本）降权，即使分数高也不可信
        关卡 4：relevance_score 综合
            最终相关性分数 = 检索分 × 覆盖度加权，让高覆盖度的 chunk 排名更靠前
        """
        from utils.tokenizer import tokenize_for_query

        # 预处理 query tokens（去停用词）
        query_tokens = set(tokenize_for_query(query))
        # query 很短时（<3 个 token）不要求覆盖度，避免误杀
        require_coverage = len(query_tokens) >= 3

        results = []
        for chunk in chunks:
            score = float(chunk.get("score", 0.0) or 0.0)
            content = (chunk.get("content") or chunk.get("text") or "") or ""

            # 关卡 1：分数阈值
            pass_score = score >= self._min_score

            # 关卡 2：query 覆盖度
            coverage = 0.0
            pass_coverage = True  # 默认通过（query 太短时不检查）
            if require_coverage and content:
                chunk_tokens = set(tokenize_for_query(content))
                if query_tokens:
                    hit = len(query_tokens & chunk_tokens)
                    coverage = hit / len(query_tokens)
                    # 高分 chunk 放宽覆盖度要求（向量检索语义匹配可能字面不重叠）
                    effective_min_cov = self._min_coverage * 0.5 if score >= self._boost_top_score else self._min_coverage
                    pass_coverage = coverage >= effective_min_cov

            # 关卡 3：长度惩罚
            pass_length = len(content) >= self._min_chunk_len

            # 综合判定：三关全过才 verified
            verified = pass_score and pass_coverage and pass_length

            # 关卡 4：relevance_score 综合计算
            # 高分 + 高覆盖度 → relevance_score 更高
            # 低覆盖度但高分 → 适当降权（避免高分但语义偏题的 chunk 排第一）
            if verified:
                relevance_score = score * (0.5 + 0.5 * coverage) if require_coverage else min(score, 1.0)
                relevance_score = min(relevance_score, 1.0)
            else:
                relevance_score = score * 0.3  # 未验证通过的 chunk 大幅降权

            # 记录各关卡结果（便于调试和日志）
            fail_reasons = []
            if not pass_score:
                fail_reasons.append(f"score={score:.3f}<{self._min_score}")
            if not pass_coverage:
                fail_reasons.append(f"coverage={coverage:.2f}<{self._min_coverage}")
            if not pass_length:
                fail_reasons.append(f"len={len(content)}<{self._min_chunk_len}")

            results.append({
                **chunk,
                "verified": verified,
                "relevance_score": relevance_score,
                "verification_detail": {
                    "score": score,
                    "coverage": round(coverage, 3),
                    "length": len(content),
                    "fail_reasons": fail_reasons,
                },
            })
        return results

    async def _verify_llm(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LLM 验证：批量判断 chunk 与查询的相关性（增强防瞎猜）

        防瞎猜三道保险：
        1. 要求输出 reason（相关性理由）+ confidence（置信度 0-1）
        2. confidence < VERIFIER_LLM_MIN_CONFIDENCE(0.5) → 不算 verified
        3. reason 为空或过短（<5 字）→ 降级到规则验证
        """
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
[{{"index": 0, "relevant": true, "confidence": 0.9, "reason": "简短说明为什么相关/不相关"}}, ...]

要求：
1. relevant: bool，是否与查询语义相关
2. confidence: 0.0-1.0，你的判断置信度（不确定时低于 0.5）
3. reason: 不少于 5 个字的相关性理由（例如"讨论了微服务性能对比"或"只是介绍架构概念，未涉及性能"）

只输出 JSON 数组，不要其他内容。"""

        client = get_async_openai_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500,
            timeout=max(self._timeout - 1.0, 3.0),
        )

        text = response.choices[0].message.content or ""
        judgments = self._parse_verification_json(text, len(chunks))

        min_confidence = float(os.getenv("VERIFIER_LLM_MIN_CONFIDENCE", "0.5"))
        min_reason_len = int(os.getenv("VERIFIER_LLM_MIN_REASON_LEN", "5"))

        results = []
        for i, chunk in enumerate(chunks):
            j = judgments.get(i, {})
            relevant = bool(j.get("relevant", False))
            confidence = float(j.get("confidence", 0.0))
            reason = str(j.get("reason", "")).strip()

            # 防瞎猜检查 1：confidence 低于阈值 → 不算 verified
            pass_confidence = confidence >= min_confidence

            # 防瞎猜检查 2：reason 为空或过短 → 降级到规则验证
            pass_reason = len(reason) >= min_reason_len

            if not pass_reason:
                # LLM 没给出有效理由，降级到规则验证这个 chunk
                logger.warning(
                    f"EvidenceVerifier: LLM reason 为空/过短 (chunk {i})，降级到规则验证"
                )
                rule_result = self._verify_rules(query, [chunk])[0]
                results.append(rule_result)
                continue

            # 通过防瞎猜检查：relevant + confidence + reason 都有效
            verified = relevant and pass_confidence
            # relevance_score 综合 confidence 和原检索分
            relevance_score = (confidence + float(chunk.get("score", 0.0))) / 2 if verified else confidence * 0.3

            results.append({
                **chunk,
                "verified": verified,
                "relevance_score": min(relevance_score, 1.0),
                "verification_detail": {
                    "source": "llm",
                    "relevant": relevant,
                    "confidence": confidence,
                    "reason": reason,
                    "pass_confidence": pass_confidence,
                    "pass_reason": pass_reason,
                },
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
        # 解析失败，默认全部相关（保守策略：宁过松不错杀，后续 confidence 会兜底）
        return {i: {"relevant": True, "confidence": 0.3, "reason": "解析失败"} for i in range(expected_count)}


class Reflector:
    """反思器：基于观察结果判断证据充分性，决定下一步动作。

    LLM 驱动模式：调用 LLM 判断充分性、识别缺口、生成下一步查询
    规则 fallback：基于验证后的证据数量、分数、梯度、覆盖率趋势的多维度判断
    """

    def __init__(self):
        self._model = None
        self._mode = (os.getenv("REFLECTOR_MODE") or "rules").strip().lower()
        self._timeout = float(os.getenv("REFLECTOR_TIMEOUT", "15"))
        self._min_verified = int(os.getenv("REFLECTOR_MIN_VERIFIED", "2"))
        self._min_top_score = float(os.getenv("REFLECTOR_MIN_TOP_SCORE", "0.3"))
        # 增强规则参数
        self._min_score_gap = float(os.getenv("REFLECTOR_MIN_SCORE_GAP", "0.1"))  # top1-top3 分数差距低于此值说明区分度差
        self._min_coverage = float(os.getenv("REFLECTOR_MIN_COVERAGE", "0.5"))    # 验证后 chunk 覆盖 query 关键词的最低比例
        self._stagnant_rounds = int(os.getenv("REFLECTOR_STAGNANT_ROUNDS", "2"))  # verified_count 连续 N 轮不增长则判定策略无效

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
        verified_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """反思：判断证据是否充分，决定下一步。

        Args:
            query: 原始用户查询
            observations: 历次观察列表（含 round, evidence_count, top_score, summary）
            verified_count: 经验证相关的证据总数
            total_retrieval_count: 已检索次数
            max_retrievals: 最大检索次数限制
            verified_chunks: 验证后的 chunk 列表（可选，用于覆盖度分析）

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
            return self._reflect_rules(query, observations, verified_count, total_retrieval_count, verified_chunks)

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
        return self._reflect_rules(query, observations, verified_count, total_retrieval_count, verified_chunks)

    def _reflect_rules(
        self,
        query: str,
        observations: List[Dict[str, Any]],
        verified_count: int,
        total_retrieval_count: int,
        verified_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """增强规则反思：多维度判断证据充分性

        维度 1：数量+分数（原有）
            verified_count >= MIN_VERIFIED(2) 且 top_score >= MIN_TOP_SCORE(0.3)
        维度 2：分数梯度（新增）
            top1 vs top3 的分数差距，差距太小（<0.1）说明检索区分度差，证据可能都不可靠
        维度 3：query 覆盖度（新增）
            验证后的 chunk 合并起来是否覆盖了 query 至少 50% 的关键词
            覆盖度低说明检索到的证据不全面，即使分数够高也可能答非所问
        维度 4：检索停滞检测（新增）
            连续 N 轮 verified_count 不增长，说明当前检索策略无效，应换策略
        """
        from utils.tokenizer import tokenize_for_query

        # 维度 1：数量 + 分数
        top_score = max((obs.get("top_score", 0) for obs in observations), default=0.0)
        pass_count_score = verified_count >= self._min_verified and top_score >= self._min_top_score

        # 维度 2：分数梯度（top1 - top3）
        all_scores = sorted(
            [float(c.get("score", 0.0) or 0.0) for c in (verified_chunks or [])],
            reverse=True
        )
        score_gap = 0.0
        if len(all_scores) >= 3:
            score_gap = all_scores[0] - all_scores[2]
        elif len(all_scores) >= 2:
            score_gap = all_scores[0] - all_scores[1]
        # 分数梯度低不直接否决，但作为 gaps 信号
        low_distinction = len(all_scores) >= 2 and score_gap < self._min_score_gap

        # 维度 3：query 覆盖度
        query_tokens = set(tokenize_for_query(query))
        coverage = 0.0
        if query_tokens and verified_chunks:
            chunk_text = " ".join(
                (c.get("content") or c.get("text") or "") for c in verified_chunks if c.get("verified")
            )
            if chunk_text:
                chunk_tokens = set(tokenize_for_query(chunk_text))
                hit = len(query_tokens & chunk_tokens)
                coverage = hit / len(query_tokens)
        low_coverage = len(query_tokens) >= 3 and coverage < self._min_coverage

        # 维度 4：检索停滞检测
        stagnant = False
        if len(observations) >= self._stagnant_rounds + 1:
            # 检查最近 N 轮的 verified_count 是否都没增长
            recent = observations[-(self._stagnant_rounds + 1):]
            counts = [obs.get("verified_count", 0) for obs in recent]
            stagnant = len(set(counts)) == 1 and counts[0] > 0  # 数值完全相同且非 0

        # 综合判定
        # 充分条件：数量+分数达标，且覆盖度达标，且未停滞
        # 如果只是数量+分数达标但覆盖度低，仍然判定不足（证据可能答非所问）
        sufficient = pass_count_score and not low_coverage

        if sufficient:
            reason_parts = [f"verified={verified_count}", f"top_score={top_score:.3f}"]
            if query_tokens:
                reason_parts.append(f"coverage={coverage:.2f}")
            return {
                "sufficient": True,
                "gaps": [],
                "next_action": "answer",
                "next_query": "",
                "reason": f"证据充分（{', '.join(reason_parts)}）",
                "verified_count": verified_count,
                "source": "rules",
                "detail": {
                    "top_score": round(top_score, 3),
                    "coverage": round(coverage, 3),
                    "score_gap": round(score_gap, 3),
                    "stagnant": stagnant,
                },
            }

        # 证据不足：收集 gaps
        gaps = []
        if verified_count < self._min_verified:
            gaps.append(f"验证相关证据不足（{verified_count}<{self._min_verified}）")
        if top_score < self._min_top_score:
            gaps.append(f"top_score 过低（{top_score:.3f}<{self._min_top_score}）")
        if low_coverage:
            gaps.append(f"query 覆盖度低（{coverage:.2f}<{self._min_coverage}），证据可能答非所问")
        if low_distinction:
            gaps.append(f"分数梯度低（gap={score_gap:.3f}<{self._min_score_gap}），检索区分度差")
        if stagnant:
            gaps.append(f"最近 {self._stagnant_rounds} 轮 verified_count 未增长，检索策略可能无效")

        # 策略改写：基于具体不足给出针对性建议
        if stagnant:
            # 停滞：换检索策略
            next_query = f"换用 keyword 策略或换关键词重试：{query}"
        elif low_coverage:
            # 覆盖度低：提示 LLM 补充未覆盖的关键词
            missed = query_tokens - set(tokenize_for_query(
                " ".join((c.get("content") or c.get("text") or "") for c in (verified_chunks or []) if c.get("verified"))
            )) if query_tokens else set()
            if missed:
                next_query = f"补充以下关键词重新检索：{' '.join(list(missed)[:5])}（原始 query：{query}）"
            else:
                next_query = f"换用关键词重新检索：{query}"
        elif total_retrieval_count >= 2:
            next_query = f"换用 keyword 策略检索：{query}"
        else:
            next_query = f"提取核心概念重新检索：{query}"

        return {
            "sufficient": False,
            "gaps": gaps,
            "next_action": "retrieve_more",
            "next_query": next_query,
            "reason": "; ".join(gaps) if gaps else "证据不足",
            "verified_count": verified_count,
            "source": "rules",
            "detail": {
                "top_score": round(top_score, 3),
                "coverage": round(coverage, 3),
                "score_gap": round(score_gap, 3),
                "stagnant": stagnant,
            },
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
