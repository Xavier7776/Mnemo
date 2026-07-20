"""Query planner — LLM 驱动 + 规则 fallback（阶段二改造）

模式控制：环境变量 QUERY_PLANNER_MODE
  - "llm"（默认）：先尝试 LLM 驱动，超时/失败时 fallback 到规则引擎
  - "rules"：纯规则引擎（回滚用，行为与改造前完全一致）

LLM 驱动新增能力：
  - 语义级意图识别（替代关键词 if-else）
  - 查询分解（sub_queries）：复杂问题拆分为独立子查询
  - 更精准的查询改写（rewritten_queries）
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

from models.rag import QueryPlan
from utils.logger import logger


def _truthy(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class QueryPlanner:
    """检索规划器 — LLM 驱动 + 规则 fallback"""

    def __init__(self):
        self._planner_model: Optional[str] = None

    @property
    def planner_model(self) -> str:
        if self._planner_model is None:
            # 处理 PLANNER_MODEL 为空字符串的情况（os.getenv 对空串不返回默认值）
            model = (os.getenv("PLANNER_MODEL") or "").strip()
            if not model:
                model = (os.getenv("LLM_MODEL") or "").strip() or "mimo-v2.5"
            self._planner_model = model
        return self._planner_model

    @property
    def mode(self) -> str:
        """规划模式：'llm' 或 'rules'"""
        return os.getenv("QUERY_PLANNER_MODE", "llm").lower()

    async def build_plan(
        self,
        query: str,
        runtime_modules: Optional[Dict[str, Any]] = None,
        runtime_params: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        planner_mode: str = "auto",
    ) -> QueryPlan:
        """构建检索计划（async，因为 LLM 调用是异步的）

        Args:
            planner_mode: 规划模式，三选一
                - "auto"（默认）：智能选择。短查询/明确意图用规则引擎（毫秒级），
                  长查询/复杂结构用 LLM（更准但慢）。兼顾速度和准确性。
                - "rules"：强制规则引擎，毫秒级，不调 LLM。
                - "llm"：强制 LLM 判断，最准但每次增加 ~1-3s 延迟。
                - "env"：按 QUERY_PLANNER_MODE 环境变量走（兼容旧用法）。
        """
        mode = planner_mode.lower()

        # auto 模式：根据 query 复杂度智能选择
        if mode == "auto":
            if self._query_needs_llm(query):
                logger.debug(f"QueryPlanner[auto]: query 复杂，用 LLM 规划 (len={len(query)})")
                # 走 LLM 路径（下方处理）
                mode = "llm"
            else:
                logger.debug(f"QueryPlanner[auto]: query 简单，用规则引擎 (len={len(query)})")
                plan = self._build_plan_rules(query, runtime_modules, runtime_params, filters)
                return plan

        # rules 模式：强制规则引擎
        if mode == "rules":
            return self._build_plan_rules(query, runtime_modules, runtime_params, filters)

        # env 模式：按 QUERY_PLANNER_MODE 环境变量走（兼容 retrieval.py 旧用法）
        if mode == "env":
            mode = self.mode

        # LLM 模式：先尝试 LLM，失败 fallback 到规则引擎
        if mode == "llm":
            try:
                plan = await asyncio.wait_for(
                    self._build_plan_llm(query, runtime_modules, runtime_params, filters),
                    timeout=float(os.getenv("PLANNER_TIMEOUT", "20.0")),
                )
                logger.info(f"QueryPlanner[llm]: LLM 规划完成 (intent={plan.intent}, sub_queries={len(plan.sub_queries)}, rewritten={len(plan.rewritten_queries)})")
                if plan.rewritten_queries:
                    for i, rq in enumerate(plan.rewritten_queries, 1):
                        logger.info(f"QueryPlanner: rewrite[{i}] = {rq}")
                if plan.sub_queries:
                    for i, sq in enumerate(plan.sub_queries, 1):
                        logger.info(f"QueryPlanner: sub_query[{i}] = {sq}")
                return plan
            except asyncio.TimeoutError:
                logger.warning(f"QueryPlanner: LLM 规划超时({os.getenv('PLANNER_TIMEOUT', '20.0')}s)，fallback 到规则引擎")
            except Exception as e:
                logger.warning(f"QueryPlanner: LLM 规划失败: {e}，fallback 到规则引擎")

        return self._build_plan_rules(query, runtime_modules, runtime_params, filters)

    def _query_needs_llm(self, query: str) -> bool:
        """判断 query 是否需要 LLM 规划（auto 模式的决策逻辑）

        规则引擎足够的情况（返回 False）：
        - 命中明确意图关键词（"对比/比较/有哪些/条款/定义"等）
        - 极短查询（<=15 字）且单句：即使误判为 general 影响也小

        需要 LLM 的情况（返回 True）：
        - 长查询（>40 字）：可能包含多重意图，规则引擎关键词匹配会漏
        - 多句查询（含多个问号/分号）：可能是复合问题
        - 中等长度（16-40 字）且无明确关键词：规则引擎会误判为 general
        """
        q = (query or "").strip()
        if not q:
            return False

        # 明确意图关键词命中：规则引擎足够准确
        explicit_keywords = (
            "对比", "比较", "差异", "优缺点", "优劣", "分别", "各自", "区别", "相同点", "不同点",
            "有哪些", "列举", "总结", "概括", "要点", "关键点", "核心观点",
            "条款", "规定", "标准", "定义", "范围", "假设", "条件",
            "风险", "限制", "不足", "漏洞",
        )
        has_explicit = any(k in q for k in explicit_keywords)

        # 命中明确关键词：规则引擎足够（无论长度）
        if has_explicit:
            return False

        # 极短查询（<=15 字）+ 单句：即使误判为 general 影响也小，规则引擎够用
        if len(q) <= 15 and q.count("？") + q.count("?") <= 1:
            return False

        # 长查询（>40 字）：可能多重意图，用 LLM
        if len(q) > 40:
            return True

        # 多句查询：复合问题，用 LLM
        if q.count("？") + q.count("?") >= 2 or q.count("；") + q.count(";") >= 2:
            return True

        # 中等长度（16-40 字）且无明确关键词：规则引擎会误判为 general，用 LLM
        return True

    # ==================== LLM 驱动 ====================

    async def _build_plan_llm(
        self,
        query: str,
        runtime_modules: Optional[Dict[str, Any]] = None,
        runtime_params: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> QueryPlan:
        """用 LLM 分析查询，生成结构化检索计划"""
        from utils.llm_client import get_async_openai_client

        runtime_modules = runtime_modules or {}
        runtime_params = runtime_params or {}

        prompt = self._build_llm_prompt(query)

        # API 请求超时：略小于 PLANNER_TIMEOUT，留时间给 asyncio.wait_for 捕获
        planner_timeout = float(os.getenv("PLANNER_TIMEOUT", "20.0"))
        api_timeout = max(planner_timeout - 1.0, 3.0)

        client = get_async_openai_client()
        response = await client.chat.completions.create(
            model=self.planner_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
            timeout=api_timeout,
        )
        text = response.choices[0].message.content or ""

        # 解析 JSON（兼容 markdown 代码块包裹）
        plan_data = self._parse_llm_json(text)
        if not plan_data:
            raise ValueError(f"LLM 返回无法解析为 JSON: {text[:200]}")

        # 构造 QueryPlan，对 LLM 输出做安全约束
        intent = plan_data.get("intent", "general")
        if intent not in ("general", "compare", "summary", "clause", "verification"):
            intent = "general"

        final_k = self._clamp(plan_data.get("final_k", 12), 5, 30)
        prefetch_k = self._clamp(plan_data.get("prefetch_k", 200), 50, 500)

        sub_queries = [q.strip() for q in plan_data.get("sub_queries", []) if q and q.strip()][:5]
        rewritten = [q.strip() for q in plan_data.get("rewritten_queries", []) if q and q.strip()][:4]
        if not rewritten:
            rewritten = [query]

        need_graph = _truthy(runtime_modules.get("kg_retrieve_enabled"), True)
        if "need_graph" in plan_data:
            need_graph = bool(plan_data["need_graph"]) and need_graph

        return QueryPlan(
            intent=intent,
            need_rewrite=len(rewritten) > 1,
            need_graph=need_graph,
            prefetch_k=prefetch_k,
            final_k=final_k,
            context_budget=int(runtime_params.get("context_budget") or os.getenv("RAG_CONTEXT_BUDGET", "30000")),
            filters=filters or {},
            rewritten_queries=rewritten,
            fusion_strategy=str(runtime_params.get("retrieval_fusion_strategy") or os.getenv("RETRIEVAL_FUSION_STRATEGY", "rrf")),
            sub_queries=sub_queries,
            planner_source="llm",
        )

    def _build_llm_prompt(self, query: str) -> str:
        """构建 LLM 规划提示词"""
        return f"""你是一个检索规划器。分析用户查询，输出结构化检索计划。

## 任务
1. 识别查询意图
2. 判断是否需要查询分解（复杂问题拆成独立子查询）
3. 生成查询改写变体（同一问题的不同表述，用于多路召回）
4. 决定检索参数

## 意图类型
- general: 通用事实查询
- compare: 对比类（对比A和B的差异/优缺点）
- summary: 总结/列举类（总结要点/列举清单）
- clause: 条款/定义类（依据/条款/定义/范围）
- verification: 验证/风险类（风险/限制/证据）

## 查询分解规则
- 简单查询（单一信息需求）：sub_queries 返回空数组 []
- 复杂查询（多个独立信息需求）：拆成 2-4 个子查询
  - 例："对比HNSW和IVF的性能差异和应用场景"
    → ["HNSW索引的性能特点", "IVF索引的性能特点", "HNSW和IVF的应用场景对比"]
  - 例："RAG系统的核心组件和评估指标是什么"
    → ["RAG系统的核心组件", "RAG系统的评估指标"]
- 每个子查询应聚焦于一个独立的信息需求

## 查询改写规则
- 生成 1-3 个改写变体（包括原始查询）
- 改写是同义替换，不是分解
- 用于多路召回提高覆盖率

## 用户查询
{query}

## 输出格式（严格 JSON，不要 markdown 代码块）
{{
  "intent": "general|compare|summary|clause|verification",
  "sub_queries": ["子查询1", "子查询2"],
  "rewritten_queries": ["原始查询", "改写变体1"],
  "final_k": 12,
  "prefetch_k": 200,
  "need_graph": true
}}"""

    def _parse_llm_json(self, text: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 返回的 JSON（兼容 markdown 代码块包裹）"""
        # 尝试直接解析
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # 尝试从 ```json ... ``` 中提取
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试从裸 { ... } 中提取
        m = re.search(r'\{[^{}]*"intent"[^{}]*\}', text, re.DOTALL)
        if m:
            # 找到最外层的大括号
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass

        return None

    @staticmethod
    def _clamp(value, lo, hi):
        try:
            v = int(value)
        except (TypeError, ValueError):
            v = lo
        return max(lo, min(hi, v))

    # ==================== 规则引擎 fallback ====================

    def _build_plan_rules(
        self,
        query: str,
        runtime_modules: Optional[Dict[str, Any]] = None,
        runtime_params: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> QueryPlan:
        """规则引擎规划（改造前的原始逻辑，作为 fallback）"""
        q = (query or "").strip()
        runtime_modules = runtime_modules or {}
        runtime_params = runtime_params or {}

        is_compare = any(k in q for k in ("对比", "比较", "差异", "优缺点", "优劣", "分别", "各自", "相同点", "不同点"))
        is_list = any(k in q for k in ("有哪些", "列举", "总结", "概括", "要点", "关键点", "核心观点", "主要结论"))
        is_clause = any(k in q for k in ("依据", "条款", "规定", "标准", "口径", "定义", "范围", "假设", "条件"))
        is_risk = any(k in q for k in ("风险", "限制", "不足", "漏洞", "反例", "校验", "证据"))

        final_k = 12
        prefetch_k = 200
        intent = "general"
        if len(q) > 80 or is_compare or is_list:
            final_k = 20
            intent = "compare" if is_compare else "summary"
        if is_clause:
            prefetch_k = 260
            final_k = max(final_k, 16)
            intent = "clause"
        if is_risk and intent == "general":
            intent = "verification"

        rewrite_enabled = _truthy(runtime_modules.get("query_rewrite_enabled"), True)
        need_rewrite = rewrite_enabled and (len(q) > 80 or is_compare or is_list or is_clause)
        rewritten_queries = self._rewrite_queries(q, intent) if need_rewrite else [q]

        return QueryPlan(
            intent=intent,
            need_rewrite=need_rewrite,
            need_graph=_truthy(runtime_modules.get("kg_retrieve_enabled"), True),
            prefetch_k=prefetch_k,
            final_k=final_k,
            context_budget=int(runtime_params.get("context_budget") or os.getenv("RAG_CONTEXT_BUDGET", "30000")),
            filters=filters or {},
            rewritten_queries=rewritten_queries,
            fusion_strategy=str(runtime_params.get("retrieval_fusion_strategy") or os.getenv("RETRIEVAL_FUSION_STRATEGY", "rrf")),
            sub_queries=[],  # 规则引擎不做查询分解
            planner_source="rules",
        )

    def _rewrite_queries(self, query: str, intent: str) -> List[str]:
        """规则引擎的查询改写（原逻辑保持不变）"""
        variants = [query]
        if intent == "compare":
            variants.append(f"{query} 对比 差异 优缺点")
            variants.append(f"{query} 共同点 不同点 依据")
        elif intent == "clause":
            variants.append(f"{query} 定义 范围 条件 例外")
            variants.append(f"{query} 条款 规定 依据")
        elif intent == "summary":
            variants.append(f"{query} 要点 结论 证据")
            variants.append(f"{query} 核心观点 关键发现")
        elif intent == "verification":
            variants.append(f"{query} 证据 风险 限制")
        else:
            variants.append(f"{query} 相关证据")

        deduped: List[str] = []
        seen = set()
        for item in variants:
            normalized = " ".join(item.split())
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(normalized)
        return deduped[:3]


query_planner = QueryPlanner()
