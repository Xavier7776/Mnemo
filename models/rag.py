"""Structured models for RAG evidence and agent orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """Chunk-level evidence passed through retrieval, generation, and agents."""

    id: str
    text: str
    document_id: Optional[str] = None
    file_id: Optional[str] = None
    conversation_id: Optional[str] = None
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    document_title: Optional[str] = None
    section_path: List[str] = Field(default_factory=list)
    page: Optional[int] = None
    score: float = 0.0
    retrieval_type: str = "vector"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # —— 阶段三新增 ——
    retrieved_at_round: Optional[int] = None
    """该证据在哪一轮工具调用中被检索到"""
    verified: bool = False
    """EvidenceVerifier 是否判定该证据与查询相关"""
    relevance_score: Optional[float] = None
    """验证器给出的语义相关性分数（0~1）"""


class QueryPlan(BaseModel):
    """Internal retrieval plan derived from a user query."""

    intent: str = "general"
    need_rewrite: bool = False
    need_graph: bool = True
    prefetch_k: int = 200
    final_k: int = 12
    context_budget: int = 30_000
    filters: Dict[str, Any] = Field(default_factory=dict)
    rewritten_queries: List[str] = Field(default_factory=list)
    fusion_strategy: str = "rrf"
    # —— 阶段二新增 ——
    sub_queries: List[str] = Field(default_factory=list)
    """查询分解的子查询列表。空列表表示不需要分解（走单查询流程）。
    非空时，RetrievalContextManager 会对每个子查询独立检索并合并证据。"""
    planner_source: str = "rules"
    """规划来源：'llm'（LLM 驱动）或 'rules'（规则引擎 fallback）"""


class Observation(BaseModel):
    """阶段三：工具调用后结构化观察结果。"""

    round: int
    """第几轮工具调用"""
    tool: str
    """工具名称"""
    query: str = ""
    """检索查询（仅 rag_retrieve）"""
    evidence_ids: List[str] = Field(default_factory=list)
    """本轮检索到的 chunk_id 列表"""
    evidence_count: int = 0
    """本轮检索到的证据数量"""
    top_score: float = 0.0
    """本轮最高分"""
    summary: str = ""
    """观察摘要（供反思器参考）"""


class Reflection(BaseModel):
    """阶段三：反思结果，驱动 Plan-Act-Observe-Reflect 循环。"""

    sufficient: bool = False
    """证据是否充分，可以给出最终回答"""
    gaps: List[str] = Field(default_factory=list)
    """识别到的证据缺口"""
    next_action: str = "answer"
    """下一步动作：'answer'（直接回答）/ 'retrieve_more'（继续检索）/ 'refine_query'（改写查询后检索）"""
    next_query: str = ""
    """下一步检索的查询（next_action=retrieve_more/refine_query 时有效）"""
    reason: str = ""
    """反思理由"""
    verified_count: int = 0
    """经验证相关的证据数量"""
    source: str = "rules"
    """反思来源：'llm' 或 'rules'"""


class AgentPlan(BaseModel):
    """Validated plan returned by the coordinator."""

    selected_agents: List[str] = Field(default_factory=list)
    agent_tasks: Dict[str, str] = Field(default_factory=dict)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    parallel_groups: List[List[str]] = Field(default_factory=list)
    reasoning: str = ""


class AgentResultModel(BaseModel):
    """Structured result emitted by expert agents."""

    agent_type: str
    content: str = ""
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    open_questions: List[str] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    error: bool = False
