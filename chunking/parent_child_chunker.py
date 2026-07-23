"""父子分块器（Small-to-Big / Parent-Child Chunking）

原理：
- 入库时切两层 —— 大块（parent）+ 小块（child）
- child 做向量检索（粒度小，命中精准）
- 命中 child 后回查 parent，返回 parent 文本给 LLM（上下文完整）

适用场景：长文档（PDF/Markdown/报告），小块精准命中 + 大块完整上下文
不适用：HotpotQA/T2Retrieval 等已是完整段落的评测数据集
"""
from typing import List, Dict, Any, Optional
from .base import BaseChunker
from .simple_chunker import SimpleChunker


class ParentChildChunker(BaseChunker):
    """父子分块器：parent 大块存原文，child 小块做 embedding"""

    def __init__(
        self,
        parent_size: int = 1500,
        parent_overlap: int = 200,
        child_size: int = 400,
        child_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        """
        Args:
            parent_size: parent 块最大字符数（存原文，不向量化）
            parent_overlap: parent 块之间重叠字符数
            child_size: child 块最大字符数（做向量化，检索粒度）
            child_overlap: child 块之间重叠字符数
            separators: 分隔符优先级列表

        默认参数适配长文档（论文全文/法律文档/技术报告，>=3000 字符）：
        - parent 1500 字符：大块保留完整上下文
        - child 400 字符：小块做精准向量化，避免长文档语义稀释
        """
        self.parent_chunker = SimpleChunker(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            separators=separators,
        )
        self.child_chunker = SimpleChunker(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=separators,
        )

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        切两层：parent + child

        Returns:
            chunk 列表，每个 chunk 字典包含：
            - text: 文本内容
            - metadata: 含 chunk_level ("parent"/"child")、parent_index、child_index
              parent chunk 额外含 child_indices (List[int])
              child chunk 额外含 parent_index (int)
            - chunk_index: 全局顺序索引（parent 先编号，child 后编号）
            - start_index / end_index: 在原文中的字符偏移
        """
        if not text.strip():
            return []

        base_meta = metadata or {}
        chunks: List[Dict[str, Any]] = []

        # 第 1 层：切 parent
        parent_chunks = self.parent_chunker.chunk(text, metadata=base_meta)
        if not parent_chunks:
            return chunks

        # 第 2 层：每个 parent 内部切 child
        for parent_idx, parent in enumerate(parent_chunks):
            parent_text = parent["text"]
            parent_chunk_index = len(chunks)  # parent 在全局列表中的索引

            # 切 child
            child_chunks = self.child_chunker.chunk(parent_text, metadata=base_meta)
            child_indices: List[int] = []

            for child_idx_in_parent, child in enumerate(child_chunks):
                child_global_index = len(chunks) + len(child_indices) + 1  # 占位，后面修正
                child_meta = base_meta.copy()
                child_meta.update({
                    "chunk_level": "child",
                    "parent_index": parent_chunk_index,
                    "child_index_in_parent": child_idx_in_parent,
                })
                chunks.append({
                    "text": child["text"],
                    "start_index": child["start_index"],
                    "end_index": child["end_index"],
                    "metadata": child_meta,
                })
                child_indices.append(len(chunks) - 1)

            # parent chunk：存原文，记录它的 child 们
            parent_meta = base_meta.copy()
            parent_meta.update({
                "chunk_level": "parent",
                "parent_index": parent_idx,
                "child_indices": child_indices,
            })
            chunks.append({
                "text": parent_text,
                "start_index": parent["start_index"],
                "end_index": parent["end_index"],
                "metadata": parent_meta,
            })

        return chunks

    def get_child_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从分块结果中筛出 child chunk（用于向量化）"""
        return [c for c in chunks if c.get("metadata", {}).get("chunk_level") == "child"]

    def get_parent_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从分块结果中筛出 parent chunk（用于回查原文）"""
        return [c for c in chunks if c.get("metadata", {}).get("chunk_level") == "parent"]
