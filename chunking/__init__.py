"""文本分块模块"""
# 保持向后兼容，导出原有模块
from .base import BaseChunker
from .simple_chunker import SimpleChunker
from .smart_chunker import SmartChunker
from .report_chunker import ReportChunker
from .parent_child_chunker import ParentChildChunker

# 导出分块路由模块
try:
    from .router import ContentAnalyzer
    from .langchain import RecursiveChunker, SemanticChunker

    __all__ = [
        "BaseChunker",
        "SimpleChunker",
        "SmartChunker",
        "ReportChunker",
        "ParentChildChunker",
        "ContentAnalyzer",
        "RecursiveChunker",
        "SemanticChunker"
    ]
except ImportError:
    # 如果路由模块或 LangChain 模块不可用，只导出原有模块
    __all__ = [
        "BaseChunker",
        "SimpleChunker",
        "SmartChunker",
        "ReportChunker",
        "ParentChildChunker"
    ]

