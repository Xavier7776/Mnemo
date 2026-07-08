"""向量化服务 - 使用 sentence-transformers 本地模型（高质量中文检索）"""
from typing import List, Optional, Dict, Any
import os
from utils.logger import logger


class EmbeddingService:
    """文本向量化服务 - 使用本地 sentence-transformers 模型"""
    
    def __init__(
        self,
        model_name: Optional[str] = None
    ):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-zh-v1.5")
        self._model = None
        self.vector_size = None
        logger.info(f"Embedding 服务初始化 - 模型: {self.model_name}")
    
    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"正在加载 embedding 模型: {self.model_name}（首次需下载 ~100MB）")
                self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
                self.vector_size = self._model.get_sentence_embedding_dimension()
                logger.info(f"Embedding 模型加载完成 - 向量维度: {self.vector_size}")
            except Exception as e:
                logger.error(f"加载 embedding 模型失败: {str(e)}", exc_info=True)
                raise
        return self._model
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        model_name: Optional[str] = None,
        is_query: bool = False
    ) -> List[List[float]]:
        if not texts:
            return []
        
        model = self._get_model()
        
        cleaned = []
        for t in texts:
            if t is None:
                cleaned.append("")
            else:
                t = str(t)
                #替换空字节
                if "\x00" in t:
                    t = t.replace("\x00", "")
                cleaned.append(t)
        
        text_list = cleaned
        # BGE 官方要求：提问时加前缀，文档入库不加
        if is_query and "bge" in self.model_name.lower():
            text_list = ["为这个句子生成表示以用于检索：" + t for t in cleaned]
        
        try:
            embeddings = model.encode(
                text_list,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"向量化失败: {str(e)}", exc_info=True)
            raise
    
    def encode_single(self, text: str, model_name: Optional[str] = None, is_query: bool = False) -> List[float]:
        result = self.encode([text], model_name=model_name, is_query=is_query)
        return result[0] if result else []
    
    @property
    def dimension(self) -> int:
        if self.vector_size is None:
            self._get_model()
        return self.vector_size


# 全局向量化服务实例
embedding_service = EmbeddingService()
