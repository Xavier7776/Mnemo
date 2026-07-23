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
        # Phase 2 优化：嵌入模型从 bge-base-zh-v1.5（768维）升级到 bge-large-zh-v1.5（1024维）
        # bge-large-zh-v1.5 是中文 SOTA 嵌入模型，在 C-MTEB 中文检索任务上表现更好
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
        self._model = None
        self.vector_size = None
        logger.info(f"Embedding 服务初始化 - 模型: {self.model_name}")
    
    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # 读取 EMBEDDING_DEVICE 环境变量，默认 cpu
                # 6GB 显存 GPU 跑 bge-m3（权重 2.3GB + 推理激活）容易 OOM，默认 cpu 更稳
                # 显存充足时（8GB+）可设 EMBEDDING_DEVICE=cuda 加速
                device = os.getenv("EMBEDDING_DEVICE", "cpu")
                logger.info(f"正在加载 embedding 模型: {self.model_name}（device={device}）")
                self._model = SentenceTransformer(self.model_name, device=device, trust_remote_code=True)
                self.vector_size = self._model.get_sentence_embedding_dimension()
                logger.info(f"Embedding 模型加载完成 - 向量维度: {self.vector_size}, device={device}")
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
        # BGE-zh 系列官方要求：query 加前缀，document 不加
        # 但 bge-m3 不需要前缀（原生支持 query/passage，自动处理）
        # bge-large-zh-v1.5 / bge-base-zh-v1.5 需要 "为这个句子生成表示以用于检索：" 前缀
        model_lower = self.model_name.lower()
        needs_query_prefix = (
            is_query
            and "bge" in model_lower
            and "zh-v1.5" in model_lower  # 仅 bge-base-zh-v1.5 / bge-large-zh-v1.5
            and "m3" not in model_lower   # 排除 bge-m3
        )
        if needs_query_prefix:
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
