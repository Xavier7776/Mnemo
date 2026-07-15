"""
直接调用后端服务上传测试文档并进行精准召回率测试
"""
import os
import sys
import time
import json
import asyncio
from pathlib import Path

# 确保从项目根目录运行
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

# 加载环境变量
from dotenv import load_dotenv
env_path = BASE_DIR / ".env.development"
if env_path.exists():
    load_dotenv(env_path, override=True)

from utils.logger import logger
from database.mongodb import get_document_repo, get_chunk_repo, mongodb_client
from chunking.simple_chunker import SimpleChunker
from embedding.embedding_service import embedding_service
from database.qdrant_client import get_qdrant_client
from retrieval.rag_retriever import RAGRetriever


TEST_DOCS_DIR = BASE_DIR / "tests" / "test_docs"
COLLECTION_NAME = "default_knowledge"


def upload_document_local(file_path: Path) -> dict:
    """直接上传文档到系统（不走API）"""
    doc_repo = get_document_repo()
    chunk_repo = get_chunk_repo()
    
    filename = file_path.name
    print(f"\n正在处理: {filename}")
    
    # 读取文件内容
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 计算哈希
    import hashlib
    file_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    # 检查重复
    duplicate = doc_repo.find_duplicate_by_hash(file_hash)
    if duplicate:
        doc_id = str(duplicate["_id"])
        print(f"  文档已存在, document_id: {doc_id}")
        return {"document_id": doc_id, "filename": filename, "status": "existing"}
    
    # 创建文档记录
    file_size = len(text.encode('utf-8'))
    doc_id = doc_repo.create_document(
        title=filename,
        file_type="md",
        file_path=str(file_path),
        file_size=file_size,
        file_hash=file_hash,
    )
    print(f"  创建文档记录, document_id: {doc_id}")
    
    # 标记为处理中
    doc_repo.update_document_status(doc_id, "processing")
    doc_repo.update_document_progress(doc_id, 25, "文本分块", "正在分块...")
    
    # 分块
    chunker = SimpleChunker()
    chunks = chunker.chunk(text, metadata={"document_id": doc_id, "file_type": ".md"})
    print(f"  分块完成, 共 {len(chunks)} 个块")
    
    # 向量化
    doc_repo.update_document_progress(doc_id, 50, "向量化", f"正在向量化 {len(chunks)} 个块...")
    chunk_texts = [chunk["text"] for chunk in chunks]
    vectors = embedding_service.encode(chunk_texts)
    print(f"  向量化完成, 向量维度: {len(vectors[0]) if vectors else 0}")
    
    # 存储到MongoDB
    doc_repo.update_document_progress(doc_id, 75, "存储向量", "正在存储到数据库...")
    chunk_ids = []
    for idx, chunk in enumerate(chunks):
        chunk_id = chunk_repo.create_chunk(
            document_id=doc_id,
            chunk_index=idx,
            text=chunk["text"],
            metadata=chunk.get("metadata", {})
        )
        chunk_ids.append(chunk_id)
    
    # 存储到Qdrant
    qdrant = get_qdrant_client(COLLECTION_NAME)
    qdrant.create_collection(vector_size=embedding_service.dimension)
    
    batch_payloads = []
    batch_ids = []
    for i, (chunk_id, chunk) in enumerate(zip(chunk_ids, chunks)):
        meta = (chunk.get("metadata") or {}).copy()
        batch_payloads.append({
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "text": chunk["text"],
            "chunk_index": i,
            "metadata": {
                "content_type": meta.get("content_type", "text"),
                "chunker_type": meta.get("chunker_type", "simple"),
                "file_type": ".md",
            }
        })
        batch_ids.append(chunk_id)
    
    qdrant.insert_vectors(
        vectors=vectors,
        payloads=batch_payloads,
        ids=batch_ids,
    )
    print(f"  向量存储完成, 共 {len(vectors)} 个向量")
    
    # 标记完成
    doc_repo.update_document_status(doc_id, "completed")
    doc_repo.update_document_progress(doc_id, 100, "完成", f"成功处理 {len(chunks)} 个文本块")
    print(f"  ✓ 文档处理完成")
    
    return {
        "document_id": doc_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "status": "completed"
    }


def test_retrieval(query: str, top_k: int = 10) -> dict:
    """测试检索"""
    try:
        retriever = RAGRetriever(collection_name=COLLECTION_NAME)
        results = retriever.search(query, top_k=top_k)
        return results
    except Exception as e:
        print(f"  检索失败: {e}")
        import traceback
        traceback.print_exc()
        return {"results": []}


def calc_recall_at_k(hit_flags, k):
    return 1.0 if any(hit_flags[:k]) else 0.0

def calc_precision_at_k(hit_flags, k):
    top_k = hit_flags[:k]
    return sum(1 for x in top_k if x) / k if k > 0 else 0.0

def calc_mrr(hit_flags):
    for idx, hit in enumerate(hit_flags, start=1):
        if hit:
            return 1.0 / idx
    return 0.0


def main():
    print("=" * 60)
    print("测试文档上传与精准召回率测试")
    print("=" * 60)
    
    # 连接数据库
    print("\n[1/4] 初始化数据库连接...")
    mongodb_client.connect()
    print("  ✓ MongoDB 连接成功")
    
    # 初始化 embedding
    print("\n[2/4] 初始化 Embedding 服务...")
    print(f"  模型: {embedding_service.model_name}")
    print(f"  维度: {embedding_service.dimension}")
    print("  ✓ Embedding 服务就绪")
    
    # 上传测试文档
    print("\n[3/4] 上传测试文档...")
    doc_files = sorted(TEST_DOCS_DIR.glob("*.md"))
    print(f"  找到 {len(doc_files)} 个测试文档")
    
    doc_info_list = []
    for doc_file in doc_files:
        result = upload_document_local(doc_file)
        doc_info_list.append(result)
    
    # 等待一下确保向量索引更新
    print("\n  等待向量索引更新...")
    time.sleep(3)
    
    # 构建测试集并测试召回率
    print("\n[4/4] 构建测试集并计算召回率...")
    
    test_cases = [
        {
            "id": "q1",
            "query": "人工智能和机器学习的关系是什么",
            "target_doc": "01_人工智能基础.md",
            "note": "人工智能基础概念"
        },
        {
            "id": "q2",
            "query": "向量数据库的工作原理是什么",
            "target_doc": "02_向量数据库原理与应用.md",
            "note": "向量数据库原理"
        },
        {
            "id": "q3",
            "query": "RAG检索增强生成技术的核心思想",
            "target_doc": "03_RAG检索增强生成技术详解.md",
            "note": "RAG技术原理"
        },
        {
            "id": "q4",
            "query": "Transformer模型的自注意力机制",
            "target_doc": "04_Transformer模型架构详解.md",
            "note": "Transformer架构"
        },
        {
            "id": "q5",
            "query": "MongoDB数据库的特点和优势",
            "target_doc": "05_MongoDB数据库入门与实践.md",
            "note": "MongoDB基础"
        },
    ]
    
    results = []
    for tc in test_cases:
        print(f"\n  测试: {tc['id']} - {tc['query']}")
        
        retrieval_result = test_retrieval(tc["query"], top_k=10)
        sources = retrieval_result.get("results", [])
        
        # 查找目标文档
        target_doc_info = next((d for d in doc_info_list if d["filename"] == tc["target_doc"]), None)
        if not target_doc_info:
            print(f"    未找到目标文档: {tc['target_doc']}")
            continue
        
        target_doc_id = target_doc_info["document_id"]
        
        # 计算命中情况
        hit_flags = []
        for src in sources:
            src_doc_id = src.get("document_id") or src.get("payload", {}).get("document_id")
            hit = src_doc_id == target_doc_id
            hit_flags.append(hit)
        
        r1 = calc_recall_at_k(hit_flags, 1)
        r3 = calc_recall_at_k(hit_flags, 3)
        r5 = calc_recall_at_k(hit_flags, 5)
        r10 = calc_recall_at_k(hit_flags, 10)
        p5 = calc_precision_at_k(hit_flags, 5)
        mrr_val = calc_mrr(hit_flags)
        
        first_hit_rank = None
        for i, hit in enumerate(hit_flags, start=1):
            if hit:
                first_hit_rank = i
                break
        
        print(f"    目标文档: {tc['target_doc']}")
        print(f"    首次命中排名: {first_hit_rank if first_hit_rank else '未命中'}")
        print(f"    Recall@1: {r1:.2f}, Recall@3: {r3:.2f}, Recall@5: {r5:.2f}, Recall@10: {r10:.2f}")
        print(f"    Precision@5: {p5:.2f}, MRR: {mrr_val:.4f}")
        
        results.append({
            "query_id": tc["id"],
            "query": tc["query"],
            "target_doc": tc["target_doc"],
            "note": tc["note"],
            "first_hit_rank": first_hit_rank,
            "recall_at_1": r1,
            "recall_at_3": r3,
            "recall_at_5": r5,
            "recall_at_10": r10,
            "precision_at_5": p5,
            "mrr": mrr_val,
            "total_sources": len(sources)
        })
    
    # 计算平均指标
    if results:
        avg_r1 = sum(r["recall_at_1"] for r in results) / len(results)
        avg_r3 = sum(r["recall_at_3"] for r in results) / len(results)
        avg_r5 = sum(r["recall_at_5"] for r in results) / len(results)
        avg_r10 = sum(r["recall_at_10"] for r in results) / len(results)
        avg_p5 = sum(r["precision_at_5"] for r in results) / len(results)
        avg_mrr = sum(r["mrr"] for r in results) / len(results)
        
        print("\n" + "=" * 60)
        print("召回率测试结果汇总")
        print("=" * 60)
        print(f"  测试用例数: {len(results)}")
        print(f"  平均 Recall@1:  {avg_r1:.4f} ({avg_r1*100:.1f}%)")
        print(f"  平均 Recall@3:  {avg_r3:.4f} ({avg_r3*100:.1f}%)")
        print(f"  平均 Recall@5:  {avg_r5:.4f} ({avg_r5*100:.1f}%)")
        print(f"  平均 Recall@10: {avg_r10:.4f} ({avg_r10*100:.1f}%)")
        print(f"  平均 Precision@5: {avg_p5:.4f}")
        print(f"  平均 MRR:       {avg_mrr:.4f}")
        
        # 保存结果
        output_file = BASE_DIR / "tests" / "retrieval_recall_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total_queries": len(results),
                    "avg_recall_at_1": avg_r1,
                    "avg_recall_at_3": avg_r3,
                    "avg_recall_at_5": avg_r5,
                    "avg_recall_at_10": avg_r10,
                    "avg_precision_at_5": avg_p5,
                    "avg_mrr": avg_mrr
                },
                "detailed_results": results,
                "documents": doc_info_list
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n  详细结果已保存到: {output_file}")
    
    print("\n✓ 测试完成!")


if __name__ == "__main__":
    main()
