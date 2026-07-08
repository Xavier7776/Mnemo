"""
性能基准测试脚本
测量 RAG 检索各阶段的耗时、模型预热时间等指标
"""
import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载 .env.development
from pathlib import Path
env_path = Path(__file__).parent.parent / ".env.development"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def print_header(title):
    print("\n" + "=" * 60)
    print(f"【{title}】")
    print("=" * 60)


async def bench_embedding_warmup():
    """测量 Embedding 模型首次加载时间（冷启动）"""
    print_header("测试1：Embedding 模型预热时间")

    from embedding.embedding_service import embedding_service

    embedding_service._model = None
    embedding_service.vector_size = None

    t0 = time.perf_counter()
    vec = embedding_service.encode_single("测试文本", is_query=True)
    t1 = time.perf_counter()
    cold_start_ms = (t1 - t0) * 1000
    print(f"  冷启动（加载模型 + 首次编码）: {cold_start_ms:.1f} ms")
    print(f"  向量维度: {len(vec)}")

    t0 = time.perf_counter()
    vec = embedding_service.encode_single("测试文本", is_query=True)
    t1 = time.perf_counter()
    warm_ms = (t1 - t0) * 1000
    print(f"  已预热（单次编码）: {warm_ms:.1f} ms")

    speedup = cold_start_ms / warm_ms if warm_ms > 0 else 0
    print(f"  预热加速比: {speedup:.1f}x")

    # 批量编码测试
    batch_texts = [f"测试文本 {i}" for i in range(10)]
    t0 = time.perf_counter()
    vecs = embedding_service.encode(batch_texts)
    t1 = time.perf_counter()
    batch_ms = (t1 - t0) * 1000
    print(f"  批量编码（10条）: {batch_ms:.1f} ms, 每条均 {batch_ms/10:.1f} ms")

    return {
        "cold_start_ms": cold_start_ms,
        "warm_ms": warm_ms,
        "speedup": speedup,
        "batch_10_ms": batch_ms,
    }


async def bench_reranker_warmup():
    """测量 CrossEncoder 重排模型预热时间"""
    print_header("测试2：CrossEncoder 重排模型预热时间")

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("  sentence-transformers 未安装，跳过此测试")
        return {"cold_start_ms": None, "warm_ms": None, "speedup": None}

    model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

    test_pairs = [
        ("什么是人工智能", f"人工智能是计算机科学的一个分支，研究如何模拟人类智能。" * 5),
    ] * 20

    # 冷启动
    t0 = time.perf_counter()
    reranker = CrossEncoder(model_name, device="cpu")
    scores = reranker.predict(test_pairs)
    t1 = time.perf_counter()
    cold_start_ms = (t1 - t0) * 1000
    print(f"  冷启动（加载模型 + 20对重排）: {cold_start_ms:.1f} ms")
    print(f"  模型: {model_name}")

    # 预热后
    t0 = time.perf_counter()
    scores = reranker.predict(test_pairs)
    t1 = time.perf_counter()
    warm_ms = (t1 - t0) * 1000
    print(f"  已预热（20对重排）: {warm_ms:.1f} ms")

    speedup = cold_start_ms / warm_ms if warm_ms > 0 else 0
    print(f"  预热加速比: {speedup:.1f}x")

    # 不同数量级
    print(f"\n  --- 不同候选数重排耗时 ---")
    for n in [10, 20, 50, 100]:
        pairs = test_pairs[: min(n, len(test_pairs))]
        pairs = pairs * (n // len(pairs) + 1)
        pairs = pairs[:n]
        t0 = time.perf_counter()
        reranker.predict(pairs)
        t1 = time.perf_counter()
        print(f"  {n:>4} 对: {(t1-t0)*1000:.1f} ms")

    return {
        "cold_start_ms": cold_start_ms,
        "warm_ms": warm_ms,
        "speedup": speedup,
    }


async def bench_base_prompt_cache():
    """测量 Base Prompt 缓存效果"""
    print_header("测试3：Base Prompt 缓存效果")

    from services.prompt_chain import PromptChain

    t0 = time.perf_counter()
    p1 = await PromptChain.get_base_prompt()
    t1 = time.perf_counter()
    first_ms = (t1 - t0) * 1000
    print(f"  首次调用: {first_ms:.1f} ms")
    print(f"  Prompt 长度: {len(p1)} 字符")

    t0 = time.perf_counter()
    p2 = await PromptChain.get_base_prompt()
    t1 = time.perf_counter()
    cached_ms = (t1 - t0) * 1000
    print(f"  二次调用（缓存）: {cached_ms:.3f} ms")

    speedup = first_ms / cached_ms if cached_ms > 0 else 0
    print(f"  缓存加速比: {speedup:.1f}x")

    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        await PromptChain.get_base_prompt()
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) * 1000 / n
    print(f"  {n} 次平均: {avg_ms:.4f} ms/次")

    return {
        "first_ms": first_ms,
        "cached_ms": cached_ms,
        "speedup": speedup,
        "avg_ms": avg_ms,
    }


async def bench_core_memory():
    """测量 Core Memory 读写性能"""
    print_header("测试4：Core Memory 读写性能")

    try:
        from services.core_memory_service import core_memory_service
        from database.mongodb import mongodb_client
        await mongodb_client.connect()
    except Exception as e:
        print(f"  Core Memory 服务初始化失败: {e}")
        return {"read_ms": None, "append_ms": None, "replace_ms": None}

    # 读
    t0 = time.perf_counter()
    mem = await core_memory_service.get("global", "default")
    t1 = time.perf_counter()
    read_ms = (t1 - t0) * 1000
    print(f"  读取: {read_ms:.2f} ms")
    print(f"  块数量: {len(mem.blocks)}")

    # 追加
    test_content = "测试追加内容" + str(time.time())
    t0 = time.perf_counter()
    result = await core_memory_service.append("global", "default", "human", test_content)
    t1 = time.perf_counter()
    append_ms = (t1 - t0) * 1000
    print(f"  追加: {append_ms:.2f} ms, 成功={result.get('success')}")

    # 替换
    t0 = time.perf_counter()
    result = await core_memory_service.replace(
        "global", "default", "human", test_content, "已替换的测试内容"
    )
    t1 = time.perf_counter()
    replace_ms = (t1 - t0) * 1000
    print(f"  替换: {replace_ms:.2f} ms, 成功={result.get('success')}")

    # 渲染
    t0 = time.perf_counter()
    text = await core_memory_service.render_for_prompt("global", "default")
    t1 = time.perf_counter()
    render_ms = (t1 - t0) * 1000
    print(f"  渲染 prompt: {render_ms:.2f} ms, 长度={len(text)}")

    return {
        "read_ms": read_ms,
        "append_ms": append_ms,
        "replace_ms": replace_ms,
        "render_ms": render_ms,
    }


async def bench_archival_memory():
    """测量 Archival Memory 性能"""
    print_header("测试5：Archival Memory 性能")

    try:
        from services.archival_memory_service import archival_memory_service
    except Exception as e:
        print(f"  Archival Memory 服务初始化失败: {e}")
        return {"insert_ms": None, "search_ms": None}

    # 插入
    test_content = f"性能测试归档内容 {time.time()}"
    t0 = time.perf_counter()
    result = await archival_memory_service.insert("global", "default", test_content)
    t1 = time.perf_counter()
    insert_ms = (t1 - t0) * 1000
    print(f"  插入: {insert_ms:.2f} ms, 成功={result.get('success')}")

    # 检索
    t0 = time.perf_counter()
    results = await archival_memory_service.search("global", "default", "测试归档", top_k=5)
    t1 = time.perf_counter()
    search_ms = (t1 - t0) * 1000
    print(f"  检索: {search_ms:.2f} ms, 结果数={len(results)}")

    return {
        "insert_ms": insert_ms,
        "search_ms": search_ms,
    }


async def bench_llm_streaming_ttfb():
    """测量 LLM 流式首 token 延迟（TTFB）"""
    print_header("测试6：LLM 流式首 Token 延迟（TTFB）")

    from utils.llm_client import get_async_openai_client
    import os

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OLLAMA_API_KEY") or "sk-test"
    if not api_key or api_key == "sk-test":
        print("  未配置 LLM API Key，跳过此测试")
        return {"ttfb_ms": None, "total_ms": None}

    client = get_async_openai_client()
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # 预热
    try:
        stream = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        async for _ in stream:
            break
    except Exception as e:
        print(f"  LLM 连接失败: {e}")
        return {"ttfb_ms": None, "total_ms": None}

    # 正式测量
    test_prompts = [
        "请用一句话解释什么是人工智能",
        "什么是机器学习",
        "深度学习的原理是什么",
    ]

    ttfb_list = []
    total_list = []
    for i, prompt in enumerate(test_prompts):
        t_start = time.perf_counter()
        stream = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        first_token = True
        ttfb = 0
        async for chunk in stream:
            if first_token and chunk.choices[0].delta.content:
                ttfb = (time.perf_counter() - t_start) * 1000
                ttfb_list.append(ttfb)
                first_token = False
        total = (time.perf_counter() - t_start) * 1000
        total_list.append(total)
        print(f"  [{i+1}] TTFB: {ttfb:.0f} ms, 总耗时: {total:.0f} ms")

    if ttfb_list:
        avg_ttfb = sum(ttfb_list) / len(ttfb_list)
        avg_total = sum(total_list) / len(total_list)
        print(f"\n  平均 TTFB: {avg_ttfb:.0f} ms")
        print(f"  平均总耗时: {avg_total:.0f} ms")
        return {"ttfb_ms": avg_ttfb, "total_ms": avg_total}

    return {"ttfb_ms": None, "total_ms": None}


async def main():
    print("\n" + "#" * 60)
    print("#  Advanced-RAG 性能基准测试")
    print("#  测试时间: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("#" * 60)

    results = {}

    tests = [
        ("embedding_warmup", bench_embedding_warmup),
        ("reranker_warmup", bench_reranker_warmup),
        ("base_prompt_cache", bench_base_prompt_cache),
        ("core_memory", bench_core_memory),
        ("archival_memory", bench_archival_memory),
        ("llm_ttfb", bench_llm_streaming_ttfb),
    ]

    for name, func in tests:
        try:
            results[name] = await func()
        except Exception as e:
            print(f"  测试失败: {e}")
            results[name] = {"error": str(e)}

    # 汇总报告
    print("\n")
    print("#" * 60)
    print("#  测试结果汇总")
    print("#" * 60)

    lines = []
    lines.append("# Advanced-RAG 性能基准测试报告\n")
    lines.append(f"> 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 核心指标\n")
    lines.append("| 指标 | 结果 | 说明 |")
    lines.append("|------|------|------|")

    def add_row(metric, value, desc):
        if value is not None:
            lines.append(f"| {metric} | {value} | {desc} |")

    if "embedding_warmup" in results:
        r = results["embedding_warmup"]
        add_row("Embedding 冷启动", f"{r['cold_start_ms']:.0f} ms" if r.get('cold_start_ms') else "N/A", "首次加载模型 + 编码")
        add_row("Embedding 预热后", f"{r['warm_ms']:.1f} ms" if r.get('warm_ms') else "N/A", "单次编码耗时")
        add_row("Embedding 加速比", f"{r['speedup']:.1f}x" if r.get('speedup') else "N/A", "预热 vs 冷启动")

    if "reranker_warmup" in results:
        r = results["reranker_warmup"]
        if r.get('cold_start_ms') is not None:
            add_row("重排模型冷启动", f"{r['cold_start_ms']:.0f} ms", "20对重排")
            add_row("重排模型预热后", f"{r['warm_ms']:.1f} ms", "20对重排")
            add_row("重排加速比", f"{r['speedup']:.1f}x", "预热 vs 冷启动")

    if "base_prompt_cache" in results:
        r = results["base_prompt_cache"]
        add_row("Base Prompt 缓存加速", f"{r['speedup']:.1f}x" if r.get('speedup') else "N/A", "缓存 vs 首次")
        add_row("Base Prompt 缓存平均", f"{r['avg_ms']:.4f} ms" if r.get('avg_ms') else "N/A", "100次平均")

    if "core_memory" in results:
        r = results["core_memory"]
        if r.get('read_ms') is not None:
            add_row("Core Memory 读取", f"{r['read_ms']:.2f} ms", "MongoDB 查询")
            add_row("Core Memory 追加", f"{r['append_ms']:.2f} ms", "写入 MongoDB")
            add_row("Core Memory 替换", f"{r['replace_ms']:.2f} ms", "精确子串替换")
            add_row("Core Memory 渲染", f"{r['render_ms']:.2f} ms", "生成 prompt 文本")

    if "archival_memory" in results:
        r = results["archival_memory"]
        if r.get('insert_ms') is not None:
            add_row("Archival Memory 插入", f"{r['insert_ms']:.2f} ms", "向量化 + Qdrant写入")
            add_row("Archival Memory 检索", f"{r['search_ms']:.2f} ms", "top_k=5 语义检索")

    if "llm_ttfb" in results:
        r = results["llm_ttfb"]
        if r.get('ttfb_ms') is not None:
            add_row("LLM 首 Token 延迟", f"{r['ttfb_ms']:.0f} ms", "流式 TTFB")
            add_row("LLM 总响应时间", f"{r['total_ms']:.0f} ms", "短回答平均")

    report_text = "\n".join(lines)
    print(report_text)

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_result.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告已保存到: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
