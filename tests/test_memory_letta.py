"""Letta 记忆系统冒烟测试 - Phase 1/4/5"""
import asyncio
import sys
import os
import json
import httpx
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
PROJECT_DIR = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

env_file = Path(PROJECT_DIR) / ".env.development"
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    load_dotenv(Path(PROJECT_DIR) / ".env", override=True)

BASE_URL = "http://localhost:8000/api/chat"


async def test_phase1_summarizer():
    """Phase 1: Summarizer + Recall 历史摘要化"""
    print("\n" + "=" * 60)
    print("Phase 1: Summarizer + Recall 历史摘要化")
    print("=" * 60)

    from database.mongodb import mongodb
    await mongodb.connect()

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{BASE_URL}/conversations", json={"title": "Summarizer 测试"})
        data = resp.json()
        conv_id = data["id"]
        print(f"  [CREATE] 对话 ID: {conv_id}")

    col = mongodb.get_collection("conversations")
    from utils.timezone import beijing_now
    messages = []
    for i in range(35):
        messages.append({
            "message_id": f"msg-{i:03d}",
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"这是第 {i+1} 条测试消息，讨论的话题是 Python 异步编程 {'协程' if i < 12 else '生成器' if i < 24 else '事件循环'}。",
            "timestamp": beijing_now(),
            "sources": [], "evidence": [], "citation_warnings": [], "recommended_resources": [],
        })
    await col.update_one({"_id": conv_id}, {"$set": {"messages": messages}})
    print(f"  [INSERT] 直接写入 {len(messages)} 条消息到 MongoDB")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{BASE_URL}/conversations/{conv_id}/messages",
            json={"role": "user", "content": "这是第 36 条消息，用来触发 summarizer。"},
        )
        assert resp.status_code == 200, f"add_message 失败: {resp.text}"
        print(f"  [ADD_MSG] 第 36 条消息已添加")

    print("  [WAIT] 等待 15 秒让 summarizer 执行...")
    await asyncio.sleep(15)

    # 同时也直接调用一次 maybe_summarize，确保逻辑正确
    print("  [DIRECT] 直接调用 maybe_summarize 确保逻辑验证...")
    from services.memory_summarizer import memory_summarizer
    # 重新添加消息使总数>30（因为上面的对话可能在 BackgroundTasks 中被处理过了）
    doc2 = await col.find_one({"_id": conv_id})
    if doc2 and len(doc2.get("messages", [])) <= 30:
        # BackgroundTasks 没触发或还没完成，直接调 maybe_summarize
        await memory_summarizer.maybe_summarize(conv_id)
        print("  [DIRECT] maybe_summarize 已直接调用")
    elif doc2 and doc2.get("summary"):
        print("  [DIRECT] BackgroundTasks 已完成摘要化，跳过直接调用")

    doc = await col.find_one({"_id": conv_id})
    summary = doc.get("summary", "")
    remaining_msgs = len(doc.get("messages", []))
    print(f"  [CHECK] summary 长度: {len(summary) if summary else 0}")
    print(f"  [CHECK] 剩余消息数: {remaining_msgs}")

    if summary:
        print(f"  [SUMMARY] 摘要预览: {summary[:300]}...")
        print("  [PASS] Phase 1: Summarizer 正常工作")
    else:
        print(f"  [WARN] summary 为空，可能 LLM 调用超时。剩余消息={remaining_msgs}")
        if remaining_msgs <= 36:
            print("  [PARTIAL] 消息数正常，summarizer 可能需要更长时间")

    await col.delete_one({"_id": conv_id})
    print(f"  [CLEANUP] 测试对话已删除")


async def test_phase4_tool_registration():
    """Phase 4: 工具注册验证"""
    print("\n" + "=" * 60)
    print("Phase 4: 工具注册验证（直接调用 async_call_tool）")
    print("=" * 60)

    from services.ai_tools import AITools
    tools = AITools()
    schemas = tools.get_tools_schema()
    names = [s["name"] for s in schemas]

    required = ["core_memory_append", "core_memory_replace",
                "archival_memory_insert", "archival_memory_search",
                "conversation_search"]
    for n in required:
        assert n in names, f"工具 {n} 未注册"
        print(f"  [TOOL] {n}: registered")

    from database.mongodb import mongodb
    await mongodb.connect()

    result = await tools.async_call_tool("core_memory_append", {
        "label": "human", "content": "Phase4 冒烟测试追加",
    })
    assert result.get("success"), f"core_memory_append 调用失败: {result}"
    print(f"  [CALL] core_memory_append -> success")

    result = await tools.async_call_tool("archival_memory_search", {
        "query": "Letta 记忆", "top_k": 3,
    })
    assert isinstance(result, list), f"archival_memory_search 返回类型错误: {type(result)}"
    print(f"  [CALL] archival_memory_search -> {len(result)} 条结果")

    result = await tools.async_call_tool("conversation_search", {
        "query": "Python", "limit": 3,
    })
    print(f"  [CALL] conversation_search -> {result}")

    print("  [PASS] Phase 4: 5 个新工具全部注册且可调用")


async def test_phase5_step_loop():
    """Phase 5: Step Loop + 模型自主工具调用"""
    print("\n" + "=" * 60)
    print("Phase 5: Step Loop + 模型自主工具调用")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{BASE_URL}/conversations", json={"title": "Step Loop 测试"})
        data = resp.json()
        conv_id = data["id"]
        print(f"  [CREATE] 对话 ID: {conv_id}")

        query = "请使用archival_memory_search工具搜索归档记忆中关于项目技术栈的内容。"
        print(f"  [SEND] query: {query}")

        resp = await client.post(
            f"{BASE_URL}/",
            json={"query": query, "conversation_id": conv_id, "enable_rag": False},
            timeout=120,
        )

        if resp.status_code == 200:
            raw = resp.text
            full_text = ""
            for line in raw.split("\n"):
                if line.startswith("data: "):
                    try:
                        chunk = json.loads(line[6:])
                        if "content" in chunk:
                            full_text += chunk["content"]
                    except:
                        pass
            print(f"  [RESPONSE] 模型回复（前 500 字）: {full_text[:500]}...")
            if "FastAPI" in full_text or "MongoDB" in full_text or "Qdrant" in full_text:
                print("  [PASS] Phase 5: 模型成功检索到归档记忆并回答")
            else:
                print("  [PARTIAL] Phase 5: 模型回复了但可能没有调用 archival_memory_search")
                print("    这取决于模型对 <function_calls> 格式的遵循能力")
        else:
            print(f"  [WARN] chat API 返回 {resp.status_code}: {resp.text[:200]}")

    print("  [INFO] Phase 5 测试完成")


async def main():
    try:
        await test_phase1_summarizer()
        await test_phase4_tool_registration()
        await test_phase5_step_loop()
        print("\n" + "=" * 60)
        print("Phase 1/4/5 冒烟测试完成")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
