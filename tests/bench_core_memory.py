"""单独测试 Core Memory 性能"""
import asyncio
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

env_path = Path(__file__).parent.parent / ".env.development"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


async def main():
    print("\n=== Core Memory 性能测试 ===")

    from database.mongodb import mongodb
    await mongodb.connect()
    print("  MongoDB 连接成功")

    from services.core_memory_service import core_memory_service

    # 读
    t0 = time.perf_counter()
    mem = await core_memory_service.get("global", "default")
    t1 = time.perf_counter()
    read_ms = (t1 - t0) * 1000
    print(f"  读取: {read_ms:.2f} ms")
    print(f"  块数量: {len(mem.blocks)}")

    # 多次读取平均
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        await core_memory_service.get("global", "default")
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    avg_read = sum(times) / len(times)
    print(f"  20次读取平均: {avg_read:.2f} ms")

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

    print("\n=== 汇总 ===")
    print(f"| Core Memory 读取 | {avg_read:.2f} ms | 20次平均 |")
    print(f"| Core Memory 追加 | {append_ms:.2f} ms | 写入 MongoDB |")
    print(f"| Core Memory 替换 | {replace_ms:.2f} ms | 精确子串替换 |")
    print(f"| Core Memory 渲染 | {render_ms:.2f} ms | 生成 prompt 文本 |")


if __name__ == "__main__":
    asyncio.run(main())
