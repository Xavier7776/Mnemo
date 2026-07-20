# Mnemo 性能优化报告

> 日期：2026-07-04  
> 版本：v1.0  
> 范围：工具链修复 + 后端性能优化 + 前端性能优化

---

## 一、优化概述

本次优化共涉及 **8 个文件，17 项优化**，涵盖工具链正确性修复、后端性能瓶颈消除、前端渲染优化三个维度。

| 维度 | 文件数 | 优化项数 |
|------|--------|----------|
| 工具链修复 | 2 | 2 |
| 后端性能优化 | 5 | 9 |
| 前端性能优化 | 2 | 7 |

---

## 二、工具链修复

### 2.1 同 Chunk 文本丢失修复

**问题**：当 LLM 的一个 delta 同时包含 `<function_calls>` 标签前的正常文本和标签本身时（如 `"根据查询结果<function_calls>..."`），代码直接 `continue` 跳过 `yield`，导致标签前的文本被永久丢弃。

**文件**：[services/llm_service.py](file:///d:/timeModel/Mnemo/services/llm_service.py)

**修改**：
```python
# 修复前
if tool_call_start in full_response:
    tool_call_detected = True
    continue

# 修复后
if tool_call_start in full_response:
    tool_call_detected = True
    idx = full_response.find(tool_call_start)
    before_text = full_response[:idx]
    if before_text:
        yield before_text
    continue
```

### 2.2 短路检测优化

**问题**：`tool_call_detected=True` 后仍对每个 chunk 做 `tool_call_start in full_response` 的 O(n) 子串扫描。

**修改**：
```python
if tool_call_detected:
    continue  # 已检测到，直接跳过
if tool_call_start in full_response:
    ...
```

### 2.3 删除不可达死代码

**问题**：`_generate_stream` 方法末尾 for 循环之后的代码不可达（循环必然在内部 return）。

**修改**：删除约 4 行死代码。

### 2.4 `\x1e` 检测精度提升

**问题**：检测 `'\x1e' not in buffer_chunk` 过于宽泛，若 LLM 输出中出现孤立的 `\x1e` 字节，后续文本会被滞留。

**文件**：[agents/general_assistant/general_assistant_agent.py](file:///d:/timeModel/Mnemo/agents/general_assistant/general_assistant_agent.py)

**修改**：
```python
# 修复前
if stream and buffer_chunk and '\x1e' not in buffer_chunk:

# 修复后
if stream and buffer_chunk and '\x1eTOOL_CALL:' not in buffer_chunk:
```

### 2.5 Import 规范

**文件**：[agents/general_assistant/general_assistant_agent.py](file:///d:/timeModel/Mnemo/agents/general_assistant/general_assistant_agent.py)

**修改**：
- `import re`、`import json` 移到文件顶部
- `tool_call_re = re.compile(...)` 移到模块级常量

---

## 三、后端性能优化

### 3.1 LLM 客户端异步化

**问题**：`_generate_once` 和 `list_models` 仍使用同步客户端，阻塞事件循环。

**文件**：[services/llm_service.py](file:///d:/timeModel/Mnemo/services/llm_service.py)

**修改**：
```python
# list_models
models = await self.async_client.models.list()

# _generate_once
response = await self.async_client.chat.completions.create(...)
```

### 3.2 Base Prompt 缓存

**问题**：每次请求都查询 DB 获取 base_prompt 和 core_memory，增加 2 次 RTT。

**文件**：[services/llm_service.py](file:///d:/timeModel/Mnemo/services/llm_service.py)

**修改**：
```python
# __init__ 中添加缓存
self._base_prompt_cache = None
self._base_prompt_cache_time = 0

# _build_messages 中实现 5 分钟 TTL 缓存
if assistant_id is None:
    now = time.time()
    if self._base_prompt_cache and (now - self._base_prompt_cache_time) < 300:
        system_instruction = self._base_prompt_cache
    else:
        system_instruction = await prompt_chain.get_base_prompt()
        self._base_prompt_cache = system_instruction
        self._base_prompt_cache_time = now
```

### 3.3 BM25 双重分词合并

**问题**：对 1200 个候选 chunk 分词两次（一次算 avgdl，一次打分），CPU 浪费。

**文件**：[retrieval/rag_retriever.py](file:///d:/timeModel/Mnemo/retrieval/rag_retriever.py)

**修改**：先一次性分词所有 chunk，复用结果计算 avgdl 和打分。

### 3.4 图谱检索同步调用包装

**问题**：`_graph_search` 中 `self.chunk_repo.get_chunk_by_id` 是同步调用，且对每个 chunk_id 串行查询（N+1）。

**文件**：[retrieval/rag_retriever.py](file:///d:/timeModel/Mnemo/retrieval/rag_retriever.py)

**修改**：
1. 收集所有 chunk_ids
2. 用 `collection.find({"_id": {"$in": oids}})` 批量查询
3. 用 `await asyncio.to_thread(...)` 包装

### 3.5 Async Tools 字典复用

**问题**：`async_call_tool` 每次调用都重建 `async_tools` 字典（9 个 lambda），并重复 import 两个 service。

**文件**：[services/ai_tools.py](file:///d:/timeModel/Mnemo/services/ai_tools.py)

**修改**：
```python
def __init__(self):
    self._async_tools = None

def _get_async_tools(self):
    if self._async_tools is None:
        from services.core_memory_service import core_memory_service
        from services.archival_memory_service import archival_memory_service
        self._async_tools = { ... }
    return self._async_tools
```

### 3.6 同步 Qdrant 调用包装

**问题**：`_aget_knowledge_base_stats` 中 `qdrant.get_collection_info()` 是同步 gRPC 调用。

**文件**：[services/ai_tools.py](file:///d:/timeModel/Mnemo/services/ai_tools.py)

**修改**：
```python
total_vectors = await asyncio.to_thread(qdrant.get_collection_info).get("points_count", 0)
```

### 3.7 Embedding 模型预热

**问题**：Embedding 模型懒加载，首次查询需下载/加载模型（~100MB，数秒）。

**文件**：[utils/lifespan.py](file:///d:/timeModel/Mnemo/utils/lifespan.py)

**修改**：
```python
from embedding.embedding_service import embedding_service
await asyncio.to_thread(embedding_service._get_model)
```

### 3.8 删除冗余 find_one

**问题**：`add_message` 中刚 `update_one` 完又 `find_one` 拉取整个文档，仅为获取 title。

**文件**：[routers/chat.py](file:///d:/timeModel/Mnemo/routers/chat.py)

**修改**：复用前面已查询的 `doc` 变量。

### 3.9 标题生成线程池优化

**问题**：每次标题生成都新建 `ThreadPoolExecutor`。

**文件**：[routers/chat.py](file:///d:/timeModel/Mnemo/routers/chat.py)

**修改**：用 `asyncio.to_thread` 替代新建线程池。

---

## 四、前端性能优化

### 4.1 Draft 消息 Key 稳定化

**问题**：`timestamp: new Date().toISOString()` 每次 render 都生成新值，导致 React 认为是新元素，每 chunk 卸载重建 MessageBubble。

**文件**：[web-tanstack/src/components/chat/chat-playground.tsx](file:///d:/timeModel/Mnemo/web-tanstack/src/components/chat/chat-playground.tsx)

**修改**：
```tsx
const draftTimestampRef = useRef<string>("")

// sendMutation 开头
draftTimestampRef.current = new Date().toISOString()

// allMessages 中
timestamp: draftTimestampRef.current
```

### 4.2 组件 Memo 化

**问题**：MessageBubble、ToolCallChain、ToolCallItem、SourceList、CopyButton 未 memo，流式期间全量 diff。

**文件**：[web-tanstack/src/components/chat/chat-playground.tsx](file:///d:/timeModel/Mnemo/web-tanstack/src/components/chat/chat-playground.tsx)

**修改**：
```tsx
const MessageBubble = memo(function MessageBubble(...) { ... })
const ToolCallChain = memo(function ToolCallChain(...) { ... })
const ToolCallItem = memo(function ToolCallItem(...) { ... })
const SourceList = memo(function SourceList(...) { ... })
const CopyButton = memo(function CopyButton(...) { ... })
```

### 4.3 派生值 UseMemo 化

**问题**：`messages`、`allMessages`、`stripToolCalls(draftAnswer)` 每次渲染重算。

**文件**：[web-tanstack/src/components/chat/chat-playground.tsx](file:///d:/timeModel/Mnemo/web-tanstack/src/components/chat/chat-playground.tsx)

**修改**：
```tsx
const messages = useMemo<ConversationMessage[]>(..., [activeConversationId, detailQuery.data])

const cleanedDraft = useMemo(() => stripToolCalls(draftAnswer), [draftAnswer])

const allMessages = useMemo<ConversationMessage[]>(..., [messages, cleanedDraft, draftAnswer, draftToolCalls])
```

### 4.4 流式滚动优化

**问题**：每个 chunk 都触发滚动，无节流，不响应用户上滑。

**文件**：[web-tanstack/src/components/chat/chat-playground.tsx](file:///d:/timeModel/Mnemo/web-tanstack/src/components/chat/chat-playground.tsx)

**修改**：
```tsx
const stickToBottomRef = useRef(true)

const handleScroll = useCallback(() => {
  const el = messageListRef.current
  stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
}, [])

useEffect(() => {
  if (!stickToBottomRef.current) return
  const id = requestAnimationFrame(() => {
    messageListRef.current.scrollTop = messageListRef.current.scrollHeight
  })
  return () => cancelAnimationFrame(id)
}, [draftAnswer, draftToolCalls, detailQuery.data, activeConversationId])
```

### 4.5 组件卸载时 Abort

**问题**：组件卸载时未 abort 流式请求，浪费 LLM 费用。

**文件**：[web-tanstack/src/components/chat/chat-playground.tsx](file:///d:/timeModel/Mnemo/web-tanstack/src/components/chat/chat-playground.tsx)

**修改**：
```tsx
useEffect(() => {
  return () => {
    abortRef.current?.abort()
  }
}, [])
```

### 4.6 AbortError 不显示为错误

**问题**：用户主动停止却看到红色错误提示。

**文件**：[web-tanstack/src/components/chat/chat-playground.tsx](file:///d:/timeModel/Mnemo/web-tanstack/src/components/chat/chat-playground.tsx)

**修改**：
```tsx
} catch (err) {
  if (err instanceof DOMException && err.name === "AbortError") {
    // 用户主动取消，不算错误
  } else {
    collectedError = err instanceof Error ? err.message : String(err)
    setStreamError(collectedError)
  }
}
```

### 4.7 SSE Reader 资源释放

**问题**：`sendChatStream` 中 reader 在异常路径未显式释放。

**文件**：[web-tanstack/src/lib/api.ts](file:///d:/timeModel/Mnemo/web-tanstack/src/lib/api.ts)

**修改**：
```tsx
try {
  while (true) { ... }
} finally {
  reader.releaseLock()
}
```

---

## 五、预期收益

| 优化项 | 预期收益 | 优先级 |
|--------|----------|--------|
| BM25 双重分词合并 | 关键词检索 CPU 减半 | P0 |
| 同步调用 asyncio.to_thread 包装 | 消除事件循环阻塞 | P0 |
| Base Prompt 缓存 | 每请求省 2 次 DB 查询 | P0 |
| Embedding 预热 | 首请求省数秒 | P0 |
| Draft 消息 key 稳定 | 流式渲染流畅度大幅提升 | P0 |
| 组件 memo 化 | 流式期间历史消息不重渲染 | P0 |
| 滚动节流 | 体验 + 性能 | P1 |
| 卸载 abort | 省 LLM 费用 | P1 |

---

## 六、验证结果

- ✅ Python 语法检查通过（6 个文件）
- ✅ TypeScript 编译通过（无错误）
