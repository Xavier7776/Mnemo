# 性能优化报告 — 后端响应速度 + 前端 TTFB 占位

> 日期：2026-07-04
> 目标：优化后端 chat 响应速度 + 前端 AI 回复前加占位内容
> 验证：后端启动 healthy，流式响应正常，Core Memory 注入验证通过

---

## 一、优化前的性能瓶颈

通过全链路走读 `routers/chat.py` → `agents/general_assistant_agent.py` → `services/rag_service.py` → `retrieval/rag_retriever.py` → `services/llm_service.py`，识别出 **10 个性能瓶颈**，按严重程度分级：

| 级别 | 瓶颈 | 位置 | 影响 |
|---|---|---|---|
| P0 | LLM 流式生成用同步客户端 `for chunk in stream` | `llm_service.py:299-320` | 阻塞事件循环，所有请求串行，吞吐量极低 |
| P0 | 实体提取同步 LLM 调用 | `knowledge_extraction_service.py:132-141` | 最多阻塞 30 秒，拖慢 TTFB |
| P1 | BM25 关键词检索全程同步 | `rag_retriever.py:276-338` | 1200 个 chunk 分词+打分，阻塞数百毫秒 |
| P1 | CrossEncoder predict 同步 CPU 推理 | `rag_retriever.py:523` | 首次加载数秒，每次推理阻塞 |
| P2 | 邻居扩展 N 次串行 DB 查询 | `rag_service.py:266-294` | 12-20 次串行 MongoDB 往返 |
| P2 | 文档信息"批量查询"实为串行 | `rag_service.py:197-215` | for 循环逐个查 MongoDB |
| 前端 | TTFB 期间无 assistant 占位气泡 | `chat-playground.tsx:351-361` | 用户感知"卡住" |

---

## 二、优化内容（7 个文件，7 项改动）

### 2.1 P0：LLM 流式生成异步化（`utils/llm_client.py` + `services/llm_service.py`）

**问题**：`_generate_stream` 使用同步 `OpenAI` 客户端的 `for chunk in stream` 迭代，在 `async def` 函数内独占事件循环，导致所有并发请求被阻塞。

**优化**：

1. `utils/llm_client.py` 新增 `get_async_openai_client()` 函数，返回全局 `AsyncOpenAI` 客户端单例：

```python
from openai import OpenAI, AsyncOpenAI

_async_client: Optional[AsyncOpenAI] = None

def get_async_openai_client() -> AsyncOpenAI:
    """用于流式生成等需要高并发的场景，避免同步客户端阻塞事件循环。"""
    global _async_client
    if _async_client is not None:
        return _async_client
    # ... 与 get_openai_client 相同的 base_url/api_key 逻辑
    _async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return _async_client
```

2. `services/llm_service.py` 新增 `async_client` property，`_generate_stream` 改用 `await` + `async for`：

```python
# 优化前（阻塞事件循环）
stream = self.client.chat.completions.create(model=..., stream=True, ...)
for chunk in stream:
    yield delta.content

# 优化后（非阻塞，允许并发）
stream = await self.async_client.chat.completions.create(model=..., stream=True, ...)
async for chunk in stream:
    yield delta.content
```

**预期收益**：并发吞吐量提升 **数倍**（从"同时只能处理 1 个流式请求"→"可并发处理多个流式请求"）

---

### 2.2 P0：实体提取异步化（`services/knowledge_extraction_service.py`）

**问题**：`extract_entities` 在 `async def` 中直接调用同步 `self.client.chat.completions.create`，最多阻塞 30 秒。

**优化**：用 `asyncio.to_thread` 包装同步调用：

```python
# 优化前（阻塞事件循环）
response = self.client.chat.completions.create(model=..., timeout=30)
content = response.choices[0].message.content

# 优化后（在线程池中执行，不阻塞事件循环）
def _sync_extract() -> str:
    response = self.client.chat.completions.create(model=..., timeout=30)
    return response.choices[0].message.content
content = await asyncio.to_thread(_sync_extract)
```

**预期收益**：消除 TTFB 中 30 秒级阻塞风险，其他请求可并发处理

---

### 2.3 P1：BM25 关键词检索异步化（`retrieval/rag_retriever.py`）

**问题**：`_keyword_search` 内的 MongoDB 查询 + jieba 分词 + BM25 打分全程同步，处理 1200 个 chunk 阻塞数百毫秒。

**优化**：整个方法体用 `asyncio.to_thread` 包装：

```python
async def _keyword_search(self, query, document_id) -> List[Dict[str, Any]]:
    def _keyword_search_sync() -> List[Dict[str, Any]]:
        # ... 原有 MongoDB 查询 + BM25 计算逻辑
        return sorted(results, key=...)[: self.prefetch_k]
    return await asyncio.to_thread(_keyword_search_sync)
```

**预期收益**：消除数百毫秒事件循环阻塞

---

### 2.4 P1：重排模型 predict 异步化 + 启动预热（`retrieval/rag_retriever.py` + `utils/lifespan.py`）

**问题 1**：`_rerank` 方法是同步 `def`，`reranker.predict(pairs)` 同步 CPU 推理阻塞事件循环。
**问题 2**：首次调用时加载 CrossEncoder 模型（~400MB），冷启动延迟数秒。

**优化 1**：`_rerank` 改为 `async def`，`predict` 用 `asyncio.to_thread` 包装：

```python
# 优化前（同步，阻塞）
def _rerank(self, query, results, reranker):
    scores = reranker.predict(pairs)
    ...

# 优化后（异步，非阻塞）
async def _rerank(self, query, results, reranker):
    scores = await asyncio.to_thread(reranker.predict, pairs)
    ...
```

调用处也改为 `await self._rerank(...)`。

**优化 2**：`utils/lifespan.py` 启动时预热重排模型：

```python
# 启动时预热重排模型（如果启用），避免首个请求卡在模型加载上
if enable_reranker:
    logger.info("正在预热重排模型...")
    def _warmup_reranker():
        retriever = RAGRetriever()
        retriever._get_reranker()
    await asyncio.to_thread(_warmup_reranker)
    logger.info("重排模型预热完成")
```

**预期收益**：消除冷启动数秒延迟，predict 不再阻塞并发请求

---

### 2.5 P2：邻居扩展并发化（`services/rag_service.py`）

**问题**：邻居扩展在 for 循环内串行调用 `chunk_repo.get_neighbor_chunks`，12-20 个 chunk 就是 12-20 次串行 DB 往返。

**优化**：先收集所有需要扩展的 chunk，用 `asyncio.gather` 并发查询，再回填到循环中：

```python
# 优化前（串行 N 次 DB 往返）
for result in results:
    neighbors = chunk_repo.get_neighbor_chunks(doc_id, chunk_index, ...)
    # 处理 neighbors...

# 优化后（并发查询）
# 1. 预取阶段：收集所有需要扩展的 chunk
neighbor_prefetch_keys = [(chunk_id, doc_id, chunk_index) for result in results ...]

# 2. 并发查询
neighbor_tasks = [
    asyncio.to_thread(_get_neighbors, did, cidx)
    for _, did, cidx in neighbor_prefetch_keys
]
neighbor_results_list = await asyncio.gather(*neighbor_tasks, return_exceptions=True)

# 3. 回填到 map
for (cid, _, _), nb_result in zip(neighbor_prefetch_keys, neighbor_results_list):
    neighbor_prefetch_map[cid] = nb_result

# 4. 循环内从 map 取结果（无 DB 往返）
for result in results:
    neighbors = neighbor_prefetch_map.get(chunk_id) or []
    # 处理 neighbors...
```

**预期收益**：N 次串行 DB 往返 → 1 次并发往返，耗时从 N×RTT → max(RTT)

---

### 2.6 P2：文档信息批量查询（`services/rag_service.py`）

**问题**：注释写"批量查询文档信息"，实际是 `for doc_id in document_ids: doc_repo.get_document(doc_id)` 串行查询。

**优化**：改为 MongoDB `$in` 一次查询：

```python
# 优化前（串行 N 次）
for doc_id in document_ids:
    doc = doc_repo.get_document(doc_id)
    document_info_map[doc_id] = {...}

# 优化后（1 次批量查询）
doc_ids_obj = [ObjectId(did) for did in document_ids]
def _batch_get_docs():
    return list(doc_repo.collection.find({"_id": {"$in": doc_ids_obj}}))
docs = await asyncio.to_thread(_batch_get_docs)
for doc in docs:
    document_info_map[str(doc["_id"])] = {...}
```

**预期收益**：N 次 DB 往返 → 1 次，节省 (N-1)×RTT

---

### 2.7 前端：TTFB 期间加占位气泡（`web-tanstack/src/components/chat/chat-playground.tsx`）

**问题**：用户发送消息后，`draftAnswer` 为空字符串时 `allMessages` 不追加 assistant 气泡，TTFB 期间用户只能看到底部微弱的"生成中…"文字，感知上像是"卡住"了。

**优化**：在消息列表末尾、`streaming && draftAnswer.length === 0` 时渲染一个占位气泡——三个脉冲点 + "正在思考…"文字，样式与 assistant 消息气泡一致：

```tsx
{/* TTFB 占位气泡：流式已开始但首个 token 还没到 */}
{streaming && draftAnswer.length === 0 ? (
  <div className="flex gap-3 justify-start">
    <div className="flex size-9 shrink-0 items-center justify-center rounded-2xl bg-neutral-100 text-neutral-900">
      <Bot className="size-4" />
    </div>
    <div className="max-w-[80%] rounded-[1.4rem] border border-[var(--blue-line)] bg-white px-4 py-3 text-sm leading-6">
      <div className="flex items-center gap-2 text-neutral-500">
        <span className="inline-flex gap-1">
          <span className="size-1.5 animate-bounce rounded-full bg-neutral-400 [animation-delay:-0.3s]" />
          <span className="size-1.5 animate-bounce rounded-full bg-neutral-400 [animation-delay:-0.15s]" />
          <span className="size-1.5 animate-bounce rounded-full bg-neutral-400 [animation-delay:0s]" />
        </span>
        <span className="text-xs">正在思考…</span>
      </div>
    </div>
  </div>
) : null}
```

**设计要点**：
- 极简白色主题：白色背景 + 灰色边框 + 灰色脉冲点
- 与 assistant 消息气泡样式完全一致（同样的圆角、边框、Bot 图标）
- 三个脉冲点用 `animate-bounce` + 不同 `animation-delay` 实现波浪效果
- 首个 token 到达后，`draftAnswer.length > 0`，占位气泡自动被真实回复替换

---

## 三、修改文件清单

| 文件 | 改动类型 | 改动内容 |
|---|---|---|
| `utils/llm_client.py` | 新增 | `get_async_openai_client()` 函数 + `AsyncOpenAI` 导入 |
| `services/llm_service.py` | 修改 | `async_client` property + `_generate_stream` 改 `async for` |
| `services/knowledge_extraction_service.py` | 修改 | `extract_entities` 用 `asyncio.to_thread` 包装 |
| `retrieval/rag_retriever.py` | 修改 | `_keyword_search` 用 `to_thread` + `_rerank` 改 `async` + `predict` 用 `to_thread` |
| `services/rag_service.py` | 修改 | 邻居扩展并发预取 + 文档信息 `$in` 批量查询 |
| `utils/lifespan.py` | 修改 | 启动时预热重排模型 |
| `web-tanstack/src/components/chat/chat-playground.tsx` | 修改 | TTFB 占位气泡 |

---

## 四、预期指标提升

| 指标 | 优化前 | 优化后 | 提升幅度 |
|---|---|---|---|
| **并发吞吐量** | 同时只能处理 1 个流式请求（事件循环被同步迭代阻塞） | 可并发处理多个流式请求 | **数倍** |
| **TTFB（首字节响应时间）** | 检索全链路串行 + 实体提取 30s 阻塞 + 重排冷启动数秒 | 检索内部并发 + 无阻塞 + 重排预热 | **降低 30-60%** |
| **邻居扩展耗时** | N 次串行 DB 往返（N=12-20） | 1 次并发往返 | **降低 (N-1)/N** |
| **文档信息查询耗时** | N 次串行 DB 往返 | 1 次 `$in` 批量查询 | **降低 (N-1)/N** |
| **冷启动首请求延迟** | 重排模型首次加载数秒 | 启动时预热，首请求无加载延迟 | **消除数秒冷启动** |
| **前端用户感知** | TTFB 期间无任何反馈，像是"卡住" | 三脉冲点 + "正在思考…"占位气泡 | **显著改善** |

---

## 五、验证结果

| 验证项 | 结果 |
|---|---|
| 6 个后端文件语法检查 | All syntax OK |
| 后端启动 | healthy（MongoDB + Qdrant 均 healthy） |
| 流式响应 | 正常（`data: {"content": "你好，"}\n\ndata: {"content": "Xavier。我是"}...`） |
| Core Memory 注入 | 正常（回复中包含 "Xavier" 和 "个人科研/开发助手"） |
| 前端占位气泡 | 代码已写入，等待前端构建验证 |

---

## 六、未做的事项（后续可选）

1. **base_prompt / Core Memory 加缓存**：每次请求查 3 次 MongoDB 构建 system prompt（assistant_prompt + base_prompt + core_memory），可加 TTL 缓存节省 10-50ms。当前未做，因为收益较小。

2. **"先流后检索"模式**：对简单问题先让 LLM 开口，检索异步并行。改动较大，需要判断问题复杂度，建议后续观察 TTFB 是否仍不满意时再考虑。

3. **知识空间 collection_name 串行查询**：`rag_service.py:96-108` 中 N 个知识空间 N 次串行 await。通常 N=1-3，影响不大，未优化。

4. **`_generate_once`（非流式生成）也可改 AsyncOpenAI**：当前只改了流式路径，非流式路径（如 summarizer）仍用同步客户端。因非流式不在热路径，暂未改。
