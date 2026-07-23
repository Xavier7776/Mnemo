# Mnemo 记忆系统完整流程

> 日期：2026-07-04（初版） / 2026-07-20（v6 / v6.1 更新）  
> 参考架构：Letta 三层记忆体系

---

## 一、整体架构

项目实现了三层记忆体系：

| 记忆类型 | 存储位置 | 作用 | 注入方式 |
|----------|----------|------|----------|
| **Core Memory**（核心记忆） | MongoDB `agent_core_memory` | 常驻 system prompt 的 persona/human 块 | 每次请求自动注入 |
| **Recall Memory**（召回记忆） | MongoDB `conversations` 的 `messages` + `compressed_summary` | 全部对话原文 + 5 段式压缩摘要 | 取最近 8 条 + 压缩摘要前置 |
| **Archival Memory**（归档记忆） | Qdrant `agent_archival_memory` | 长期归档，按需语义检索 | 模型主动调用工具检索 |

> **v6 改动**：Recall Memory 不再裁剪 `messages` 数组（原版 Letta 行为），新增 `compressed_summary` 字段存压缩结果，`last_summarized_count` 记录增量压缩位置。
>
> **v6.2 改动**：9 段式 → 5 段式精简压缩。压缩比从 114.76% 降至 58.59%，保真度损失从 37.5% 降至 12.5%。新增 Key Facts 段显式保留硬细节（数字/API/决策/错误/专名）。详见 tests/compression_fidelity_results.md。

通过 `scope_type` + `scope_id` 统一抽象记忆归属：
- `scope_type="global"` + `scope_id="default"`：全局共享（当前默认）
- `scope_type="assistant"` + `scope_id=<assistant_id>`：同助手共享
- `scope_type="conversation"` + `scope_id=<conversation_id>`：会话级隔离

---

## 二、Core Memory（核心记忆）

### 2.1 数据结构

文件：[models/memory.py](file:///d:/timeModel/Mnemo/models/memory.py)

```python
class CoreMemoryBlock(BaseModel):
    value: str = ""
    limit: int = 2000  # 字符数上限

class CoreMemory(BaseModel):
    scope_type: str          # "global" | "assistant" | "conversation"
    scope_id: str
    blocks: Dict[str, CoreMemoryBlock] = {
        "persona": CoreMemoryBlock(value="你是用户的个人科研/开发助手..."),
        "human": CoreMemoryBlock(value=""),
    }
    updated_at: Optional[datetime] = None
```

默认两个块：`persona`（AI 人设）和 `human`（用户画像）。

### 2.2 读写流程

文件：[services/core_memory_service.py](file:///d:/timeModel/Mnemo/services/core_memory_service.py)

| 操作 | 方法 | 行号 | 说明 |
|------|------|------|------|
| 读取 | `get()` | 24-38 | 按 scope 查找，不存在则用默认值初始化 |
| 保存 | `_save()` | 40-51 | `upsert` 写入 MongoDB |
| 追加 | `append()` | 53-89 | 拼接 `原值 + "\n" + content`，超限返回错误 |
| 替换 | `replace()` | 91-130 | 精确子串匹配，仅替换首次匹配 |
| 渲染 | `render_for_prompt()` | 132-143 | 输出为 Markdown 格式注入 system prompt |

### 2.3 注入时机

文件：[services/prompt_chain.py](file:///d:/timeModel/Mnemo/services/prompt_chain.py) 第 323-333 行

在 `build_prompt_chain` 中，拼接完 `base_prompt + assistant_prompt` 后追加 Core Memory：

```python
core_memory_text = await core_memory_service.render_for_prompt("global", "default")
if core_memory_text:
    system_instruction = f"{system_instruction}\n\n{core_memory_text}"
```

渲染格式：
```
## 核心记忆（长期有效，除非你主动修改）
### persona
<AI人设内容>
### human
<用户画像内容>
```

---

## 三、Archival Memory（归档记忆）

### 3.1 数据结构

文件：[models/memory.py](file:///d:/timeModel/Mnemo/models/memory.py) 第 38-45 行

```python
class ArchivalMemoryItem(BaseModel):
    scope_type: str
    scope_id: str
    content: str
    source: str = "manual"  # manual | auto_summary | recall_migration
    created_at: datetime
    conversation_id: Optional[str] = None
```

### 3.2 读写流程

文件：[services/archival_memory_service.py](file:///d:/timeModel/Mnemo/services/archival_memory_service.py)

| 操作 | 方法 | 行号 | 说明 |
|------|------|------|------|
| 插入 | `insert()` | 37-88 | 向量化 → 写入 Qdrant |
| 检索 | `search()` | 90-138 | 向量化 → 语义搜索 → 返回 top_k 结果 |

**插入流程**：
1. `embedding_service.encode_single(content)` 向量化
2. 生成 `uuid4` 作为 point_id
3. 构造 payload（含 scope/source/conversation_id 等）
4. 调用 `qdrant_client.insert_vectors()`

**检索流程**：
1. `embedding_service.encode_single(query, is_query=True)` 向量化
2. 调用 `qdrant_client.search()`，按 scope 过滤
3. 返回 `[{id, content, score, created_at, source, conversation_id}, ...]`

---

## 四、Memory Summarizer（记忆摘要器）

### 4.1 触发条件

文件：[services/memory_summarizer.py](file:///d:/timeModel/Mnemo/services/memory_summarizer.py)

```python
CONTEXT_WINDOW_RESERVE = 50_000   # 当 context_window - messages_tokens < 此值时触发
KEEP_RECENT = 8                    # 压缩后保留最近 8 条原文注入 prompt
DEFAULT_CONTEXT_WINDOW_FOR_COMPRESS = 128_000  # 未识别模型时的兜底 context window
```

触发位置：[routers/chat.py](file:///d:/timeModel/Mnemo/routers/chat.py) `add_message` 端点中通过 `BackgroundTasks` 后台触发，不阻塞主流程。

**v6.1 触发逻辑**：
1. 读取 conversation 文档的 `last_model_name` 字段推断 context_window（由 `chat` 端点在请求时写入）
2. 用 `estimate_tokens` 估算 messages 数组总 token 数
3. 当 `context_window - total_tokens < CONTEXT_WINDOW_RESERVE` 时触发压缩
4. **小窗口保护**：当 `context_window <= CONTEXT_WINDOW_RESERVE`（如 32k 的 mimo-v2.5）时，降级为预留 `context_window // 4`（至少 4k），避免每条消息都触发

触发示例：
| 模型 | context_window | 触发阈值（已用 token） |
|------|----------------|------------------------|
| gpt-4o | 128k | 78k（剩余 50k 触发） |
| claude-3-opus | 200k | 150k（剩余 50k 触发） |
| mimo-v2.5 | 32k | 24k（剩余 8k 触发，小窗口保护） |
| 未知模型 | 128k（兜底） | 78k（剩余 50k 触发） |

### 4.2 v6.2 压缩策略（5 段式精简 + 不裁剪 messages）

1. **token 检查**：`context_window - messages_tokens < effective_reserve` 则继续，否则 return
2. **增量切分**：`to_summarize = messages[last_summarized_count : -KEEP_RECENT]`
   - `last_summarized_count` 从 MongoDB 字段读取，记录上次压缩到的位置
   - 避免不裁剪 messages 后重复压缩旧消息
3. **5 段式 LLM 压缩**：用 `_build_compact_prompt()` 生成精简摘要（目标压缩比 < 60%）
4. **三写**：
   - 归档到 Archival Memory（`source="auto_summary"`，可追溯）
   - 更新 `conversations.compressed_summary`：5 段式压缩结果全文
   - 更新 `conversations.summary`：压缩结果前 500 字符预览（向后兼容）
   - 更新 `conversations.last_summarized_count`：本次压缩到的位置
   - **不再裁剪 `messages` 数组**（保留全部原始消息，便于审计和 `conversation_search` 全文检索）

> **v6.1 移除**：暂不实施"清理可重新获取工具结果"（Read/Bash/Grep/Glob/WebSearch/Edit/Write）。
> 未来方案：基于"长时间（1 小时）模型没有调用"触发清理，而非压缩时清理。
> 代码中保留 `REGENERATABLE_TOOL_PATTERNS` 和 `_is_regeneratable_tool()` 定义以便未来启用。

> **v6.2 改动**：9 段式 → 5 段式精简。压缩比从 114.76% 降至 58.59%（实测 8 个用例），保真度损失从 37.5% 降至 12.5%。
> Key Facts 段显式保留数字/API 合同/决策结果/错误日志/专名，解决 9 段式对这三类细节完全失真的问题。

### 4.3 5 段式压缩结构（v6.2）

| # | 章节 | 核心要求 |
|---|------|----------|
| 1 | Request | 用户要什么。一句话 |
| 2 | Key Facts | 硬细节原文保留：数字/配置/API 合同/决策结果/错误日志/专名 |
| 3 | Files | 涉及的文件路径 + 函数名/行号（不复制代码段） |
| 4 ⭐⭐⭐ | User Messages | 按顺序列出每条用户发言要点（省略问候/客套） |
| 5 | Status | 当前进度 + 待办 + 下一步，每项一行简短 |

第 2 段（Key Facts）是 v6.2 新增的核心改进，解决 9 段式对 api_contract/decision_history/error_log 三类硬细节完全失真（100% 损失）的问题。

#### v6.1 vs v6.2 实测对比（8 个测试用例）

| 指标 | v6.1（9 段式） | v6.2（5 段式） | 改进 |
|---|---|---|---|
| 压缩比 | 114.76%（比原文还长） | 58.59% | -56.17 个百分点 |
| 保真度损失 | 37.5% | 12.5% | -25.0 个百分点 |
| api_contract | 100% 损失 | 0% 损失 | 完全修复 |
| decision_history | 100% 损失 | 0% 损失 | 完全修复 |
| error_log | 100% 损失 | 33% 损失 | 显著改善 |

### 4.4 大工具结果落盘（与压缩正交的优化）

文件：[services/tool_result_store.py](file:///d:/timeModel/Mnemo/services/tool_result_store.py)

落盘与压缩是两道独立的防线：
- **落盘**：在工具调用发生的瞬间就处理（[llm_service.py](file:///d:/timeModel/Mnemo/services/llm_service.py) 第 542-562 行），>2000 字符的结果存磁盘，messages 里只放引用 + 前 200 字符预览
- **压缩**：在 token 剩余不足时触发，5 段式 LLM 压缩（v6.2）

落盘配置：
```python
TOOL_RESULT_STORE_THRESHOLD = 2000   # 触发落盘的字符数阈值
TOOL_RESULT_PREVIEW_LENGTH = 200     # 落盘后保留的预览长度
TOOL_RESULT_TTL_DAYS = 7             # 自动清理天数
```

落盘文件位置：`logs/tool_results/{YYYYMMDD_HHMMSS}_{call_id前8位}.json`

---

## 五、记忆相关工具

文件：[services/ai_tools.py](file:///d:/timeModel/Mnemo/services/ai_tools.py)

| 工具名 | 行号 | 参数 | 作用 |
|--------|------|------|------|
| `core_memory_append` | 96-116 | `label`, `content` | 向核心记忆块追加内容 |
| `core_memory_replace` | 119-137 | `label`, `old_content`, `new_content` | 替换核心记忆中过时片段 |
| `archival_memory_insert` | 140-157 | `content` | 归档长期保存的信息 |
| `archival_memory_search` | 160-177 | `query`, `top_k=5` | 语义检索归档记忆 |
| `conversation_search` | 180-195 | `query`, `limit=5` | 全文检索历史对话 |

工具调用机制：
- `async_call_tool()`（第 293-331 行）统一调度
- `_get_async_tools()`（第 263-291 行）懒加载字典复用
- 记忆工具走 async 路径（直接 await），系统查询工具走 `asyncio.to_thread`

---

## 六、完整请求处理流程

以 `POST /api/chat/` 为例：

```
用户提问
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. 路由入口 chat.py (第748行)                            │
│    接收 ChatRequest                                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 2. 读取 Recall Memory (chat.py 第771-793行)              │
│    从 conversations 读取 messages + compressed_summary   │
│    取最近 8 条原文 + 5 段式压缩摘要前置为 system 消息    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Agent 执行 (general_assistant_agent.py)               │
│    若 enable_rag=True → RAG 检索                         │
│    调用 llm_service.generate()                           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 4. 构建 System Prompt (llm_service._build_messages)      │
│    ├─ base_prompt (数据库优先 / 默认值)                  │
│    ├─ assistant_prompt (特定助手提示词)                  │
│    ├─ ★ Core Memory 注入 (prompt_chain 第323行)         │
│    │  core_memory_service.render_for_prompt()            │
│    │  → 读取 MongoDB agent_core_memory                   │
│    │  → 追加到 system_instruction                        │
│    ├─ 工具调用规则引导                                   │
│    ├─ RAG 上下文 (若有)                                  │
│    └─ conversation_history (最近20条)                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 5. 流式生成 + 工具调用循环 (llm_service._generate_stream) │
│    max_tool_rounds = 4                                   │
│                                                          │
│    每轮:                                                 │
│    ├─ AsyncOpenAI stream=True                            │
│    ├─ 检测 <function_calls> 标签                         │
│    ├─ 标签前文本 → yield 给前端                          │
│    ├─ 解析工具调用 (name + params)                       │
│    ├─ 执行工具:                                          │
│    │  ├─ core_memory_append → MongoDB 写入               │
│    │  ├─ core_memory_replace → MongoDB 更新              │
│    │  ├─ archival_memory_insert → Qdrant 写入            │
│    │  ├─ archival_memory_search → Qdrant 检索            │
│    │  └─ conversation_search → MongoDB 全文检索          │
│    ├─ 工具结果拼回 messages                              │
│    └─ yield \x1eTOOL_CALL:{json}\x1e 事件给前端          │
│                                                          │
│    终止: 无工具调用 或 达到 4 轮上限                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Agent 解析标记 (general_assistant_agent.py 第133行)   │
│    正则匹配 \x1eTOOL_CALL:...\x1e                       │
│    分离为 chunk / tool_call 事件                         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 7. SSE 转发 (chat.py 第819-871行)                        │
│    chunk → {"content": "..."}                            │
│    tool_call → {"tool_call": {"round", "tools"}}         │
│    complete → {"done": true, "sources", ...}             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 8. 前端持久化 (POST /conversations/{id}/messages)        │
│    用户消息 + 助手回复 $push 到 messages 数组            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 9. 后台压缩触发 (chat.py BackgroundTasks)                │
│    if context_window - messages_tokens < 50_000:        │
│    ├─ 增量取 to_summarize = messages[last_summarized_count:-8] │
│    ├─ LLM 生成 5 段式 compressed_summary                 │
│    ├─ 旧消息归档到 Archival Memory (source=auto_summary) │
│    └─ 更新 conversations:                                │
│       ├─ compressed_summary（5 段式全文）                │
│       ├─ summary（前 500 字符预览，向后兼容）            │
│       ├─ last_summarized_count（增量压缩位置）           │
│       └─ messages 数组不裁剪（保留全部原文）             │
└─────────────────────────────────────────────────────────┘
```

> 此外，第 5 步工具调用循环中，>2000 字符的工具结果会即时落盘到 `logs/tool_results/`，
> messages 里只放引用 + 前 200 字符预览（见 [tool_result_store.py](file:///d:/timeModel/Mnemo/services/tool_result_store.py)）。
> `chat` 端点在请求开始时把 `model_name` 写入 conversation 文档的 `last_model_name` 字段，
> 供 `maybe_summarize` 推断 context_window（[chat.py](file:///d:/timeModel/Mnemo/routers/chat.py) 第 792-800 行）。

---

## 七、服务依赖关系

```
routers/chat.py
   │
   ├──> agents/general_assistant_agent.py
   │       ├──> services/rag_service.py (RAG 检索)
   │       └──> services/llm_service.py (LLM 生成)
   │               │
   │               ├──> services/prompt_chain.py
   │               │       └──> services/core_memory_service.py ★ Core Memory 注入
   │               │               └──> MongoDB (agent_core_memory)
   │               │
   │               └──> services/ai_tools.py (工具调用循环)
   │                       ├──> services/core_memory_service.py
   │                       ├──> services/archival_memory_service.py
   │                       │       ├──> Qdrant (agent_archival_memory)
   │                       │       └──> embedding_service.py
   │                       └──> MongoDB (conversation_search)
   │
   └──> services/memory_summarizer.py (后台任务)
           ├──> services/llm_service.py (生成摘要)
           ├──> services/archival_memory_service.py (旧对话归档)
           └──> MongoDB (更新 conversations)
```

### 各文件核心职责

| 文件 | 职责 |
|------|------|
| [models/memory.py](file:///d:/timeModel/Mnemo/models/memory.py) | 三层记忆数据模型 |
| [core_memory_service.py](file:///d:/timeModel/Mnemo/services/core_memory_service.py) | Core Memory 读写 + 渲染，存 MongoDB |
| [archival_memory_service.py](file:///d:/timeModel/Mnemo/services/archival_memory_service.py) | Archival Memory 向量化插入 + 语义检索，存 Qdrant |
| [memory_summarizer.py](file:///d:/timeModel/Mnemo/services/memory_summarizer.py) | v6.2: 5 段式精简压缩（压缩比 58.59%）+ token 剩余触发（50k）+ 不裁剪 messages（增量压缩） |
| [tool_result_store.py](file:///d:/timeModel/Mnemo/services/tool_result_store.py) | v6: 大工具结果（>2000 字符）落盘到 `logs/tool_results/`，TTL 7 天 |
| [prompt_chain.py](file:///d:/timeModel/Mnemo/services/prompt_chain.py) | base_prompt + assistant_prompt + Core Memory 注入 |
| [llm_service.py](file:///d:/timeModel/Mnemo/services/llm_service.py) | 构建 messages、流式生成 + 工具调用循环（max 4轮）、第 542-562 行工具结果落盘 |
| [ai_tools.py](file:///d:/timeModel/Mnemo/services/ai_tools.py) | 注册 9 个工具（5个记忆 + 4个系统查询） |
| [chat.py](file:///d:/timeModel/Mnemo/routers/chat.py) | HTTP 端点、读取 Recall Memory（第 771-793 行）、SSE 转发、持久化、后台压缩触发 |
| [general_assistant_agent.py](file:///d:/timeModel/Mnemo/agents/general_assistant/general_assistant_agent.py) | RAG 检索 + LLM 生成编排、解析工具调用事件 |

---

## 八、关键调用链路总结

### Core Memory
- **读**：`chat.py → llm_service._build_messages → prompt_chain.build_prompt_chain → core_memory_service.render_for_prompt → MongoDB`
- **写**：`LLM 工具调用 → ai_tools.async_call_tool("core_memory_append") → core_memory_service.append → MongoDB`
- **生效时机**：写入后**下一次请求**的 system prompt 构建时重新读取注入

### Archival Memory
- **手动写**：`LLM 工具调用 → ai_tools.async_call_tool("archival_memory_insert") → archival_memory_service.insert → Qdrant`
- **自动写**：`memory_summarizer.maybe_summarize → archival_memory_service.insert(source="auto_summary") → Qdrant`
- **读**：`LLM 工具调用 → ai_tools.async_call_tool("archival_memory_search") → archival_memory_service.search → Qdrant`

### Recall Memory
- **读**：`chat.py 直接读 conversations 集合 messages + compressed_summary 字段`（取最近 8 条 + 压缩摘要前置）
- **写**：`chat.py add_message 端点 $push messages`（永不裁剪）；`chat` 端点写入 `last_model_name` 供压缩器推断 context_window
- **检索**：`LLM 工具调用 → ai_tools.async_call_tool("conversation_search") → MongoDB $text 全文检索`（基于全部原始消息，覆盖范围比 v5 更广）
- **压缩触发**：`chat.py add_message → BackgroundTasks → memory_summarizer.maybe_summarize`（token 剩余 < 50k 触发，增量压缩 `messages[last_summarized_count:-8]`）
- **大结果落盘**：`llm_service._generate_stream → tool_result_store.store`（>2000 字符即时落盘）

---

## 九、已知限制

1. **Scope 硬编码**：所有记忆工具和 Core Memory 注入都写死 `("global", "default")`，尚未实现多用户隔离
2. **Core Memory 超限不截断**：`limit=2000`，超限时返回错误而非截断（设计正确）
3. **工具调用循环上限 4 轮**：超过后强制结束
4. **Recall Memory 历史窗口**：`_build_messages` 只取 `conversation_history[-20:]`，超长依赖压缩摘要补全
5. **压缩器幂等**：仅在 `context_window - messages_tokens < effective_reserve` 时触发，增量压缩 `messages[last_summarized_count:-8]`
6. **MongoDB messages 数组持续增长**：v6 不再裁剪，长期对话会导致 document 变大，需关注 MongoDB 16MB document 上限
7. **大工具结果落盘文件未加密**：`logs/tool_results/` 中明文存储，需确保日志目录权限
8. **token 估算为近似值**：`estimate_tokens` 用 `ascii/4 + cjk + other/2` 近似，与真实 tokenizer 有偏差，但作为触发阈值足够安全
9. **小窗口模型触发频率高**：32k 模型在 24k 已用时即触发压缩，压缩频率高于大窗口模型（设计权衡，保证回答空间）

---

## 十、v6 / v6.1 改动详解（2026-07-20）

参考 Letta 原版做法 + Claude Code 的对话总结格式，对记忆系统压缩环节做优化：

### 10.1 大工具结果直接存磁盘（v6）

**问题**：Read 大文件、WebSearch 返回大量内容、Bash 长输出直接塞进 messages 会撑爆 LLM 的 context window。

**方案**：在 [llm_service.py](file:///d:/timeModel/Mnemo/services/llm_service.py) 第 542-562 行，工具调用后判断结果长度，>2000 字符调 `tool_result_store.store()` 落盘到 `logs/tool_results/{timestamp}_{call_id}.json`，messages 里只放引用 + 前 200 字符预览：

```
[工具结果已落盘: tool=mcp__filesystem__read_file, file=20260720_143012_a1b2c3d4.json]
预览（前 200 字符）：
xxxxxxxx...
```

- SSE 事件已发送完整结果给前端，落盘只影响后续 LLM 上下文
- TTL 7 天自动清理（节流：每小时最多触发一次清理）
- 失败时降级为使用原始结果（非关键路径）

### 10.2 清理「可重新获取」的工具结果（v6 → v6.1 暂不实施）

**原 v6 方案**：压缩时清理 Read/Bash/Grep/Glob/WebSearch/Edit/Write 等工具的结果，只保留元信息。

**v6.1 调整**：暂不实施。用户反馈："做的话是长时间（1个小时）模型没有调用就删除"。
- v6 的"压缩时清理"逻辑会在每次压缩时都清理，不够精细
- 未来方案：基于"长时间未调用"触发清理（需要追踪每个工具结果的最后访问时间）
- 代码中保留 `REGENERATABLE_TOOL_PATTERNS` 和 `_is_regeneratable_tool()` 定义，以便未来启用

### 10.3 5 段式精简压缩（v6.2）

**问题**：v6 的 9 段式压缩实测压缩比 114.76%（比原文还长），且对 api_contract/decision_history/error_log 三类硬细节完全失真（100% 损失）。

**方案**：用 `_build_compact_prompt()` 生成 5 段精简摘要（见 4.3 节）。强制目标压缩比 < 60%，新增 Key Facts 段显式保留硬细节。

**实测效果**（8 个测试用例）：压缩比 58.59%，保真度损失 12.5%（vs 9 段式 37.5%）。api_contract 和 decision_history 完全修复，error_log 显著改善。

### 10.4 Recall Memory 保留全部消息（v6）

**问题**：原 v5 实现把过往消息归档到 Archival Memory 后直接删除 MongoDB 的 messages 数组，与 Letta 原版"Recall Memory 存全部消息，不归档"的做法不一致，导致 `conversation_search` 全文检索范围受限。

**方案**：
- MongoDB `conversations.messages` 数组永不裁剪（保留全部原始消息）
- 新增 `compressed_summary` 字段存 5 段式压缩结果全文
- 新增 `last_summarized_count` 字段记录增量压缩位置（避免重复压缩）
- `summary` 字段保留向后兼容（写入压缩结果的前 500 字符预览）
- 旧消息归档到 Archival Memory，可追溯

读取时（[chat.py](file:///d:/timeModel/Mnemo/routers/chat.py) 第 771-802 行）：取 `compressed_summary`（或回退到 `summary`）前置为 system 消息 + 最近 8 条原文。

### 10.5 触发条件改为 token 剩余（v6.1）

**问题**：v6 用"消息数 > 30"作为触发条件，但消息长度差异巨大（一条 Read 大文件的结果可能等于 50 条普通对话），消息数无法准确反映上下文压力。

**方案**：改为基于 token 剩余触发：
- 新增 `CONTEXT_WINDOW_RESERVE = 50_000` 常量
- `chat` 端点在请求时把 `model_name` 写入 conversation 文档的 `last_model_name` 字段
- `maybe_summarize` 从 conversation 文档读 `last_model_name`，调 `get_model_context_window` 推断 context_window
- 用 `estimate_tokens` 估算 messages 数组总 token 数
- 当 `context_window - total_tokens < CONTEXT_WINDOW_RESERVE` 时触发压缩

**小窗口保护**：当 `context_window <= CONTEXT_WINDOW_RESERVE`（如 32k 的 mimo-v2.5）时，50k 预留永远不可能满足，此时降级为预留 `context_window // 4`（至少 4k），避免每条消息都触发压缩。

### 10.6 v6.1 与 Letta 原版对比

| 维度 | Letta 原版 | Mnemo v5 | Mnemo v6 | Mnemo v6.1 |
|------|-----------|----------|----------|-----------|
| Recall Memory 存储 | 全部消息，不归档 | 裁剪到 8 条，旧消息归档 | 全部消息不裁剪 | 全部消息不裁剪 |
| 摘要格式 | 无结构纯文本 | 无结构纯文本 | 9 段式结构化 | 9 段式结构化 |
| 摘要字段 | summary | summary | compressed_summary | compressed_summary |
| 增量压缩 | 无 | 无 | last_summarized_count | last_summarized_count |
| 大工具结果处理 | 无 | 无 | >2000 字符落盘 | >2000 字符落盘 |
| 可重新获取工具结果 | 无 | 无 | 压缩时清理 | **暂不实施**（未来基于空闲时间） |
| 压缩触发条件 | 消息数 | 消息数 > 30 | 消息数 > 30 | **token 剩余 < 50k** |
| context_window 感知 | 无 | 无 | 无 | **last_model_name 推断** |
