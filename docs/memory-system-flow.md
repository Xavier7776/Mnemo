# Advanced-RAG 记忆系统完整流程

> 日期：2026-07-04  
> 参考架构：Letta 三层记忆体系

---

## 一、整体架构

项目实现了三层记忆体系：

| 记忆类型 | 存储位置 | 作用 | 注入方式 |
|----------|----------|------|----------|
| **Core Memory**（核心记忆） | MongoDB `agent_core_memory` | 常驻 system prompt 的 persona/human 块 | 每次请求自动注入 |
| **Recall Memory**（召回记忆） | MongoDB `conversations` 的 `messages` + `summary` | 对话历史 + 摘要 | 取最近 8 条 + 摘要前置 |
| **Archival Memory**（归档记忆） | Qdrant `agent_archival_memory` | 长期归档，按需语义检索 | 模型主动调用工具检索 |

通过 `scope_type` + `scope_id` 统一抽象记忆归属：
- `scope_type="global"` + `scope_id="default"`：全局共享（当前默认）
- `scope_type="assistant"` + `scope_id=<assistant_id>`：同助手共享
- `scope_type="conversation"` + `scope_id=<conversation_id>`：会话级隔离

---

## 二、Core Memory（核心记忆）

### 2.1 数据结构

文件：[models/memory.py](file:///d:/timeModel/advanced-rag/models/memory.py)

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

文件：[services/core_memory_service.py](file:///d:/timeModel/advanced-rag/services/core_memory_service.py)

| 操作 | 方法 | 行号 | 说明 |
|------|------|------|------|
| 读取 | `get()` | 24-38 | 按 scope 查找，不存在则用默认值初始化 |
| 保存 | `_save()` | 40-51 | `upsert` 写入 MongoDB |
| 追加 | `append()` | 53-89 | 拼接 `原值 + "\n" + content`，超限返回错误 |
| 替换 | `replace()` | 91-130 | 精确子串匹配，仅替换首次匹配 |
| 渲染 | `render_for_prompt()` | 132-143 | 输出为 Markdown 格式注入 system prompt |

### 2.3 注入时机

文件：[services/prompt_chain.py](file:///d:/timeModel/advanced-rag/services/prompt_chain.py) 第 323-333 行

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

文件：[models/memory.py](file:///d:/timeModel/advanced-rag/models/memory.py) 第 38-45 行

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

文件：[services/archival_memory_service.py](file:///d:/timeModel/advanced-rag/services/archival_memory_service.py)

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

文件：[services/memory_summarizer.py](file:///d:/timeModel/advanced-rag/services/memory_summarizer.py)

```python
TRIGGER_MESSAGE_COUNT = 30  # 超过 30 条消息触发
KEEP_RECENT = 8             # 摘要后保留最近 8 条原文
```

触发位置：[routers/chat.py](file:///d:/timeModel/advanced-rag/routers/chat.py) 第 451-462 行，在 `add_message` 端点中通过 `BackgroundTasks` 后台触发，不阻塞主流程。

### 4.2 摘要策略

1. **幂等检查**：`len(messages) <= 30` 则直接 return
2. **切分**：`to_summarize = messages[:-8]`（旧消息）
3. **拼转录文本**：`"\n".join(f"{role}: {content}")`
4. **LLM 生成摘要**：把已有摘要 + 新增对话合并，只保留事实/结论/偏好
5. **双写**：
   - 归档到 Archival Memory（`source="auto_summary"`）
   - 更新 `conversations` 集合：`summary` 字段保存摘要，`messages` 裁剪到最近 8 条

---

## 五、记忆相关工具

文件：[services/ai_tools.py](file:///d:/timeModel/advanced-rag/services/ai_tools.py)

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
│ 2. 读取 Recall Memory (chat.py 第774-790行)              │
│    从 conversations 集合读取 messages + summary          │
│    取最近 8 条原文 + 摘要前置为 system 消息              │
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
│ 9. 后台摘要触发 (chat.py 第451行 BackgroundTasks)        │
│    if len(messages) > 30:                                │
│    ├─ LLM 生成旧对话摘要                                 │
│    ├─ 旧对话归档到 Archival Memory (source=auto_summary) │
│    └─ 更新 conversations: summary + 裁剪 messages 到 8条 │
└─────────────────────────────────────────────────────────┘
```

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
| [models/memory.py](file:///d:/timeModel/advanced-rag/models/memory.py) | 三层记忆数据模型 |
| [core_memory_service.py](file:///d:/timeModel/advanced-rag/services/core_memory_service.py) | Core Memory 读写 + 渲染，存 MongoDB |
| [archival_memory_service.py](file:///d:/timeModel/advanced-rag/services/archival_memory_service.py) | Archival Memory 向量化插入 + 语义检索，存 Qdrant |
| [memory_summarizer.py](file:///d:/timeModel/advanced-rag/services/memory_summarizer.py) | 超长对话摘要化（>30条触发，保留8条） |
| [prompt_chain.py](file:///d:/timeModel/advanced-rag/services/prompt_chain.py) | base_prompt + assistant_prompt + Core Memory 注入 |
| [llm_service.py](file:///d:/timeModel/advanced-rag/services/llm_service.py) | 构建 messages、流式生成 + 工具调用循环（max 4轮） |
| [ai_tools.py](file:///d:/timeModel/advanced-rag/services/ai_tools.py) | 注册 9 个工具（5个记忆 + 4个系统查询） |
| [chat.py](file:///d:/timeModel/advanced-rag/routers/chat.py) | HTTP 端点、读取 Recall Memory、SSE 转发、持久化、后台摘要 |
| [general_assistant_agent.py](file:///d:/timeModel/advanced-rag/agents/general_assistant/general_assistant_agent.py) | RAG 检索 + LLM 生成编排、解析工具调用事件 |

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
- **读**：`chat.py 直接读 conversations 集合 messages + summary 字段`
- **写**：`chat.py add_message 端点 $push messages`
- **检索**：`LLM 工具调用 → ai_tools.async_call_tool("conversation_search") → MongoDB $text 全文检索`
- **摘要触发**：`chat.py add_message → BackgroundTasks → memory_summarizer.maybe_summarize`

---

## 九、已知限制

1. **Scope 硬编码**：所有记忆工具和 Core Memory 注入都写死 `("global", "default")`，尚未实现多用户隔离
2. **Core Memory 超限不截断**：`limit=2000`，超限时返回错误而非截断（设计正确）
3. **工具调用循环上限 4 轮**：超过后强制结束
4. **Recall Memory 历史窗口**：`_build_messages` 只取 `conversation_history[-20:]`，超长依赖 summary 补全
5. **摘要器幂等**：仅在 `len(messages) > 30` 时触发，摘要后裁剪到 8 条
