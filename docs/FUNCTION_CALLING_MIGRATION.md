# Function Calling 改造文档

## 改造概述

将 Mnemo 项目的工具调用机制从**自定义 XML 标签 + regex 解析**全面改造为**原生 Function Calling**，并新建统一适配层支持多厂商模型。

### 改造前

- LLM 输出 `<function_calls><invoke name="..."><parameter name="...">...</parameter></invoke></function_calls>` XML 标签
- 后端用正则 `r'<function_calls>\s*<invoke\s+name="([^"]+)">\s*(.*?)\s*</invoke>\s*</function_calls>'` 解析
- 参数类型靠手动推断（int/float/bool/JSON/字符串），容易出错（如 `sources` 数组变字符串）
- 不支持原生 Function Calling 的结构化优势
- 工具调用失败后提示词禁止重试（"不要再输出 `<function_calls>`"）

### 改造后

- 使用 OpenAI Chat Completions 原生 `tools=` / `tool_choice=` 参数
- 流式响应通过 `delta.tool_calls` 接收结构化工具调用
- 参数类型由 LLM 原生保证，无需手动推断
- 统一适配层自动适配 OpenAI / Claude / Gemini / DeepSeek / Qwen / MiMo 等厂商
- 工具调用失败后引导 LLM 修正参数重试

---

## 改动文件清单

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `services/tool_call_adapter.py` | **新建** | Function Calling 统一适配层 |
| `services/llm_service.py` | **重写核心方法** | 移除 regex 解析，改用适配层 |
| `services/prompt_chain.py` | **修改** | 移除 XML 格式说明，简化工具调用规则 |
| `services/ai_tools.py` | **无改动** | 工具 schema 已是 JSON Schema，直接复用 |

---

## 详细改动

### 1. 新建：`services/tool_call_adapter.py`

统一适配层，以 OpenAI Chat Completions 格式作为 Canonical（规范中立）格式。

#### 核心组件

| 组件 | 作用 |
|---|---|
| `CanonicalTool` / `CanonicalToolCall` / `CanonicalToolResult` | 统一数据结构 |
| `ProviderCapability` | 厂商能力声明（strict/parallel/required/streaming/reasoning） |
| `detect_provider(base_url, model_name)` | 自动探测厂商 |
| `ToolSchemaConverter` | 工具定义双向转换（OpenAI/Claude/Gemini） |
| `StreamAggregator` | 流式工具调用增量聚合（按 index/block 聚合参数片段） |
| `MessageBuilder` | 构造厂商特定的工具结果消息 |
| `serialize_tool_result(result)` | 工具执行结果序列化为字符串 |
| `ToolCallAdapter` | Facade 顶层适配器 |

#### 支持的厂商

| 厂商 | 探测关键词 | 兼容方式 |
|---|---|---|
| OpenAI | 默认 | 直接兼容 |
| DeepSeek | `deepseek` | OpenAI 兼容 |
| Qwen | `dashscope` / `qwen` | OpenAI 兼容端点 |
| MiMo | `mimo` / `xiaomi` | OpenAI 兼容 |
| OpenRouter | `openrouter` | OpenAI 兼容 |
| Together | `together` | OpenAI 兼容 |
| Claude | `anthropic` / `claude` | 特殊适配（input_schema / tool_use block / tool_result in user） |
| Gemini | `gemini` / `generativelanguage` | 特殊适配（function_declarations / functionCall / functionResponse） |

#### 使用示例

```python
from services.tool_call_adapter import ToolCallAdapter, ToolSchemaConverter, serialize_tool_result, CanonicalToolResult

# 1. 创建适配器（自动探测厂商）
adapter = ToolCallAdapter.from_env(base_url=os.getenv("OPENAI_BASE_URL"), model_name="mimo-v2.5")

# 2. 转换工具定义
canonical_tools = ToolSchemaConverter.from_ai_tools_schema(ai_tools.get_tools_schema())
tools_payload = adapter.convert_tools(canonical_tools)
tool_choice = adapter.convert_tool_choice("auto")

# 3. 流式处理
aggregator = adapter.create_stream_aggregator()
stream = await client.chat.completions.create(model=..., messages=..., tools=tools_payload, tool_choice=tool_choice, stream=True)
async for chunk in stream:
    for event in aggregator.process_chunk(chunk):
        if event["type"] == "text":
            yield event["content"]
        elif event["type"] == "thinking":
            yield thinking_marker
        elif event["type"] == "tool_call":
            call = event["tool_call"]  # CanonicalToolCall

# 4. 构造工具结果消息
result_messages = adapter.build_tool_result_messages(
    assistant_content="...",
    tool_calls=[call],
    tool_results=[CanonicalToolResult(call.id, serialize_tool_result(result))],
)
current_messages.extend(result_messages)
```

---

### 2. 重写：`services/llm_service.py`

#### 新增

- `self._adapter_cache = None`（`__init__` 中）
- `_adapter` property：通过 `ToolCallAdapter.from_env()` 创建适配器，自动探测厂商

#### 移除

- `_process_tool_calls_in_messages` 方法（整个方法删除）
- 三个 regex pattern（`<function_calls>` / `<invoke name=>` / `<parameter name=>`）
- 流式文本累积 + `<function` 前缀检测逻辑
- `yielded_length` 文本去重机制
- 参数类型手动推断（int/float/bool/JSON/字符串）

#### 改造 `_generate_stream` 方法

| 方面 | 改造前 | 改造后 |
|---|---|---|
| LLM 调用 | `create(model, messages, stream, timeout)` | `create(model, messages, stream, timeout, tools, tool_choice)` |
| 流式处理 | 累积 `full_response`，检测 `<function` 前缀 | `StreamAggregator.process_chunk()` 事件驱动 |
| 工具调用检测 | regex 匹配 `<function_calls>` 标签 | `event["type"] == "tool_call"` |
| 参数解析 | regex `([^<]+)` + 类型推断 | `CanonicalToolCall.arguments`（原生 dict） |
| 工具结果消息 | `{"role": "user", "content": "工具调用结果..."}` | `adapter.build_tool_result_messages()`（厂商特定） |
| 前端事件标记 | `\x1eTOOL_CALL:{json}\x1e` | **不变** |
| 思考链标记 | `\x1eTHINKING:{json}\x1e` | **不变** |

#### 保留的逻辑

- `max_tool_rounds=20` 和 PAR 循环（Plan-Act-Observe-Reflect）
- `_observe_and_reflect` 方法（只对 rag_retrieve 生效）
- `tool_execution_context` 注入和 `seen_chunk_ids` 去重
- `rag_retrieve` 特殊路径（注入检索范围和去重状态）
- `assistant_id` 自动注入
- 工具调用失败重试引导
- `logs/prompts/prompt_<timestamp>.json` 调试日志

---

### 3. 修改：`services/prompt_chain.py`

#### 移除

- **Section 6.1 工具函数调用格式**：整个 `<function_calls>` XML 格式说明段落
- **Section 6.3 参数类型指引**：移除"数组类型参数应直接写 JSON 数组"等指引
- **build_prompt_chain 工具调用规则**：移除"不要输出 `<function_calls>`"

#### 修改

- Section 6 开头加一句："工具通过 Function Calling 机制调用，系统会自动处理参数格式。"
- Section 6.3 简化为："如果工具函数调用失败，请检查参数并重试"
- 工具调用规则改为："如果当前问题不需要调用任何工具，直接给出最终回答。"

#### 保留

- `_format_tools_description` 方法和工具描述注入
- Agentic RAG 检索指引
- 阶段三反思指引
- `beijing_now()` 当前时间注入

---

### 4. 无改动：`services/ai_tools.py`

工具 schema 已是标准 JSON Schema 格式，适配层通过 `ToolSchemaConverter.from_ai_tools_schema()` 转换。`async_call_tool` 路由逻辑不变。

---

## 架构对比

### 改造前

```
用户消息
  ↓
_build_messages（系统提示词含 XML 格式说明）
  ↓
_generate_stream
  ├─ LLM 输出文本（含 <function_calls> XML）
  ├─ regex 解析 XML → 工具名 + 参数字符串
  ├─ 参数类型推断（int/float/bool/JSON/string）
  ├─ 执行工具
  ├─ 工具结果作为 user 消息追加
  └─ 下一轮循环
```

### 改造后

```
用户消息
  ↓
_build_messages（系统提示词移除 XML 格式说明）
  ↓
_generate_stream
  ├─ ToolCallAdapter.from_env() → 自动探测厂商
  ├─ adapter.convert_tools() → 厂商特定工具格式
  ├─ LLM 调用（tools=, tool_choice=）
  ├─ StreamAggregator.process_chunk() → 事件流
  │   ├─ text 事件 → yield 给前端
  │   ├─ thinking 事件 → yield THINKING 标记
  │   └─ tool_call 事件 → CanonicalToolCall（原生 dict 参数）
  ├─ 执行工具（rag_retrieve 特殊路径保留）
  ├─ adapter.build_tool_result_messages() → 厂商特定结果消息
  └─ 下一轮循环
```

---

## 厂商适配差异

| 差异点 | OpenAI 系 | Claude | Gemini |
|---|---|---|---|
| 工具定义外层 | `{type:"function", function:{...}}` | `{name, description, input_schema}` | `{type:"function", name, ...}` |
| 参数 schema 字段名 | `parameters` | `input_schema` | `parameters` |
| 响应工具调用载体 | `message.tool_calls[].function` | `content[].tool_use` block | `parts[].functionCall` |
| 参数类型 | JSON **字符串**（需 json.loads） | **对象**（直接用） | **对象**（直接用） |
| 工具结果消息 | `{"role":"tool", "tool_call_id", "content"}` | `{"role":"user", "content":[{type:"tool_result",...}]}` | `{"role":"user", "parts":[{functionResponse:...}]}` |
| 流式参数增量 | `delta.tool_calls[i].function.arguments` | `content_block_delta.partial_json` | `args` 增量 |
| 流式完成信号 | `finish_reason:"tool_calls"` | `stop_reason:"tool_use"` | part/step 结束 |
| strict 模式 | 支持 | 支持 | 不支持 |
| reasoning + tools | 支持（回传 reasoning items） | 仅 auto/none | 支持（thought signatures） |

适配层通过 `ProviderCapability` 声明能力，`ToolSchemaConverter` / `StreamAggregator` / `MessageBuilder` 按厂商转换，`ToolCallAdapter` 作为 Facade 统一调用。

---

## 验证清单

- [x] `tool_call_adapter.py` 语法检查通过
- [x] `llm_service.py` 语法检查通过
- [x] `prompt_chain.py` 语法检查通过
- [x] `ai_tools.py` 语法检查通过
- [x] 前端 `tsc --noEmit` 类型检查通过
- [ ] 后端启动测试（重启后端验证）
- [ ] 工具调用功能测试（问一个需要检索的问题）
- [ ] MCP 工具调用测试（问一个需要 MCP 工具的问题）
- [ ] 多厂商适配测试（切换不同模型验证）

---

## 回滚方案

如果改造后出现问题，可以回滚到 regex 方案：

1. `llm_service.py`：恢复 `_process_tool_calls_in_messages` 方法和 regex 解析逻辑，移除 `tools=` 参数
2. `prompt_chain.py`：恢复 Section 6.1 XML 格式说明
3. `tool_call_adapter.py`：可保留（不影响 regex 方案），或删除

回滚时注意 `_generate_stream` 方法的完整重构，建议从 git 历史恢复。

---

## 后续优化方向

1. **strict 模式**：对支持 strict 的厂商（OpenAI/Claude）启用严格 schema 校验
2. **并行工具调用**：利用 `tool_calls` 支持单轮多工具并行执行
3. **能力探测缓存**：首次调用时探测厂商能力，后续复用
4. **自动重试增强**：工具调用失败时，适配层自动调整参数格式重试
5. **更多厂商支持**：Cohere、Mistral、百川等
