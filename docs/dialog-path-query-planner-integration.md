# 对话路径 QueryPlanner 前置规划集成

> 弥补对话 RAG 路径缺失的前置规划，让 `chat.py → GeneralAssistantAgent → PAOR 循环` 也能享受结构化检索参数规划。

## 一、背景

### 1.1 问题

项目的检索规划器 `QueryPlanner` 原本只在检索 API 路径生效：

```
retrieval.py → rag_service.retrieve_context() → query_planner.build_plan()  ← Plan 在这里
```

而主对话场景的调用链是：

```
chat.py → GeneralAssistantAgent.execute() → LLMService._generate_stream()  ← PAOR 循环
```

**对话路径没有调用 `QueryPlanner`**，`rag_context` 初始化为空字符串直接进入 PAOR 循环，LLM 在 Reasoning 阶段实时决定要不要搜、搜什么。这导致两个问题：

1. **检索参数用默认值**：`final_k`/`prefetch_k`/`fusion_strategy` 不会按查询意图动态调整（比如 compare 类查询应该 `final_k=20`，general 类 `final_k=12`）
2. **复杂查询无预检索**：compare/summary/clause 类查询本可提前检索证据注入 LLM 首轮 prompt，减少工具调用轮次

### 1.2 两条路径对比

| 路径 | 原 Plan? | 说明 |
|------|----------|------|
| 检索 API (`retrieval.py`) | 有 | `QueryPlanner.build_plan()` 前置规划 |
| 对话 RAG (`chat.py`) | **没有** | `rag_context=""`，LLM 在 PAOR 里即时决策 |

### 1.3 目标

让对话路径在进入 PAOR 循环前，先用 `QueryPlanner` 做一次前置规划，产出 LLM reasoning 不会生成的结构化检索参数，并按意图决定是否预检索。

## 二、核心概念澄清：前置 Plan ≠ PAOR 的 P

PAOR 循环里的 **P（Plan）就是 LLM reasoning**。LLM 每轮收到 messages + 工具 schema + 系统提示词后，自主思考"要不要调工具、调哪个、传什么参数"，这个 `reasoning_content` 就是 Plan。

**前置 `build_plan` 不是 reasoning**，是**检索参数规划**（retrieval parameter planning），产出 LLM reasoning 不会生成的结构化参数：

| 产出 | PAOR 的 P（LLM reasoning） | 前置 build_plan |
|------|---------------------------|-----------------|
| 要不要调工具 | ✅ LLM 实时决策 | ❌ 不参与 |
| 调什么参数（query/top_k）| ✅ LLM 决定 | ❌ 不参与 |
| `final_k` / `prefetch_k` | ❌ 用默认值 | ✅ 按 intent 动态调整 |
| `fusion_strategy` | ❌ 用默认值 | ✅ 规划（rrf/auto）|
| `sub_queries` 查询分解 | ⚠️ LLM 可能会做但不会结构化 | ✅ 结构化产出 |
| 是否预检索 | ❌ 不参与 | ✅ 按 intent 决定 |

**两者各司其职，不冲突**：
- 前置 `build_plan` = 检索参数预规划（决定 `final_k`、是否预检索）
- PAOR 的 P = LLM 实时 reasoning（决定要不要调工具）

## 三、方案设计

### 3.1 三态规划模式

通过环境变量 `AGENT_PLANNER_MODE` 控制（默认 `auto`）：

| 模式 | 行为 | 延迟 | 适用场景 |
|------|------|------|----------|
| `auto`（默认）| 短查询/命中关键词用规则引擎，长查询/复杂结构用 LLM | 大多数 <1ms，复杂查询 ~1-3s | 生产默认 |
| `rules` | 强制规则引擎 | 始终 <1ms | 延迟敏感场景 |
| `llm` | 强制 LLM 判断 | 始终 ~1-3s | 准确性优先场景 |
| `env` | 按 `QUERY_PLANNER_MODE` 走 | 取决于配置 | retrieval.py API 路径兼容 |

### 3.2 auto 模式的智能决策

`_query_needs_llm(query)` 函数根据 query 复杂度智能选择：

**用规则引擎（快）的情况**：
1. **命中明确意图关键词**（27 个词）→ 规则引擎关键词匹配够准
   - 对比类：`对比`、`比较`、`差异`、`优缺点`、`优劣`、`分别`、`各自`、`区别`、`相同点`、`不同点`
   - 列举类：`有哪些`、`列举`、`总结`、`概括`、`要点`、`关键点`、`核心观点`
   - 条款类：`条款`、`规定`、`标准`、`定义`、`范围`、`假设`、`条件`
   - 风险类：`风险`、`限制`、`不足`、`漏洞`
2. **极短查询（<=15 字）且单句** → 即使误判为 general 影响也小，PAOR 循环里 LLM reasoning 会自己纠错

**用 LLM（准）的情况**：
1. **长查询（>40 字）且无明确关键词** → 可能多重意图，规则引擎会漏
2. **多句查询（>=2 个问号/分号）且无明确关键词** → 复合问题
3. **中等长度（16-40 字）且无明确关键词** → 规则引擎会误判为 general

### 3.3 条件预检索

前置 Plan 完成后，按 `plan.intent` 决定是否预检索：

| intent | 预检索? | 理由 |
|--------|---------|------|
| `general` | ❌ 不预检索 | 简单查询让 LLM 自主决定，避免增加首轮延迟 |
| `compare` | ✅ 预检索 | 复杂查询提前给 LLM 证据，减少工具调用轮次 |
| `summary` | ✅ 预检索 | 同上 |
| `clause` | ✅ 预检索 | 同上 |
| `verification` | ✅ 预检索 | 同上 |

预检索结果处理：
- `context` → 注入 LLM 首轮 prompt（让 LLM 首轮就能看到证据）
- `evidence` → 写入 `tool_execution_context.retrieval_context.collected_evidence`（供 Agent 阶段三回收）
- `chunk_id` → 加入 `seen_chunk_ids`（跨轮次去重，避免 PAOR 工具调用重复返回）

### 3.4 三级降级策略

1. **QueryPlanner 失败**（LLM 超时等）→ 降级到无 Plan 的 PAOR 循环（当前行为），日志 warning
2. **预检索失败**（Qdrant/RediSearch 异常）→ 降级到只有 Plan 提示无预检索证据，日志 warning
3. **Plan 成功但 intent=general** → 不预检索，只把 plan 的参数提示注入 LLM

### 3.5 Plan 参数复用

LLM 在 PAOR 循环里调 `rag_retrieve` 工具时，工具会从 `tool_execution_context["query_plan"]` 读取前置 plan，传给 `rag_service.retrieve_context(plan=plan)`，复用：
- `rewritten_queries`（多路召回改写）
- `sub_queries`（查询分解）
- `final_k` / `prefetch_k`（检索参数）
- `fusion_strategy`（融合策略）
- `need_graph`（是否启用图谱）

这样避免了 LLM 每次调工具都重新规划一次，也保证了对话路径和检索 API 路径的 plan 行为一致。

## 四、完整调用链

```
chat.py → GeneralAssistantAgent.execute()
  ↓
[新增] 前置参数规划：query_planner.build_plan(planner_mode=auto)
  ├─ auto 决策：短查询/命中关键词 → 规则引擎（<1ms）
  └─ auto 决策：长查询/复杂结构 → LLM（~1-3s）
  ↓ plan
[新增] 注入 tool_execution_context["query_plan"]
  ↓
[新增] 条件预检索（intent != "general" 才做）：
  - rag_service.retrieve_context(plan=plan)  ← 复用 plan，避免重复规划
  - context → LLM 首轮 prompt
  - evidence → collected_evidence（供阶段三回收）
  - chunk_id → seen_chunk_ids（跨轮次去重）
  ↓
进入 PAOR 循环（LLMService._generate_stream）
  ↓
LLM 首轮看到：规划提示 + 预检索证据（如果有）
  ↓
LLM 自主决策：直接回答 or 调 rag_retrieve 工具
  ↓
如果调 rag_retrieve：
  - 从 tool_execution_context["query_plan"] 读取前置 plan
  - 传给 ai_tools.rag_retrieve_with_context(plan=plan)
  - 复用 plan 的 rewritten_queries/final_k/fusion_strategy
```

## 五、修改文件清单

### 5.1 `services/query_planner.py`

**修改点**：`build_plan` 方法签名和实现

- 旧签名：`build_plan(query, runtime_modules, runtime_params, filters)` + `fast_mode: bool = False`
- 新签名：`build_plan(query, runtime_modules, runtime_params, filters, planner_mode: str = "auto")`

新增 `_query_needs_llm(query)` 方法，auto 模式的决策函数。

**四种模式**：
- `auto`：智能选择（默认）
- `rules`：强制规则引擎
- `llm`：强制 LLM
- `env`：按 `QUERY_PLANNER_MODE` 环境变量走（兼容 retrieval.py 旧用法）

### 5.2 `agents/general_assistant/general_assistant_agent.py`

**修改点**：`execute` 方法加入前置 Plan + 条件预检索

- 读 `AGENT_PLANNER_MODE` 环境变量（默认 `auto`）
- 调用 `query_planner.build_plan(planner_mode=agent_planner_mode)`
- 把 plan 注入 `tool_execution_context["query_plan"]`
- 按 `plan.intent` 决定是否预检索
- 把 plan 提示和预检索证据注入 LLM 首轮 context

### 5.3 `services/rag_service.py`

**修改点**：`retrieve_context` 新增 `plan` 参数

- 旧签名：`retrieve_context(query, ..., exclude_chunk_ids)`
- 新签名：`retrieve_context(query, ..., exclude_chunk_ids, plan: Optional[Any] = None)`

如果调用方传入 plan（如对话路径前置规划），直接复用，跳过内部 `build_plan`，避免重复调用 LLM。

retrieval.py API 路径（不传 plan）显式传 `planner_mode="env"`，保持原行为（按 `QUERY_PLANNER_MODE` 走）。

### 5.4 `services/ai_tools.py`

**修改点**：`rag_retrieve_with_context` 新增 `plan` 参数

- 旧签名：`rag_retrieve_with_context(query, ..., exclude_chunk_ids)`
- 新签名：`rag_retrieve_with_context(query, ..., exclude_chunk_ids, plan: Optional[Any] = None)`

透传给 `rag_service.retrieve_context(plan=plan)`。

### 5.5 `services/llm_service.py`

**修改点**：rag_retrieve 工具调用时从 `tool_execution_context` 读取 plan

在 `_generate_stream` 的 Act 阶段，调用 `rag_retrieve_with_context` 时：

```python
# 从 tool_execution_context 读取前置 plan（对话路径注入），让工具复用 plan 的参数
tool_plan = tool_execution_context.get("query_plan")
result = await ai_tools.rag_retrieve_with_context(
    query=params.get("query", ""),
    ...
    plan=tool_plan,
)
```

## 六、配置说明

### 6.1 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `AGENT_PLANNER_MODE` | `auto` | 对话路径规划模式：auto/rules/llm |
| `QUERY_PLANNER_MODE` | `llm` | retrieval.py API 路径规划模式（旧变量，保留兼容）|
| `PLANNER_TIMEOUT` | `20.0` | LLM 规划超时（秒），超时后 fallback 到规则引擎 |
| `PLANNER_MODEL` | 同 `LLM_MODEL` | 规划器用的 LLM 模型 |

### 6.2 推荐配置

**生产环境**（默认，平衡速度和准确性）：
```bash
AGENT_PLANNER_MODE=auto
```

**延迟敏感场景**（如实时聊天）：
```bash
AGENT_PLANNER_MODE=rules
```

**准确性优先场景**（如知识库问答）：
```bash
AGENT_PLANNER_MODE=llm
```

## 七、测试结果

### 7.1 auto 模式决策逻辑测试

14 个测试用例全部通过：

| query | 长度 | 决策 | 理由 |
|-------|------|------|------|
| `你好` | 2 | 规则 | 极短查询 |
| `对比 A 和 B 的差异` | 12 | 规则 | 命中"对比""差异" |
| `有哪些优点` | 5 | 规则 | 命中"有哪些" |
| `这个产品的定义是什么` | 10 | 规则 | 命中"定义" |
| `列举关键点` | 5 | 规则 | 命中"列举""关键点" |
| `A 方案和 B 方案有什么区别` | 15 | 规则 | 命中"区别" |
| `请详细分析这两个方案在性能、成本、可维护性方面的差异，并给出推荐` | 32 | 规则 | 命中"差异"（即使长查询）|
| `为什么这个设计这样` | 9 | 规则 | 极短查询 |
| `请帮我看看这个问题怎么解决` | 13 | 规则 | 极短查询 |
| `在考虑是否采用微服务架构时...风险？...限制？` | 40 | 规则 | 命中"风险""限制" |
| `请分析这个系统的架构设计是否合理` | 16 | **LLM** | 16-40 字无明确关键词 |
| `这个方案的性能怎么样` | 10 | 规则 | 极短查询 |
| `帮我总结一下这个项目的核心要点和主要结论` | 20 | 规则 | 命中"总结""要点""结论" |
| `如何优化数据库查询性能` | 11 | 规则 | 极短查询 |

### 7.2 关键设计验证

- ✅ 关键词命中优先于长度判断（长查询+关键词也用规则引擎）
- ✅ 极短查询即使无关键词也用规则引擎（PAOR 里 LLM reasoning 会纠错）
- ✅ 中等长度无关键词用 LLM（避免规则引擎误判为 general）
- ✅ LLM 失败自动 fallback 到规则引擎（原有逻辑保留）

## 八、与 PAOR 循环的协同

### 8.1 两个 Plan 的职责区分

| 层级 | 谁做 | 产出什么 | 频率 |
|------|------|----------|------|
| 前置参数规划（`build_plan`）| 规则引擎或 LLM | `final_k`/`prefetch_k`/`fusion_strategy`/`intent` | 每次查询 1 次 |
| PAOR 的 P（LLM reasoning）| LLM | "要不要调 rag_retrieve、用什么 query、top_k 设多少" | 每轮 PAOR 1 次 |

### 8.2 PAOR 循环完整流程（更新后）

**Plan 阶段**（LLM reasoning）：
- 用户消息 + `tool_execution_context` 一起传入 LLM
- LLM 收到 messages 历史、工具 JSON Schema、三段系统提示词
- LLM 自主思考要不要调工具、调哪个、传什么参数
- `reasoning_content` 就是 Plan 阶段的思考链产物

**Act 阶段**：
- LLM 返回 `tool_call` 后立即执行
- 如果是 `rag_retrieve`，从 `tool_execution_context` 读取：
  - `seen_chunk_ids`（跨轮次去重）
  - `query_plan`（前置 plan，复用参数）
- 检查 `max_retrievals=5` 上限

**Observe 阶段**：
- 从 `rag_retrieve` 返回结果提取所有 chunk
- `EvidenceVerifier` 验证相关性（LLM 模式或规则 fallback）
- 验证后证据写入 `collected_evidence`
- 构造 `Observation` 记录追加到 `observations`

**Reflect 阶段**：
- `Reflector` 判断 `verified_count >= 2` 且 `top_score >= 0.3`
- 满足则 `sufficient=True`，不满足生成 `gaps` 和 `next_query`
- 反思结论文本化注入下一轮 messages（`⚠️ 检索反思：...`）
- 硬安全阀：`total_retrieval_count >= max_retrievals` 强制 `next_action="answer"`

**闭环**：
- Observe 写入全局上下文
- Reflect 文本化注入 messages
- LLM 下一轮 Plan 时读到反思文本，动态调整策略

## 九、回滚方案

如需回滚到原行为（无前置 Plan），两种方式：

**方式 1：环境变量**（推荐，无需改代码）
```bash
AGENT_PLANNER_MODE=rules
```
然后注释掉 `general_assistant_agent.py` 的预检索代码块（第 167-213 行）。

**方式 2：代码回滚**

还原以下 4 个文件：
- `services/query_planner.py`：移除 `planner_mode` 参数和 `_query_needs_llm` 方法
- `agents/general_assistant/general_assistant_agent.py`：移除前置 Plan 代码块（第 121-222 行）
- `services/rag_service.py`：移除 `plan` 参数
- `services/ai_tools.py`：移除 `plan` 参数
- `services/llm_service.py`：移除 `tool_plan` 读取逻辑

## 十、后续优化方向

1. **缓存 Plan 结果**：相同 query 短时间内复用 plan，避免重复规划
2. **Plan 质量监控**：记录 plan 的 intent 分布、预检索命中率、PAOR 轮次减少比例
3. **A/B 测试**：对比 auto/rules/llm 三种模式在真实流量下的效果
4. **动态阈值**：`_query_needs_llm` 的长度阈值（15/40 字）可根据实际 query 分布调整
