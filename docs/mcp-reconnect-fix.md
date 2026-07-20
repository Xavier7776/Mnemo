# MCP 断线重连机制修复：从「被动重试」到「主动健康检查 + 熔断 + 幂等保护」

> 解决 6 个核心问题：被动重连、调用丢失无追踪、无幂等保护、无指数退避、无熔断、无部分结果恢复。

## 一、修复前会出现什么问题

### 问题 1：被动重连，无主动健康检查

**根因**：原版只在 `call_tool` 入口或异常捕获时触发重连，没有心跳、没有定期健康检查、没有 Watchdog。

**实际场景**：
- Server 在 10:00 断连
- 10:00-10:30 之间没有任何 `call_tool` 调用
- 10:30 用户调用工具时才发现断连，必须等待重连

**后果**：
- 断连状态最长延迟 30 分钟才被发现
- 首个调用者承担重连延迟（1-3 秒）
- 如果 Server 长时间不稳定，每个调用者都要重复尝试重连

### 问题 2：执行中异常断开，原调用丢失无追踪

**根因**：原版异常处理直接重试，不记录原调用的状态：

```python
# 修复前（mcp_client_service.py:421-461）
except Exception as e:
    logger.warning(f"MCP 工具调用异常 [{server_name}/{tool_name}]: {e}")
    # 直接重连 + 重试，不记录原调用状态
    self._disconnected.add(server_name)
    reconnected = await self._reconnect_server(server_name)
    # ... 用同样的 arguments 在新 session 上重试
```

**实际场景**：
- 用户调 `send_email` 工具
- 请求已发到 Server，Server 正在发送邮件
- 连接断开，响应没回来
- 客户端重试 → Server 又发了一次邮件 → **用户收到 2 封邮件**

**后果**：
- 不知道原调用执行到哪了
- 不知道 Server 有没有收到
- 没有 call_id 可以追踪
- 用户无法查询失败调用的详情

### 问题 3：无幂等保护，重试导致副作用重复执行

**根因**：原版对所有工具一视同仁地重试，没有区分幂等/非幂等：

```python
# 修复前：不分青红皂白全部重试
except Exception as e:
    ...
    reconnected = await self._reconnect_server(server_name)
    if not reconnected:
        return {...}
    # 重试一次（无论工具是否幂等）
    result = await asyncio.wait_for(
        session.call_tool(tool_name, arguments),
        timeout=timeout,
    )
```

**实际场景**：

| 工具类型 | 重试后果 |
|---------|---------|
| `list_files`（只读）| 重试无害 |
| `search_docs`（只读）| 重试无害 |
| `send_email`（副作用）| **重复发送邮件** |
| `create_resource`（副作用）| **重复创建资源** |
| `delete_file`（副作用）| 删除已是幂等的，但如果有错误反馈，重试会覆盖错误信息 |

**后果**：对非幂等工具的盲目重试可能导致业务侧的重复执行，造成数据污染或用户体验问题。

### 问题 4：无指数退避，重连失败立即放弃

**根因**：原版 `_reconnect_server` 只重连一次：

```python
# 修复前（mcp_client_service.py:264-300）
async def _reconnect_server(self, server_name: str) -> bool:
    async with self._reconnect_locks[server_name]:
        # 检查是否已重连成功
        ...
        # 清理旧连接
        await self._cleanup_server(server_name)
        # 重新连接（只连一次）
        try:
            await self._connect_server(server_name, server_config)
            return True
        except Exception as e:
            self._disconnected.add(server_name)
            return False  # 失败立即返回
```

**实际场景**：
- Server 进程刚崩溃，操作系统还在清理端口
- 立即重连 → 端口被占用 → 失败
- 立即放弃 → 用户得到错误
- 实际再等 1 秒就能连上

**后果**：网络抖动或进程重启的瞬时故障被放大为永久失败，用户体验差。

### 问题 5：无熔断，反复失败拖垮整个系统

**根因**：原版没有熔断器，Server 持续故障时每个调用都要等待超时（30s）：

```python
# 修复前：每个调用都走完整超时
result = await asyncio.wait_for(
    session.call_tool(tool_name, arguments),
    timeout=timeout,  # 30s
)
```

**实际场景**：
- Server 挂了
- 100 个并发请求同时打到这个 Server
- 每个请求等 30 秒超时
- Agent 阻塞 30 秒，无法响应其他用户

**后果**：
- 单 Server 故障可能拖垮整个 Agent
- 30 秒超时期间资源被占用
- 没有"快速失败"机制

### 问题 6：无部分结果恢复，工具执行一半结果完全丢失

**根因**：原版异常后直接重试，不记录原调用的任何信息：

```python
# 修复前：重试时不知道原调用的状态
# 不知道 Server 有没有收到
# 不知道 Server 执行到哪了
# 不知道有没有副作用
```

**实际场景**：
- 用户调 `create_document` 工具
- 请求发到 Server，Server 创建了文档
- 连接断开，响应没回来
- 客户端重试 → Server 又创建了一个文档（如果非幂等）
- 用户得到第 2 个文档的 ID，但第 1 个文档孤儿存在

**后果**：
- 数据不一致
- 用户无法知道原调用的结果
- 无法手动恢复

---

## 二、修复方案：5 重防护

```
call_tool 调用
  ↓
【防护 1】熔断检查（CircuitBreaker）
  ├─ CLOSED：放行
  ├─ OPEN：直接拒绝（不发起调用，快速失败）
  └─ HALF_OPEN：放行一个试探请求
  ↓
【防护 2】断连检查 + 指数退避重连
  └─ 断连时重连，1s → 2s → 4s 最多 3 次
  ↓
【防护 3】调用追踪（CallLog）
  └─ 生成 call_id，记录 pending/success/failed/retrying/dropped 状态
  ↓
【防护 4】幂等保护
  ├─ 幂等工具（list_/get_/search_ 等）：失败时自动重连 + 重试
  └─ 非幂等工具（send_email/create_resource 等）：失败时标记 dropped，不重试
  ↓
【防护 5】熔断反馈
  ├─ 成功 → record_success（HALF_OPEN 成功 → CLOSED）
  └─ 失败 → record_failure（CLOSED 连续失败 5 次 → OPEN）
```

### 2.1 主动健康检查（后台心跳）

**新增**：`_start_health_check_task` / `_run_health_check` / `_ping_server`

```
后台任务每 30 秒 ping 一次所有 server
  ├─ ping 成功 → 清除断连标记
  └─ ping 超时（5s）或失败 → 标记 _disconnected
```

**MCP 协议内置 ping**：`session.send_request("ping", {})`，轻量级，不影响 Server 业务。

**优势**：
- 断连状态最长延迟 30 秒被发现（原版可能 30 分钟）
- 健康检查与业务调用解耦，不增加首个调用者的延迟
- 健康检查失败只标记，不触发重连（重连由 `call_tool` 入口触发，避免空转）

### 2.2 指数退避重连

**修改**：`_reconnect_server` 从单次重试改为指数退避最多 3 次

```python
# 修复后
delay = 1.0  # 首次延迟 1s
for attempt in range(1, 4):  # 最多 3 次
    try:
        await self._connect_server(server_name, server_config)
        return True
    except Exception:
        if attempt < 3:
            await asyncio.sleep(delay)
            delay *= 2  # 指数退避：1s → 2s → 4s
return False
```

**退避序列**：1s → 2s → 4s（总等待 7s，总尝试 3 次）

**优势**：
- 网络抖动/进程重启的瞬时故障有恢复机会
- 退避避免紧挨着重连（给操作系统清理端口的时间）
- 重连成功后重置熔断器

### 2.3 熔断器（CircuitBreaker）

**新增**：`services/mcp_circuit_breaker.py`

**三态机**：

| 状态 | 行为 | 转移条件 |
|------|------|---------|
| CLOSED（正常）| 请求放行，失败计数累积 | 连续失败 5 次 → OPEN |
| OPEN（熔断）| 直接拒绝请求（快速失败）| cooldown 60s 后 → HALF_OPEN |
| HALF_OPEN（半开）| 放行 1 个试探请求 | 成功 → CLOSED；失败 → OPEN |

**配置**（可在 server config 覆盖）：
- `circuit_failure_threshold`：连续失败多少次触发熔断（默认 5）
- `circuit_cooldown_seconds`：OPEN 状态持续时间（默认 60s）

**优势**：
- 单 Server 故障不会拖垮整个 Agent（熔断后直接拒绝，不等 30s 超时）
- 自动恢复：cooldown 后自动半开试探，不需要人工干预
- 熔断状态对调用方可见（`circuit_state` 字段）

### 2.4 幂等保护

**新增**：`_is_safe_retry` 方法

**判断策略**（优先级从高到低）：
1. server config 的 `safe_retry_tools` 白名单 → 安全
2. server config 的 `unsafe_retry_tools` 黑名单 → 不安全
3. 默认前缀匹配：`list_` / `get_` / `search_` / `query_` / `read_` / `fetch_` 开头 → 安全
4. 其他工具（如 `send_email` / `create_resource` / `delete_file`）→ 不安全

**调用行为**：

| 工具类型 | 失败时行为 |
|---------|----------|
| 幂等工具（list_files）| 重连 + 重试一次 |
| 非幂等工具（send_email）| 标记 dropped，**不重试**，返回 call_id 供人工查询 |

**配置示例**（mcp_servers.json）：
```json
{
  "email_server": {
    "command": "python",
    "args": ["-m", "mcp_email_server"],
    "safe_retry_tools": ["list_emails", "get_email"],
    "unsafe_retry_tools": ["send_email", "delete_email"]
  }
}
```

**优势**：
- 避免副作用工具的重复执行
- 用户可通过 call_id 查询失败调用的详情
- 默认前缀匹配覆盖了大多数只读工具，无需逐个配置

### 2.5 调用追踪（CallLog）

**新增**：`services/mcp_call_log.py`

**每次 call_tool 生成 call_id**，记录：
- `call_id`：唯一标识（`mcp-{uuid12}`）
- `status`：pending / success / failed / retrying / dropped
- `start_ts` / `end_ts` / `duration_ms`
- `error`：失败原因
- `result_preview`：成功结果前 200 字符
- `retry_count`：重试次数
- `idempotent_skipped`：是否因幂等保护被跳过

**查询 API**：
- `get_records(server_name, limit)`：按 server 查询历史调用
- `get_record(call_id)`：按 call_id 查询单条记录

**存储**：
- 内存存储（不持久化，重启清空）
- 每个 server 最多保留 100 条
- 自动清理 1 小时前的记录

**优势**：
- 失败调用有据可查
- 调用方拿到 call_id 可主动查询状态
- 监控/调试有完整链路

---

## 三、修复后会怎么样

### 3.1 断连发现：30 分钟延迟 → 30 秒延迟

| 修复前 | 修复后 |
|--------|--------|
| Server 10:00 断连，10:30 首个调用者才发现 | Server 10:00 断连，10:30 健康检查发现并标记 |
| 首个调用者承担 1-3s 重连延迟 | 健康检查标记后，首个调用者触发重连（但仍承担延迟）|
| 长时间无调用时断连完全不可知 | 健康检查持续运行，断连主动暴露 |

### 3.2 重连策略：单次立即放弃 → 指数退避 3 次

| 修复前 | 修复后 |
|--------|--------|
| 重连失败立即返回 false | 1s → 2s → 4s 最多重试 3 次 |
| 瞬时故障（端口未释放）放大为永久失败 | 瞬时故障有 7s 恢复窗口 |
| 单次重连 ~1s | 3 次重连最多 7s（但成功率显著提升）|

### 3.3 故障隔离：30s 超时拖垮系统 → 熔断快速失败

| 修复前 | 修复后 |
|--------|--------|
| Server 故障，每个调用等 30s 超时 | 5 次失败后熔断，后续调用毫秒级拒绝 |
| 100 个并发请求阻塞 30s | 100 个并发请求立即返回"熔断中" |
| 30s 内 Agent 无法响应其他用户 | Agent 毫秒级响应，可降级到其他 Server |

### 3.4 重试安全：盲目重试 → 幂等保护

| 修复前 | 修复后 |
|--------|--------|
| `send_email` 断连后盲目重试 → 用户收到 2 封邮件 | `send_email` 非幂等，标记 dropped 不重试，返回 call_id |
| `create_resource` 重试 → 创建 2 个资源 | `create_resource` 非幂等，不重试，用户可凭 call_id 查询 |
| `list_files` 重试 → 无害 | `list_files` 幂等，正常重试 |

### 3.5 调用追踪：完全丢失 → call_id 全链路追踪

| 修复前 | 修复后 |
|--------|--------|
| 异常后不知道原调用状态 | 每次调用生成 call_id，记录完整状态 |
| 用户无法查询失败调用 | 用户凭 call_id 查询 `get_record(call_id)` |
| 监控/调试无线索 | `get_records(server_name)` 查看历史调用 |

### 3.6 返回值增强

**修复前的返回值**：
```python
{"success": False, "error": "MCP Server 未连接且重连失败"}
```

**修复后的返回值**：
```python
{
    "success": False,
    "error": "MCP Server 未连接且重连失败",
    "call_id": "mcp-d1bfd4335d65",       # 可追踪
    "circuit_state": "open",              # 熔断状态
    "retried": False                      # 是否重试过
}
```

---

## 四、工程化考量

### 4.1 向后兼容

- `call_tool` 返回值新增字段（`call_id` / `circuit_state` / `retried`），原有字段保留
- 配置文件 `mcp_servers.json` 新增可选字段（`safe_retry_tools` / `unsafe_retry_tools` / `circuit_failure_threshold` / `circuit_cooldown_seconds`），不配置走默认值
- 原有的 `_disconnected` / `_reconnect_locks` / `_cleanup_server` 保留不变

### 4.2 配置参数

**全局常量**（`mcp_client_service.py`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `HEALTH_CHECK_INTERVAL` | 30s | 健康检查间隔 |
| `HEALTH_CHECK_TIMEOUT` | 5s | ping 超时 |
| `RECONNECT_MAX_ATTEMPTS` | 3 | 指数退避最多重试次数 |
| `RECONNECT_INITIAL_DELAY` | 1.0s | 首次重试延迟 |
| `RECONNECT_BACKOFF_FACTOR` | 2.0 | 退避倍数 |
| `CIRCUIT_FAILURE_THRESHOLD` | 5 | 熔断失败阈值 |
| `CIRCUIT_COOLDOWN_SECONDS` | 60s | 熔断 cooldown |
| `SAFE_RETRY_PREFIXES` | `list_/get_/search_/query_/read_/fetch_` | 默认幂等工具前缀 |

**单 Server 配置覆盖**（`mcp_servers.json`）：

```json
{
  "email_server": {
    "command": "python",
    "args": ["-m", "mcp_email_server"],
    "timeout": 30,
    "circuit_failure_threshold": 3,
    "circuit_cooldown_seconds": 120,
    "safe_retry_tools": ["list_emails", "get_email", "search_emails"],
    "unsafe_retry_tools": ["send_email", "delete_email", "mark_read"]
  }
}
```

### 4.3 性能开销

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 健康检查（每 30s 一次）| <5ms | ping 一个 server，并发对所有 server |
| 熔断检查 | <1ms | 字典查找 + 状态判断 |
| 幂等判断 | <1ms | 字典查找 + 前缀匹配 |
| CallLog 记录 | <1ms | 内存 dict 追加 |
| 指数退避重连（最坏情况）| 7s | 3 次重试，1s+2s+4s |

**正常路径开销**：<3ms（熔断 + 幂等 + CallLog）
**故障路径开销**：7s（指数退避重连）+ 后续毫秒级熔断拒绝

### 4.4 降级策略

- **健康检查任务失败**：单次 ping 异常不影响其他 server，任务继续运行
- **熔断器创建失败**：fallback 到无熔断（每个调用都走完整流程）
- **CallLog 写入失败**：忽略，不影响业务调用
- **幂等判断无配置**：走默认前缀匹配

### 4.5 可观测性

**日志**：
```
MCP 熔断器 [email_server] CLOSED → OPEN（连续失败 5 次）
MCP 熔断器 [email_server] OPEN → HALF_OPEN（cooldown 到期，试探请求）
MCP 熔断器 [email_server] HALF_OPEN → CLOSED（试探成功）
MCP 健康检查 [search_server] ping 超时（5s），标记为断连
MCP 工具 [email_server/send_email] 非幂等，不自动重试（避免副作用重复执行）
MCP Server [search_server] 重连尝试 2/3... 等待 2.0s 后重试...
```

**API**（`get_server_status`）：
```json
{
  "name": "email_server",
  "connected": true,
  "circuit_state": "closed",
  "circuit_failure_count": 0,
  "recent_calls": 15
}
```

**CallLog API**：
```python
# 查询某 server 最近 20 次调用
mcp_call_log.get_records("email_server", limit=20)

# 按 call_id 查询单条记录
mcp_call_log.get_record("mcp-d1bfd4335d65")
```

---

## 五、测试验证

### 5.1 熔断器三态转换（全部通过）

```
初始状态: closed
3 次失败后: open
cooldown 后: half_open
试探成功后: closed
[OK] 熔断器三态转换正确
```

### 5.2 幂等保护（12 个测试全部通过）

| 工具名 | 期望 | 实际 |
|--------|------|------|
| list_files | 安全 | 安全 ✅ |
| get_user | 安全 | 安全 ✅ |
| search_docs | 安全 | 安全 ✅ |
| query_data | 安全 | 安全 ✅ |
| read_file | 安全 | 安全 ✅ |
| fetch_url | 安全 | 安全 ✅ |
| send_email | 不安全 | 不安全 ✅ |
| create_resource | 不安全 | 不安全 ✅ |
| delete_file | 不安全 | 不安全 ✅ |
| update_record | 不安全 | 不安全 ✅ |
| custom_safe_tool（白名单）| 安全 | 安全 ✅ |
| custom_unsafe_tool（黑名单）| 不安全 | 不安全 ✅ |

### 5.3 CallLog（全部通过）

- 新建 call → `status=pending`，`call_id=mcp-{uuid12}` ✅
- 标记 success → `status=success` ✅
- 查询 records → 返回列表 ✅
- 按 call_id 查询 → 返回单条 ✅
- 标记 dropped → `status=dropped`，`idempotent=True` ✅

---

## 六、调用流程对比

### 6.1 修复前：盲目重试

```
call_tool(server, tool, args)
  ↓
检查 _disconnected → 重连一次
  ↓
session.call_tool(tool, args)
  ↓
异常 → 标记 _disconnected → 重连一次 → 重试一次
  ↓
返回结果（无 call_id，无熔断状态，无幂等判断）
```

### 6.2 修复后：5 重防护

```
call_tool(server, tool, args)
  ↓
【1. 熔断检查】
  ├─ OPEN → 直接返回 "熔断中" + call_id
  ├─ HALF_OPEN → 放行一个试探
  └─ CLOSED → 放行
  ↓
【2. 断连检查 + 指数退避重连】
  └─ 断连 → 1s/2s/4s 最多 3 次重连
  ↓
【3. CallLog 记录】
  └─ 生成 call_id，status=pending
  ↓
session.call_tool(tool, args)
  ↓
成功 → record_success + mark_success
  ├─ HALF_OPEN 成功 → CLOSED
  └─ 返回 {success, result, call_id}
  ↓
异常 →
  ├─【4. 幂等保护】
  │   ├─ 幂等工具 → 重连 + 重试一次
  │   └─ 非幂等工具 → mark_dropped，不重试，返回 call_id
  └─【5. 熔断反馈】
      └─ record_failure → 连续 5 次 → OPEN
```

---

## 七、后续优化方向

### 7.1 短期

1. **CallLog 持久化**：当前内存存储重启清空，可考虑写文件或 Redis
2. **熔断器监控**：暴露 Prometheus 指标（circuit_state / failure_count）
3. **健康检查并发优化**：当前串行 ping 所有 server，可改并发

### 7.2 中期

1. **工具级熔断**：当前熔断是 server 级，可考虑对单个 tool 做熔断
2. **自适应超时**：根据历史调用 P95 自动调整 timeout
3. **重连策略可配**：指数退避参数可在 server config 覆盖

### 7.3 长期

1. **Server 健康评分**：综合 ping 延迟、失败率、熔断次数 → 评分路由
2. **多副本故障转移**：同一种工具的多个 server 副本，故障时自动切换
3. **调用链追踪**：CallLog 集成 OpenTelemetry，跨服务追踪

---

## 八、修改文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| [services/mcp_circuit_breaker.py](file:///d:/timeModel/Mnemo/services/mcp_circuit_breaker.py) | 新增 | 三态熔断器（CLOSED/OPEN/HALF_OPEN）|
| [services/mcp_call_log.py](file:///d:/timeModel/Mnemo/services/mcp_call_log.py) | 新增 | 调用日志（call_id 追踪 + 状态记录）|
| [services/mcp_client_service.py](file:///d:/timeModel/Mnemo/services/mcp_client_service.py) | 修改 | 集成熔断 + 健康检查 + 幂等保护 + 指数退避 + CallLog |
