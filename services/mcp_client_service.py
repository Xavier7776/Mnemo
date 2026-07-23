"""MCP Client 服务 — 连接外部 MCP Server，扩展 Agent 可用工具

阶段二增强（v2 修复版）：
- 主动健康检查：后台心跳任务定期 ping Server，断连主动发现
- 断线重连：指数退避重试（1s/2s/4s，最多 3 次）
- 熔断保护：连续失败 N 次触发熔断，cooldown 后半开试探
- 幂等保护：非安全工具不自动重试，避免重复执行副作用
- 调用追踪：每次 call_tool 生成 call_id，记录状态供查询
- 热加载：运行时动态增删改 Server 配置，无需重启
- 工具按需加载：工具数量超过阈值时，只注入摘要，LLM 通过 mcp_list_tools 发现
- 配置持久化：增删改 Server 时自动写回 mcp_servers.json

设计要点：
- 对现有工具调用链路零侵入：MCP 工具注册为 ai_tools 后，LLM 和 Agent 无感知
- 工具命名：mcp__{server_name}__{tool_name}，避免与内置工具冲突
- async wrapper：每个 MCP 工具生成一个 async 闭包，注册到 ai_tools._async_tools
"""
import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from utils.logger import logger
from services.mcp_circuit_breaker import CircuitBreaker
from services.mcp_call_log import mcp_call_log, CallRecord


# 工具按需加载的阈值：超过此数量时启用 compact 模式（按需加载）
# v6: 用户指定阈值 30，低于此值走全量注入，高于此值走 ToolIndex 按需加载
# v6.2: 阈值改为 100。实测 30 以下工具全量注入仍会因 ToolIndex 索引构建拖慢启动，
#       100 以内的 schema 总 token 量在主流模型 context window 里仍可承受。
TOOL_COUNT_THRESHOLD = 100

# 健康检查配置
HEALTH_CHECK_INTERVAL = 30.0  # 每 30 秒 ping 一次
HEALTH_CHECK_TIMEOUT = 5.0    # ping 超时 5 秒视为断连

# 重连配置
RECONNECT_MAX_ATTEMPTS = 3    # 指数退避最多重试 3 次
RECONNECT_INITIAL_DELAY = 1.0 # 首次重试延迟 1 秒
RECONNECT_BACKOFF_FACTOR = 2.0  # 每次延迟翻倍

# 熔断配置
CIRCUIT_FAILURE_THRESHOLD = 5  # 连续失败 5 次触发熔断
CIRCUIT_COOLDOWN_SECONDS = 60.0  # 熔断 60 秒后进入半开

# 默认幂等性：以下前缀的工具视为可安全重试
SAFE_RETRY_PREFIXES = ("list_", "get_", "search_", "query_", "read_", "fetch_")


# ==================== v3 新增：模型 context window 自适应阈值 ====================

# 常见 LLM 模型的 context window（按 tokens 计，model_name 小写前缀匹配）
# 未列出的模型按 DEFAULT_CONTEXT_WINDOW 兜底
MODEL_CONTEXT_WINDOW: Dict[str, int] = {
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_384,
    # Anthropic
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3.5-sonnet": 200_000,
    # 国内模型
    "mimo-v2.5": 32_768,
    "deepseek-chat": 32_768,
    "deepseek-coder": 16_384,
    "qwen2.5": 32_768,
    "qwen2": 32_768,
    "glm-4": 128_000,
    # 开源模型
    "llama3.1": 128_000,
    "llama3": 8_192,
}

DEFAULT_CONTEXT_WINDOW = 16_384  # 未识别模型的默认 context window


def get_model_context_window(model_name: str) -> int:
    """根据模型名查询 context window 大小（精确 → 前缀模糊匹配 → 默认值）

    Args:
        model_name: LLM 模型名（如 "mimo-v2.5"、"gpt-4o-2024-08-06"）

    Returns:
        context window 大小（tokens）
    """
    if not model_name:
        return DEFAULT_CONTEXT_WINDOW

    model_lower = model_name.lower()

    # 精确匹配
    if model_lower in MODEL_CONTEXT_WINDOW:
        return MODEL_CONTEXT_WINDOW[model_lower]

    # 前缀匹配（model_name 可能带版本号或日期后缀）
    for prefix, ctx in MODEL_CONTEXT_WINDOW.items():
        if model_lower.startswith(prefix):
            return ctx

    return DEFAULT_CONTEXT_WINDOW


def get_adaptive_tool_threshold(model_name: str) -> int:
    """根据 LLM 模型的 context window 自适应计算工具数量阈值

    阈值映射逻辑（基于实验5的拐点验证 + token 占用推算）：
    - ≤ 8K  context: prompt 预算紧张，阈值 15（早启用 compact 省 token）
    - ≤ 16K context: 阈值 18
    - ≤ 32K context: 阈值 22
    - ≤ 64K context: 阈值 28
    - ≤ 128K context: 阈值 35
    - > 128K context: 阈值 40（大窗口晚启用，减少 mcp_list_tools 调用）

    推算依据：单个 MCP 工具 schema 平均 200~300 tokens，compact summary 每工具 30~50 tokens。
    context window 越大，能容纳的 schema 越多，越晚启用 compact mode 越划算（避免 LLM 多一次
    list_tools 调用带来的延迟和误判）。context window 越小，越早启用 compact mode 节省 token，
    防止工具 schema 挤占 RAG context 空间。

    Args:
        model_name: LLM 模型名

    Returns:
        该模型对应的工具数量阈值
    """
    ctx = get_model_context_window(model_name)

    if ctx <= 8_192:
        return 15
    elif ctx <= 16_384:
        return 18
    elif ctx <= 32_768:
        return 22
    elif ctx <= 65_536:
        return 28
    elif ctx <= 131_072:
        return 35
    else:
        return 40


class MCPClientManager:
    """MCP Client 管理器：多 Server 连接、工具路由、生命周期、热加载"""

    def __init__(self):
        # server_name → ClientSession
        self._sessions: Dict[str, Any] = {}
        # server_name → transport 上下文（用于关闭）
        self._transports: Dict[str, Any] = {}
        # server_name → tool list（原始 MCP Tool 对象）
        self._tools: Dict[str, List[Any]] = {}
        # server_name → 配置（timeout 等）
        self._server_configs: Dict[str, dict] = {}
        # 全局配置
        self._config: dict = {"enabled": False, "servers": {}}
        # 是否已初始化
        self._initialized = False
        # 配置文件路径（用于热加载时持久化）
        self._config_path: Optional[str] = None
        # 断连的 Server 集合（用于健康检查）
        self._disconnected: Set[str] = set()
        # 重连锁（避免并发重连同一 Server）
        self._reconnect_locks: Dict[str, asyncio.Lock] = {}
        # 已注册到 ai_tools 的工具名集合（用于热更新时清理旧工具）
        self._registered_tool_names: Set[str] = set()
        # v2 新增：熔断器（server_name → CircuitBreaker）
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        # v2 新增：健康检查任务
        self._health_check_task: Optional[asyncio.Task] = None
        # v2 新增：是否正在关闭（避免关闭时健康检查任务报错）
        self._shutting_down = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def is_enabled(self) -> bool:
        return self._config.get("enabled", False) and self._initialized

    @property
    def total_tool_count(self) -> int:
        """当前已发现的工具总数"""
        return sum(len(tools) for tools in self._tools.values())

    async def initialize(self, config_path: str) -> None:
        """读取配置，连接所有启用的 Server，获取工具列表

        Args:
            config_path: mcp_servers.json 配置文件路径
        """
        self._config_path = config_path

        # 读取配置
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
        except FileNotFoundError:
            logger.warning(f"MCP 配置文件不存在: {config_path}，跳过 MCP 初始化")
            return
        except json.JSONDecodeError as e:
            logger.error(f"MCP 配置文件 JSON 解析失败: {e}")
            return

        if not self._config.get("enabled", False):
            logger.info("MCP 全局开关未启用（enabled=false），跳过 MCP 初始化")
            return

        servers = self._config.get("servers", {})
        if not servers:
            logger.info("MCP 配置中无 Server 定义，跳过")
            return

        logger.info(f"MCP 初始化开始，共 {len(servers)} 个 Server 配置")

        # 逐个连接 Server
        success_count = 0
        for server_name, server_config in servers.items():
            if not server_config.get("enabled", True):
                logger.info(f"MCP Server [{server_name}] 未启用，跳过")
                continue

            try:
                await self._connect_server(server_name, server_config)
                success_count += 1
            except Exception as e:
                logger.error(f"MCP Server [{server_name}] 连接失败: {e}", exc_info=True)
                self._disconnected.add(server_name)
                self._server_configs[server_name] = server_config

            # 为每个 server 创建熔断器
            if server_name not in self._circuit_breakers:
                self._circuit_breakers[server_name] = CircuitBreaker(
                    name=server_name,
                    failure_threshold=int(server_config.get("circuit_failure_threshold", CIRCUIT_FAILURE_THRESHOLD)),
                    cooldown_seconds=float(server_config.get("circuit_cooldown_seconds", CIRCUIT_COOLDOWN_SECONDS)),
                )

        self._initialized = True
        total_tools = self.total_tool_count
        logger.info(
            f"MCP 初始化完成: {success_count}/{len(servers)} 个 Server 连接成功，"
            f"共发现 {total_tools} 个工具"
        )

        # v3 新增：构建工具语义索引（用于 compact mode 按需加载）
        try:
            self.build_tool_index()
        except Exception as e:
            logger.warning(f"ToolIndex 构建失败（非关键路径，降级到全量模式）: {e}", exc_info=True)

        # v2 新增：启动后台健康检查任务
        self._start_health_check_task()

    async def _connect_server(self, server_name: str, server_config: dict) -> None:
        """连接单个 MCP Server

        Args:
            server_name: Server 名称（用作工具前缀）
            server_config: Server 配置
        """
        transport = server_config.get("transport", "stdio")

        logger.info(f"MCP Server [{server_name}] 正在连接 (transport={transport})...")

        # 动态导入 MCP SDK（避免未安装时影响整个项目启动）
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.error(
                "MCP SDK 未安装，请运行: pip install mcp>=1.6.0。"
                "跳过 MCP Server 连接。"
            )
            return

        if transport == "stdio":
            # stdio 传输：启动子进程
            command = server_config.get("command")
            args = server_config.get("args", [])
            env = self._resolve_env(server_config.get("env", {}))

            if not command:
                logger.error(f"MCP Server [{server_name}] stdio 传输缺少 command 配置")
                return

            # === 诊断日志 1/5：打印启动配置 ===
            env_keys = list(env.keys()) if env else []
            logger.info(
                f"[MCP诊断] [{server_name}] 启动配置: command={command!r}, "
                f"args={args}, env_keys={env_keys}, timeout={server_config.get('connect_timeout', 60)}s"
            )

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env if env else None,
            )

            # === 诊断日志 2/5：启动子进程 ===
            logger.info(f"[MCP诊断] [{server_name}] 正在启动子进程: {command} {' '.join(args)}")
            try:
                # stderr 默认输出到 sys.stderr，server 的错误信息会直接打印到终端/日志
                transport_ctx = stdio_client(server_params)
                read, write = await transport_ctx.__aenter__()
            except Exception as e:
                logger.error(
                    f"[MCP诊断] [{server_name}] 子进程启动失败: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                raise Exception(f"子进程启动失败: {type(e).__name__}: {e}")
            logger.info(f"[MCP诊断] [{server_name}] 子进程启动成功，stdio 管道已建立")
            self._transports[server_name] = transport_ctx

            # === 诊断日志 3/5：建立 ClientSession ===
            logger.info(f"[MCP诊断] [{server_name}] 正在建立 ClientSession...")
            try:
                session = ClientSession(read, write)
                await session.__aenter__()
            except Exception as e:
                logger.error(
                    f"[MCP诊断] [{server_name}] ClientSession 建立失败: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                await self._cleanup_server(server_name)
                raise Exception(f"ClientSession 建立失败: {type(e).__name__}: {e}")
            logger.info(f"[MCP诊断] [{server_name}] ClientSession 建立成功")
            self._sessions[server_name] = session

        elif transport == "sse":
            # SSE 传输：连接远程 HTTP Server
            try:
                from mcp.client.sse import sse_client
            except ImportError:
                logger.error(f"MCP Server [{server_name}] SSE 传输需要 mcp[sse] 扩展")
                return

            url = server_config.get("url")
            if not url:
                logger.error(f"MCP Server [{server_name}] SSE 传输缺少 url 配置")
                return

            transport_ctx = sse_client(url)
            read, write = await transport_ctx.__aenter__()
            self._transports[server_name] = transport_ctx

            session = ClientSession(read, write)
            await session.__aenter__()
            self._sessions[server_name] = session

        else:
            logger.error(f"MCP Server [{server_name}] 不支持的传输方式: {transport}")
            return

        # === 诊断日志 4/5：MCP 握手 ===
        connect_timeout = server_config.get("connect_timeout", 60)
        logger.info(f"[MCP诊断] [{server_name}] 开始 MCP 握手 (initialize, 超时={connect_timeout}s)...")
        try:
            await asyncio.wait_for(session.initialize(), timeout=connect_timeout)
        except asyncio.TimeoutError:
            logger.error(f"[MCP诊断] [{server_name}] 握手超时：子进程在 {connect_timeout}s 内未完成 initialize")
            await self._cleanup_server(server_name)
            raise Exception(f"MCP 握手超时（{connect_timeout}秒），子进程可能未正确启动")
        except Exception as e:
            logger.error(
                f"[MCP诊断] [{server_name}] 握手失败: {type(e).__name__}: {e}",
                exc_info=True,
            )
            await self._cleanup_server(server_name)
            raise Exception(f"握手失败: {type(e).__name__}: {e}")
        logger.info(f"[MCP诊断] [{server_name}] MCP 握手成功")

        # === 诊断日志 5/5：获取工具列表 ===
        logger.info(f"[MCP诊断] [{server_name}] 正在获取工具列表 (list_tools, 超时=30s)...")
        try:
            tools_result = await asyncio.wait_for(session.list_tools(), timeout=30)
        except asyncio.TimeoutError:
            logger.error(f"[MCP诊断] [{server_name}] 获取工具列表超时（30秒）")
            await self._cleanup_server(server_name)
            raise Exception("获取工具列表超时（30秒）")
        except Exception as e:
            logger.error(
                f"[MCP诊断] [{server_name}] 获取工具列表失败: {type(e).__name__}: {e}",
                exc_info=True,
            )
            await self._cleanup_server(server_name)
            raise Exception(f"获取工具列表失败: {type(e).__name__}: {e}")
        logger.info(f"[MCP诊断] [{server_name}] 获取工具列表成功: {len(tools_result.tools)} 个工具")
        self._tools[server_name] = tools_result.tools
        self._server_configs[server_name] = server_config
        self._disconnected.discard(server_name)

        # 初始化重连锁
        if server_name not in self._reconnect_locks:
            self._reconnect_locks[server_name] = asyncio.Lock()

        tool_names = [t.name for t in tools_result.tools]
        logger.info(
            f"MCP Server [{server_name}] 连接成功，发现 {len(tools_result.tools)} 个工具: "
            f"{tool_names}"
        )

    async def _reconnect_server(self, server_name: str) -> bool:
        """重连单个 Server（含清理旧连接 + 指数退避重试）

        v2 修复：原版只重连一次，失败立即返回 false。新版用指数退避最多重试 3 次。
        退避序列：1s → 2s → 4s

        Args:
            server_name: Server 名称

        Returns:
            是否重连成功
        """
        # 使用锁避免并发重连
        if server_name not in self._reconnect_locks:
            self._reconnect_locks[server_name] = asyncio.Lock()

        async with self._reconnect_locks[server_name]:
            # 检查是否已重连成功（其他协程可能已完成）
            if server_name in self._sessions and server_name not in self._disconnected:
                return True

            server_config = self._server_configs.get(server_name)
            if not server_config:
                logger.error(f"MCP Server [{server_name}] 无配置信息，无法重连")
                return False

            # 指数退避重试
            delay = RECONNECT_INITIAL_DELAY
            for attempt in range(1, RECONNECT_MAX_ATTEMPTS + 1):
                logger.info(f"MCP Server [{server_name}] 重连尝试 {attempt}/{RECONNECT_MAX_ATTEMPTS}...")

                # 清理旧连接（只在首次和重试前清理）
                if attempt == 1:
                    await self._cleanup_server(server_name)

                try:
                    await self._connect_server(server_name, server_config)
                    logger.info(f"MCP Server [{server_name}] 重连成功（第 {attempt} 次尝试）")
                    # 重置熔断器（重连成功视为恢复）
                    cb = self._circuit_breakers.get(server_name)
                    if cb:
                        cb.reset()
                    # v3: 重连后工具列表可能变化，重建 ToolIndex
                    try:
                        self.build_tool_index()
                    except Exception as e:
                        logger.warning(f"重连后重建 ToolIndex 失败（非关键路径）: {e}")
                    return True
                except Exception as e:
                    logger.warning(f"MCP Server [{server_name}] 第 {attempt} 次重连失败: {e}")
                    self._disconnected.add(server_name)
                    # 最后一次不用 sleep
                    if attempt < RECONNECT_MAX_ATTEMPTS:
                        logger.info(f"MCP Server [{server_name}] 等待 {delay:.1f}s 后重试...")
                        await asyncio.sleep(delay)
                        delay *= RECONNECT_BACKOFF_FACTOR

            logger.error(f"MCP Server [{server_name}] 重连失败（已尝试 {RECONNECT_MAX_ATTEMPTS} 次）")
            return False

    # ==================== v2 新增：主动健康检查 ====================

    def _start_health_check_task(self) -> None:
        """启动后台健康检查任务（每 30s ping 一次所有 server）"""
        if self._health_check_task and not self._health_check_task.done():
            return  # 已在运行

        async def _health_check_loop():
            logger.info(f"MCP 健康检查任务已启动（间隔 {HEALTH_CHECK_INTERVAL}s）")
            while not self._shutting_down:
                try:
                    await asyncio.sleep(HEALTH_CHECK_INTERVAL)
                    if self._shutting_down:
                        break
                    await self._run_health_check()
                except asyncio.CancelledError:
                    logger.info("MCP 健康检查任务被取消")
                    break
                except Exception as e:
                    logger.warning(f"MCP 健康检查异常: {e}")
            logger.info("MCP 健康检查任务已停止")

        self._health_check_task = asyncio.create_task(_health_check_loop())

    async def _run_health_check(self) -> None:
        """对所有 server 做一次 ping 健康检查"""
        # 复制一份 server 列表，避免迭代时被修改
        server_names = list(self._server_configs.keys())
        for server_name in server_names:
            if self._shutting_down:
                break
            await self._ping_server(server_name)

    async def _ping_server(self, server_name: str) -> None:
        """对单个 server 做一次 ping 检查

        策略：
        - 用 session.send_request 发送 ping 请求（MCP 协议内置的 ping method）
        - 超时 5s 视为断连
        - 失败时标记 _disconnected，但不立即触发重连（重连由 call_tool 入口触发）
        """
        if server_name not in self._sessions:
            return  # 本来就没连上，不重复处理

        session = self._sessions[server_name]
        try:
            # MCP 协议内置的 ping 方法
            # v6.2 修复：用 session.send_ping() 替代手写的 send_request("ping", {})
            # mcp 库的 send_request 签名是 send_request(request, result_type, ...)
            # 旧代码传 ("ping", {}) 会被当作 (request="ping", result_type={})
            # result_type={} 在内部访问 .model_dump() 时报 'str' object has no attribute 'model_dump'
            await asyncio.wait_for(
                session.send_ping(),
                timeout=HEALTH_CHECK_TIMEOUT,
            )
            # ping 成功，清除断连标记
            if server_name in self._disconnected:
                logger.info(f"MCP 健康检查 [{server_name}] ping 成功，清除断连标记")
                self._disconnected.discard(server_name)
        except asyncio.TimeoutError:
            logger.warning(f"MCP 健康检查 [{server_name}] ping 超时（{HEALTH_CHECK_TIMEOUT}s），标记为断连")
            self._disconnected.add(server_name)
        except Exception as e:
            logger.warning(f"MCP 健康检查 [{server_name}] ping 失败: {e}，标记为断连")
            self._disconnected.add(server_name)

    def _stop_health_check_task(self) -> None:
        """停止健康检查任务"""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()

    # ==================== v2 新增：幂等性判断 ====================

    def _is_safe_retry(self, server_name: str, tool_name: str) -> bool:
        """判断工具是否可安全重试（幂等）

        策略：
        1. 优先看 server config 里的 safe_retry_tools 白名单
        2. 其次看 server config 里的 unsafe_retry_tools 黑名单
        3. 默认：以 list_/get_/search_/query_/read_/fetch_ 开头的工具视为安全
        4. 其他工具（如 send_email, create_resource）视为不安全
        """
        config = self._server_configs.get(server_name, {})

        # 显式白名单
        safe_whitelist = config.get("safe_retry_tools", [])
        if tool_name in safe_whitelist:
            return True

        # 显式黑名单
        unsafe_blacklist = config.get("unsafe_retry_tools", [])
        if tool_name in unsafe_blacklist:
            return False

        # 默认前缀匹配
        return tool_name.lower().startswith(SAFE_RETRY_PREFIXES)

    # ==================== v2 新增：熔断器访问 ====================

    def _get_circuit_breaker(self, server_name: str) -> CircuitBreaker:
        """获取或创建熔断器"""
        if server_name not in self._circuit_breakers:
            config = self._server_configs.get(server_name, {})
            self._circuit_breakers[server_name] = CircuitBreaker(
                name=server_name,
                failure_threshold=int(config.get("circuit_failure_threshold", CIRCUIT_FAILURE_THRESHOLD)),
                cooldown_seconds=float(config.get("circuit_cooldown_seconds", CIRCUIT_COOLDOWN_SECONDS)),
            )
        return self._circuit_breakers[server_name]

    async def _cleanup_server(self, server_name: str) -> None:
        """清理单个 Server 的所有资源（session、transport、工具）

        异常容错：即使 session/transport 清理失败，也确保内存状态被清除，
        保证 remove_server 操作总能成功。
        """
        # 关闭旧 session
        session = self._sessions.pop(server_name, None)
        if session:
            try:
                await session.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"MCP Server [{server_name}] session 清理异常（已忽略）: {e}")

        # 关闭旧 transport
        transport_ctx = self._transports.pop(server_name, None)
        if transport_ctx:
            try:
                await transport_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"MCP Server [{server_name}] transport 清理异常（已忽略）: {e}")

        # 清理工具列表
        self._tools.pop(server_name, None)

    def _resolve_env(self, env_config: dict) -> dict:
        """解析环境变量配置，支持 ${VAR} 插值"""
        resolved = {}
        for key, value in env_config.items():
            if isinstance(value, str) and "${" in value:
                var_name = value.strip("${}").strip()
                resolved[key] = os.getenv(var_name, "")
            else:
                resolved[key] = value
        return resolved

    def get_all_tools(self) -> List[dict]:
        """返回所有 Server 的工具列表（已转换为 OpenAI Function Calling 格式）"""
        result = []
        for server_name, tools in self._tools.items():
            for tool in tools:
                result.append({
                    "name": f"mcp__{server_name}__{tool.name}",
                    "description": f"[MCP/{server_name}] {tool.description or tool.name}",
                    "parameters": tool.inputSchema if tool.inputSchema else {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                })
        return result

    def get_tool_map(self) -> Dict[str, Tuple[str, str]]:
        """返回工具名映射：mcp__{server}__{tool} → (server_name, tool_name)"""
        result = {}
        for server_name, tools in self._tools.items():
            for tool in tools:
                full_name = f"mcp__{server_name}__{tool.name}"
                result[full_name] = (server_name, tool.name)
        return result

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        """路由调用到对应 Server 的 session.call_tool

        v2 修复版，含 5 重防护：
        1. 熔断检查：熔断中（OPEN）直接拒绝，不发起调用
        2. 断连检查：断连时先重连（指数退避最多 3 次）
        3. 幂等保护：非安全工具不自动重试，避免副作用重复执行
        4. 调用追踪：生成 call_id，记录状态供查询
        5. 熔断反馈：成功/失败都通知熔断器

        Args:
            server_name: Server 名称
            tool_name: 工具名称（不含前缀）
            arguments: 工具参数

        Returns:
            调用结果字典（含 call_id 供追踪）
        """
        # v2 新增：创建 call 记录
        record = mcp_call_log.new_call(server_name, tool_name, arguments)

        # 1. 熔断检查
        cb = self._get_circuit_breaker(server_name)
        if not cb.allow_request():
            logger.warning(f"MCP 熔断中 [{server_name}]，拒绝调用 {tool_name}")
            cb.record_failure()  # 熔断状态下的拒绝也算失败
            mcp_call_log.mark_failed(record, f"熔断中（state={cb.state}）")
            return {
                "success": False,
                "error": f"MCP Server [{server_name}] 熔断中，请稍后重试",
                "call_id": record.call_id,
                "circuit_state": cb.state,
            }

        # 2. 断连检查 + 重连
        if server_name in self._disconnected or server_name not in self._sessions:
            reconnected = await self._reconnect_server(server_name)
            if not reconnected:
                cb.record_failure()
                mcp_call_log.mark_failed(record, "未连接且重连失败")
                return {
                    "success": False,
                    "error": f"MCP Server [{server_name}] 未连接且重连失败",
                    "call_id": record.call_id,
                    "circuit_state": cb.state,
                }

        session = self._sessions.get(server_name)
        if session is None:
            cb.record_failure()
            mcp_call_log.mark_failed(record, "session 不存在")
            return {
                "success": False,
                "error": f"MCP Server [{server_name}] 未连接",
                "call_id": record.call_id,
            }

        server_config = self._server_configs.get(server_name, {})
        timeout = server_config.get("timeout", 30)

        # 3. 首次调用
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments),
                timeout=timeout,
            )

            if result.isError:
                error_text = self._extract_text(result.content)
                logger.warning(f"MCP 工具调用返回错误 [{server_name}/{tool_name}]: {error_text}")
                # 工具执行返回错误（如参数错误），不算连接问题，不触发重连
                cb.record_success()  # 通信成功，工具业务错误不算熔断信号
                mcp_call_log.mark_failed(record, error_text or "MCP 工具返回错误")
                return {
                    "success": False,
                    "error": error_text or "MCP 工具返回错误",
                    "call_id": record.call_id,
                }

            text_content = self._extract_text(result.content)
            cb.record_success()
            mcp_call_log.mark_success(record, text_content)
            return {
                "success": True,
                "result": text_content,
                "call_id": record.call_id,
            }

        except asyncio.TimeoutError:
            # 超时不视为断连，不触发重连
            logger.error(f"MCP 工具调用超时 [{server_name}/{tool_name}] ({timeout}s)")
            cb.record_failure()
            mcp_call_log.mark_failed(record, f"工具调用超时（{timeout}s）")
            return {
                "success": False,
                "error": f"工具调用超时（{timeout}s）",
                "call_id": record.call_id,
                "circuit_state": cb.state,
            }
        except Exception as e:
            logger.warning(f"MCP 工具调用异常 [{server_name}/{tool_name}]: {e}")

            # 4. 幂等保护：非安全工具不自动重试
            if not self._is_safe_retry(server_name, tool_name):
                logger.warning(
                    f"MCP 工具 [{server_name}/{tool_name}] 非幂等，不自动重试（避免副作用重复执行）"
                )
                cb.record_failure()
                self._disconnected.add(server_name)
                mcp_call_log.mark_dropped(record, f"非幂等工具不重试，原异常: {str(e)}")
                return {
                    "success": False,
                    "error": f"MCP 工具调用失败（非幂等，未重试）: {str(e)}",
                    "call_id": record.call_id,
                    "retried": False,
                    "circuit_state": cb.state,
                }

            # 5. 幂等工具：重连 + 重试一次
            mcp_call_log.mark_retrying(record)
            logger.info(f"MCP Server [{server_name}] 尝试重连后重试（幂等工具）...")
            self._disconnected.add(server_name)
            reconnected = await self._reconnect_server(server_name)

            if not reconnected:
                cb.record_failure()
                mcp_call_log.mark_failed(record, f"重连失败: {str(e)}")
                return {
                    "success": False,
                    "error": f"MCP Server [{server_name}] 连接断开且重连失败: {str(e)}",
                    "call_id": record.call_id,
                    "circuit_state": cb.state,
                }

            # 重试一次
            try:
                session = self._sessions[server_name]
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments),
                    timeout=timeout,
                )

                if result.isError:
                    error_text = self._extract_text(result.content)
                    cb.record_failure()
                    mcp_call_log.mark_failed(record, f"重试后工具返回错误: {error_text}")
                    return {
                        "success": False,
                        "error": error_text or "MCP 工具返回错误",
                        "call_id": record.call_id,
                        "retried": True,
                        "circuit_state": cb.state,
                    }

                text_content = self._extract_text(result.content)
                cb.record_success()
                mcp_call_log.mark_success(record, text_content)
                logger.info(f"MCP 工具重试成功 [{server_name}/{tool_name}]")
                return {
                    "success": True,
                    "result": text_content,
                    "call_id": record.call_id,
                    "retried": True,
                }
            except Exception as e2:
                logger.error(f"MCP 工具重试仍失败 [{server_name}/{tool_name}]: {e2}")
                cb.record_failure()
                mcp_call_log.mark_failed(record, f"重试失败: {str(e2)}")
                return {
                    "success": False,
                    "error": f"工具调用重试失败: {str(e2)}",
                    "call_id": record.call_id,
                    "retried": True,
                    "circuit_state": cb.state,
                }

    def _extract_text(self, content_list: list) -> str:
        """从 MCP CallToolResult.content 中提取文本内容"""
        texts = []
        for item in content_list:
            if hasattr(item, "text"):
                texts.append(item.text)
            elif isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts)

    # —— 热加载 API —— 

    async def add_server(self, server_name: str, server_config: dict) -> dict:
        """热加载：添加新 Server 并连接

        Args:
            server_name: Server 名称
            server_config: Server 配置

        Returns:
            {"success": bool, "message": str, "tool_count": int}
        """
        if server_name in self._server_configs:
            return {"success": False, "message": f"Server [{server_name}] 已存在"}

        # 先连接，成功后再写入配置（避免失败时配置污染）
        try:
            await self._connect_server(server_name, server_config)
        except Exception as e:
            # 连接失败：清理可能残留的 session/transport，不写入配置
            try:
                await self._cleanup_server(server_name)
            except Exception:
                pass
            self._disconnected.add(server_name)
            return {"success": False, "message": f"连接失败: {str(e)}", "tool_count": 0}

        # 连接成功，写入配置
        self._config.setdefault("servers", {})[server_name] = server_config

        # 持久化配置
        self._save_config()

        # v3: 工具列表变化，重建 ToolIndex
        try:
            self.build_tool_index()
        except Exception as e:
            logger.warning(f"add_server 后重建 ToolIndex 失败（非关键路径）: {e}")

        tool_count = len(self._tools.get(server_name, []))
        return {
            "success": True,
            "message": f"Server [{server_name}] 添加成功，发现 {tool_count} 个工具",
            "tool_count": tool_count,
        }

    async def remove_server(self, server_name: str) -> dict:
        """热加载：移除 Server 并清理资源

        Args:
            server_name: Server 名称

        Returns:
            {"success": bool, "message": str}
        """
        if server_name not in self._server_configs:
            return {"success": False, "message": f"Server [{server_name}] 不存在"}

        # 清理连接（异常容错：即使清理失败也继续删除配置）
        try:
            await self._cleanup_server(server_name)
        except Exception as e:
            logger.warning(f"MCP Server [{server_name}] 清理过程异常（继续删除配置）: {e}")

        self._disconnected.discard(server_name)
        self._tools.pop(server_name, None)
        self._server_configs.pop(server_name, None)

        # 更新配置
        self._config.get("servers", {}).pop(server_name, None)
        self._save_config()

        # v3: 工具列表变化，重建 ToolIndex
        try:
            self.build_tool_index()
        except Exception as e:
            logger.warning(f"remove_server 后重建 ToolIndex 失败（非关键路径）: {e}")

        return {"success": True, "message": f"Server [{server_name}] 已移除"}

    async def update_server(self, server_name: str, server_config: dict) -> dict:
        """热加载：更新 Server 配置（先移除再添加）

        Args:
            server_name: Server 名称
            server_config: 新的 Server 配置

        Returns:
            {"success": bool, "message": str, "tool_count": int}
        """
        if server_name not in self._server_configs:
            return {"success": False, "message": f"Server [{server_name}] 不存在"}

        # 先移除
        await self._cleanup_server(server_name)

        # 更新配置
        self._config["servers"][server_name] = server_config
        self._server_configs[server_name] = server_config

        # 重新连接
        try:
            await self._connect_server(server_name, server_config)
        except Exception as e:
            self._disconnected.add(server_name)
            self._save_config()
            return {"success": False, "message": f"重连失败: {str(e)}", "tool_count": 0}

        self._save_config()

        # v3: 工具列表变化，重建 ToolIndex
        try:
            self.build_tool_index()
        except Exception as e:
            logger.warning(f"update_server 后重建 ToolIndex 失败（非关键路径）: {e}")

        tool_count = len(self._tools.get(server_name, []))
        return {
            "success": True,
            "message": f"Server [{server_name}] 更新成功，发现 {tool_count} 个工具",
            "tool_count": tool_count,
        }

    async def reconnect_server(self, server_name: str) -> dict:
        """手动触发重连

        Args:
            server_name: Server 名称

        Returns:
            {"success": bool, "message": str}
        """
        if server_name not in self._server_configs:
            return {"success": False, "message": f"Server [{server_name}] 不存在"}

        success = await self._reconnect_server(server_name)
        if success:
            tool_count = len(self._tools.get(server_name, []))
            return {
                "success": True,
                "message": f"Server [{server_name}] 重连成功，{tool_count} 个工具",
            }
        else:
            return {"success": False, "message": f"Server [{server_name}] 重连失败"}

    def get_server_status(self) -> List[dict]:
        """获取所有 Server 的状态

        Returns:
            Server 状态列表（含熔断器状态和最近调用记录）
        """
        result = []
        for name, config in self._server_configs.items():
            cb = self._circuit_breakers.get(name)
            result.append({
                "name": name,
                "transport": config.get("transport", "stdio"),
                "enabled": config.get("enabled", True),
                "connected": name in self._sessions and name not in self._disconnected,
                "tool_count": len(self._tools.get(name, [])),
                "tools": [t.name for t in self._tools.get(name, [])],
                "timeout": config.get("timeout", 30),
                # v2 新增字段
                "circuit_state": cb.state if cb else "unknown",
                "circuit_failure_count": cb._failure_count if cb else 0,
                "recent_calls": len(mcp_call_log.get_records(name, limit=10)),
            })
        return result

    def _save_config(self) -> bool:
        """持久化配置到文件

        Returns:
            True 成功，False 失败
        """
        if not self._config_path:
            logger.warning("MCP 配置未持久化：_config_path 为空（initialize 未执行？）")
            return False

        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
            logger.info(f"MCP 配置已持久化到 {self._config_path}")
            return True
        except Exception as e:
            logger.error(f"MCP 配置持久化失败: {e}", exc_info=True)
            return False

    # —— 工具按需加载 —— 

    def should_use_compact_mode(self, model_name: Optional[str] = None) -> bool:
        """是否启用 compact 模式（工具数 > 阈值时启用按需加载）

        v6 恢复阈值判断：用户指定阈值 100（v6.2 从 30 调整为 100）。
        - 工具数 ≤ 100：走全量注入（所有 MCP 工具 schema 进 tools_payload）
        - 工具数 > 100：走按需加载（ToolIndex 检索相关工具，只注入 top_k 个 schema）
        - model_name 参数保留向后兼容，但 v6 改为固定阈值，不再自适应

        Args:
            model_name: LLM 模型名（v6 后不再使用，保留向后兼容）

        Returns:
            是否启用 compact 模式
        """
        return self.total_tool_count > TOOL_COUNT_THRESHOLD

    def get_compact_tool_summary(self) -> str:
        """获取所有 MCP 工具的摘要文本（用于状态展示/调试）

        v5 改造：不再用于 system prompt 注入（按需加载模式下 LLM 直接拿到相关工具的完整 schema）。
        保留方法用于 routers/mcp.py 状态查询和调试。

        Returns:
            工具摘要文本
        """
        if not self._tools:
            return ""

        lines = ["MCP 工具摘要:"]
        for server_name, tools in self._tools.items():
            lines.append(f"\n[{server_name}] {len(tools)} 个工具:")
            for tool in tools:
                # 提取 description 的第一行/第一句，截断到 60 字符
                desc_raw = (tool.description or "").strip()
                desc_first_line = desc_raw.split("\n")[0].strip() if desc_raw else ""
                if "。" in desc_first_line:
                    desc_first_line = desc_first_line.split("。")[0].strip()
                elif ". " in desc_first_line:
                    desc_first_line = desc_first_line.split(". ")[0].strip()
                if len(desc_first_line) > 60:
                    desc_first_line = desc_first_line[:57] + "..."
                short_name = tool.name
                if short_name.startswith(f"mcp__{server_name}__"):
                    short_name = short_name[len(f"mcp__{server_name}__"):]

                if desc_first_line:
                    lines.append(f"  - {short_name}: {desc_first_line}")
                else:
                    lines.append(f"  - {short_name}")
        return "\n".join(lines)

    # ==================== v3 新增：ToolIndex 按需加载 ====================

    def build_tool_index(self) -> None:
        """构建工具语义索引（基于 embedding）

        在 MCP 初始化完成、工具列表稳定后调用。构建后可以用 get_relevant_tool_schemas(query)
        按 query 检索相关工具的完整 schema，避免把所有工具 schema 都塞进 LLM prompt。

        幂等：重复调用会清空旧索引重建。建议在以下时机调用：
        - initialize() 完成后
        - 热加载增删 Server 后
        - 重连成功后（工具列表可能变化）

        v6.2: 工具数 ≤ TOOL_COUNT_THRESHOLD(100) 时跳过构建，避免无谓的 embedding
        索引开销拖慢启动。compact 模式未启用时 get_relevant_tool_schemas 也走全量返回。
        注意：import mcp_tool_index 会触发模块级单例 ToolIndex() 的 embedding 初始化，
        所以必须延迟到阈值检查通过后再 import。
        """
        if not self._tools:
            logger.info("ToolIndex 构建：无工具可索引（MCP 未连接或未启用）")
            from services.mcp_tool_index import mcp_tool_index
            mcp_tool_index.clear()
            return

        # v6.2: 工具数未达阈值时跳过索引构建（全量注入路径不需要 ToolIndex）
        # 不 import mcp_tool_index，避免触发 embedding 服务初始化
        if self.total_tool_count <= TOOL_COUNT_THRESHOLD:
            logger.info(
                f"ToolIndex 跳过构建：工具数 {self.total_tool_count} ≤ 阈值 {TOOL_COUNT_THRESHOLD}，"
                f"走全量注入模式"
            )
            return

        from services.mcp_tool_index import mcp_tool_index
        mcp_tool_index.build(self._tools)
        logger.info(
            f"ToolIndex 构建完成: {mcp_tool_index.size} 个工具，"
            f"mode={'embedding' if not mcp_tool_index._use_keyword_fallback else 'keyword'}"
        )

    def get_relevant_tool_schemas(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """按 query 检索最相关的 N 个工具，返回完整 schema

        v3 按需加载核心方法。compact mode 下替代 get_all_tools() 全量注入。

        Args:
            query: 用户查询（通常是 user prompt）
            top_k: 返回的工具数量（建议 3~10）
            min_score: 最低相似度阈值（embedding 模式下 0~1，关键词模式下不太准确）
                默认 0.0 表示不过滤

        Returns:
            工具 schema 列表（OpenAI Function Calling 格式），按相关性降序
            格式：[{"name": "mcp__server__tool", "description": "...", "parameters": {...}}, ...]
        """
        # v6.2: 工具数未达阈值时直接全量返回，不 import mcp_tool_index
        # 避免 embedding 服务初始化开销
        if self.total_tool_count <= TOOL_COUNT_THRESHOLD:
            return self.get_all_tools()

        from services.mcp_tool_index import mcp_tool_index

        if not mcp_tool_index.is_built:
            logger.warning("ToolIndex 未构建，回退到全量工具列表")
            return self.get_all_tools()

        if not query or not query.strip():
            logger.debug("get_relevant_tool_schemas: query 为空，返回空列表")
            return []

        # 检索相关工具名
        relevant_tool_names = mcp_tool_index.retrieve(query, top_k=top_k)
        if not relevant_tool_names:
            logger.debug(f"get_relevant_tool_schemas: query={query!r} 未检索到相关工具")
            return []

        # 根据 tool_full_name 找回完整 schema
        # 构建 full_name → tool_schema 的映射
        result: List[Dict[str, Any]] = []
        for full_name in relevant_tool_names:
            meta = mcp_tool_index.get_tool_meta(full_name)
            if not meta:
                continue
            server_name, original_name = meta
            tools = self._tools.get(server_name, [])
            for tool in tools:
                if tool.name == original_name:
                    result.append({
                        "name": full_name,
                        "description": f"[MCP/{server_name}] {tool.description or tool.name}",
                        "parameters": tool.inputSchema if tool.inputSchema else {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    })
                    break

        logger.info(
            f"ToolIndex 按需加载: query={query!r}, 检索到 {len(result)}/{top_k} 个工具: "
            f"{[r['name'] for r in result]}"
        )
        return result

    async def list_tools_detail(self, server_name: Optional[str] = None) -> dict:
        """列出工具详情（供 mcp_list_tools 元工具调用）

        Args:
            server_name: 可选，指定 Server 名称。None 则返回所有。

        Returns:
            工具详情字典
        """
        result = {}
        servers = [server_name] if server_name else list(self._tools.keys())

        for name in servers:
            if name not in self._tools:
                continue
            tools = self._tools[name]
            result[name] = [
                {
                    "name": f"mcp__{name}__{t.name}",
                    "description": t.description or t.name,
                    "parameters": t.inputSchema if t.inputSchema else {},
                }
                for t in tools
            ]

        return {"success": True, "servers": result}

    async def shutdown(self) -> None:
        """关闭所有 session 和 transport"""
        logger.info("MCP 正在关闭所有连接...")

        # v2 新增：停止健康检查任务
        self._shutting_down = True
        self._stop_health_check_task()

        for server_name in list(self._server_configs.keys()):
            await self._cleanup_server(server_name)

        self._disconnected.clear()
        self._initialized = False

        logger.info("MCP 关闭完成")


# 全局单例
mcp_client_manager = MCPClientManager()
