"""MCP Client 服务 — 连接外部 MCP Server，扩展 Agent 可用工具

阶段二增强：
- 断线重连：call_tool 失败时自动重连一次再重试
- 热加载：运行时动态增删改 Server 配置，无需重启
- 健康检查：定期检测 Server 连接状态
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


# 工具按需加载的阈值：超过此数量时启用摘要模式
TOOL_COUNT_THRESHOLD = 20


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

        self._initialized = True
        total_tools = self.total_tool_count
        logger.info(
            f"MCP 初始化完成: {success_count}/{len(servers)} 个 Server 连接成功，"
            f"共发现 {total_tools} 个工具"
        )

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
        """重连单个 Server（含清理旧连接）

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

            logger.info(f"MCP Server [{server_name}] 正在重连...")

            # 清理旧连接
            await self._cleanup_server(server_name)

            # 重新连接
            server_config = self._server_configs.get(server_name)
            if not server_config:
                logger.error(f"MCP Server [{server_name}] 无配置信息，无法重连")
                return False

            try:
                await self._connect_server(server_name, server_config)
                logger.info(f"MCP Server [{server_name}] 重连成功")
                return True
            except Exception as e:
                logger.error(f"MCP Server [{server_name}] 重连失败: {e}", exc_info=True)
                self._disconnected.add(server_name)
                return False

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

        含断线重连逻辑：首次调用失败时自动重连一次再重试。

        Args:
            server_name: Server 名称
            tool_name: 工具名称（不含前缀）
            arguments: 工具参数

        Returns:
            调用结果字典
        """
        # 检查是否断连，尝试重连
        if server_name in self._disconnected or server_name not in self._sessions:
            reconnected = await self._reconnect_server(server_name)
            if not reconnected:
                return {
                    "success": False,
                    "error": f"MCP Server [{server_name}] 未连接且重连失败",
                }

        session = self._sessions.get(server_name)
        if session is None:
            return {
                "success": False,
                "error": f"MCP Server [{server_name}] 未连接",
            }

        server_config = self._server_configs.get(server_name, {})
        timeout = server_config.get("timeout", 30)

        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments),
                timeout=timeout,
            )

            if result.isError:
                error_text = self._extract_text(result.content)
                logger.warning(f"MCP 工具调用返回错误 [{server_name}/{tool_name}]: {error_text}")
                return {
                    "success": False,
                    "error": error_text or "MCP 工具返回错误",
                }

            text_content = self._extract_text(result.content)
            return {
                "success": True,
                "result": text_content,
            }

        except asyncio.TimeoutError:
            logger.error(f"MCP 工具调用超时 [{server_name}/{tool_name}] ({timeout}s)")
            return {
                "success": False,
                "error": f"工具调用超时（{timeout}s）",
            }
        except Exception as e:
            logger.warning(f"MCP 工具调用异常 [{server_name}/{tool_name}]: {e}")

            # 尝试重连一次后重试
            logger.info(f"MCP Server [{server_name}] 尝试重连后重试...")
            self._disconnected.add(server_name)
            reconnected = await self._reconnect_server(server_name)

            if not reconnected:
                return {
                    "success": False,
                    "error": f"MCP Server [{server_name}] 连接断开且重连失败: {str(e)}",
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
                    return {
                        "success": False,
                        "error": error_text or "MCP 工具返回错误",
                    }

                text_content = self._extract_text(result.content)
                logger.info(f"MCP 工具重试成功 [{server_name}/{tool_name}]")
                return {
                    "success": True,
                    "result": text_content,
                }
            except Exception as e2:
                logger.error(f"MCP 工具重试仍失败 [{server_name}/{tool_name}]: {e2}")
                return {
                    "success": False,
                    "error": f"工具调用重试失败: {str(e2)}",
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
            Server 状态列表
        """
        result = []
        for name, config in self._server_configs.items():
            result.append({
                "name": name,
                "transport": config.get("transport", "stdio"),
                "enabled": config.get("enabled", True),
                "connected": name in self._sessions and name not in self._disconnected,
                "tool_count": len(self._tools.get(name, [])),
                "tools": [t.name for t in self._tools.get(name, [])],
                "timeout": config.get("timeout", 30),
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

    def should_use_compact_mode(self) -> bool:
        """是否应该使用紧凑模式（工具数量超过阈值时）

        紧凑模式下，只注入工具摘要到 system prompt，
        LLM 通过 mcp_list_tools 发现具体工具后再调用。

        Returns:
            是否启用紧凑模式
        """
        return self.total_tool_count > TOOL_COUNT_THRESHOLD

    def get_compact_tool_summary(self) -> str:
        """获取紧凑模式的工具摘要（用于注入 system prompt）

        Returns:
            工具摘要文本
        """
        if not self.should_use_compact_mode():
            return ""

        lines = ["MCP 工具摘要（使用 mcp_list_tools 查看详情）:"]
        for server_name, tools in self._tools.items():
            tool_names = [t.name for t in tools]
            lines.append(f"  [{server_name}] {len(tools)} 个工具: {', '.join(tool_names)}")
        return "\n".join(lines)

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

        for server_name in list(self._server_configs.keys()):
            await self._cleanup_server(server_name)

        self._disconnected.clear()
        self._initialized = False

        logger.info("MCP 关闭完成")


# 全局单例
mcp_client_manager = MCPClientManager()
