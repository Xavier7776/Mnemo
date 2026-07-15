"""MCP 管理 API 路由 — 动态增删改查 MCP Server 配置

阶段二新增：热加载管理后台，无需重启即可管理 MCP Server。

API 列表：
  GET    /api/mcp/servers              列出所有 Server 及状态
  POST   /api/mcp/servers/{name}       添加新 Server（热加载）
  PUT    /api/mcp/servers/{name}       更新 Server 配置（热重载）
  DELETE /api/mcp/servers/{name}       移除 Server（热卸载）
  POST   /api/mcp/servers/{name}/reconnect  手动重连
  GET    /api/mcp/tools                列出所有已注册的 MCP 工具
  GET    /api/mcp/status               获取 MCP 整体状态
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from utils.logger import logger

router = APIRouter()


class ServerConfigRequest(BaseModel):
    """Server 配置请求体"""
    transport: str = Field(default="stdio", description="传输方式: stdio / sse")
    command: Optional[str] = Field(default=None, description="stdio: 启动命令")
    args: List[str] = Field(default_factory=list, description="stdio: 命令参数")
    env: Dict[str, str] = Field(default_factory=dict, description="环境变量")
    url: Optional[str] = Field(default=None, description="sse: 远程 URL")
    enabled: bool = Field(default=True, description="是否启用")
    timeout: int = Field(default=30, description="工具调用超时（秒）")


class ServerStatusResponse(BaseModel):
    name: str
    transport: str
    enabled: bool
    connected: bool
    tool_count: int
    tools: List[str]
    timeout: int


class MCPStatusResponse(BaseModel):
    initialized: bool
    enabled: bool
    total_servers: int
    connected_servers: int
    total_tools: int
    compact_mode: bool
    servers: List[ServerStatusResponse]


class OperationResponse(BaseModel):
    success: bool
    message: str
    tool_count: Optional[int] = None


class ToolDetail(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolsListResponse(BaseModel):
    success: bool
    servers: Dict[str, List[ToolDetail]]


@router.get("/servers", response_model=List[ServerStatusResponse])
async def list_servers():
    """列出所有 MCP Server 及其状态"""
    from services.mcp_client_service import mcp_client_manager

    if not mcp_client_manager.is_initialized:
        return []

    status_list = mcp_client_manager.get_server_status()
    return [ServerStatusResponse(**s) for s in status_list]


@router.get("/status", response_model=MCPStatusResponse)
async def get_mcp_status():
    """获取 MCP 整体状态"""
    from services.mcp_client_service import mcp_client_manager

    servers = mcp_client_manager.get_server_status()
    connected = sum(1 for s in servers if s["connected"])

    return MCPStatusResponse(
        initialized=mcp_client_manager.is_initialized,
        enabled=mcp_client_manager.is_enabled,
        total_servers=len(servers),
        connected_servers=connected,
        total_tools=mcp_client_manager.total_tool_count,
        compact_mode=mcp_client_manager.should_use_compact_mode(),
        servers=[ServerStatusResponse(**s) for s in servers],
    )


@router.post("/servers/{server_name}", response_model=OperationResponse)
async def add_server(server_name: str, config: ServerConfigRequest):
    """添加新 MCP Server（热加载）

    添加后自动连接并注册工具到 ai_tools，无需重启。
    """
    from services.mcp_client_service import mcp_client_manager

    if not mcp_client_manager.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MCP 未初始化，请设置 MCP_ENABLED=true 并重启服务",
        )

    server_config = config.model_dump()
    result = await mcp_client_manager.add_server(server_name, server_config)

    if result["success"]:
        # 热注册工具到 ai_tools
        _register_tools_to_ai_tools()
        return OperationResponse(**result)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"],
        )


@router.put("/servers/{server_name}", response_model=OperationResponse)
async def update_server(server_name: str, config: ServerConfigRequest):
    """更新 MCP Server 配置（热重载）

    先断开旧连接，再用新配置重新连接。
    """
    from services.mcp_client_service import mcp_client_manager

    if not mcp_client_manager.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MCP 未初始化",
        )

    server_config = config.model_dump()
    result = await mcp_client_manager.update_server(server_name, server_config)

    if result["success"]:
        _register_tools_to_ai_tools()
        return OperationResponse(**result)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"],
        )


@router.delete("/servers/{server_name}", response_model=OperationResponse)
async def remove_server(server_name: str):
    """移除 MCP Server（热卸载）

    断开连接并清理所有资源，配置持久化到文件。
    """
    from services.mcp_client_service import mcp_client_manager

    if not mcp_client_manager.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MCP 未初始化",
        )

    result = await mcp_client_manager.remove_server(server_name)

    if result["success"]:
        # 从 ai_tools 中移除该 Server 的工具
        _unregister_tools_from_ai_tools(server_name)
        return OperationResponse(**result)
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["message"],
        )


@router.post("/servers/{server_name}/reconnect", response_model=OperationResponse)
async def reconnect_server(server_name: str):
    """手动触发 Server 重连"""
    from services.mcp_client_service import mcp_client_manager

    if not mcp_client_manager.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MCP 未初始化",
        )

    result = await mcp_client_manager.reconnect_server(server_name)

    if result["success"]:
        _register_tools_to_ai_tools()
        return OperationResponse(**result)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"],
        )


@router.get("/tools", response_model=ToolsListResponse)
async def list_tools(server: Optional[str] = None):
    """列出所有已注册的 MCP 工具

    Args:
        server: 可选，筛选特定 Server 的工具
    """
    from services.mcp_client_service import mcp_client_manager

    if not mcp_client_manager.is_initialized:
        return ToolsListResponse(success=True, servers={})

    result = await mcp_client_manager.list_tools_detail(server)

    servers = {}
    for server_name, tools in result.get("servers", {}).items():
        servers[server_name] = [ToolDetail(**t) for t in tools]

    return ToolsListResponse(success=result["success"], servers=servers)


# —— 辅助函数 ——

def _register_tools_to_ai_tools():
    """将 MCP 工具重新注册到 ai_tools（全量刷新）"""
    try:
        from services.ai_tools import ai_tools
        from services.mcp_client_service import mcp_client_manager

        # 先移除旧工具
        for old_name in list(mcp_client_manager._registered_tool_names):
            ai_tools.tools.pop(old_name, None)
            ai_tools.functions.pop(old_name, None)
            if ai_tools._async_tools:
                ai_tools._async_tools.pop(old_name, None)

        # 重新注册
        count = ai_tools.register_mcp_tools(mcp_client_manager)
        mcp_client_manager._registered_tool_names = set(
            ai_tools.tools.keys() - {"get_available_ollama_models",
                                     "get_knowledge_base_documents",
                                     "get_system_info",
                                     "get_knowledge_base_stats",
                                     "core_memory_append",
                                     "core_memory_replace",
                                     "archival_memory_insert",
                                     "archival_memory_search",
                                     "conversation_search",
                                     "rag_retrieve"}
        )
        logger.info(f"MCP 工具热注册完成: {count} 个工具")
    except Exception as e:
        logger.error(f"MCP 工具热注册失败: {e}", exc_info=True)


def _unregister_tools_from_ai_tools(server_name: str):
    """从 ai_tools 中移除指定 Server 的工具"""
    try:
        from services.ai_tools import ai_tools
        from services.mcp_client_service import mcp_client_manager

        # 找到该 Server 的所有工具名
        prefix = f"mcp__{server_name}__"
        to_remove = [name for name in ai_tools.tools.keys() if name.startswith(prefix)]

        for name in to_remove:
            ai_tools.tools.pop(name, None)
            ai_tools.functions.pop(name, None)
            if ai_tools._async_tools:
                ai_tools._async_tools.pop(name, None)
            mcp_client_manager._registered_tool_names.discard(name)

        logger.info(f"MCP 工具热卸载完成: [{server_name}] 移除 {len(to_remove)} 个工具")
    except Exception as e:
        logger.error(f"MCP 工具热卸载失败: {e}", exc_info=True)
