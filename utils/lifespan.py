"""应用生命周期管理"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from database.mongodb import mongodb
from utils.logger import logger


async def _connect_mongodb_with_retry(max_retries: int = 3, delay_seconds: float = 2.0):
    """带重试的 MongoDB 连接，启动时使用。失败不抛异常，返回是否连接成功。"""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"正在连接 MongoDB... (尝试 {attempt}/{max_retries})")
            await mongodb.connect()
            return True
        except Exception as e:
            logger.warning(f"MongoDB 连接失败 (尝试 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                import asyncio
                await asyncio.sleep(delay_seconds)
    logger.error(
        "MongoDB 启动时连接失败，服务将先启动，依赖 MongoDB 的接口可能不可用。"
        "请确认：1) MongoDB 已启动；2) .env 中 MONGODB_URI 或 MONGODB_HOST/PORT 正确；"
        "3) 若在 Docker 内访问宿主机请使用 host.docker.internal 或 127.0.0.1。"
    )
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。MongoDB 连接失败时仍允许服务启动，便于本地调试。"""
    mongodb_ready = False
    try:
        mongodb_ready = await _connect_mongodb_with_retry()
        app.state.mongodb_ready = mongodb_ready

        if mongodb_ready:
            # 启动时数据修复/初始化：
            # 1) 仅保留一个默认“通用助手”（course_assistants）——对话用
            # 2) 初始化至少一个默认知识空间（knowledge_spaces）——入库/检索用
            try:
                from utils.timezone import beijing_now

                assistants = mongodb.get_collection("course_assistants")
                # 确保至少有一个默认助手
                default_assistant = await assistants.find_one({"is_default": True})
                if not default_assistant:
                    now = beijing_now()
                    await assistants.insert_one(
                        {
                            "name": "默认助手",
                            "description": "系统默认对话助手（GeneralAssistantAgent）",
                            "system_prompt": "",
                            "collection_name": "default_knowledge",
                            "is_default": True,
                            "created_at": now,
                            "updated_at": now,
                        }
                    )
                    default_assistant = await assistants.find_one({"is_default": True})

                # 删除非默认的“通用助手”（保留一个默认即可）
                await assistants.delete_many({"is_default": {"$ne": True}})

                # 初始化默认知识空间
                spaces = mongodb.get_collection("knowledge_spaces")
                default_space = await spaces.find_one({"is_default": True})
                if not default_space:
                    now = beijing_now()
                    await spaces.insert_one(
                        {
                            "name": "默认知识空间",
                            "description": "系统默认知识库空间",
                            "collection_name": "default_knowledge",
                            "is_default": True,
                            "created_at": now,
                            "updated_at": now,
                        }
                    )
            except Exception as e:
                logger.warning(f"启动初始化（助手/知识空间）失败: {e}")

            # 预热重排模型（如果启用），避免首个请求卡在模型加载上
            try:
                import os
                enable_reranker = os.getenv("ENABLE_RERANKER", "0").strip().lower() in {"1", "true", "yes", "on"}
                if enable_reranker:
                    logger.info("正在预热重排模型...")
                    from retrieval.rag_retriever import RAGRetriever
                    # 创建临时实例触发模型加载，并复用到 EvidenceVerifier 做分层验证
                    warmup_retriever = RAGRetriever()
                    def _warmup_reranker():
                        return warmup_retriever._get_reranker()
                    reranker = await asyncio.to_thread(_warmup_reranker)
                    logger.info("重排模型预热完成")

                    # 如果启用证据验证的 CrossEncoder 分层确认，注入已加载的 reranker
                    use_ce_verify = os.getenv("VERIFIER_USE_CROSS_ENCODER", "0").strip().lower() in {"1", "true", "yes", "on"}
                    if use_ce_verify and reranker is not None:
                        from services.evidence_verifier import evidence_verifier
                        evidence_verifier.attach_cross_encoder(reranker)
                        logger.info("EvidenceVerifier 已注入 CrossEncoder，启用分层证据验证")
            except Exception as e:
                logger.warning(f"重排模型预热失败（不影响服务启动）: {e}")

            # 预热 Embedding 模型
            try:
                from embedding.embedding_service import embedding_service
                await asyncio.to_thread(embedding_service._get_model)
                logger.info("Embedding 模型预热完成")
            except Exception as e:
                logger.warning(f"Embedding 模型预热失败（不影响启动）: {e}")

        # —— MCP Client 初始化 ——
        import os
        if os.getenv("MCP_ENABLED", "false").lower() in {"1", "true", "yes", "on"}:
            try:
                from services.mcp_client_service import mcp_client_manager
                from services.ai_tools import ai_tools
                # 项目根目录 = lifespan.py 所在 utils 目录的上一级
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                config_path = os.path.join(base_dir, "config", "mcp_servers.json")
                await mcp_client_manager.initialize(config_path)
                if mcp_client_manager.is_enabled:
                    count = ai_tools.register_mcp_tools(mcp_client_manager)
                    logger.info(f"MCP 工具已注册 {count} 个到 ai_tools")
                else:
                    logger.info("MCP 初始化完成但未启用（无 Server 连接成功或配置 disabled）")
            except Exception as e:
                logger.warning(f"MCP 初始化失败（不影响服务启动）: {e}", exc_info=True)
        else:
            logger.info("MCP_ENABLED 未启用，跳过 MCP 初始化")

    except Exception as e:
        logger.error(f"lifespan 异常: {str(e)}", exc_info=True)
        app.state.mongodb_ready = False

    yield

    # 关闭时执行
    try:
        await mongodb.disconnect()
    except Exception as e:
        logger.error(f"关闭数据库连接时出错: {str(e)}", exc_info=True)

    # 关闭 MCP 连接
    try:
        from services.mcp_client_service import mcp_client_manager
        if mcp_client_manager.is_initialized:
            await mcp_client_manager.shutdown()
    except Exception as e:
        logger.warning(f"关闭 MCP 连接时出错: {e}")

