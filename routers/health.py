"""健康检查路由"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any, Optional
from database.mongodb import mongodb
from database.qdrant_client import qdrant_client
from database.neo4j_client import neo4j_client
from utils.logger import logger
from utils.monitoring import performance_monitor
import psutil
import os

router = APIRouter()


class HealthStatus(BaseModel):
    """健康状态模型"""
    status: str
    version: str
    services: Dict[str, Any]
    system: Optional[Dict[str, Any]] = None


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    健康检查端点
    检查所有服务的连接状态

    覆盖范围（v0.8.6+）：
    - MongoDB：核心元数据 + chunk 内容存储（强依赖，挂掉影响所有功能）
    - Qdrant：向量检索（强依赖，挂掉退化到 keyword+graph 两路）
    - Redis：BM25 关键词检索（弱依赖，挂掉 fallback 到 MongoDB 全表扫描）
    - Neo4j：知识图谱检索（弱依赖，挂掉返回空结果，不影响其他两路）

    降级策略详见 docs/failure-modes.md
    """
    services_status = {}
    overall_status = "healthy"

    # MongoDB健康检查
    try:
        collection = mongodb.get_collection("documents")
        await collection.find_one({}, limit=1)
        services_status["mongodb"] = {
            "status": "healthy",
            "connected": True
        }
    except Exception as e:
        logger.warning(f"MongoDB健康检查失败: {str(e)}")
        services_status["mongodb"] = {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)[:100]
        }
        overall_status = "degraded"

    # Qdrant健康检查
    try:
        is_healthy = qdrant_client.check_health()
        services_status["qdrant"] = {
            "status": "healthy" if is_healthy else "unhealthy",
            "connected": is_healthy
        }
        if not is_healthy:
            overall_status = "degraded"
    except Exception as e:
        logger.warning(f"Qdrant健康检查失败: {str(e)}")
        services_status["qdrant"] = {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)[:100]
        }
        overall_status = "degraded"

    # Redis健康检查（v0.8.6+ 新增）
    # Redis 是弱依赖：挂掉后 BM25 关键词检索会 fallback 到 MongoDB 全表扫描
    # 因此 Redis 不可用不应让整体状态变成 degraded（只是性能下降）
    try:
        from utils.redis_client import is_available as redis_is_available, get_redis_client
        r = get_redis_client()
        if r is not None:
            # 双重确认：client 存在且 ping 通
            r.ping()
            services_status["redis"] = {
                "status": "healthy",
                "connected": True,
                "redisearch_available": redis_is_available(),
            }
        else:
            services_status["redis"] = {
                "status": "unavailable",
                "connected": False,
                "fallback": "mongo_bm25",
                "note": "Redis 不可用，BM25 关键词检索降级到 MongoDB 全表扫描",
            }
            # Redis 是弱依赖，不改变 overall_status
    except Exception as e:
        logger.warning(f"Redis健康检查失败: {str(e)}")
        services_status["redis"] = {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)[:100],
            "fallback": "mongo_bm25",
        }
        # Redis 是弱依赖，不改变 overall_status

    # Neo4j健康检查（v0.8.6+ 新增）
    # Neo4j 是弱依赖：挂掉后图谱检索返回空结果，不影响其他两路
    try:
        if neo4j_client.driver is None:
            neo4j_client.connect()
        if neo4j_client.driver is not None:
            # verify_connectivity 是同步阻塞调用，用 to_thread 包装
            import asyncio
            await asyncio.to_thread(neo4j_client.driver.verify_connectivity)
            services_status["neo4j"] = {
                "status": "healthy",
                "connected": True,
            }
        else:
            services_status["neo4j"] = {
                "status": "unavailable",
                "connected": False,
                "fallback": "empty_results",
                "note": "Neo4j 不可用，图谱检索返回空结果（不影响向量+关键词两路）",
            }
            # Neo4j 是弱依赖，不改变 overall_status
    except Exception as e:
        logger.warning(f"Neo4j健康检查失败: {str(e)}")
        services_status["neo4j"] = {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)[:100],
            "fallback": "empty_results",
        }
        # Neo4j 是弱依赖，不改变 overall_status

    # 系统资源信息（可选）
    system_info = None
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        system_info = {
            "cpu_percent": round(cpu_percent, 2),
            "memory_percent": round(memory.percent, 2),
            "memory_available_mb": round(memory.available / 1024 / 1024, 2),
            "memory_total_mb": round(memory.total / 1024 / 1024, 2),
        }
    except Exception as e:
        logger.debug(f"获取系统资源信息失败: {str(e)}")
        # 系统信息获取失败不影响健康检查

    return HealthStatus(
        status=overall_status,
        version="v0.8.6",
        services=services_status,
        system=system_info
    )


@router.get("/health/liveness")
async def liveness_check():
    """
    Kubernetes存活探针
    简单的存活检查，不检查依赖服务
    """
    return {"status": "alive"}


@router.get("/health/readiness")
async def readiness_check():
    """
    Kubernetes就绪探针
    检查关键服务是否就绪
    """
    try:
        # 检查MongoDB
        collection = mongodb.get_collection("documents")
        await collection.find_one({}, limit=1)
        
        # 如果所有关键服务都正常，返回就绪
        return {"status": "ready"}
    except Exception as e:
        logger.warning(f"就绪检查失败: {str(e)}")
        return {"status": "not_ready", "error": str(e)[:100]}


@router.get("/health/metrics")
async def metrics():
    """
    性能指标端点
    返回请求统计和系统资源使用情况
    """
    try:
        request_stats = await performance_monitor.get_stats()
        system_metrics = await performance_monitor.get_system_metrics()
        
        return {
            "request_stats": request_stats,
            "system_metrics": system_metrics
        }
    except Exception as e:
        logger.error(f"获取性能指标失败: {str(e)}", exc_info=True)
        return {"error": str(e)}

