"""工具结果落盘服务 — 大的工具结果存磁盘，messages 里只放引用

设计目的：
    工具调用结果（如 Read 大文件、WebSearch 返回大量内容、Bash 长输出）如果全部塞进
    messages 上下文，会迅速撑爆 LLM 的 context window。本服务把超过阈值的工具结果
    落盘到磁盘文件，messages 里只保留引用（文件路径 + 前 200 字符预览）。

存储位置：
    logs/tool_results/{call_id}.json

文件格式：
    {
        "call_id": "uuid",
        "tool_name": "mcp__filesystem__read_file",
        "arguments": {"path": "/test.txt"},
        "result": "完整工具结果...",
        "result_length": 12345,
        "created_at": "2026-07-20T12:34:56",
        "conversation_id": "xxx"
    }

清理策略：
    - 自动清理：超过 7 天的文件自动删除（在 store 时顺带清理）
    - 手动清理：提供 cleanup_all() 接口
"""
import os
import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from utils.logger import logger


# 触发落盘的阈值（字符数）：超过此长度的工具结果落盘
TOOL_RESULT_STORE_THRESHOLD = 2000

# 预览长度：落盘后 messages 里保留的前 N 字符
TOOL_RESULT_PREVIEW_LENGTH = 200

# 自动清理：超过此天数的文件自动删除
TOOL_RESULT_TTL_DAYS = 7

# 落盘目录
TOOL_RESULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "tool_results"
)


class ToolResultStore:
    """工具结果落盘服务（单例）"""

    def __init__(self):
        self._ensure_dir()
        self._cleanup_lock = asyncio.Lock()
        self._last_cleanup_time = 0  # 上次清理时间（用于节流，每小时最多清理一次）

    def _ensure_dir(self) -> None:
        """确保落盘目录存在"""
        os.makedirs(TOOL_RESULT_DIR, exist_ok=True)

    def should_store(self, result: Any) -> bool:
        """判断工具结果是否需要落盘

        Args:
            result: 工具调用结果（任意类型）

        Returns:
            是否需要落盘
        """
        if result is None:
            return False
        # 转成字符串计算长度
        if isinstance(result, str):
            text = result
        elif isinstance(result, (dict, list)):
            try:
                text = json.dumps(result, ensure_ascii=False)
            except Exception:
                text = str(result)
        else:
            text = str(result)
        return len(text) > TOOL_RESULT_STORE_THRESHOLD

    def store(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        conversation_id: Optional[str] = None,
    ) -> Optional[str]:
        """把工具结果落盘，返回文件路径（同步接口，在工具调用线程中执行）

        Args:
            tool_name: 工具名（如 "mcp__filesystem__read_file"）
            arguments: 工具参数
            result: 工具结果
            conversation_id: 对话 ID（可选，便于追溯）

        Returns:
            落盘文件的绝对路径；失败返回 None
        """
        try:
            self._ensure_dir()

            # 生成 call_id 和文件名
            call_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()
            file_name = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{call_id[:8]}.json"
            file_path = os.path.join(TOOL_RESULT_DIR, file_name)

            # 序列化结果
            if isinstance(result, (dict, list)):
                result_str = json.dumps(result, ensure_ascii=False, indent=2)
            elif isinstance(result, str):
                result_str = result
            else:
                result_str = str(result)

            # 构造落盘数据
            store_data = {
                "call_id": call_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result_str,
                "result_length": len(result_str),
                "created_at": timestamp.isoformat(),
                "conversation_id": conversation_id,
            }

            # 写入文件
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(store_data, f, ensure_ascii=False, indent=2)

            logger.info(
                f"工具结果落盘: tool={tool_name}, len={len(result_str)}, "
                f"file={file_name}, call_id={call_id[:8]}"
            )

            # 节流清理（每小时最多一次）
            self._maybe_cleanup()

            return file_path

        except Exception as e:
            logger.warning(f"工具结果落盘失败（非关键路径）: {e}", exc_info=True)
            return None

    def make_reference(self, tool_name: str, result: Any, file_path: str) -> str:
        """生成落盘后的引用文本（替换原始结果放入 messages）

        格式：
            [工具结果已落盘: tool=mcp__filesystem__read_file, file=xxx.json]
            预览（前 200 字符）：
            xxxxxxxx...

        Args:
            tool_name: 工具名
            result: 原始结果（用于生成预览）
            file_path: 落盘文件路径

        Returns:
            引用文本
        """
        # 生成预览
        if isinstance(result, (dict, list)):
            try:
                preview_text = json.dumps(result, ensure_ascii=False)
            except Exception:
                preview_text = str(result)
        elif isinstance(result, str):
            preview_text = result
        else:
            preview_text = str(result)

        preview = preview_text[:TOOL_RESULT_PREVIEW_LENGTH]
        if len(preview_text) > TOOL_RESULT_PREVIEW_LENGTH:
            preview += "...(truncated)"

        file_name = os.path.basename(file_path)
        return (
            f"[工具结果已落盘: tool={tool_name}, file={file_name}]\n"
            f"预览（前 {TOOL_RESULT_PREVIEW_LENGTH} 字符）：\n{preview}"
        )

    def load(self, file_name: str) -> Optional[Dict[str, Any]]:
        """读取落盘的工具结果（用于审计/调试）

        Args:
            file_name: 文件名（不含路径）

        Returns:
            落盘的数据字典；文件不存在返回 None
        """
        file_path = os.path.join(TOOL_RESULT_DIR, file_name)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"读取落盘工具结果失败: {file_name}, {e}")
            return None

    def _maybe_cleanup(self) -> None:
        """节流清理：每小时最多触发一次"""
        now = datetime.utcnow().timestamp()
        if now - self._last_cleanup_time < 3600:  # 1 小时
            return
        self._last_cleanup_time = now
        try:
            self._cleanup_old_files()
        except Exception as e:
            logger.debug(f"清理旧工具结果文件失败（非关键路径）: {e}")

    def _cleanup_old_files(self) -> int:
        """清理超过 TTL 的旧文件

        Returns:
            清理的文件数
        """
        if not os.path.exists(TOOL_RESULT_DIR):
            return 0

        cutoff = datetime.utcnow() - timedelta(days=TOOL_RESULT_TTL_DAYS)
        cutoff_ts = cutoff.timestamp()
        cleaned = 0

        for file_name in os.listdir(TOOL_RESULT_DIR):
            if not file_name.endswith(".json"):
                continue
            file_path = os.path.join(TOOL_RESULT_DIR, file_name)
            try:
                mtime = os.path.getmtime(file_path)
                if mtime < cutoff_ts:
                    os.remove(file_path)
                    cleaned += 1
            except Exception:
                continue

        if cleaned > 0:
            logger.info(f"清理过期工具结果文件: {cleaned} 个（TTL={TOOL_RESULT_TTL_DAYS}天）")
        return cleaned

    def cleanup_all(self) -> int:
        """清理所有落盘文件（手动触发）"""
        if not os.path.exists(TOOL_RESULT_DIR):
            return 0
        cleaned = 0
        for file_name in os.listdir(TOOL_RESULT_DIR):
            if not file_name.endswith(".json"):
                continue
            try:
                os.remove(os.path.join(TOOL_RESULT_DIR, file_name))
                cleaned += 1
            except Exception:
                continue
        logger.info(f"手动清理所有工具结果文件: {cleaned} 个")
        return cleaned

    def get_stats(self) -> Dict[str, Any]:
        """获取落盘统计信息"""
        if not os.path.exists(TOOL_RESULT_DIR):
            return {"total_files": 0, "total_size_bytes": 0}
        files = [f for f in os.listdir(TOOL_RESULT_DIR) if f.endswith(".json")]
        total_size = sum(
            os.path.getsize(os.path.join(TOOL_RESULT_DIR, f))
            for f in files
            if os.path.exists(os.path.join(TOOL_RESULT_DIR, f))
        )
        return {
            "total_files": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "store_dir": TOOL_RESULT_DIR,
            "threshold": TOOL_RESULT_STORE_THRESHOLD,
            "ttl_days": TOOL_RESULT_TTL_DAYS,
        }


# 全局单例
tool_result_store = ToolResultStore()
