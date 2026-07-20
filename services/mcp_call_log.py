"""MCP Call Log：追踪每次 tool call 的状态，支持失败后查询

解决的问题：
- 修复前：执行中异常断开，原调用状态丢失，不知道 Server 有没有收到/执行
- 修复后：每次 call_tool 生成 call_id，记录状态（pending/success/failed/retrying），
  失败时返回 call_id 给调用方，可查询调用历史

设计：
- 内存存储（不持久化，重启清空）
- 按 server_name 分组
- 自动清理超过 1 小时的记录
"""
import time
import uuid
from typing import Dict, List, Optional, Any
from threading import Lock
from utils.logger import logger


# 记录保留时间（秒）
RECORD_TTL = 3600  # 1 小时
# 每个 server 最多保留多少条记录
MAX_RECORDS_PER_SERVER = 100


class CallRecord:
    """单次 tool call 的记录"""

    def __init__(
        self,
        call_id: str,
        server_name: str,
        tool_name: str,
        arguments: dict,
    ):
        self.call_id = call_id
        self.server_name = server_name
        self.tool_name = tool_name
        self.arguments = arguments
        self.status: str = "pending"  # pending / success / failed / retrying / dropped
        self.start_ts: float = time.time()
        self.end_ts: Optional[float] = None
        self.error: Optional[str] = None
        self.result: Optional[str] = None
        self.retry_count: int = 0
        self.idempotent: Optional[bool] = None  # 是否被幂等保护跳过重试

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "server_name": self.server_name,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "status": self.status,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration_ms": int((self.end_ts or time.time() - self.start_ts) * 1000),
            "error": self.error,
            "result_preview": (self.result[:200] + "...") if self.result and len(self.result) > 200 else self.result,
            "retry_count": self.retry_count,
            "idempotent_skipped": self.idempotent,
        }


class McpCallLog:
    """MCP 调用日志：按 server 分组存储 call 记录"""

    def __init__(self):
        self._records: Dict[str, List[CallRecord]] = {}
        self._lock = Lock()

    def new_call(self, server_name: str, tool_name: str, arguments: dict) -> CallRecord:
        """新建一条 call 记录，返回 record（含 call_id）"""
        call_id = f"mcp-{uuid.uuid4().hex[:12]}"
        record = CallRecord(call_id, server_name, tool_name, arguments)
        with self._lock:
            self._records.setdefault(server_name, []).append(record)
            # 清理过期 + 限制数量
            self._cleanup_locked(server_name)
        return record

    def mark_success(self, record: CallRecord, result: str) -> None:
        record.status = "success"
        record.end_ts = time.time()
        record.result = result

    def mark_failed(self, record: CallRecord, error: str) -> None:
        record.status = "failed"
        record.end_ts = time.time()
        record.error = error

    def mark_retrying(self, record: CallRecord) -> None:
        record.status = "retrying"
        record.retry_count += 1

    def mark_dropped(self, record: CallRecord, reason: str) -> None:
        """标记为「丢弃」（非幂等工具不重试）"""
        record.status = "dropped"
        record.end_ts = time.time()
        record.error = reason
        record.idempotent = True

    def get_records(self, server_name: Optional[str] = None, limit: int = 20) -> List[dict]:
        """查询调用记录

        Args:
            server_name: 指定 server，None 查所有
            limit: 最多返回多少条
        """
        with self._lock:
            if server_name:
                records = list(self._records.get(server_name, []))
            else:
                records = []
                for recs in self._records.values():
                    records.extend(recs)
        # 按时间倒序
        records.sort(key=lambda r: r.start_ts, reverse=True)
        return [r.to_dict() for r in records[:limit]]

    def get_record(self, call_id: str) -> Optional[dict]:
        """按 call_id 查询单条记录"""
        with self._lock:
            for recs in self._records.values():
                for r in recs:
                    if r.call_id == call_id:
                        return r.to_dict()
        return None

    def _cleanup_locked(self, server_name: str) -> None:
        """清理过期记录 + 限制数量（调用前需持有锁）"""
        records = self._records.get(server_name, [])
        now = time.time()
        # 清理 TTL 过期
        records[:] = [r for r in records if now - r.start_ts < RECORD_TTL]
        # 限制数量（保留最近的 N 条）
        if len(records) > MAX_RECORDS_PER_SERVER:
            records[:] = records[-MAX_RECORDS_PER_SERVER:]
        self._records[server_name] = records


# 全局单例
mcp_call_log = McpCallLog()
