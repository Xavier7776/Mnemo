"""MCP 熔断器：防止反复失败的 Server 拖垮整个系统

三态机：
- CLOSED（正常）：请求正常通过，失败计数累积
- OPEN（熔断）：直接拒绝请求，等待 cooldown 后进入 HALF_OPEN
- HALF_OPEN（半开试探）：放一个请求试探，成功 → CLOSED，失败 → OPEN

触发条件：
- 连续失败 N 次（默认 5）→ CLOSED → OPEN
- OPEN 持续 cooldown 秒（默认 60）→ OPEN → HALF_OPEN
- HALF_OPEN 成功一次 → CLOSED（重置失败计数）
- HALF_OPEN 失败一次 → OPEN（重新计时）
"""
import time
from typing import Optional
from utils.logger import logger


class CircuitBreaker:
    """单 Server 的熔断器"""

    CLOSED = "closed"       # 正常
    OPEN = "open"           # 熔断
    HALF_OPEN = "half_open" # 半开试探

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        """
        Args:
            name: 熔断器名称（通常是 server_name）
            failure_threshold: 连续失败多少次触发熔断
            cooldown_seconds: OPEN 状态持续时间，到时间转 HALF_OPEN
            half_open_max_calls: HALF_OPEN 状态下最多放行多少个试探请求
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_ts: float = 0.0
        self._opened_at: float = 0.0  # 进入 OPEN 的时间
        self._half_open_calls: int = 0  # HALF_OPEN 状态下已放行的请求数

    @property
    def state(self) -> str:
        """当前状态（带自动状态转移）"""
        if self._state == self.OPEN:
            # 检查 cooldown 是否到期
            if time.time() - self._opened_at >= self.cooldown_seconds:
                self._state = self.HALF_OPEN
                self._half_open_calls = 0
                logger.info(f"MCP 熔断器 [{self.name}] OPEN → HALF_OPEN（cooldown 到期，试探请求）")
        return self._state

    def allow_request(self) -> bool:
        """是否允许请求通过

        Returns:
            True 放行，False 拒绝（熔断中）
        """
        current = self.state
        if current == self.CLOSED:
            return True
        if current == self.OPEN:
            return False
        # HALF_OPEN：限制试探请求数
        if self._half_open_calls < self.half_open_max_calls:
            self._half_open_calls += 1
            return True
        return False

    def record_success(self) -> None:
        """记录一次成功"""
        if self._state == self.HALF_OPEN:
            # HALF_OPEN 成功 → CLOSED
            logger.info(f"MCP 熔断器 [{self.name}] HALF_OPEN → CLOSED（试探成功）")
            self._state = self.CLOSED
            self._failure_count = 0
            self._success_count += 1
            self._half_open_calls = 0
        elif self._state == self.CLOSED:
            self._success_count += 1
            # 成功重置失败计数
            self._failure_count = 0

    def record_failure(self) -> None:
        """记录一次失败"""
        self._last_failure_ts = time.time()

        if self._state == self.HALF_OPEN:
            # HALF_OPEN 失败 → OPEN
            logger.warning(f"MCP 熔断器 [{self.name}] HALF_OPEN → OPEN（试探失败）")
            self._state = self.OPEN
            self._opened_at = time.time()
            self._half_open_calls = 0
        elif self._state == self.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                logger.warning(
                    f"MCP 熔断器 [{self.name}] CLOSED → OPEN（连续失败 {self._failure_count} 次）"
                )
                self._state = self.OPEN
                self._opened_at = time.time()

    def reset(self) -> None:
        """重置熔断器（如手动重连成功时）"""
        self._state = self.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0

    def get_status(self) -> dict:
        """获取熔断器状态（用于 API/日志）"""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.failure_threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "opened_at": self._opened_at,
            "last_failure_ts": self._last_failure_ts,
        }
