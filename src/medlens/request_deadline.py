"""Dependency-free full-request wall-clock budget for the shipping application.

A budget starts at transport ingress, before validation and admission. Native work
ends early enough to reserve time for process cleanup and terminal delivery.
These bounds are enforcement settings, not a claim about observed client latency.
"""

from dataclasses import dataclass
import math
import time

from .mvp import (REQUEST_TIMEOUT_SECONDS, REQUEST_CLEANUP_RESERVE_SECONDS,
                  REQUEST_DELIVERY_RESERVE_SECONDS)


@dataclass(frozen=True)
class RequestBudget:
    started: float
    seconds: float = REQUEST_TIMEOUT_SECONDS
    cleanup_reserve: float = REQUEST_CLEANUP_RESERVE_SECONDS
    delivery_reserve: float = REQUEST_DELIVERY_RESERVE_SECONDS

    def __post_init__(self):
        values = (self.started, self.seconds, self.cleanup_reserve, self.delivery_reserve)
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("Request budget values must be finite numbers")
        if self.seconds <= 0 or min(self.cleanup_reserve, self.delivery_reserve) < 0:
            raise ValueError("Invalid request budget")
        if self.cleanup_reserve + self.delivery_reserve >= self.seconds:
            raise ValueError("Request budget has no computation allowance")

    @property
    def work_deadline(self):
        return self.started + self.seconds - self.cleanup_reserve - self.delivery_reserve

    @property
    def cleanup_deadline(self):
        return self.started + self.seconds - self.delivery_reserve

    def elapsed(self):
        return max(0.0, time.monotonic() - self.started)
