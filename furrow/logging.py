"""Logging configuration for Furrow.

Configures structlog to wrap stdlib logging when available, with a defensive
fallback to stdlib logging if structlog is missing.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any


def _configure_stdlib_fallback(level: str = "INFO") -> None:
    """Configure standard library logging as a fallback."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


try:
    import structlog
    _STRUCTLOG_AVAILABLE = True
except ImportError:
    structlog = None  # type: ignore[assignment]
    _STRUCTLOG_AVAILABLE = False


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog to wrap stdlib logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        stream=sys.stdout,
    )

    if not _STRUCTLOG_AVAILABLE:
        # Fall back to stdlib logging if structlog is not available
        _configure_stdlib_fallback(level)
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Return a structlog logger, falling back to stdlib if needed."""
    if _STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return logging.getLogger(name)


class ProgressTracker:
    """Track progress of a Furrow run with structured logging."""
    
    def __init__(self, logger: Any, run_id: str = ""):
        self.logger = logger
        self.run_id = run_id
        self.start_time = time.time()
        self.cycle_count = 0
        self.task_count = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.tokens_used = 0
        self.cost_usd = 0.0
    
    def cycle_started(self, cycle: int, goal: str) -> None:
        """Log cycle start."""
        self.cycle_count = cycle
        self.logger.info(
            "cycle_started",
            run_id=self.run_id,
            cycle=cycle,
            goal=goal[:200],
            elapsed_seconds=time.time() - self.start_time,
        )
    
    def cycle_finished(self, cycle: int, tasks_planned: int, tests_passed: bool | None, test_summary: str) -> None:
        """Log cycle completion."""
        self.logger.info(
            "cycle_finished",
            run_id=self.run_id,
            cycle=cycle,
            tasks_planned=tasks_planned,
            tests_passed=tests_passed,
            test_summary=test_summary[:200],
            elapsed_seconds=time.time() - self.start_time,
        )
    
    def task_started(self, task_id: str, description: str) -> None:
        """Log task start."""
        self.task_count += 1
        self.logger.info(
            "task_started",
            run_id=self.run_id,
            task_id=task_id,
            description=description[:200],
            cycle=self.cycle_count,
        )
    
    def task_finished(self, task_id: str, success: bool, summary: str, attempts: int) -> None:
        """Log task completion."""
        if success:
            self.completed_tasks += 1
        else:
            self.failed_tasks += 1
        self.logger.info(
            "task_finished",
            run_id=self.run_id,
            task_id=task_id,
            success=success,
            summary=summary[:200],
            attempts=attempts,
            cycle=self.cycle_count,
        )
    
    def usage_updated(self, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        """Log token usage update."""
        self.tokens_used = input_tokens + output_tokens
        self.cost_usd = cost_usd
        self.logger.info(
            "usage_updated",
            run_id=self.run_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=self.tokens_used,
            cost_usd=cost_usd,
        )
    
    def run_finished(self, stopped_reason: str, passed: bool | None) -> None:
        """Log run completion."""
        elapsed = time.time() - self.start_time
        self.logger.info(
            "run_finished",
            run_id=self.run_id,
            stopped_reason=stopped_reason,
            passed=passed,
            total_cycles=self.cycle_count,
            total_tasks=self.task_count,
            completed_tasks=self.completed_tasks,
            failed_tasks=self.failed_tasks,
            total_tokens=self.tokens_used,
            total_cost_usd=self.cost_usd,
            elapsed_seconds=elapsed,
        )
    
    @contextmanager
    def task_timer(self, task_id: str, description: str):
        """Context manager for timing task execution."""
        start = time.time()
        self.task_started(task_id, description)
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.logger.debug(
                "task_timed",
                run_id=self.run_id,
                task_id=task_id,
                elapsed_seconds=elapsed,
            )
