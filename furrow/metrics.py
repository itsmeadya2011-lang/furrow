"""Metrics collection and reporting for Furrow.

This module provides a lightweight metrics system for tracking run statistics,
performance data, and operational metrics. It uses a simple in-memory store
with optional periodic reporting.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any


@dataclass
class MetricValue:
    """A single metric value with metadata."""
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Counter:
    """A monotonically increasing counter."""
    name: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        if labels:
            self.labels.update(labels)
        self.value += amount


@dataclass
class Gauge:
    """A gauge that can go up and down."""
    name: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        if labels:
            self.labels.update(labels)
        self.value = value

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        if labels:
            self.labels.update(labels)
        self.value += amount

    def dec(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        if labels:
            self.labels.update(labels)
        self.value -= amount


@dataclass
class Histogram:
    """A histogram for tracking distributions."""
    name: str
    buckets: list[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf")]
    )
    values: list[float] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    _count: int = 0
    _sum: float = 0.0

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        if labels:
            self.labels.update(labels)
        self.values.append(value)
        self._count += 1
        self._sum += value

    @property
    def count(self) -> int:
        return self._count

    @property
    def sum(self) -> float:
        return self._sum

    def buckets_counts(self) -> dict[float, int]:
        """Return count of values in each bucket."""
        counts: dict[float, int] = defaultdict(int)
        for v in self.values:
            for bucket in self.buckets:
                if v <= bucket:
                    counts[bucket] += 1
                    break
        return counts


class MetricsRegistry:
    """Thread-safe registry for all metrics."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = Lock()

    def counter(self, name: str, labels: dict[str, str] | None = None) -> Counter:
        """Get or create a counter."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name=name, labels=labels or {})
            return self._counters[name]

    def gauge(self, name: str, labels: dict[str, str] | None = None) -> Gauge:
        """Get or create a gauge."""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name=name, labels=labels or {})
            return self._gauges[name]

    def histogram(
        self,
        name: str,
        buckets: list[float] | None = None,
        labels: dict[str, str] | None = None,
    ) -> Histogram:
        """Get or create a histogram."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(
                    name=name,
                    buckets=buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float('inf')],
                    labels=labels or {}
                )
            return self._histograms[name]

    def get_all(self) -> dict[str, Any]:
        """Get all metrics as a serializable dict."""
        with self._lock:
            return {
                "counters": {
                    name: {"value": c.value, "labels": c.labels}
                    for name, c in self._counters.items()
                },
                "gauges": {
                    name: {"value": g.value, "labels": g.labels}
                    for name, g in self._gauges.items()
                },
                "histograms": {
                    name: {
                        "count": h.count,
                        "sum": h.sum,
                        "buckets": h.buckets_counts(),
                        "labels": h.labels,
                    }
                    for name, h in self._histograms.items()
                },
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# Global registry instance
_REGISTRY = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    return _REGISTRY


def counter(name: str, labels: dict[str, str] | None = None) -> Counter:
    """Get or create a counter from the global registry."""
    return _REGISTRY.counter(name, labels)


def gauge(name: str, labels: dict[str, str] | None = None) -> Gauge:
    """Get or create a gauge from the global registry."""
    return _REGISTRY.gauge(name, labels)


def histogram(
    name: str,
    buckets: list[float] | None = None,
    labels: dict[str, str] | None = None,
) -> Histogram:
    """Get or create a histogram from the global registry."""
    return _REGISTRY.histogram(name, buckets, labels)


class RunMetrics:
    """Metrics specific to a single Furrow run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.start_time = time.monotonic()
        self.cycles_completed = counter("furrow_cycles_completed_total", {"run_id": run_id})
        self.tasks_completed = counter("furrow_tasks_completed_total", {"run_id": run_id})
        self.tasks_failed = counter("furrow_tasks_failed_total", {"run_id": run_id})
        self.tasks_blocked = counter("furrow_tasks_blocked_total", {"run_id": run_id})
        self.tasks_skipped = counter("furrow_tasks_skipped_total", {"run_id": run_id})
        self.cycle_duration = histogram("furrow_cycle_duration_seconds", labels={"run_id": run_id})
        self.task_duration = histogram("furrow_task_duration_seconds", labels={"run_id": run_id})
        self.llm_calls = counter("furrow_llm_calls_total", {"run_id": run_id})
        self.llm_tokens = counter("furrow_llm_tokens_total", {"run_id": run_id})
        self.llm_cost = counter("furrow_llm_cost_usd_total", {"run_id": run_id})
        self.errors = counter("furrow_errors_total", {"run_id": run_id})
        self.state_saves = counter("furrow_state_saves_total", {"run_id": run_id})
        self.state_save_failures = counter("furrow_state_save_failures_total", {"run_id": run_id})
        self.active_tasks = gauge("furrow_active_tasks", {"run_id": run_id})
        self.current_cycle = gauge("furrow_current_cycle", {"run_id": run_id})

    def record_cycle_start(self, cycle: int) -> None:
        """Record the start of a cycle."""
        self.current_cycle.set(cycle)
        self._cycle_start = time.monotonic()

    def record_cycle_end(self, cycle: int) -> None:
        """Record the end of a cycle."""
        duration = time.monotonic() - self._cycle_start
        self.cycle_duration.observe(duration)
        self.cycles_completed.inc()
        self.current_cycle.set(cycle)

    def record_task_start(self) -> None:
        """Record the start of a task."""
        self.active_tasks.inc()
        self._task_start = time.monotonic()

    def record_task_end(self, status: str) -> None:
        """Record the end of a task."""
        duration = time.monotonic() - self._task_start
        self.task_duration.observe(duration)
        self.active_tasks.dec()
        if status == "completed":
            self.tasks_completed.inc()
        elif status == "failed":
            self.tasks_failed.inc()
        elif status == "blocked":
            self.tasks_blocked.inc()
        elif status == "skipped":
            self.tasks_skipped.inc()

    def record_llm_usage(self, calls: int, tokens: int, cost: float) -> None:
        """Record LLM usage."""
        self.llm_calls.inc(calls)
        self.llm_tokens.inc(tokens)
        self.llm_cost.inc(cost)

    def record_error(self, error_type: str) -> None:
        """Record an error."""
        self.errors.inc(labels={"type": error_type})

    def record_state_save(self, success: bool) -> None:
        """Record a state save attempt."""
        if success:
            self.state_saves.inc()
        else:
            self.state_save_failures.inc()

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of run metrics."""
        elapsed = time.monotonic() - self.start_time
        return {
            "run_id": self.run_id,
            "elapsed_seconds": elapsed,
            "cycles_completed": self.cycles_completed.value,
            "tasks_completed": self.tasks_completed.value,
            "tasks_failed": self.tasks_failed.value,
            "tasks_blocked": self.tasks_blocked.value,
            "tasks_skipped": self.tasks_skipped.value,
            "llm_calls": self.llm_calls.value,
            "llm_tokens": self.llm_tokens.value,
            "llm_cost_usd": self.llm_cost.value,
            "errors": self.errors.value,
            "state_saves": self.state_saves.value,
            "state_save_failures": self.state_save_failures.value,
        }


async def periodic_metrics_reporter(interval: float = 60.0) -> None:
    """Periodically report metrics to structured logs."""
    from furrow.logging import get_logger
    logger = get_logger(__name__)
    while True:
        await asyncio.sleep(interval)
        metrics = _REGISTRY.get_all()
        if any(metrics["counters"].values()) or any(metrics["gauges"].values()):
            logger.info("metrics_snapshot", extra={"metrics": metrics})