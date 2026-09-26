"""Tests for Orchestrator task lifecycle and max_cycles enforcement."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import TaskModel
from furrow.core.orchestrator import Orchestrator


def _orch(client=None) -> Orchestrator:
    """Build an Orchestrator with a mock client to avoid real LLM calls."""
    return Orchestrator(goal="test", client=client or MagicMock())


# ---------------------------------------------------------------------------
# _get_tasks / _is_done lifecycle
# ---------------------------------------------------------------------------

def test_get_tasks_initially_empty():
    orch = _orch()
    assert orch._get_tasks() == []


def test_is_done_empty_tasks():
    orch = _orch()
    assert orch._is_done() is True


def test_get_tasks_returns_set_tasks():
    orch = _orch()
    tasks = [TaskModel(id="1", description="a")]
    orch.tasks = tasks
    assert orch._get_tasks() is tasks


def test_is_done_all_completed():
    orch = _orch()
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_is_done_has_pending():
    orch = _orch()
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orch._is_done() is False


def test_is_done_has_failed():
    orch = _orch()
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    # Failed tasks take priority — should return False even if others completed.
    assert orch._is_done() is False


def test_is_done_single_pending():
    orch = _orch()
    orch.tasks = [TaskModel(id="1", description="a", status="pending")]
    assert orch._is_done() is False


# ---------------------------------------------------------------------------
# max_cycles enforcement
# ---------------------------------------------------------------------------

async def test_run_halts_at_max_cycles():
    """With max_cycles=1 the loop should stop after exactly 1 cycle."""
    client = MagicMock()
    client.settings.max_cycles = 1
    orch = Orchestrator(goal="test", client=client)
    orch._cycle = AsyncMock()
    orch._is_done = MagicMock(return_value=False)
    await orch.run()
    assert orch.cycles == 1


async def test_run_max_cycles_zero_unlimited():
    """With max_cycles=0 the loop should NOT halt due to max_cycles;
    it stops only when _is_done() returns True."""
    client = MagicMock()
    client.settings.max_cycles = 0
    orch = Orchestrator(goal="test", client=client)
    orch._cycle = AsyncMock()
    # Stop after one cycle via _is_done to avoid an infinite loop.
    call_count = 0

    def _is_done():
        nonlocal call_count
        call_count += 1
        return call_count >= 1

    orch._is_done = _is_done
    await orch.run()
    assert orch.cycles == 1
    assert client.settings.max_cycles == 0
