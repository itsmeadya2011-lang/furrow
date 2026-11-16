"""Testing-related functionality for the orchestrator."""

from __future__ import annotations

from typing import Any

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.core.events import Event, EventBus, EventType
from furrow.llm import LLMClient


async def run_tests(
    original_goal: str,
    tasks: list[TaskModel],
    client: LLMClient,
    settings: Settings,
    workspace: Any,
    event_bus: EventBus,
) -> TestResult:
    """Run tests and return the result."""
    tester = TesterAgent(client=client, settings=settings, workspace=workspace)
    event_bus.emit(Event(type=EventType.TEST_STARTED, data={}))
    try:
        test_result = await tester.run(original_goal, tasks)
    except Exception as e:
        event_bus.emit(Event(type=EventType.ERROR, message=str(e)))
        raise
    event_bus.emit(
        Event(
            type=EventType.TEST_FINISHED,
            data={"passed": test_result.passed, "summary": test_result.summary},
        )
    )
    return test_result