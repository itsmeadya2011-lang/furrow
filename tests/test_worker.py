"""Tests for the Furrow worker agent."""

from __future__ import annotations

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel


class FakeLLMClient:
    def __init__(self) -> None:
        self.settings = Settings()
        self.tool_call_results: list[str] = []

    async def complete_with_tools(self, prompt: str, system: str = "", **kwargs) -> str:
        self.tool_call_results.append(prompt)
        return "Task implemented."

    async def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        return "Task implemented."


@pytest.mark.asyncio
async def test_worker_returns_completion_message():
    fake = FakeLLMClient()
    task = TaskModel(id="1", description="Write a README", files=["README.md"])
    worker = WorkerAgent(task=task, client=fake)  # type: ignore[arg-type]
    result = await worker.run()
    assert result == "Task implemented."
    assert len(fake.tool_call_results) == 1


@pytest.mark.asyncio
async def test_worker_prompt_includes_task_description():
    fake = FakeLLMClient()
    task = TaskModel(id="1", description="Implement user auth", files=["src/auth.py"])
    worker = WorkerAgent(task=task, client=fake)  # type: ignore[arg-type]
    await worker.run()
    prompt = fake.tool_call_results[0]
    assert "Implement user auth" in prompt
    assert "src/auth.py" in prompt


@pytest.mark.asyncio
async def test_worker_prompt_handles_no_files():
    fake = FakeLLMClient()
    task = TaskModel(id="1", description="Refactor module", files=[])
    worker = WorkerAgent(task=task, client=fake)  # type: ignore[arg-type]
    await worker.run()
    prompt = fake.tool_call_results[0]
    assert "any relevant" in prompt


@pytest.mark.asyncio
async def test_worker_default_tools():
    from furrow.tools import get_default_tools
    worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=FakeLLMClient())  # type: ignore[arg-type]
    assert len(worker.tools) == 5


@pytest.mark.asyncio
async def test_worker_custom_tools():
    from furrow.tools.base import Tool, ToolResult

    class CustomTool(Tool):
        @property
        def name(self) -> str:
            return "custom"

        @property
        def description(self) -> str:
            return "A custom tool"

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, output="custom")

    fake = FakeLLMClient()
    task = TaskModel(id="1", description="test")
    worker = WorkerAgent(task=task, client=fake, tools=[CustomTool()])  # type: ignore[arg-type]
    assert len(worker.tools) == 1
    assert worker.tools[0].name == "custom"
