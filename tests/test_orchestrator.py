from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel
from furrow.core.orchestrator import Orchestrator
from furrow.llm import CompletionResult, LLMClient, ToolCall

PLAN_JSON = (
    '{"tasks":[{"id":"1","description":"Create result.py placeholder","files":["result.py"],'
    '"dependencies":[]}],"rationale":"ok"}'
)
TEST_PASS_JSON = '{"passed": true, "summary": "All tests passed", "failures": []}'
TEST_FAIL_JSON = '{"passed": false, "summary": "fail", "failures": ["broken thing"]}'


class FakeLLMClient(LLMClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings=settings)
        self.complete_calls: list[str] = []
        self.cwt_calls: int = 0
        self.force_test_fail: bool = False

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls.append(prompt)
        if "Break the user's goal" in prompt:
            return PLAN_JSON
        if "determine if tests" in prompt:
            if self.force_test_fail:
                return TEST_FAIL_JSON
            return TEST_PASS_JSON
        return "ok"

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        system: str = "",
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        self.cwt_calls += 1
        has_tool_msg = any(m.get("role") == "tool" for m in messages)
        if not has_tool_msg:
            return CompletionResult(
                text="I will create the file.",
                tool_calls=[
                    ToolCall(
                        id="t1", name="write_file", args={"path": "result.py", "content": "# done"}
                    )
                ],
            )
        return CompletionResult(text="Created result.py.", tool_calls=[])


def _make_settings(tmp_path: Path, max_cycles: int = 0, max_parallel: int = 5) -> Settings:
    return Settings(
        max_cycles=max_cycles,
        max_parallel_tasks=max_parallel,
        workspace=tmp_path,
        provider=Provider.ANTHROPIC,
        model="x",
        planner_model="x",
        worker_model="x",
        tester_model="x",
    )


@pytest.mark.asyncio
async def test_orchestrator_completes_when_tests_pass(tmp_path):
    settings = _make_settings(tmp_path, max_cycles=0)
    fake = FakeLLMClient(settings)
    collected: list[str] = []

    async def out(msg: str) -> None:
        collected.append(msg)

    orch = Orchestrator(goal="Create a placeholder file", client=fake, output=out)
    await orch.run()

    assert orch.cycles == 1
    assert orch.last_test_passed is True
    assert orch._is_done() is True
    assert (tmp_path / "result.py").exists()
    assert any("Goal complete" in m for m in collected)


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles(tmp_path):
    settings = _make_settings(tmp_path, max_cycles=2, max_parallel=5)
    fake = FakeLLMClient(settings)
    fake.force_test_fail = True
    collected: list[str] = []

    async def out(msg: str) -> None:
        collected.append(msg)

    orch = Orchestrator(goal="Keep failing", client=fake, output=out)
    await orch.run()

    assert orch.cycles == 3
    assert orch.last_test_passed is False
    assert any("Max cycles" in m for m in collected)


@pytest.mark.asyncio
async def test_orchestrator_no_output_uses_console(tmp_path):
    settings = _make_settings(tmp_path, max_cycles=0)
    fake = FakeLLMClient(settings)
    orch = Orchestrator(goal="Console path", client=fake)
    await orch.run()
    assert orch.cycles == 1
    assert (tmp_path / "result.py").exists()


@pytest.mark.asyncio
async def test_orchestrator_parallel_tasks(tmp_path):
    settings = _make_settings(tmp_path, max_cycles=0, max_parallel=2)
    fake = FakeLLMClient(settings)
    collected: list[str] = []

    async def out(msg: str) -> None:
        collected.append(msg)

    orch = Orchestrator(goal="Two files", client=fake, output=out)

    async def fake_plan(goal: str) -> Plan:
        return Plan(
            tasks=[
                TaskModel(id="a", description="file a", files=["a.py"]),
                TaskModel(id="b", description="file b", files=["b.py"]),
            ],
            rationale="two",
        )

    orch.planner.plan = fake_plan  # type: ignore[assignment]

    async def patched_cwt(
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        system: str = "",
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        has_tool = any(m.get("role") == "tool" for m in messages)
        if not has_tool:
            user_text = ""
            for m in messages:
                if m.get("role") == "user":
                    user_text = m.get("content", "") or ""
                    break
            fname = "a.py" if "file a" in user_text else "b.py"
            return CompletionResult(
                text="creating",
                tool_calls=[
                    ToolCall(id="tc", name="write_file", args={"path": fname, "content": "# x"})
                ],
            )
        return CompletionResult(text="done", tool_calls=[])

    fake.complete_with_tools = patched_cwt  # type: ignore[method-assign]

    await orch.run()
    assert orch.cycles == 1
    assert orch.last_test_passed is True
    assert (tmp_path / "a.py").exists()
    assert (tmp_path / "b.py").exists()
