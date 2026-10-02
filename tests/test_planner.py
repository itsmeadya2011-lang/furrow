"""Tests for the Furrow planner agent."""

from __future__ import annotations

import json
from typing import Any

import pytest

from furrow.agents.planner import PlannerAgent, _strip_markdown_fence
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient


class FakeLLMClient:
    """A fake LLM client that returns canned responses."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.complete_calls: list[tuple[str, str]] = []
        self.settings = type("FakeSettings", (), {"planner_model": "fake"})()

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls.append((prompt, model or ""))
        return self._response


# --- _strip_markdown_fence ---


def test_strip_markdown_plain_json():
    text = '{"tasks": [], "rationale": "ok"}'
    assert _strip_markdown_fence(text) == text


def test_strip_markdown_json_block():
    text = '```json\n{"tasks": [], "rationale": "ok"}\n```'
    result = _strip_markdown_fence(text)
    assert result == '{"tasks": [], "rationale": "ok"}'


def test_strip_markdown_plain_fence():
    text = '''
{"tasks": []}
'''
    result = _strip_markdown_fence(text)
    assert result.startswith("{")
    assert result.endswith("}")


def test_strip_markdown_extra_whitespace():
    text = '```json\n{"tasks": []}\n```\n\n'
    result = _strip_markdown_fence(text)
    assert result == '{"tasks": []}'


# --- PlannerAgent ---


@pytest.mark.asyncio
async def test_planner_parses_json_response():
    response = json.dumps({
        "tasks": [
            {"id": "1", "description": "Write tests", "files": ["test.py"], "dependencies": []}
        ],
        "rationale": "Test coverage needed",
    })
    planner = PlannerAgent(client=FakeLLMClient(response))  # type: ignore[arg-type]
    plan = await planner.plan("Add tests")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "Write tests"
    assert plan.rationale == "Test coverage needed"


@pytest.mark.asyncio
async def test_planner_parses_fenced_json():
    json_str = json.dumps({
        "tasks": [{"id": "1", "description": "Fix bug", "files": [], "dependencies": []}],
        "rationale": "Bug fix",
    })
    response = f"```json\n{json_str}\n```"
    planner = PlannerAgent(client=FakeLLMClient(response))  # type: ignore[arg-type]
    plan = await planner.plan("Fix the bug")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"


@pytest.mark.asyncio
async def test_planner_extracts_json_from_text():
    """If the LLM includes extra text, the parser should extract the JSON."""
    json_str = json.dumps({
        "tasks": [],
        "rationale": "Already complete",
    })
    response = f"Some preamble\n{json_str}\nSome trailing text"
    planner = PlannerAgent(client=FakeLLMClient(response))  # type: ignore[arg-type]
    plan = await planner.plan("Already done task")
    # This should still parse via regex fallback
    assert plan.rationale == "Already complete"


@pytest.mark.asyncio
async def test_planner_raises_on_garbage():
    planner = PlannerAgent(client=FakeLLMClient("Not JSON at all"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Some goal")


@pytest.mark.asyncio
async def test_planner_calls_complete_with_model():
    response = json.dumps({"tasks": [], "rationale": "ok"})
    fake = FakeLLMClient(response)
    planner = PlannerAgent(client=fake)  # type: ignore[arg-type]
    await planner.plan("My goal")
    prompt, model = fake.complete_calls[0]
    assert "My goal" in prompt
    assert model == "fake"  # planner_model


@pytest.mark.asyncio
async def test_planner_empty_plan():
    response = json.dumps({"tasks": [], "rationale": "Already complete"})
    planner = PlannerAgent(client=FakeLLMClient(response))  # type: ignore[arg-type]
    plan = await planner.plan("Done")
    assert len(plan.tasks) == 0


@pytest.mark.asyncio
async def test_planner_multiple_tasks():
    response = json.dumps({
        "tasks": [
            {"id": "1", "description": "Task A", "files": ["a.py"], "dependencies": []},
            {"id": "2", "description": "Task B", "files": ["b.py"], "dependencies": ["1"]},
        ],
        "rationale": "Two tasks",
    })
    planner = PlannerAgent(client=FakeLLMClient(response))  # type: ignore[arg-type]
    plan = await planner.plan("Build something")
    assert len(plan.tasks) == 2
    assert plan.tasks[1].dependencies == ["1"]
