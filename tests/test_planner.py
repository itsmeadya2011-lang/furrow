"""Regression tests for the ``furrow.agents.planner`` rewrite.

Self-contained on purpose: this module defines its own stub LLM client so it can
run without a shared ``conftest.py`` and without any network access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from furrow.agents.planner import (
    MAX_PLAN_ATTEMPTS,
    MAX_RESPONSE_EXCERPT,
    STRICT_JSON_INSTRUCTION,
    PlannerAgent,
    extract_json,
)

_PLAN_OBJECT: dict[str, Any] = {
    "tasks": [{"id": "1", "description": "do the thing"}],
    "rationale": "because",
}


@dataclass
class _StubSettings:
    planner_model: str = "stub-planner-model"
    worker_model: str = "stub-worker-model"
    tester_model: str = "stub-tester-model"
    workspace: Path = Path("/tmp/furrow-planner-stub-workspace")


class _StubClient:
    """Replays canned ``complete`` replies and records every call."""

    def __init__(self, responses: list[str]) -> None:
        self.settings = _StubSettings()
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.written: dict[str, str] = {}

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def _next(self) -> str:
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        return self._next()

    async def read_file(self, path: str | Path) -> str:
        return ""

    async def write_file(self, path: str | Path, content: str) -> None:
        self.written[str(path)] = content

    def list_files(self, directory: str | Path) -> list[str]:
        return []


def _as_text(payload: Any) -> str:
    return payload if isinstance(payload, str) else json.dumps(payload)


async def _plan_for(payload: Any, goal: str = "ship it") -> tuple[Any, _StubClient]:
    client = _StubClient([_as_text(payload)])
    plan = await PlannerAgent(client=client).plan(goal)
    return plan, client


# --------------------------------------------------------------------------- #
# extract_json
# --------------------------------------------------------------------------- #

#: Text that must not parse, plus the cases where it does.
_BAD_JSON = [
    "I cannot help with that.",
    "{not valid json",
    "```\nstill not json\n```",
    "",
    "[1, 2,",
]

_GOOD_JSON = [
    pytest.param(_as_text(_PLAN_OBJECT), _PLAN_OBJECT, id="raw-object"),
    pytest.param("```json\n" + _as_text(_PLAN_OBJECT) + "\n```", _PLAN_OBJECT, id="json-fence"),
    pytest.param("```\n" + _as_text(_PLAN_OBJECT) + "\n```", _PLAN_OBJECT, id="bare-fence"),
    pytest.param(
        "Here is the plan: " + _as_text(_PLAN_OBJECT) + " Hope that helps!",
        _PLAN_OBJECT,
        id="prose-around-object",
    ),
    pytest.param(
        "```json\n" + _as_text(_PLAN_OBJECT) + "\n```\nAnything else?",
        _PLAN_OBJECT,
        id="fence-then-prose",
    ),
    pytest.param(
        "Plan:\n[{\"id\": \"1\", \"description\": \"one\"}]\nLet me know!",
        [{"id": "1", "description": "one"}],
        id="prose-around-array",
    ),
    pytest.param(
        'Here: {"tasks": [{"id": "1", "description": "brace } and [bracket] inside"}]} done!',
        {"tasks": [{"id": "1", "description": "brace } and [bracket] inside"}]},
        id="braces-inside-string-literal",
    ),
    pytest.param(
        'Plan: {"tasks": [{"description": "escaped \\" quote then }"}], "rationale": "r"} -- ok',
        {"tasks": [{"description": 'escaped " quote then }'}], "rationale": "r"},
        id="escaped-quote-inside-string-literal",
    ),
    pytest.param(
        '{"tasks": [{"id": "1", "files": ["a.py", {"nested": [1, 2]}], "description": "d"}]}',
        {"tasks": [{"id": "1", "files": ["a.py", {"nested": [1, 2]}], "description": "d"}]},
        id="nested-json-value-inside-string",
    ),
]


@pytest.mark.parametrize(("text", "expected"), _GOOD_JSON)
def test_extract_json_recovers_json_from_replies(text: str, expected: Any) -> None:
    assert extract_json(text) == expected


@pytest.mark.parametrize("text", _BAD_JSON)
def test_extract_json_rejects_unparseable_replies(text: str) -> None:
    with pytest.raises(ValueError):
        extract_json(text)


# --------------------------------------------------------------------------- #
# plan(): task normalization
# --------------------------------------------------------------------------- #


async def test_plan_normalizes_degenerate_task_fields() -> None:
    """Missing/blank ids and descriptions, and non-list files/deps, are repaired."""
    plan, _ = await _plan_for(
        {
            "tasks": [
                {"description": "no id here"},
                {"id": "   ", "description": "blank id"},
                {"id": "named", "files": "not-a-list"},
                {
                    "id": "coerced",
                    "description": 42,
                    "files": [1, "b.py"],
                    "dependencies": [3, "x"],
                },
                {"id": "blank-desc", "description": "   "},
                "this entry is not a dict",
                {"id": "no-deps", "description": "d", "dependencies": {"a": 1}},
            ],
            "rationale": "because",
        }
    )

    tasks = plan.tasks
    assert [t.id for t in tasks] == ["0", "1", "named", "coerced", "blank-desc", "no-deps"]
    assert tasks[0].description == "no id here"
    assert tasks[1].description == "blank id"
    assert tasks[2].files == []
    assert tasks[2].dependencies == []
    assert tasks[3].description == "No description provided"
    assert tasks[3].files == ["1", "b.py"]
    assert tasks[3].dependencies == ["3", "x"]
    assert tasks[4].description == "No description provided"
    assert tasks[5].dependencies == []
    assert plan.rationale == "because"


async def test_plan_drops_non_dict_task_entries_without_failing() -> None:
    plan, _ = await _plan_for({"tasks": ["nope", 7, None, {"id": "1", "description": "ok"}]})
    assert [t.id for t in plan.tasks] == ["1"]


async def test_plan_supplies_default_rationale() -> None:
    plan, _ = await _plan_for({"tasks": []})
    assert plan.tasks == []
    assert plan.rationale == "No rationale provided."


async def test_plan_accepts_a_bare_json_array_response() -> None:
    plan, _ = await _plan_for([{"id": "a", "description": "x"}, {"id": "b", "description": "y"}])
    assert [(t.id, t.description) for t in plan.tasks] == [("a", "x"), ("b", "y")]
    assert plan.rationale == "Plan generated from task list."


async def test_plan_accepts_plan_wrapper_response() -> None:
    plan, _ = await _plan_for(
        {
            "plan": {"tasks": [{"id": "1", "description": "wrapped"}], "rationale": "inner"},
            "notes": "ignored",
        }
    )
    assert [(t.id, t.description) for t in plan.tasks] == [("1", "wrapped")]
    assert plan.rationale == "inner"


# --------------------------------------------------------------------------- #
# plan(): prompting and retries
# --------------------------------------------------------------------------- #


async def test_plan_prompt_carries_goal_and_strict_json_instruction() -> None:
    _, client = await _plan_for({"tasks": [], "rationale": "r"}, goal="make it fast")
    assert client.call_count == 1
    prompt = client.calls[0]["prompt"]
    assert "make it fast" in prompt
    assert STRICT_JSON_INSTRUCTION in prompt


async def test_plan_retries_once_and_recovers() -> None:
    valid = {"tasks": [{"id": "1", "description": "do it"}], "rationale": "r"}
    client = _StubClient(["I am afraid I cannot do that", _as_text(valid)])

    plan = await PlannerAgent(client=client).plan("ship it")

    assert client.call_count == 2
    assert [(t.id, t.description) for t in plan.tasks] == [("1", "do it")]
    assert [call["model"] for call in client.calls] == [client.settings.planner_model] * 2
    # The retry re-prompts with the parse failure so the model can correct itself.
    assert "could not be parsed" in client.calls[1]["prompt"]


async def test_plan_retry_truncates_a_huge_previous_response() -> None:
    garbage = "x" * (MAX_RESPONSE_EXCERPT * 3)
    client = _StubClient([garbage, _as_text({"tasks": [], "rationale": "r"})])

    await PlannerAgent(client=client).plan("keep prompts small")

    retry_prompt = client.calls[1]["prompt"]
    assert garbage not in retry_prompt
    assert "x" * MAX_RESPONSE_EXCERPT in retry_prompt


async def test_plan_raises_after_exhausting_retries() -> None:
    client = _StubClient(["nope"])

    with pytest.raises(ValueError) as excinfo:
        await PlannerAgent(client=client).plan("doomed goal")

    assert client.call_count == MAX_PLAN_ATTEMPTS
    assert [call["model"] for call in client.calls] == [client.settings.planner_model] * 3
    assert "doomed goal" in str(excinfo.value)


async def test_plan_returns_default_settings_client_when_given_a_falsy_one() -> None:
    """A client object that is falsy must not be silently replaced."""
    client = _StubClient([_as_text({"tasks": [], "rationale": "r"})])
    agent = PlannerAgent(client=client)
    assert agent.client is client
    await agent.plan("g")
    assert client.call_count == 1