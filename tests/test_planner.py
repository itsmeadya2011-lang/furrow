"""Hermetic planner tests: no network, no API keys, no filesystem access.

The LLM client is replaced by ``FakeClient``, which returns canned strings, so these
assert the planner's own parsing/normalization contract rather than model output.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from furrow.agents.planner import (
    MAX_TASKS,
    PlannerAgent,
    build_plan,
    extract_json_payload,
)
from furrow.config import Plan, Settings


def make_settings(**overrides: Any) -> Settings:
    """Settings isolated from the environment so a stray .env cannot change a test."""
    return Settings(_env_file=None, workspace=Path("/nonexistent/workspace"), **overrides)


class FakeClient:
    """Stands in for LLMClient: canned responses, fixed listing, recorded prompts."""

    def __init__(
        self,
        responses: list[str],
        files: list[str] | None = None,
        files_error: bool = False,
        settings: Settings | None = None,
    ) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.list_calls: list[Any] = []
        self.settings = settings or make_settings()
        self._files = files or []
        self._files_error = files_error

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.responses) - 1)
        return self.responses[index]

    def list_files(self, directory: str | Path | None = None) -> list[str]:
        self.list_calls.append(directory)
        if self._files_error:
            raise OSError("workspace is unavailable")
        return list(self._files)


def plan(agent: PlannerAgent, *test_failures: str) -> Any:
    """Run one planning cycle without needing an async test plugin."""
    return asyncio.run(agent.plan("Add a health endpoint", test_failures or None))


def task(task_id: str = "1", **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": task_id,
        "description": f"do task {task_id}",
        "files": ["furrow/app.py"],
        "dependencies": [],
    }
    entry.update(overrides)
    return entry


# --------------------------------------------------------------------------- parsing


def test_clean_raw_json_parses():
    response = json.dumps({"tasks": [task()], "rationale": "small goal"})
    client = FakeClient([response])
    result = plan(PlannerAgent(client=client))
    assert [t.id for t in result.tasks] == ["1"]
    assert result.rationale == "small goal"


def test_fenced_json_parses():
    response = '```json\n{"tasks": [{"id": "a", "description": "x"}], "rationale": "ok"}\n```'
    result = plan(PlannerAgent(client=FakeClient([response])))
    assert [t.id for t in result.tasks] == ["a"]


def test_json_surrounded_by_prose_parses():
    response = (
        "Sure! Here is the plan I came up with:\n"
        '{"tasks": [{"id": "p1", "description": "wire the route", '
        '"dependencies": []}], "rationale": "first slice"}\n'
        "Let me know if you want a different breakdown."
    )
    result = plan(PlannerAgent(client=FakeClient([response])))
    assert [t.id for t in result.tasks] == ["p1"]
    assert result.rationale == "first slice"


def test_prose_with_nested_braces_in_strings_parses():
    response = (
        'Note: {"a": 1} is not a plan. Use this:\n'
        '{"tasks": [{"id": "7", "description": "print {x}"}]}'
    )
    result = plan(PlannerAgent(client=FakeClient([response])))
    assert [t.id for t in result.tasks] == ["7"]


@pytest.mark.parametrize(
    "decoy",
    [
        '{"a": 1}',
        '{"config": {"retries": 2, "note": "a } brace"}}',
        '"just a quoted string"',
        "",
    ],
)
def test_decoy_json_before_the_plan_is_skipped(decoy):
    # A chatty reply shows an example object before the answer; the decoy parses
    # cleanly, so only picking the plan-shaped region avoids returning it.
    prefix = f"{decoy} is not a plan. Use this:\n" if decoy else ""
    response = prefix + json.dumps({"tasks": [task("7")]})
    assert extract_json_payload(response)["tasks"][0]["id"] == "7"


def test_decoy_json_after_the_plan_is_skipped():
    response = json.dumps({"tasks": [task("7")]}) + '\n\nExample of a task: {"id": "x"}'
    assert extract_json_payload(response)["tasks"][0]["id"] == "7"


def test_a_bare_array_before_the_plan_wins_as_a_bare_plan():
    # A top-level array is itself an accepted plan shape, so unlike an object decoy it
    # is not skipped; the scalar entries then fail loudly instead of being guessed at.
    response = "[1, 2, 3] is not a plan. Use this:\n" + json.dumps({"tasks": [task("7")]})
    assert extract_json_payload(response) == [1, 2, 3]
    with pytest.raises(ValueError, match="no usable tasks"):
        build_plan(extract_json_payload(response), MAX_TASKS)


def test_only_decoy_json_is_reported_as_a_shape_error():
    # No plan-shaped payload anywhere: the decoy is still returned so build_plan can
    # report the real problem, instead of the caller seeing a parse error.
    with pytest.raises(ValueError, match="'tasks' must be a JSON array"):
        build_plan(extract_json_payload('{"a": 1} is all I have'), MAX_TASKS)


def test_top_level_list_of_tasks_becomes_plan():
    response = json.dumps([task("a"), task("b")])
    result = plan(PlannerAgent(client=FakeClient([response])))
    assert [t.id for t in result.tasks] == ["a", "b"]
    assert result.rationale == ""


def test_invalid_json_is_not_silently_repaired():
    # A trailing comma is a syntax error we deliberately do not paper over: the empty
    # "[]" nested inside must not be returned in its place.
    with pytest.raises(ValueError, match="no parseable JSON"):
        extract_json_payload('{"tasks": [],}')


def test_empty_json_reply_is_not_a_plan():
    with pytest.raises(ValueError, match="only empty JSON"):
        extract_json_payload("[]")


# --------------------------------------------------------------------------- repair


def test_malformed_output_repairs_once_then_succeeds():
    good = json.dumps({"tasks": [task()], "rationale": "r"})
    client = FakeClient(["I am thinking about it...", good])
    result = plan(PlannerAgent(client=client))
    assert [t.id for t in result.tasks] == ["1"]
    assert len(client.prompts) == 2
    repair = client.prompts[1]
    assert "raw JSON only" in repair
    assert "I am thinking about it" in repair
    assert "Add a health endpoint" in repair


def test_malformed_output_raises_clear_error_after_one_repair():
    client = FakeClient(["not json at all", "still not json"])
    with pytest.raises(ValueError) as excinfo:
        plan(PlannerAgent(client=client))
    assert len(client.prompts) == 2  # exactly one repair attempt, never a loop
    message = str(excinfo.value)
    assert "not json at all" in message
    assert "still not json" in message
    assert "raw JSON only" in message


def test_empty_response_is_reported():
    with pytest.raises(ValueError, match="empty response"):
        extract_json_payload("   ")


# --------------------------------------------------------------------- normalization


def test_missing_and_duplicate_ids_are_normalized():
    payload = {
        "tasks": [
            {"id": "dup", "description": "first"},
            {"id": "dup", "description": "second"},
            {"description": "third"},
        ],
        "rationale": "r",
    }
    result = build_plan(payload, MAX_TASKS)
    ids = [t.id for t in result.tasks]
    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert ids[0] == "dup"


def test_unknown_and_self_dependencies_are_dropped():
    payload = {
        "tasks": [
            task("1", dependencies=["2", "1", "nope"]),
            task("2"),
        ],
        "rationale": "r",
    }
    result = build_plan(payload, MAX_TASKS)
    assert result.tasks[0].dependencies == ["2"]
    assert result.tasks[1].dependencies == []


def test_dependency_cycle_is_broken():
    payload = {
        "tasks": [
            task("1", dependencies=["2"]),
            task("2", dependencies=["1"]),
        ],
        "rationale": "r",
    }
    result = build_plan(payload, MAX_TASKS)
    assert len(result.tasks) == 2
    edges = {t.id: list(t.dependencies) for t in result.tasks}
    assert all(dep in edges for deps in edges.values() for dep in deps)
    # Acyclic: repeatedly removing nodes with no remaining deps must finish the graph.
    remaining = dict(edges)
    while True:
        ready = [node for node, deps in remaining.items() if not deps]
        if not ready:
            break
        for node in ready:
            remaining.pop(node)
        remaining = {node: [d for d in deps if d in remaining] for node, deps in remaining.items()}
    assert remaining == {}


def test_malformed_task_entries_are_dropped_not_fatal():
    payload = {
        "tasks": ["not an object", {"id": "1"}, task("2")],
        "rationale": "r",
    }
    result = build_plan(payload, MAX_TASKS)
    assert [t.id for t in result.tasks] == ["2"]


def test_missing_files_and_dependencies_default_to_empty():
    payload = {"tasks": [{"id": "1", "description": "bare"}], "rationale": "r"}
    result = build_plan(payload, MAX_TASKS)
    assert result.tasks[0].files == []
    assert result.tasks[0].dependencies == []


def test_empty_task_list_is_rejected():
    with pytest.raises(ValueError, match="at least one task"):
        build_plan({"tasks": [], "rationale": "nothing to do"}, MAX_TASKS)


def test_task_count_is_clamped_to_max_parallel_tasks():
    payload = {"tasks": [task(str(i)) for i in range(1, 8)], "rationale": "big goal"}
    result = build_plan(payload, 2)
    assert len(result.tasks) == 2
    assert [t.id for t in result.tasks] == ["1", "2"]
    assert "max_parallel_tasks=2" in result.rationale


def test_task_count_never_exceeds_five():
    payload = {"tasks": [task(str(i)) for i in range(1, 30)], "rationale": "huge"}
    result = build_plan(payload, 99)
    assert len(result.tasks) == MAX_TASKS == 5
    assert re.search(r"Clamped to the first 5", result.rationale)


# ------------------------------------------------------------------------ grounding


def test_prompt_states_cap_and_raw_json_requirement():
    payload = json.dumps({"tasks": [task()], "rationale": "r"})
    client = FakeClient([payload], files=["furrow/app.py"])
    plan(PlannerAgent(client=client))
    prompt = client.prompts[0]
    assert f"at most {MAX_TASKS} tasks" in prompt
    # Case-fold both sides: folding only the haystack silently fails the assertion.
    assert "raw json only" in prompt.lower()
    assert "furrow/app.py" in prompt  # workspace grounding
    assert client.list_calls == [client.settings.workspace]


def test_prompt_states_id_and_dependency_rules():
    client = FakeClient([json.dumps({"tasks": [task()], "rationale": "r"})])
    plan(PlannerAgent(client=client))
    # Collapse wrapping so the rules stay asserted verbatim but not line-broken.
    prompt = " ".join(client.prompts[0].split())
    assert "Ids must be unique within this response" in prompt
    assert '"dependencies" must list ids that appear in the "tasks" array' in prompt
    assert "never create a dependency cycle" in prompt


def test_list_files_failure_is_tolerated():
    client = FakeClient([json.dumps({"tasks": [task()], "rationale": "r"})], files_error=True)
    result = plan(PlannerAgent(client=client))
    assert [t.id for t in result.tasks] == ["1"]
    assert "Workspace files" not in client.prompts[0]


def test_file_listing_is_bounded():
    files = [f"pkg/module_{i}.py" for i in range(500)]
    client = FakeClient([json.dumps({"tasks": [task()], "rationale": "r"})], files=files)
    plan(PlannerAgent(client=client))
    listed = [line for line in client.prompts[0].splitlines() if line.startswith("- pkg/")]
    assert 0 < len(listed) < 500


def test_previous_test_failures_are_grounded():
    client = FakeClient([json.dumps({"tasks": [task()], "rationale": "r"})])
    plan(PlannerAgent(client=client), "tests/test_app.py::test_health FAILED")
    assert "tests/test_app.py::test_health FAILED" in client.prompts[0]


# ---------------------------------------------------------------------- construction


def test_constructs_with_injected_client_and_with_settings():
    client = FakeClient(["{}"], settings=make_settings(max_parallel_tasks=3))
    agent = PlannerAgent(client=client)
    assert agent.client is client
    assert agent.settings is client.settings
    assert isinstance(agent.settings, Settings)

    configured = PlannerAgent(settings=make_settings(max_parallel_tasks=3))
    assert configured.client is not None
    assert isinstance(configured.settings, Settings)
    assert configured.settings.max_parallel_tasks == 3


def test_client_settings_outrank_a_borrowed_settings_argument():
    # The client owns the model actually being called, so its settings (model name,
    # workspace, task cap) must not be redirected by an unrelated Settings instance.
    client = FakeClient(["{}"], settings=make_settings(max_parallel_tasks=3))
    agent = PlannerAgent(client=client, settings=make_settings(max_parallel_tasks=1))
    assert agent.settings is client.settings
    assert agent.settings.max_parallel_tasks == 3


def test_plan_returns_a_plan_model():
    client = FakeClient([json.dumps({"tasks": [task()], "rationale": "r"})])
    result = plan(PlannerAgent(client=client))
    assert isinstance(result, Plan)


def test_settings_honoured_end_to_end():
    client = FakeClient(
        [json.dumps({"tasks": [task(str(i)) for i in range(1, 6)], "rationale": "r"})],
        settings=make_settings(max_parallel_tasks=1),
    )
    result = plan(PlannerAgent(client=client))
    assert len(result.tasks) == 1
