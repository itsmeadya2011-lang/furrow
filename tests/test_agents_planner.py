"""Regression tests for :class:`furrow.agents.planner.PlannerAgent`.

Load-bearing regressions pinned here:

* a **fenced** ```json plan parses - the naive ``json.loads`` path failed on it;
* a malformed plan is normalised instead of blowing the whole run up;
* an unparseable reply raises a :class:`ValueError` that quotes the offending text;
* the planner uses ``settings.planner_model``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, TaskModel, TaskStatus
from furrow.llm import LLMClient
from tests.conftest import PLANNER, ScriptedCompletion, install_scripted, make_settings, plan_reply


def fenced(body: str, language: str = "json") -> str:
    """Wrap *body* in a markdown fence, as a chatty model would."""
    return f"Here is the plan:\n```{language}\n{body}\n```\nHope that helps!"


def make_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reply: str | Any
) -> tuple[PlannerAgent, ScriptedCompletion]:
    """A planner wired to an isolated client whose LLM answers with *reply*."""
    settings = make_settings(tmp_path)
    scripted = install_scripted(monkeypatch, planner=reply)
    return PlannerAgent(client=LLMClient(settings=settings), settings=settings), scripted


# ---------------------------------------------------------------- happy paths


async def test_parses_a_bare_json_plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent, _ = make_agent(monkeypatch, tmp_path, plan_reply({"id": "1", "description": "do it"}))
    plan = await agent.plan("Goal: make it work")
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "do it"
    assert plan.tasks[0].status is TaskStatus.PENDING


async def test_parses_a_fenced_json_plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The regression: a fenced reply must parse, not blow up on ``json.loads``."""
    body = json.dumps(
        {
            "tasks": [
                {"id": "1", "description": "first", "files": ["a.py"], "dependencies": []},
                {"id": "2", "description": "second", "dependencies": ["1"]},
            ],
            "rationale": "two steps",
        }
    )
    agent, _ = make_agent(monkeypatch, tmp_path, fenced(body))
    plan = await agent.plan("Goal: make it work")
    assert [t.id for t in plan.tasks] == ["1", "2"]
    assert plan.tasks[0].files == ["a.py"]
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.rationale == "two steps"


async def test_parses_a_plan_wrapped_in_prose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = "Sure thing!\n" + plan_reply({"id": "1", "description": "d"}) + "\nDone."
    agent, _ = make_agent(monkeypatch, tmp_path, raw)
    plan = await agent.plan("Goal")
    assert len(plan.tasks) == 1


async def test_round_trips_into_real_task_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent, _ = make_agent(
        monkeypatch,
        tmp_path,
        plan_reply(
            {"id": "1", "description": "d1", "files": ["x.py"], "dependencies": []},
            {"id": "2", "description": "d2", "dependencies": ["1"]},
        ),
    )
    plan = await agent.plan("Goal")
    assert all(isinstance(task, TaskModel) for task in plan.tasks)
    assert plan.model_dump()["tasks"][0]["id"] == "1"
    assert plan.tasks[1].status is TaskStatus.PENDING


async def test_uses_the_planner_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = make_settings(tmp_path, planner_model="planner-model-x")
    scripted = install_scripted(monkeypatch, planner=plan_reply({"id": "1", "description": "d"}))
    await PlannerAgent(client=LLMClient(settings=settings), settings=settings).plan("Goal")
    assert scripted.models(PLANNER) == ["planner-model-x"]


async def test_the_goal_reaches_the_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent, scripted = make_agent(monkeypatch, tmp_path, plan_reply({"id": "1", "description": "d"}))
    await agent.plan("Goal: ship the thing")
    assert "ship the thing" in scripted.only_prompt(PLANNER)


# ------------------------------------------------------------- normalisation


async def test_duplicate_ids_are_made_unique(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent, _ = make_agent(
        monkeypatch,
        tmp_path,
        plan_reply(
            {"id": "1", "description": "first"},
            {"id": "1", "description": "second"},
            {"id": "1", "description": "third"},
        ),
    )
    plan = await agent.plan("Goal")
    ids = [task.id for task in plan.tasks]
    assert len(set(ids)) == len(ids) == 3
    assert ids[0] == "1"
    assert all(task.description != "" for task in plan.tasks)


async def test_a_self_dependency_is_dropped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent, _ = make_agent(
        monkeypatch,
        tmp_path,
        plan_reply({"id": "1", "description": "d", "dependencies": ["1"]}),
    )
    plan = await agent.plan("Goal")
    assert plan.tasks[0].dependencies == []


async def test_a_dependency_on_an_unknown_id_is_dropped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent, _ = make_agent(
        monkeypatch,
        tmp_path,
        plan_reply(
            {"id": "1", "description": "d1"},
            {"id": "2", "description": "d2", "dependencies": ["1", "ghost", "42"]},
        ),
    )
    plan = await agent.plan("Goal")
    assert plan.tasks[1].dependencies == ["1"]


async def test_a_plan_above_the_documented_cap_is_truncated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``PLANNER_PROMPT`` documents a hard cap of five tasks."""
    agent, _ = make_agent(
        monkeypatch,
        tmp_path,
        plan_reply(*[{"id": str(i), "description": f"d{i}"} for i in range(9)]),
    )
    plan = await agent.plan("Goal")
    assert len(plan.tasks) == 5
    assert [task.id for task in plan.tasks] == ["0", "1", "2", "3", "4"]


async def test_a_plan_exactly_at_the_cap_is_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent, _ = make_agent(
        monkeypatch,
        tmp_path,
        plan_reply(*[{"id": str(i), "description": f"d{i}"} for i in range(5)]),
    )
    plan = await agent.plan("Goal")
    assert len(plan.tasks) == 5


async def test_a_missing_id_gets_a_generated_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent, _ = make_agent(
        monkeypatch,
        tmp_path,
        plan_reply({"description": "no id here"}, {"description": "nor here"}),
    )
    plan = await agent.plan("Goal")
    assert [task.id for task in plan.tasks] == ["task-1", "task-2"]


async def test_a_blank_id_gets_a_generated_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent, _ = make_agent(monkeypatch, tmp_path, plan_reply({"id": "   ", "description": "d"}))
    plan = await agent.plan("Goal")
    assert plan.tasks[0].id == "task-1"


async def test_a_missing_description_gets_a_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent, _ = make_agent(monkeypatch, tmp_path, plan_reply({"id": "1"}))
    plan = await agent.plan("Goal")
    assert plan.tasks[0].description == "implement 1"


async def test_non_dict_task_entries_are_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reply = json.dumps(
        {"tasks": ["nope", 42, {"id": "1", "description": "real"}], "rationale": "r"}
    )
    agent, _ = make_agent(monkeypatch, tmp_path, reply)
    plan = await agent.plan("Goal")
    assert [task.id for task in plan.tasks] == ["1"]


async def test_non_string_files_and_dependencies_are_coerced_or_dropped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reply = json.dumps(
        {
            "tasks": [
                {"id": "1", "description": "d", "files": ["a.py", "  ", 7], "dependencies": "nope"}
            ],
            "rationale": "r",
        }
    )
    agent, _ = make_agent(monkeypatch, tmp_path, reply)
    plan = await agent.plan("Goal")
    assert plan.tasks[0].files == ["a.py", "7"]
    assert plan.tasks[0].dependencies == []


async def test_an_empty_rationale_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reply = json.dumps({"tasks": [{"id": "1", "description": "d"}], "rationale": "   "})
    agent, _ = make_agent(monkeypatch, tmp_path, reply)
    plan = await agent.plan("Goal")
    assert plan.rationale == "planner"


# -------------------------------------------------------------------- errors


async def test_an_empty_plan_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent, _ = make_agent(monkeypatch, tmp_path, plan_reply())
    with pytest.raises(ValueError, match="at least one task"):
        await agent.plan("Goal")


async def test_a_plan_without_a_tasks_key_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent, _ = make_agent(monkeypatch, tmp_path, json.dumps({"rationale": "forgot the tasks"}))
    with pytest.raises(ValueError, match="at least one task"):
        await agent.plan("Goal")


async def test_a_non_list_tasks_key_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent, _ = make_agent(monkeypatch, tmp_path, json.dumps({"tasks": {"id": "1"}}))
    with pytest.raises(ValueError, match="must be a list"):
        await agent.plan("Goal")


async def test_plain_prose_raises_with_a_snippet_of_the_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reply = "I am not going to answer that."
    agent, _ = make_agent(monkeypatch, tmp_path, reply)
    with pytest.raises(ValueError) as excinfo:
        await agent.plan("Goal")
    message = str(excinfo.value)
    assert "Failed to parse plan from LLM" in message
    assert "I am not going to answer that." in message


async def test_a_truncated_reply_quotes_only_a_snippet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reply = "no json here at all, just " + ("prose " * 200)
    agent, _ = make_agent(monkeypatch, tmp_path, reply)
    with pytest.raises(ValueError) as excinfo:
        await agent.plan("Goal")
    message = str(excinfo.value)
    assert "no json here at all" in message
    assert "..." in message
    assert len(message) < 1500


async def test_an_empty_reply_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent, _ = make_agent(monkeypatch, tmp_path, "")
    with pytest.raises(ValueError):
        await agent.plan("Goal")


async def test_a_top_level_array_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent, _ = make_agent(monkeypatch, tmp_path, "[1, 2, 3]")
    with pytest.raises(ValueError):
        await agent.plan("Goal")
