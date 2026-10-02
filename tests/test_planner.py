"""Regression tests for ``furrow.agents.planner``.

A planner payload is model output, so it arrives with duplicate ids, missing
ids, ``tasks`` as an object instead of a list, more tasks than the concurrency
cap allows, or no rationale at all.  ``build_plan`` normalises all of that;
these tests pin the normalisation down.  The real ``PlannerAgent.plan`` is
exercised with a stubbed ``LLMClient.complete``, so no LLM call is ever made.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from furrow.agents.planner import PlannerAgent, build_plan
from furrow.config import Settings
from furrow.llm import LLMClient


def make_agent(response: str, max_parallel_tasks: int = 5) -> PlannerAgent:
    """A real PlannerAgent whose client returns ``response`` verbatim."""
    client = LLMClient(settings=Settings(max_parallel_tasks=max_parallel_tasks))

    async def fake_complete(prompt: str, system: str = "", model: str | None = None) -> str:
        return response

    client.complete = fake_complete  # type: ignore[method-assign]
    return PlannerAgent(client=client)


def raw_plan(task_ids: list[str], rationale: str = "because") -> str:
    payload = {
        "tasks": [{"id": ident, "description": f"do {ident}"} for ident in task_ids],
        "rationale": rationale,
    }
    return json.dumps(payload)


async def plan_from(response: str, max_parallel_tasks: int = 5):
    return await make_agent(response, max_parallel_tasks).plan("the goal")


# --------------------------------------------------------------------- happy path


async def test_a_well_formed_plan_round_trips():
    plan = await plan_from(raw_plan(["a", "b"]))
    assert [task.id for task in plan.tasks] == ["a", "b"]
    assert [task.description for task in plan.tasks] == ["do a", "do b"]
    assert plan.rationale == "because"


async def test_a_fenced_json_response_parses():
    response = f"```json\n{raw_plan(['a'])}\n```"
    plan = await plan_from(response)
    assert [task.id for task in plan.tasks] == ["a"]


async def test_a_prose_wrapped_response_parses():
    response = f"Sure! Here is the plan: {raw_plan(['a', 'b'])} hope that helps."
    plan = await plan_from(response)
    assert [task.id for task in plan.tasks] == ["a", "b"]


async def test_tasks_default_to_pending_with_no_result():
    plan = await plan_from(raw_plan(["a"]))
    assert plan.tasks[0].status == "pending"
    assert plan.tasks[0].result is None
    assert plan.tasks[0].files == []


# ------------------------------------------------------------------ id hygiene


def test_duplicate_task_ids_keep_the_first_occurrence():
    # Duplicate ids break task correlation and status reporting downstream.
    data = {
        "tasks": [
            {"id": "a", "description": "first"},
            {"id": "a", "description": "second"},
            {"id": "b", "description": "other"},
        ],
        "rationale": "r",
    }
    plan = build_plan(data, 5)
    assert [task.id for task in plan.tasks] == ["a", "b"]
    assert plan.tasks[0].description == "first"
    assert "duplicated id" in plan.rationale


def test_a_task_with_a_missing_id_is_dropped():
    data = {
        "tasks": [{"description": "nameless"}, {"id": "b", "description": "named"}],
        "rationale": "r",
    }
    plan = build_plan(data, 5)
    assert [task.id for task in plan.tasks] == ["b"]
    assert "Dropped 1 planned task(s)" in plan.rationale


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_task_with_a_blank_id_is_dropped(blank: str):
    data = {"tasks": [{"id": blank, "description": "blank"}], "rationale": "r"}
    plan = build_plan(data, 5)
    assert plan.tasks == []
    assert "Dropped 1 planned task(s)" in plan.rationale


def test_a_null_id_is_dropped():
    data = {"tasks": [{"id": None, "description": "null id"}], "rationale": "r"}
    assert build_plan(data, 5).tasks == []


def test_a_non_dict_entry_is_dropped():
    data = {"tasks": ["just a string", {"id": "a", "description": "real"}], "rationale": "r"}
    plan = build_plan(data, 5)
    assert [task.id for task in plan.tasks] == ["a"]


def test_numeric_ids_are_stringified_and_collide_with_their_string_form():
    data = {
        "tasks": [{"id": 1, "description": "num"}, {"id": "1", "description": "str"}],
        "rationale": "r",
    }
    plan = build_plan(data, 5)
    assert [task.id for task in plan.tasks] == ["1"]
    assert plan.tasks[0].description == "num"


def test_the_rationale_reports_how_many_tasks_were_dropped():
    data = {
        "tasks": [
            {"id": "a", "description": "keep"},
            {"description": "drop 1"},
            {"id": "a", "description": "drop 2"},
            {"id": "", "description": "drop 3"},
        ],
        "rationale": "start here",
    }
    plan = build_plan(data, 5)
    assert plan.rationale.startswith("start here")
    assert "Dropped 3 planned task(s)" in plan.rationale


# ---------------------------------------------------------------- shape coercion


def test_tasks_given_as_a_dict_is_coerced_to_a_one_element_list():
    # Bug: `{"tasks": {...}}` failed pydantic validation and lost the plan.
    data = {"tasks": {"id": "a", "description": "solo"}, "rationale": "r"}
    plan = build_plan(data, 5)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "a"


def test_tasks_given_as_none_becomes_an_empty_plan():
    assert build_plan({"tasks": None, "rationale": "r"}, 5).tasks == []


def test_a_missing_tasks_key_becomes_an_empty_plan():
    assert build_plan({"rationale": "r"}, 5).tasks == []


def test_a_non_object_payload_raises_type_error():
    with pytest.raises(TypeError, match="expected a JSON object"):
        build_plan(["not", "an", "object"], 5)


# ----------------------------------------------------------------- truncation


def test_a_nine_task_plan_is_truncated_to_the_parallel_cap():
    # Bug: a 9-task plan with max_parallel_tasks=5 ran 9 workers at once.
    ids = [str(n) for n in range(9)]
    plan = build_plan({"tasks": [{"id": i, "description": i} for i in ids], "rationale": "r"}, 5)
    assert len(plan.tasks) == 5
    assert [task.id for task in plan.tasks] == ["0", "1", "2", "3", "4"]
    assert "max_parallel_tasks=5" in plan.rationale
    assert "Truncated the plan to the first 5 task(s)" in plan.rationale
    assert "4 further task(s) deferred" in plan.rationale


async def test_a_nine_task_llm_response_is_truncated_to_five():
    plan = await plan_from(raw_plan([str(n) for n in range(9)]), max_parallel_tasks=5)
    assert len(plan.tasks) == 5
    assert "max_parallel_tasks=5" in plan.rationale


def test_a_plan_within_the_cap_is_not_truncated():
    data = {"tasks": [{"id": str(n), "description": "d"} for n in range(5)], "rationale": "r"}
    plan = build_plan(data, 5)
    assert len(plan.tasks) == 5
    assert "Truncated" not in plan.rationale


def test_a_negative_parallel_cap_drops_everything_instead_of_crashing():
    data = {"tasks": [{"id": "a", "description": "d"}], "rationale": "r"}
    plan = build_plan(data, -1)
    assert plan.tasks == []
    assert "max_parallel_tasks=0" in plan.rationale


# ------------------------------------------------------------------- rationale


def test_a_missing_rationale_defaults_to_the_empty_string():
    plan = build_plan({"tasks": [{"id": "a", "description": "d"}]}, 5)
    assert plan.rationale == ""


def test_a_non_string_rationale_defaults_to_the_empty_string():
    plan = build_plan({"tasks": [{"id": "a", "description": "d"}], "rationale": 42}, 5)
    assert plan.rationale == ""


def test_a_null_rationale_defaults_to_the_empty_string():
    plan = build_plan({"tasks": [{"id": "a", "description": "d"}], "rationale": None}, 5)
    assert plan.rationale == ""


async def test_a_response_without_a_rationale_still_plans():
    plan = await plan_from(json.dumps({"tasks": [{"id": "a", "description": "d"}]}))
    assert plan.rationale == ""
    assert [task.id for task in plan.tasks] == ["a"]


def test_a_rationale_is_stripped():
    plan = build_plan({"tasks": [{"id": "a", "description": "d"}], "rationale": "  padded  "}, 5)
    assert plan.rationale == "padded"


def test_dedupe_and_truncation_notes_are_both_reported():
    ids = [str(n) for n in range(7)] + ["0"]
    data = {"tasks": [{"id": i, "description": "d"} for i in ids], "rationale": "r"}
    plan = build_plan(data, 5)
    assert len(plan.tasks) == 5
    assert "Dropped 1 planned task(s)" in plan.rationale
    assert "Truncated the plan" in plan.rationale


# ------------------------------------------------------------------- llm errors


async def test_an_unparseable_response_raises_value_error_quoting_the_response():
    with pytest.raises(ValueError) as excinfo:
        await plan_from("I am not going to answer that.")
    assert "I am not going to answer that." in str(excinfo.value)


async def test_a_non_object_payload_raises_value_error():
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await plan_from("[1, 2, 3]")


async def test_a_truncated_response_is_snippeted_not_dumped_whole():
    long_response = "x" * 2000
    with pytest.raises(ValueError) as excinfo:
        await plan_from(long_response)
    assert "[truncated]" in str(excinfo.value)
    assert "x" * 600 not in str(excinfo.value)


async def test_plan_sends_the_goal_and_uses_the_planner_model():
    seen: dict[str, Any] = {}
    client = LLMClient(settings=Settings(planner_model="planner-model-x"))

    async def fake_complete(prompt: str, system: str = "", model: str | None = None) -> str:
        seen["prompt"] = prompt
        seen["model"] = model
        return raw_plan(["a"])

    client.complete = fake_complete  # type: ignore[method-assign]
    agent = PlannerAgent(client=client)

    await agent.plan("build a parser")

    assert "build a parser" in seen["prompt"]
    assert seen["model"] == "planner-model-x"
