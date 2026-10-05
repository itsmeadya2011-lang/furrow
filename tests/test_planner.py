"""Regression tests for :class:`furrow.agents.planner.PlannerAgent`.

The planner is the first place untrusted model output becomes structure, so most of these
tests pin down what happens when that output is fenced, wrapped in prose, missing an ``id``,
reusing an ``id``, or unusable entirely.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from furrow.agents.planner import DEFAULT_RATIONALE, PlannerAgent
from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Settings, TaskModel

from .conftest import StubLLMClient


def stub(response: str, settings: Settings | None = None) -> StubLLMClient:
    return StubLLMClient(settings=settings, plans=[response])


def plan(*tasks: object, rationale: str = "because") -> str:
    return json.dumps({"tasks": list(tasks), "rationale": rationale})


def entry(task_id: str | None, description: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {"description": description}
    if task_id is not None:
        payload["id"] = task_id
    payload.update(extra)
    return payload


async def parse(response: str, goal: str = "GOAL") -> tuple:
    return await PlannerAgent(client=stub(response)).plan(goal)


# ------------------------------------------------------------------- accepted shapes


@pytest.mark.parametrize(
    ("label", "response"),
    [
        ("plain json", plan(entry("1", "Implement the widget"), rationale="r")),
        (
            "json fenced",
            "```json\n" + plan(entry("1", "Implement the widget"), rationale="r") + "\n```",
        ),
        (
            "fenced without a language tag",
            "```\n" + plan(entry("1", "Implement the widget"), rationale="r") + "\n```",
        ),
        (
            "prose before and after",
            "Sure, here is the plan:\n"
            + plan(entry("1", "Implement the widget"), rationale="r")
            + "\nLet me know if you want changes.",
        ),
        (
            "trailing commas",
            '{"tasks": [{"id": "1", "description": "Implement the widget",},], "rationale": "r",}',
        ),
        (
            "trailing commas inside a fence",
            '```json\n{"tasks": [{"id": "1", "description": "Implement the widget",}],'
            ' "rationale": "r",}\n```',
        ),
    ],
)
def test_plan_shapes_all_parse(label: str, response: str) -> None:
    result = asyncio.run(parse(response))

    assert [task.id for task in result.tasks] == ["1"], label
    assert result.tasks[0].description == "Implement the widget", label
    assert result.rationale == "r", label


def test_plan_preserves_files_and_dependencies() -> None:
    response = plan(
        entry("1", "Write it", files=["src/a.py", "tests/test_a.py"]),
        entry("2", "Test it", dependencies=["1"]),
    )

    result = asyncio.run(parse(response))

    assert result.tasks[0].files == ["src/a.py", "tests/test_a.py"]
    assert result.tasks[1].dependencies == ["1"]


def test_tasks_default_to_pending_with_no_result() -> None:
    result = asyncio.run(parse(plan(entry("1", "Write it"))))

    assert result.tasks[0].status == "pending"
    assert result.tasks[0].result is None


def test_plan_returns_task_models() -> None:
    result = asyncio.run(parse(plan(entry("1", "Write it"))))

    assert isinstance(result.tasks[0], TaskModel)


# ------------------------------------------------------------------------- identifiers


def test_a_task_without_an_id_gets_a_positional_one() -> None:
    response = plan(
        entry(None, "First"),
        entry(None, "Second"),
        entry(None, "Third"),
        entry("keep-me", "Fourth"),
    )

    result = asyncio.run(parse(response))

    assert [task.id for task in result.tasks] == ["1", "2", "3", "keep-me"]


def test_an_empty_id_is_treated_as_missing() -> None:
    result = asyncio.run(parse(plan(entry("", "First"), entry("   ", "Second"))))

    assert [task.id for task in result.tasks] == ["1", "2"]


def test_duplicate_ids_are_suffixed() -> None:
    response = plan(entry("1", "A"), entry("1", "B"), entry("1", "C"))

    result = asyncio.run(parse(response))

    assert [task.id for task in result.tasks] == ["1", "1-2", "1-3"]


def test_a_suffix_generated_by_the_planner_is_also_respected() -> None:
    # "1" collides with the generated "1-2", so the third entry skips past it.
    response = plan(entry("1", "A"), entry("1", "B"), entry("1-2", "C"))

    result = asyncio.run(parse(response))

    assert [task.id for task in result.tasks] == ["1", "1-2", "1-2-2"]


def test_descriptions_are_kept_in_order_after_renaming() -> None:
    response = plan(entry("1", "A"), entry("1", "B"))

    result = asyncio.run(parse(response))

    assert [(task.id, task.description) for task in result.tasks] == [("1", "A"), ("1-2", "B")]


# ------------------------------------------------------------------- defensive coercion


def test_a_bare_string_task_becomes_a_description() -> None:
    result = asyncio.run(parse(plan("just do the thing")))

    assert [(task.id, task.description) for task in result.tasks] == [("1", "just do the thing")]


def test_a_task_with_no_prose_field_keeps_whatever_the_model_sent() -> None:
    result = asyncio.run(parse(plan({"id": "1", "files": ["a.py"]})))

    assert result.tasks[0].id == "1"
    assert "a.py" in result.tasks[0].description


def test_a_bare_string_dependency_is_wrapped_into_a_list() -> None:
    result = asyncio.run(parse(plan(entry("2", "B", dependencies="1"))))

    assert result.tasks[0].dependencies == ["1"]


def test_an_alternate_dependency_key_is_accepted() -> None:
    result = asyncio.run(parse(plan({"id": "2", "description": "B", "depends_on": ["1"]})))

    assert result.tasks[0].dependencies == ["1"]


def test_unusable_entries_are_skipped_but_the_plan_survives() -> None:
    response = plan(entry("1", "good"), None, 17, "", entry("2", "also good"))

    result = asyncio.run(parse(response))

    assert [task.id for task in result.tasks] == ["1", "2"]


def test_a_plan_keyed_by_id_is_expanded() -> None:
    result = asyncio.run(parse('{"tasks": {"7": {"description": "Seven"}}}'))

    assert [(task.id, task.description) for task in result.tasks] == [("7", "Seven")]


def test_a_missing_rationale_gets_a_placeholder() -> None:
    result = asyncio.run(parse('{"tasks": [{"id": "1", "description": "A"}]}'))

    assert result.rationale == DEFAULT_RATIONALE


# ---------------------------------------------------------------------- failure modes


@pytest.mark.parametrize(
    "response",
    [
        "I am afraid I cannot help with that.",
        "",
        '{"tasks": "not a list"}',
        '{"tasks": []}',
        "[]",
        '"a string"',
        '{"tasks": [null, 3, ""]}',
    ],
)
def test_an_unusable_plan_raises_value_error(response: str) -> None:
    with pytest.raises(ValueError, match="Planner returned no usable plan"):
        asyncio.run(parse(response))


def test_the_value_error_names_the_expected_shape() -> None:
    with pytest.raises(ValueError) as excinfo:
        asyncio.run(parse('{"tasks": []}'))

    assert '"rationale"' in str(excinfo.value)


def test_a_json_parse_failure_chains_the_underlying_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        asyncio.run(parse("total gibberish, no JSON here"))

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "Could not extract JSON" in str(excinfo.value.__cause__)


def test_the_value_error_truncates_a_huge_response() -> None:
    response = json.dumps({"tasks": [], "rationale": "x" * 5000})

    with pytest.raises(ValueError) as excinfo:
        asyncio.run(parse(response))

    assert "[truncated," in str(excinfo.value)


def test_a_non_object_top_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="not an object"):
        asyncio.run(parse("[1, 2, 3]"))


# --------------------------------------------------------------------------- the call


def test_the_planner_prompt_contains_the_goal_and_uses_the_planner_model() -> None:
    client = stub(plan(entry("1", "A")))

    asyncio.run(PlannerAgent(client=client).plan("Add JWT auth"))

    assert client.planner_prompts[0] == f"{PLANNER_PROMPT}\n\nGoal: Add JWT auth\n"
    assert client.calls[0].model == client.settings.planner_model


def test_the_planner_model_setting_is_honoured() -> None:
    client = stub(plan(entry("1", "A")), settings=Settings(planner_model="cheap-model"))

    asyncio.run(PlannerAgent(client=client).plan("GOAL"))

    assert client.calls[0].model == "cheap-model"


def test_a_client_is_built_when_none_is_supplied() -> None:
    agent = PlannerAgent()

    assert agent.client is not None
    assert agent.client.settings is not None


def test_plan_returns_every_parsed_task() -> None:
    response = plan(*[entry(str(index), f"Task {index}") for index in range(1, 6)])

    result = asyncio.run(parse(response))

    assert len(result.tasks) == 5
