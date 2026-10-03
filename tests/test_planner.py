"""Unit tests for plan parsing/validation in ``furrow.agents.planner``."""

from __future__ import annotations

from pathlib import Path

import pytest

from furrow.agents.planner import PlannerAgent, _build_task, _parse_plan, _validate
from furrow.config import Plan, Settings, TaskModel

# --------------------------------------------------------------------------- #
# _parse_plan: accepted shapes
# --------------------------------------------------------------------------- #


def test_parse_plan_fenced_json():
    plan = _parse_plan(
        '```json\n{"tasks": [{"id": "a", "description": "write a"}], "rationale": "because"}\n```'
    )
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "a"
    assert plan.tasks[0].description == "write a"
    assert plan.rationale == "because"


def test_parse_plan_unfenced_json():
    plan = _parse_plan('{"tasks": [{"id": "a", "description": "write a"}], "rationale": "because"}')
    assert [t.id for t in plan.tasks] == ["a"]
    assert plan.rationale == "because"


def test_parse_plan_bare_list_of_task_dicts():
    plan = _parse_plan(
        '[{"id": "1", "description": "first"}, {"id": "2", "description": "second"}]'
    )
    assert [t.id for t in plan.tasks] == ["1", "2"]
    assert [t.description for t in plan.tasks] == ["first", "second"]


def test_parse_plan_single_task_dict_without_a_tasks_key():
    plan = _parse_plan('{"id": "solo", "description": "do the only thing", "files": ["a.py"]}')
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "solo"
    assert plan.tasks[0].files == ["a.py"]


def test_parse_plan_rationale_defaults_to_empty_string():
    plan = _parse_plan('{"tasks": [{"id": "1", "description": "d"}]}')
    assert plan.rationale == ""


def test_parse_plan_without_tasks_yields_an_empty_plan():
    plan = _parse_plan('{"rationale": "nothing to do"}')
    assert plan.tasks == []
    assert plan.rationale == "nothing to do"


def test_parse_plan_bare_string_task():
    plan = _parse_plan('{"tasks": ["first", "second"]}')
    assert [t.description for t in plan.tasks] == ["first", "second"]


def test_parse_plan_coerces_integer_id_to_string():
    plan = _parse_plan('{"tasks": [{"id": 7, "description": "d"}]}')
    assert plan.tasks[0].id == "7"


def test_parse_plan_missing_id_uses_the_one_based_index():
    plan = _parse_plan(
        '{"tasks": [{"description": "a"}, {"description": "b"}, {"id": "x", "description": "c"}]}'
    )
    assert [t.id for t in plan.tasks] == ["1", "2", "x"]


def test_parse_plan_stringifies_file_and_dependency_entries():
    # Task "2" must exist in the plan: _validate() prunes dependencies on unknown
    # ids, so a dangling dep would be dropped and prove nothing about stringifying.
    plan = _parse_plan(
        '{"tasks": [{"id": "1", "description": "d", "files": [1], "dependencies": [2]},'
        ' {"id": "2", "description": "e"}]}'
    )
    assert plan.tasks[0].files == ["1"]
    assert plan.tasks[0].dependencies == ["2"]


def test_parse_plan_drops_dependencies_on_unknown_task_ids():
    plan = _parse_plan(
        '{"tasks": [{"id": "1", "description": "d", "dependencies": ["2"]}]}'
    )
    assert plan.tasks[0].dependencies == []


def test_parse_plan_keeps_known_dependencies():
    plan = _parse_plan(
        '{"tasks": [{"id": "1", "description": "a"},'
        ' {"id": "2", "description": "b", "dependencies": ["1"]}]}'
    )
    assert plan.tasks[1].dependencies == ["1"]


# --------------------------------------------------------------------------- #
# _parse_plan: rejected shapes
# --------------------------------------------------------------------------- #


def test_parse_plan_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        _parse_plan(
            '{"tasks": [{"id": "a", "description": "one"}, {"id": "a", "description": "two"}]}'
        )


def test_parse_plan_self_dependency_rejected():
    with pytest.raises(ValueError):
        _parse_plan('{"tasks": [{"id": "a", "description": "one", "dependencies": ["a"]}]}')


def test_parse_plan_dependency_on_unknown_id_is_pruned():
    plan = _parse_plan('{"tasks": [{"id": "a", "description": "one", "dependencies": ["ghost"]}]}')
    assert plan.tasks[0].dependencies == []


def test_parse_plan_empty_description_rejected():
    with pytest.raises(ValueError):
        _parse_plan('{"tasks": [{"id": "a", "description": "   "}]}')


def test_parse_plan_missing_description_rejected():
    with pytest.raises(ValueError):
        _parse_plan('{"tasks": [{"id": "a"}]}')


def test_parse_plan_non_object_task_rejected():
    with pytest.raises(ValueError):
        _parse_plan('{"tasks": [42]}')


def test_parse_plan_tasks_must_be_a_list():
    with pytest.raises(ValueError):
        _parse_plan('{"tasks": "nope"}')


def test_parse_plan_scalar_top_level_rejected():
    with pytest.raises(ValueError):
        _parse_plan('"just a string"')


def test_parse_plan_unparseable_output_rejected():
    with pytest.raises(ValueError):
        _parse_plan("I am not going to emit JSON today.")


# --------------------------------------------------------------------------- #
# _build_task / _validate in isolation
# --------------------------------------------------------------------------- #


def test_build_task_accepts_a_bare_string():
    task = _build_task("do it", 0)
    assert task.id == "1"
    assert task.description == "do it"
    assert task.status == "pending"


def test_build_task_coerces_integer_id():
    assert _build_task({"id": 7, "description": "d"}, 0).id == "7"


def test_build_task_empty_id_falls_back_to_the_index():
    assert _build_task({"id": "", "description": "d"}, 4).id == "5"
    assert _build_task({"description": "d"}, 0).id == "1"


def test_build_task_strips_the_description():
    assert _build_task({"id": "a", "description": "  spaced  "}, 0).description == "spaced"


def test_build_task_stringifies_dependency_ids():
    task = _build_task({"id": "a", "description": "d", "dependencies": [2]}, 0)
    assert task.dependencies == ["2"]


def test_validate_prunes_unknown_dependencies_in_place():
    task = TaskModel(id="a", description="a", dependencies=["ghost"])
    _validate([task])
    assert task.dependencies == []


def test_validate_rejects_duplicate_ids():
    tasks = [TaskModel(id="a", description="1"), TaskModel(id="a", description="2")]
    with pytest.raises(ValueError):
        _validate(tasks)


def test_validate_rejects_self_dependency():
    with pytest.raises(ValueError):
        _validate([TaskModel(id="a", description="1", dependencies=["a"])])


# --------------------------------------------------------------------------- #
# PlannerAgent
# --------------------------------------------------------------------------- #


class _FakeClient:
    """Just enough of ``LLMClient`` for the planner."""

    def __init__(self, response: str, settings: Settings) -> None:
        self.response = response
        self.settings = settings
        self.calls: list[tuple[str, str | None]] = []

    async def complete(self, prompt, system: str = "", model: str | None = None) -> str:
        self.calls.append((prompt, model))
        return self.response


def _settings(tmp_path) -> Settings:
    return Settings(_env_file=None, workspace=Path(tmp_path))


async def test_plan_returns_a_plan_from_fenced_json(tmp_path):
    settings = _settings(tmp_path)
    client = _FakeClient(
        '```json\n{"tasks": [{"id": "1", "description": "add a feature"}],'
        ' "rationale": "small"}\n```',
        settings,
    )
    agent = PlannerAgent(client=client, settings=settings)

    plan = await agent.plan("build a thing")

    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "add a feature"
    assert plan.rationale == "small"
    # The goal reaches the prompt and the role-specific model is passed through.
    prompt, model = client.calls[0]
    assert "build a thing" in prompt
    assert model == settings.planner_model


async def test_plan_propagates_a_parse_failure(tmp_path):
    settings = _settings(tmp_path)
    client = _FakeClient("no json here", settings)
    agent = PlannerAgent(client=client, settings=settings)
    with pytest.raises(ValueError):
        await agent.plan("build a thing")


def test_settings_fall_back_to_the_client_attribute(tmp_path):
    settings = _settings(tmp_path)
    client = _FakeClient('{"tasks": []}', settings)
    agent = PlannerAgent(client=client)
    assert agent.settings is settings
    assert agent.model == settings.planner_model
