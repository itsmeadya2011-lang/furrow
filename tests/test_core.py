import json

import pytest
from pydantic import ValidationError

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult


class FakeClient:
    def __init__(self, responses: list[str], settings: Settings | None = None) -> None:
        self.responses = list(responses)
        self.settings = settings
        self.calls: list[tuple[str, str | None]] = []

    async def complete(self, prompt: str, model: str | None = None) -> str:
        self.calls.append((prompt, model))
        return self.responses.pop(0)


def plan_response(tasks: list[dict[str, object]] | None = None, rationale: str = "ok") -> str:
    return json.dumps({"tasks": tasks if tasks is not None else [], "rationale": rationale})


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_empty_plan_is_allowed():
    plan = Plan(tasks=[], rationale="already complete")
    assert plan.tasks == []


def test_text_fields_are_trimmed_and_required():
    task = TaskModel(id="  task-1  ", description="  do thing  ")
    plan = Plan(tasks=[task], rationale="  rationale  ")

    assert task.id == "task-1"
    assert task.description == "do thing"
    assert plan.rationale == "rationale"


@pytest.mark.parametrize("value", ["", "   "])
def test_required_text_cannot_be_blank(value: str):
    with pytest.raises(ValidationError):
        TaskModel(id=value, description="do thing")
    with pytest.raises(ValidationError):
        TaskModel(id="1", description=value)
    with pytest.raises(ValidationError):
        Plan(tasks=[], rationale=value)


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/path",
        "C:\\absolute\\path",
        "C:/absolute/path",
        "\\\\host\\share\\path",
        "~/home/path",
        "../outside",
        "pkg/../outside",
        "bad\0path",
        "bad\npath",
    ],
)
def test_unsafe_file_paths_are_rejected(path: str):
    with pytest.raises(ValidationError):
        TaskModel(id="1", description="do thing", files=[path])


def test_safe_relative_file_paths_are_accepted():
    task = TaskModel(
        id="1",
        description="do thing",
        files=["src/app.py", "tests/test_app.py", "./readme.md"],
    )
    assert task.files == ["src/app.py", "tests/test_app.py", "./readme.md"]


def test_dependency_references_must_exist():
    with pytest.raises(ValidationError, match="unknown dependencies"):
        Plan(
            tasks=[TaskModel(id="1", description="do thing", dependencies=["missing"])],
            rationale="ok",
        )


def test_dependency_graph_must_be_acyclic():
    with pytest.raises(ValidationError, match="cycle"):
        Plan(
            tasks=[
                TaskModel(id="1", description="first", dependencies=["2"]),
                TaskModel(id="2", description="second", dependencies=["1"]),
            ],
            rationale="ok",
        )


def test_task_ids_and_dependencies_must_be_unique():
    with pytest.raises(ValidationError, match="task ids must be unique"):
        Plan(
            tasks=[
                TaskModel(id="1", description="first"),
                TaskModel(id="1", description="duplicate"),
            ],
            rationale="ok",
        )

    with pytest.raises(ValidationError, match="dependencies must be unique"):
        TaskModel(id="1", description="do thing", dependencies=["2", "2"])


@pytest.mark.parametrize("status", ["pending", "running", "completed", "failed", "blocked"])
def test_task_status_literals(status: str):
    assert TaskModel(id="1", description="do thing", status=status).status == status


def test_invalid_task_status_is_rejected():
    with pytest.raises(ValidationError):
        TaskModel(id="1", description="do thing", status="archived")


def test_settings_defaults_and_constraints():
    defaults = Settings()
    assert defaults.max_cycles == 10
    assert defaults.test_timeout > 0
    assert defaults.max_test_output_chars >= 1000

    with pytest.raises(ValidationError):
        Settings(max_parallel_tasks=0)
    with pytest.raises(ValidationError):
        Settings(max_cycles=0)
    with pytest.raises(ValidationError):
        Settings(test_timeout=0)
    with pytest.raises(ValidationError):
        Settings(max_test_output_chars=999)


@pytest.mark.asyncio
async def test_planner_retries_malformed_json_and_uses_effective_settings():
    settings = Settings(planner_model="planner-test")
    client = FakeClient(
        [
            "not json",
            plan_response(
                [{"id": "1", "description": "do thing", "dependencies": []}],
            ),
        ],
        settings=settings,
    )
    planner = PlannerAgent(client=client, settings=settings)

    plan = await planner.plan("build it", context="existing code")

    assert [task.id for task in plan.tasks] == ["1"]
    assert len(client.calls) == 2
    assert all(model == "planner-test" for _, model in client.calls)
    assert "<goal>\nbuild it\n</goal>" in client.calls[0][0]
    assert "<context>\nexisting code\n</context>" in client.calls[0][0]


@pytest.mark.asyncio
async def test_planner_raises_with_response_after_final_validation_failure():
    settings = Settings(planner_model="planner-test")
    client = FakeClient(["not json", plan_response(rationale="   ")], settings=settings)
    planner = PlannerAgent(client=client, settings=settings)

    with pytest.raises(ValueError, match="after 2 attempts.*Response:") as exc:
        await planner.plan("build it")

    assert "not json" in str(exc.value)
    assert "rationale" in str(exc.value)
    assert len(client.calls) == 2
