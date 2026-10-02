"""Tests for the PlannerAgent."""

import json

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan


class FakeLLMClient:
    """Fake LLM client that returns scripted JSON responses."""

    def __init__(self, json_response: str, list_files_result: list[str] | None = None) -> None:
        self._json_response = json_response
        self._list_files_result = list_files_result or []
        self.settings = type(
            "S",
            (),
            {
                "workspace": ".",
                "planner_model": "fake-planner",
            },
        )()

    async def list_files(self, directory):
        return self._list_files_result

    async def complete_json(
        self, prompt, system="", model=None, temperature=0.7, max_tokens=4096, retries=2
    ):
        return json.loads(self._json_response)


@pytest.fixture
def valid_plan_json():
    return json.dumps(
        {
            "tasks": [
                {
                    "id": "1",
                    "description": "Create auth module",
                    "files": ["src/auth.py", "tests/test_auth.py"],
                    "dependencies": [],
                },
                {
                    "id": "2",
                    "description": "Add user model",
                    "files": ["src/models.py"],
                    "dependencies": ["1"],
                },
            ],
            "rationale": "Break auth into module and model",
        }
    )


@pytest.fixture
def empty_plan_json():
    return json.dumps({"tasks": [], "rationale": "Nothing to do"})


@pytest.fixture
def invalid_json():
    return "This is not valid JSON"


def test_planner_init():
    """PlannerAgent initializes with a client."""
    planner = PlannerAgent()
    assert planner.client is not None


@pytest.fixture
def planner_with_client(valid_plan_json):
    return PlannerAgent(client=FakeLLMClient(valid_plan_json, ["main.py", "tests/test_main.py"]))


@pytest.mark.asyncio
async def test_planner_returns_valid_plan(planner_with_client):
    """Planner returns a Plan with parsed tasks."""
    plan = await planner_with_client.plan("Add auth system")
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 2
    assert plan.tasks[0].description == "Create auth module"
    assert plan.tasks[0].files == ["src/auth.py", "tests/test_auth.py"]
    assert plan.tasks[1].dependencies == ["1"]
    assert "auth" in plan.rationale.lower()


@pytest.mark.asyncio
async def test_planner_includes_workspace_context(valid_plan_json):
    """Planner includes existing files in the prompt."""
    client = FakeLLMClient(valid_plan_json, ["main.py", "utils.py"])
    planner = PlannerAgent(client=client)
    plan = await planner.plan("Some goal")
    assert plan is not None


@pytest.mark.asyncio
async def test_planner_fallback_on_empty_plan(empty_plan_json):
    """Planner creates fallback plan when LLM returns no tasks."""
    client = FakeLLMClient(empty_plan_json, [])
    planner = PlannerAgent(client=client)
    plan = await planner.plan("Do something")
    assert len(plan.tasks) == 1
    assert "fallback" in plan.rationale.lower()


@pytest.mark.asyncio
async def test_planner_fallback_on_invalid_json(invalid_json):
    """Planner creates fallback plan when LLM returns invalid JSON."""
    client = FakeLLMClient(invalid_json, [])
    planner = PlannerAgent(client=client)
    plan = await planner.plan("Do something")
    assert len(plan.tasks) == 1
    assert "fallback" in plan.rationale.lower()


@pytest.mark.asyncio
async def test_planner_fallback_on_missing_keys():
    """Planner handles JSON that parses but has missing task keys."""
    bad_json = json.dumps({"rationale": "test"})  # no tasks key
    client = FakeLLMClient(bad_json, [])
    planner = PlannerAgent(client=client)
    plan = await planner.plan("Do something")
    assert len(plan.tasks) == 1  # fallback
