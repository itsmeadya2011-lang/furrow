import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Provider, Settings


class FakePlannerClient:
    def __init__(self, settings: Settings, response: str):
        self.settings = settings
        self._response = response

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        return self._response


@pytest.fixture
def planner_settings(tmp_path) -> Settings:
    return Settings(
        provider=Provider.OLLAMA,
        workspace=tmp_path,
        max_parallel_tasks=3,
        max_cycles=0,
        model="test-model",
        planner_model="test-planner",
        worker_model="test-worker",
        tester_model="test-tester",
    )


@pytest.mark.asyncio
async def test_planner_caps_tasks_to_max_parallel(planner_settings):
    seven_tasks = {
        "tasks": [
            {"id": str(i), "description": f"Task {i}"} for i in range(1, 8)
        ],
        "rationale": "Seven tasks"
    }
    import json
    response = json.dumps(seven_tasks)
    fake_client = FakePlannerClient(planner_settings, response)

    planner = PlannerAgent(client=fake_client)
    plan = await planner.plan("test goal")

    assert len(plan.tasks) == 3
    assert plan.tasks[0].id == "1"
    assert plan.tasks[2].id == "3"


@pytest.mark.asyncio
async def test_planner_parses_json_wrapped_in_prose(planner_settings):
    wrapped_response = 'Here is the plan:\n{"tasks": [{"id": "1", "description": "Task 1"}], "rationale": "Test"}\nThanks!'
    fake_client = FakePlannerClient(planner_settings, wrapped_response)

    planner = PlannerAgent(client=fake_client)
    plan = await planner.plan("test goal")

    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "Task 1"


@pytest.mark.asyncio
async def test_planner_raises_on_garbage(planner_settings):
    garbage_response = "This is not JSON at all"
    fake_client = FakePlannerClient(planner_settings, garbage_response)

    planner = PlannerAgent(client=fake_client)

    with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
        await planner.plan("test goal")


@pytest.mark.asyncio
async def test_planner_raises_on_missing_tasks_key(planner_settings):
    missing_tasks = {"rationale": "No tasks here"}
    import json
    response = json.dumps(missing_tasks)
    fake_client = FakePlannerClient(planner_settings, response)

    planner = PlannerAgent(client=fake_client)

    with pytest.raises(ValueError, match="missing 'tasks' key"):
        await planner.plan("test goal")


@pytest.mark.asyncio
async def test_planner_raises_on_invalid_json_in_braces(planner_settings):
    invalid_json = 'Here is the plan: {"tasks": [{"id": "1", "description": "Task 1"}'
    fake_client = FakePlannerClient(planner_settings, invalid_json)

    planner = PlannerAgent(client=fake_client)

    with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
        await planner.plan("test goal")