import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan


class FakeLLMClient:
    """Duck-typed LLMClient returning canned responses in order."""

    def __init__(self, responses, settings=None):
        from furrow.config import Settings

        self.responses = list(responses)
        self.settings = settings or Settings()
        self.prompts: list[str] = []

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


async def test_plan_parses_fenced_json():
    response = """```json
{
  "tasks": [
    {"description": "task one"},
    {"id": "2", "description": "task two", "files": ["a.py"]}
  ],
  "rationale": "ok"
}
```"""
    fake = FakeLLMClient([response])
    planner = PlannerAgent(client=fake)
    plan = await planner.plan("goal")
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].files == []
    assert plan.tasks[1].id == "2"
    assert plan.tasks[1].files == ["a.py"]
    assert plan.rationale == "ok"


async def test_plan_parse_failure_raises():
    fake = FakeLLMClient(["not json at all"])
    planner = PlannerAgent(client=fake)
    with pytest.raises(ValueError):
        await planner.plan("goal")


async def test_plan_skips_invalid_entries():
    response = """```json
{
  "tasks": [
    {"id": "1", "description": "valid one"},
    "garbage string",
    {"id": "3", "description": "valid two"}
  ],
  "rationale": "ok"
}
```"""
    fake = FakeLLMClient([response])
    planner = PlannerAgent(client=fake)
    plan = await planner.plan("goal")
    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "1"
    assert plan.tasks[1].id == "3"