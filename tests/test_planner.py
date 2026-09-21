import json

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Settings


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.settings = Settings(workspace=".", test_timeout=1.0)

    async def complete(self, prompt, model=None):
        self.calls += 1
        index = min(self.calls - 1, len(self.responses) - 1)
        return self.responses[index]


def plan_response(tasks, rationale="plan"):
    return json.dumps({"tasks": tasks, "rationale": rationale})


def task(task_id, description, dependencies=None):
    return {
        "id": task_id,
        "description": description,
        "dependencies": [] if dependencies is None else dependencies,
    }


@pytest.mark.asyncio
async def test_plan_parses_valid_tasks():
    client = FakeClient(
        [
            plan_response(
                [
                    task("1", "Implement authentication"),
                    task("2", "Add login tests", ["1"]),
                ]
            )
        ]
    )

    plan = await PlannerAgent(client=client, settings=client.settings).plan("add login")

    assert [item.id for item in plan.tasks] == ["1", "2"]
    assert plan.tasks[1].dependencies == ["1"]
    assert client.calls == 1


@pytest.mark.asyncio
async def test_plan_retries_malformed_and_invalid_responses():
    valid = plan_response([task("1", "Do the work")])
    client = FakeClient(["not json", "{}", valid])

    plan = await PlannerAgent(client=client, settings=client.settings).plan("do work")

    assert [item.id for item in plan.tasks] == ["1"]
    assert client.calls == 3


@pytest.mark.asyncio
async def test_plan_rejects_duplicate_and_unknown_dependencies():
    duplicate = plan_response(
        [task("1", "First"), task("1", "Duplicate")]
    )
    unknown = plan_response([task("1", "First", ["missing"])])

    for response, message in (
        (duplicate, "unique"),
        (unknown, "unknown dependency"),
    ):
        client = FakeClient([response])
        with pytest.raises(ValueError, match=message):
            await PlannerAgent(client=client, settings=client.settings).plan("goal")
        assert client.calls == 3


@pytest.mark.asyncio
async def test_plan_rejects_dependency_cycle():
    response = plan_response(
        [
            task("1", "First", ["2"]),
            task("2", "Second", ["1"]),
        ]
    )
    client = FakeClient([response])

    with pytest.raises(ValueError, match="cycle"):
        await PlannerAgent(client=client, settings=client.settings).plan("goal")

    assert client.calls == 3


@pytest.mark.asyncio
async def test_plan_allows_explicit_empty_task_list():
    client = FakeClient([plan_response([])])

    plan = await PlannerAgent(client=client, settings=client.settings).plan("nothing")

    assert plan.tasks == []
    assert client.calls == 1
