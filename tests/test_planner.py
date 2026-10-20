from __future__ import annotations

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings


class FakeClient:
    def __init__(self, responses, settings=None):
        self.responses = list(responses)
        self.settings = settings or Settings()
        self.calls = 0

    async def complete(self, prompt, system="", model=None):
        self.calls += 1
        if self.calls > len(self.responses):
            return self.responses[-1]
        return self.responses[self.calls - 1]


def _plan_json(task_ids):
    parts = [
        f'{{"id":"{i}","description":"do thing {i}","files":["f.py"],"dependencies":[]}}'
        for i in task_ids
    ]
    return '{"tasks":[' + ",".join(parts) + '],"rationale":"because"}'


async def test_bare_json():
    agent = PlannerAgent(client=FakeClient([_plan_json(["1", "2"])]))
    plan = await agent.plan("build something")
    assert isinstance(plan, Plan)
    assert [t.id for t in plan.tasks] == ["1", "2"]
    assert plan.rationale == "because"


async def test_markdown_fence():
    body = _plan_json(["3"])
    agent = PlannerAgent(client=FakeClient(["```json\n" + body + "\n```"]))
    plan = await agent.plan("fenced")
    assert [t.id for t in plan.tasks] == ["3"]


async def test_prose_wrapped():
    body = _plan_json(["4"])
    agent = PlannerAgent(client=FakeClient(["Here you go: " + body + " done"]))
    plan = await agent.plan("prose")
    assert [t.id for t in plan.tasks] == ["4"]


async def test_retry_then_success():
    agent = PlannerAgent(client=FakeClient(["not json at all", _plan_json(["5"])]))
    plan = await agent.plan("retry")
    assert [t.id for t in plan.tasks] == ["5"]
    assert agent.client.calls == 2


async def test_both_invalid_raises():
    agent = PlannerAgent(client=FakeClient(["garbage", "still garbage"]))
    try:
        await agent.plan("fail")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


async def test_wrong_shape_raises():
    agent = PlannerAgent(client=FakeClient(['{"tasks": "nope"}']))
    try:
        await agent.plan("shape")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
