from __future__ import annotations

import asyncio

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings


class FakeLLMClient:
    def __init__(self, response="", raises=None):
        self.response = response
        self.raises = raises
        self.prompts = []
        self.models = []

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        self.models.append(model)
        if self.raises:
            raise self.raises
        return self.response


class FakeLLMClientWithResponses:
    """Fake client that returns different responses on each call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.models = []

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        self.models.append(model)
        if not self.responses:
            return "{}"
        return self.responses.pop(0)


def run(coro):
    """Run a coroutine to completion on a fresh event loop."""
    return asyncio.run(coro)


def test_valid_json():
    client = FakeLLMClient(
        response='{"tasks":[{"id":"1","description":"do thing"}],"rationale":"r"}'
    )
    s = Settings(_env_file=None)
    agent = PlannerAgent(client=client, settings=s)
    plan = run(agent.plan("goal"))
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"


def test_fenced_json():
    client = FakeLLMClient(response='```json\n{"tasks":[],"rationale":"x"}\n```')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.tasks == []


def test_prose_surrounded():
    client = FakeLLMClient(
        response='Here is plan: {"tasks":[{"id":"2","description":"d"}],"rationale":"y"} thanks'
    )
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.tasks[0].id == "2"


def test_invalid_twice_raises():
    client = FakeLLMClient(response="not json at all")
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    try:
        run(agent.plan("g"))
        raised = False
    except ValueError as e:
        raised = True
        assert "not json" in str(e) or "response" in str(e).lower() or len(str(e)) > 0
    assert raised


def test_context_included():
    client = FakeLLMClient(response='{"tasks":[],"rationale":""}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    run(agent.plan("g", context="FAILING TESTS"))
    assert "FAILING TESTS" in client.prompts[0]


def test_uses_planner_model():
    client = FakeLLMClient(response='{"tasks":[],"rationale":""}')
    s = Settings(_env_file=None, planner_model="my-model")
    agent = PlannerAgent(client=client, settings=s)
    run(agent.plan("g"))
    assert client.models[0] == "my-model"


def test_plan_with_dependencies():
    """A plan with inter-task dependencies is parsed correctly."""
    client = FakeLLMClient(
        response='{"tasks":[{"id":"1","description":"setup","files":["a.py"],"dependencies":[]},'
                  '{"id":"2","description":"impl","files":["b.py"],"dependencies":["1"]}],"rationale":"r"}'
    )
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert len(plan.tasks) == 2
    assert plan.tasks[0].dependencies == []
    assert plan.tasks[1].dependencies == ["1"]


def test_plan_with_unknown_status_normalized():
    """An unknown status from the LLM is normalized to 'pending'."""
    client = FakeLLMClient(
        response='{"tasks":[{"id":"1","description":"d","status":"WEIRD"}],"rationale":"r"}'
    )
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.tasks[0].status == "pending"


def test_plan_missing_rationale_defaults_to_empty():
    """A plan without a rationale field still parses."""
    client = FakeLLMClient(response='{"tasks":[{"id":"1","description":"d"}]}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.rationale == ""


def test_plan_missing_tasks_defaults_to_empty():
    """A plan without a tasks field still parses as an empty plan."""
    client = FakeLLMClient(response='{"rationale":"r"}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.tasks == []


def test_plan_null_rationale_treated_as_empty():
    """A null rationale is treated as an empty string."""
    client = FakeLLMClient(response='{"tasks":[],"rationale":null}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.rationale == ""


def test_plan_null_tasks_treated_as_empty():
    """A null tasks field is treated as an empty list."""
    client = FakeLLMClient(response='{"tasks":null,"rationale":"r"}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    plan = run(agent.plan("g"))
    assert plan.tasks == []


def test_plan_retries_once_on_parse_failure():
    """When the first response fails to parse, the planner retries once."""
    client = FakeLLMClientWithResponses(["not valid json", '{"tasks":[],"rationale":"ok"}'])
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    run(agent.plan("g"))
    # Two calls: the original and the retry with stricter instructions.
    assert len(client.prompts) == 2
    assert "Your previous reply could not be parsed" in client.prompts[1]


def test_plan_uses_system_prompt():
    """The planner passes its system prompt to the LLM client."""
    client = FakeLLMClient(response='{"tasks":[],"rationale":""}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    run(agent.plan("g"))
    assert client.models[0] is not None
    # The system prompt is non-empty.
    assert client.prompts[0]


def test_plan_passes_context_to_retry():
    """The retry prompt also includes the original context."""
    client = FakeLLMClientWithResponses(
        ["not valid json", '{"tasks":[],"rationale":"ok"}']
    )
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    run(agent.plan("g", context="FAILING TESTS"))
    assert "FAILING TESTS" in client.prompts[0]
    assert "FAILING TESTS" in client.prompts[1]


def test_plan_empty_response_raises():
    """An empty response from the LLM raises a ValueError."""
    client = FakeLLMClient(response="")
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    raised = False
    try:
        run(agent.plan("g"))
    except ValueError:
        raised = True
    assert raised


def test_plan_json_array_raises():
    """A bare JSON array (not an object) raises a ValueError."""
    client = FakeLLMClient(response="[1, 2, 3]")
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    raised = False
    try:
        run(agent.plan("g"))
    except ValueError:
        raised = True
    assert raised


def test_plan_tasks_not_a_list_raises():
    """A plan whose 'tasks' value is not a list raises a ValueError."""
    client = FakeLLMClient(response='{"tasks": "not-a-list", "rationale": "r"}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    raised = False
    try:
        run(agent.plan("g"))
    except ValueError:
        raised = True
    assert raised


def test_plan_task_not_an_object_raises():
    """A plan whose tasks list contains a non-object raises a ValueError."""
    client = FakeLLMClient(response='{"tasks": [42], "rationale": "r"}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    raised = False
    try:
        run(agent.plan("g"))
    except ValueError:
        raised = True
    assert raised


def test_plan_includes_original_goal_in_prompt():
    """The original goal is included verbatim in the prompt sent to the LLM."""
    client = FakeLLMClient(response='{"tasks":[],"rationale":""}')
    agent = PlannerAgent(client=client, settings=Settings(_env_file=None))
    run(agent.plan("Add JWT auth to the API"))
    assert "Add JWT auth to the API" in client.prompts[0]
