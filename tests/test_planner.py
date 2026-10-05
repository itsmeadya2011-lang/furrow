"""Tests for :class:`furrow.agents.planner.PlannerAgent`.

The planner is the only place a plan can enter the system, so these tests pin three
things: the persona/model plumbing around the LLM call, the unwrapping of the
markdown fences and prose that real models wrap JSON in, and the promise that an
unusable response always surfaces as a :class:`ValueError` carrying the raw response
- never as an escaping ``TypeError``.
"""

from __future__ import annotations

import json

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings
from tests.conftest import FakeLLM

GOAL = "Add JWT authentication to the API"
PARSE_ERROR = "Failed to parse plan from LLM"

PLAN_PAYLOAD: dict[str, object] = {
    "tasks": [
        {
            "id": "1",
            "description": "Implement the JWT helpers",
            "files": ["src/auth.py"],
            "dependencies": [],
        },
        {
            "id": "2",
            "description": "Expose the login route",
            "files": ["src/api/routes.py", "tests/test_auth.py"],
            "dependencies": ["1"],
        },
    ],
    "rationale": "The route needs the helpers, so it depends on task 1.",
}

RAW_PLAN = json.dumps(PLAN_PAYLOAD)
FENCED_PLAN = f"```json\n{RAW_PLAN}\n```"
PROSE_PLAN = f"Sure! Here is the plan:\n{RAW_PLAN}\nLet me know if you want changes."


def _planner(response: str, settings: Settings | None = None) -> PlannerAgent:
    """A planner wired to a fake client, so no test can reach a real LLM."""
    return PlannerAgent(client=FakeLLM(response=response), settings=settings)


async def test_plan_parses_raw_json() -> None:
    plan = await _planner(RAW_PLAN).plan(GOAL)

    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 2

    first, second = plan.tasks
    assert first.id == "1"
    assert first.description == "Implement the JWT helpers"
    assert first.files == ["src/auth.py"]
    assert first.dependencies == []
    assert first.status == "pending"
    assert first.result is None

    assert second.id == "2"
    assert second.description == "Expose the login route"
    assert second.files == ["src/api/routes.py", "tests/test_auth.py"]
    assert second.dependencies == ["1"]
    assert plan.rationale == PLAN_PAYLOAD["rationale"]


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(FENCED_PLAN, id="fenced-json"),
        pytest.param(PROSE_PLAN, id="prose-wrapped"),
    ],
)
async def test_plan_parses_json_the_model_wrapped(response: str) -> None:
    """Fenced and prose-wrapped JSON used to crash the planner hard."""
    plan = await _planner(response).plan(GOAL)

    assert [task.id for task in plan.tasks] == ["1", "2"]
    assert plan.rationale == PLAN_PAYLOAD["rationale"]


async def test_plan_accepts_an_empty_task_list() -> None:
    """An empty plan is schema-valid; the planner must not invent a requirement."""
    plan = await _planner('{"tasks": [], "rationale": "nothing to do"}').plan(GOAL)

    assert plan.tasks == []
    assert plan.rationale == "nothing to do"


@pytest.mark.parametrize(
    "response",
    [
        pytest.param("null", id="json-null"),
        pytest.param("[1, 2, 3]", id="json-array"),
        pytest.param('["a", "b"]', id="json-array-of-strings"),
        pytest.param('"just a string"', id="json-string"),
        pytest.param("42", id="json-number"),
        pytest.param("true", id="json-bool"),
    ],
)
async def test_non_mapping_json_raises_value_error(response: str) -> None:
    """JSON that is not an object must be reported as a ``ValueError``.

    The planner used to call ``Plan(**data)``, which raises ``TypeError`` for
    anything that is not a mapping and escaped the agent's own error handling.
    """
    agent = _planner(response)

    with pytest.raises(ValueError, match=PARSE_ERROR) as excinfo:
        await agent.plan(GOAL)

    # TypeError is not a ValueError subclass, so the context manager above already
    # fails loudly for the old bug; assert it explicitly to document the regression.
    assert not isinstance(excinfo.value, TypeError)
    assert response in str(excinfo.value)


@pytest.mark.parametrize(
    ("payload", "missing_field"),
    [
        pytest.param(
            json.dumps({"tasks": [{"id": "1", "description": "do it"}]}),
            "rationale",
            id="missing-rationale",
        ),
        pytest.param(json.dumps({"rationale": "nothing to do"}), "tasks", id="missing-tasks"),
    ],
)
async def test_incomplete_plan_raises_value_error(payload: str, missing_field: str) -> None:
    agent = _planner(payload)

    with pytest.raises(ValueError, match=PARSE_ERROR) as excinfo:
        await agent.plan(GOAL)

    assert missing_field in str(excinfo.value)
    assert payload in str(excinfo.value)


async def test_plan_sends_the_persona_as_the_system_message(settings: Settings) -> None:
    fake_llm = FakeLLM(response=RAW_PLAN)
    agent = PlannerAgent(client=fake_llm, settings=settings)

    await agent.plan(GOAL)

    prompt, system, _model = fake_llm.calls[0]
    assert len(fake_llm.calls) == 1
    assert system == PLANNER_PROMPT
    assert GOAL in prompt
    # The persona must not be duplicated into the user message.
    assert PLANNER_PROMPT not in prompt


async def test_plan_requests_the_configured_planner_model(settings: Settings) -> None:
    fake_llm = FakeLLM(response=RAW_PLAN)
    agent = PlannerAgent(client=fake_llm, settings=settings)

    await agent.plan(GOAL)

    assert fake_llm.calls[0][2] == settings.planner_model
    assert agent.model_attr == "planner_model"


async def test_plan_propagates_transport_errors() -> None:
    """A failing LLM call is not a plan problem, so it is not disguised as one."""
    agent = PlannerAgent(client=FakeLLM(error=RuntimeError("llm exploded")))

    with pytest.raises(RuntimeError, match="llm exploded"):
        await agent.plan(GOAL)


def test_planner_constructs_with_an_injected_client(settings: Settings) -> None:
    fake_llm = FakeLLM(response=RAW_PLAN)

    agent = PlannerAgent(client=fake_llm, settings=settings)

    assert agent.client is fake_llm
    assert agent.settings is settings
    assert agent.model_attr == "planner_model"
