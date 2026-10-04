"""Tests for :mod:`furrow.agents.planner`.

The planner is the one place where untrusted model text becomes scheduling
state, so these tests pin down three properties:

* a messy-but-recoverable response (markdown fence, prose on both sides)
  still yields the plan the model meant;
* a *hostile* plan -- duplicate ids, a cycle, a self-dependency, a blank
  description, a dangling dependency -- comes back schedulable: ids
  ``"1".."n"``, acyclic, no blank description, no dangling edge;
* an unrecoverable response gets exactly one corrective retry, after which
  :meth:`PlannerAgent.plan` raises :class:`~furrow.errors.StructuredOutputError`.

An empty task list is asserted as a *success*, not a failure: it is the run
loop's termination signal, and the planner never retries it.

Everything runs against the ``fake_client`` stand-in from ``conftest``, so no
test here reaches a provider, and every ``Settings`` is built by
``make_settings`` so an ambient ``FURROW_*`` env var cannot reach the planner's
model name or task cap.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings
from furrow.errors import StructuredOutputError
from furrow.llm import CompletionResult

# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

#: A goal string that appears verbatim in the user message the planner builds.
GOAL = "add exponential backoff to the HTTP client in furrow/llm.py"

#: A response with no recoverable JSON in it at all.
GARBAGE = "I'm sorry, I cannot produce a JSON plan for that request."


def _plan_text(tasks: list[dict[str, Any]], rationale: str = "the model's own words") -> str:
    """Serialise a raw model response for ``tasks``/``rationale``."""
    return json.dumps({"tasks": tasks, "rationale": rationale})


def _planner_returning(fake_client: Any, settings: Settings, *texts: str) -> Any:
    """A :class:`PlannerAgent` whose client answers ``complete_detailed`` with
    ``texts`` in order (the last one repeats if the planner asks again)."""
    client = fake_client(settings)
    remaining = list(texts)
    record_and_reply = client.complete_detailed

    async def complete_detailed(
        prompt: str, system: str = "", model: str | None = None, **kwargs: Any
    ) -> CompletionResult:
        if remaining:
            client.detailed = replace(client.detailed, text=remaining.pop(0))
        return await record_and_reply(prompt, system=system, model=model, **kwargs)

    client.complete_detailed = complete_detailed
    return PlannerAgent(client, settings=settings)


def _is_acyclic(plan: Plan) -> bool:
    """Kahn-style check: repeatedly drop tasks whose dependencies are all done."""
    remaining = {task.id: set(task.dependencies) for task in plan.tasks}
    done: set[str] = set()
    while remaining:
        ready = [node for node, deps in remaining.items() if deps <= done]
        if not ready:
            return False
        done |= set(ready)
        for node in ready:
            del remaining[node]
    return True


def _assert_schedulable(plan: Plan) -> None:
    """Assert the four invariants the scheduler relies on."""
    ids = [task.id for task in plan.tasks]
    assert ids == [str(index) for index in range(1, len(plan.tasks) + 1)]
    assert len(set(ids)) == len(ids)
    assert all(task.description.strip() for task in plan.tasks)
    assert all(dep in set(ids) for task in plan.tasks for dep in task.dependencies)
    assert _is_acyclic(plan)


# --------------------------------------------------------------------------
# 1. Recovery of a fenced, prose-wrapped response
# --------------------------------------------------------------------------


async def test_fenced_prose_wrapped_json_is_parsed(fake_client: Any, make_settings: Any) -> None:
    payload = _plan_text(
        [
            {
                "id": "1",
                "description": "Retry the anthropic call in furrow/llm.py",
                "files": ["furrow/llm.py"],
                "dependencies": [],
            },
            {
                "id": "2",
                "description": "Cover the retry path in tests/test_llm.py",
                "files": ["tests/test_llm.py"],
                "dependencies": ["1"],
            },
        ],
        rationale="Retry first, then pin it down with a test.",
    )
    text = (
        "Certainly -- here is the plan you asked for:\n"
        f"```json\n{payload}\n```\n"
        "Let me know if you would rather I split the first task."
    )
    settings = make_settings()
    planner = _planner_returning(fake_client, settings, text)

    plan = await planner.plan(GOAL)

    assert isinstance(plan, Plan)
    assert not plan.is_empty
    assert [task.id for task in plan.tasks] == ["1", "2"]
    assert [task.description for task in plan.tasks] == [
        "Retry the anthropic call in furrow/llm.py",
        "Cover the retry path in tests/test_llm.py",
    ]
    assert plan.tasks[0].files == ["furrow/llm.py"]
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.rationale == "Retry first, then pin it down with a test."
    _assert_schedulable(plan)


# --------------------------------------------------------------------------
# 2. Renumbering to "1".."n", with dependencies remapped
# --------------------------------------------------------------------------


async def test_ids_are_renumbered_and_dependencies_remapped(
    fake_client: Any, make_settings: Any
) -> None:
    # Deliberately awkward: non-numeric ids, and a task listed *before* the task
    # it depends on, so neither renumbering nor remapping can be a no-op.
    text = _plan_text(
        [
            {"id": "step-c", "description": "third", "dependencies": ["step-a", "step-b"]},
            {"id": "step-a", "description": "first", "dependencies": []},
            {"id": "step-b", "description": "second", "dependencies": ["step-a"]},
        ]
    )
    settings = make_settings()
    planner = _planner_returning(fake_client, settings, text)

    plan = await planner.plan(GOAL, max_tasks=3)

    assert [task.id for task in plan.tasks] == ["1", "2", "3"]
    # Reading order is preserved; only the ids change.
    assert [task.description for task in plan.tasks] == ["third", "first", "second"]
    assert [task.dependencies for task in plan.tasks] == [["2", "3"], [], ["2"]]
    _assert_schedulable(plan)


# --------------------------------------------------------------------------
# 3. Normalisation of hostile plans -- one case each
# --------------------------------------------------------------------------

NORMALISATION_CASES: list[tuple[str, list[dict[str, Any]], list[str], list[list[str]]]] = [
    (
        "duplicate-ids",
        [
            {"id": "1", "description": "first"},
            {"id": "1", "description": "second"},
            {"id": "2", "description": "third", "dependencies": ["1"]},
        ],
        ["first", "second", "third"],
        [[], [], ["1"]],
    ),
    (
        "dependency-cycle",
        [
            {"id": "1", "description": "first", "dependencies": ["2"]},
            {"id": "2", "description": "second", "dependencies": ["1"]},
        ],
        ["first", "second"],
        [["2"], []],
    ),
    (
        "self-dependency",
        [{"id": "1", "description": "only", "dependencies": ["1"]}],
        ["only"],
        [[]],
    ),
    (
        "blank-description",
        [
            {"id": "1", "description": "keep me"},
            {"id": "2", "description": "   "},
            {"id": "3", "description": "", "dependencies": ["2"]},
        ],
        ["keep me"],
        [[]],
    ),
    (
        "unknown-dependency",
        [
            {"id": "1", "description": "first", "dependencies": ["99"]},
            {"id": "2", "description": "second", "dependencies": ["1", "nope"]},
        ],
        ["first", "second"],
        [[], ["1"]],
    ),
]


@pytest.mark.parametrize(
    ("tasks", "expected_descriptions", "expected_dependencies"),
    [case[1:] for case in NORMALISATION_CASES],
    ids=[case[0] for case in NORMALISATION_CASES],
)
async def test_plan_is_normalised(
    tasks: list[dict[str, Any]],
    expected_descriptions: list[str],
    expected_dependencies: list[list[str]],
    fake_client: Any,
    make_settings: Any,
) -> None:
    settings = make_settings()
    planner = _planner_returning(fake_client, settings, _plan_text(tasks))

    plan = await planner.plan(GOAL, max_tasks=5)

    assert [task.description for task in plan.tasks] == expected_descriptions
    assert [task.dependencies for task in plan.tasks] == expected_dependencies
    _assert_schedulable(plan)


# --------------------------------------------------------------------------
# 4. max_tasks truncation
# --------------------------------------------------------------------------

#: Task "b" depends on "c", which is the task that truncation removes.
TRUNCATION_PAYLOAD = _plan_text(
    [
        {"id": "a", "description": "first"},
        {"id": "b", "description": "second", "dependencies": ["c"]},
        {"id": "c", "description": "third"},
    ]
)


async def test_max_tasks_truncation_drops_dependencies_on_removed_tasks(
    fake_client: Any, make_settings: Any
) -> None:
    settings = make_settings()
    planner = _planner_returning(fake_client, settings, TRUNCATION_PAYLOAD)

    plan = await planner.plan(GOAL, max_tasks=2)

    assert [task.id for task in plan.tasks] == ["1", "2"]
    # "c" was renumbered to "3" and truncated away, so the edge pointing at it
    # has nothing left to point at and must not survive.
    assert plan.tasks[1].dependencies == []
    _assert_schedulable(plan)


async def test_truncation_falls_back_to_settings_max_parallel_tasks(
    fake_client: Any, make_settings: Any
) -> None:
    settings = make_settings(max_parallel_tasks=2)
    planner = _planner_returning(fake_client, settings, TRUNCATION_PAYLOAD)

    plan = await planner.plan(GOAL)

    assert [task.id for task in plan.tasks] == ["1", "2"]
    assert plan.tasks[1].dependencies == []


# --------------------------------------------------------------------------
# 5. One corrective retry, then StructuredOutputError
# --------------------------------------------------------------------------


async def test_garbage_then_valid_response_succeeds_after_one_retry(
    fake_client: Any, make_settings: Any
) -> None:
    text = _plan_text([{"id": "1", "description": "only task"}])
    settings = make_settings()
    planner = _planner_returning(fake_client, settings, GARBAGE, text)

    plan = await planner.plan(GOAL)

    assert [task.description for task in plan.tasks] == ["only task"]
    # Exactly two requests: the rejected one and the corrective retry.
    assert len(planner.client.calls) == 2
    first, retry = planner.client.calls
    assert GOAL in first.prompt
    assert GOAL in retry.prompt
    assert retry.prompt.startswith(first.prompt)
    assert "rejected" in retry.prompt


async def test_two_garbage_responses_raise_structured_output_error(
    fake_client: Any, make_settings: Any
) -> None:
    settings = make_settings()
    planner = _planner_returning(fake_client, settings, GARBAGE, GARBAGE)

    with pytest.raises(StructuredOutputError) as excinfo:
        await planner.plan(GOAL)

    err = excinfo.value
    assert "did not return a valid plan after one retry" in err.message
    assert isinstance(err.__cause__, StructuredOutputError)
    assert "Could not extract JSON" in err.__cause__.message
    # The retry really was spent; there is no third attempt.
    assert len(planner.client.calls) == 2


# --------------------------------------------------------------------------
# 6. An empty task list is a valid plan
# --------------------------------------------------------------------------


async def test_empty_task_list_is_a_valid_plan_not_an_error(
    fake_client: Any, make_settings: Any
) -> None:
    rationale = "The code already satisfies the goal, so there is nothing to plan."
    settings = make_settings()
    planner = _planner_returning(fake_client, settings, _plan_text([], rationale=rationale))

    plan = await planner.plan(GOAL)

    assert isinstance(plan, Plan)
    assert plan.tasks == []
    assert plan.is_empty
    assert plan.rationale == rationale
    assert plan.task_map() == {}
    assert plan.duplicate_ids() == []
    # An empty plan is valid output, so no corrective retry is spent on it.
    assert len(planner.client.calls) == 1


async def test_blank_rationale_is_replaced_by_a_synthesised_one(
    fake_client: Any, make_settings: Any
) -> None:
    settings = make_settings()
    planner = _planner_returning(
        fake_client,
        settings,
        _plan_text([{"id": "1", "description": "only task"}], rationale="   "),
    )

    plan = await planner.plan(GOAL)

    assert plan.rationale.strip()
    assert GOAL in plan.rationale


# --------------------------------------------------------------------------
# 7. What the planner actually sends
# --------------------------------------------------------------------------


async def test_system_prompt_is_planner_system_and_goal_reaches_the_user_message(
    fake_client: Any, make_settings: Any
) -> None:
    settings = make_settings(planner_model="test-planner-model")
    planner = _planner_returning(
        fake_client, settings, _plan_text([{"id": "1", "description": "only task"}])
    )

    await planner.plan(GOAL)

    (call,) = planner.client.calls
    assert call.system == PLANNER_SYSTEM
    assert call.model == settings.model_for("planner") == "test-planner-model"
    assert call.prompt == PLANNER_PROMPT.format(goal=GOAL)
    assert GOAL in call.prompt
    # The persona lives in the system message, not the user message.
    assert PLANNER_SYSTEM not in call.prompt
