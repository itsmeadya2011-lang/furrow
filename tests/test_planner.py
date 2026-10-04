"""Contracts for :mod:`furrow.agents.planner`.

Three layers are pinned here, in increasing order of how much machinery they need:

1. the pure functions (:func:`~furrow.agents.planner.extract_json`,
   :func:`~furrow.agents.planner.parse_plan`) against a table of realistic-but-awkward
   model responses -- no client at all;
2. :meth:`PlannerAgent.plan` against a scripted ``FakeLLM`` -- the one-repair-retry
   contract, the argument shapes the orchestrator relies on, and the bounded project
   listing;
3. the constructor -- ``PlannerAgent()`` with no arguments must not raise.
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Any

import pytest

from furrow.agents.planner import (
    ERROR_SNIPPET_CHARS,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_FILES,
    MAX_TASKS,
    REPAIR_SNIPPET_CHARS,
    PlannerAgent,
    extract_json,
    parse_plan,
)
from furrow.config import Plan, TaskStatus
from tests.conftest import FakeLLM, Settings

VALID_PLAN_JSON = '{"tasks": [{"id": "1", "description": "ship it"}], "rationale": "small slice"}'


# --------------------------------------------------------------------------
# JSON extraction table -- no client required
# --------------------------------------------------------------------------

#: ``(label, raw response, payload extract_json is expected to surface)``.
#: Every case decodes to the same one-task plan, because the failure mode being
#: guarded against is a *silently empty* plan, not a subtly different one.
_EXTRACTION_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("plain_object", VALID_PLAN_JSON, json.loads(VALID_PLAN_JSON)),
    ("json_fence", f"```json\n{VALID_PLAN_JSON}\n```", json.loads(VALID_PLAN_JSON)),
    ("bare_fence", f"```\n{VALID_PLAN_JSON}\n```", json.loads(VALID_PLAN_JSON)),
    (
        "prose_then_fence",
        f"Sure! Here is the plan:\n```json\n{VALID_PLAN_JSON}\n```\nHope that helps.",
        json.loads(VALID_PLAN_JSON),
    ),
    ("trailing_prose", f"{VALID_PLAN_JSON}\nThat concludes my plan.", json.loads(VALID_PLAN_JSON)),
    (
        "brace_inside_a_string_value",
        '{"tasks": [{"id": "1", "description": "match the } inside a string"}], "rationale": "r"}',
        {
            "tasks": [{"id": "1", "description": "match the } inside a string"}],
            "rationale": "r",
        },
    ),
    (
        "missing_rationale",
        '{"tasks": [{"id": "1", "description": "ship it"}]}',
        {"tasks": [{"id": "1", "description": "ship it"}]},
    ),
]


@pytest.mark.parametrize(
    ("label", "response", "expected"), _EXTRACTION_CASES, ids=[c[0] for c in _EXTRACTION_CASES]
)
def test_extract_json_surfaces_the_whole_payload(
    label: str, response: str, expected: dict[str, Any]
) -> None:
    """A stray brace or a fence must never truncate the payload."""
    assert json.loads(extract_json(response)) == expected


@pytest.mark.parametrize(
    ("label", "response", "expected"), _EXTRACTION_CASES, ids=[c[0] for c in _EXTRACTION_CASES]
)
def test_parse_plan_reads_every_extraction_shape(
    label: str, response: str, expected: dict[str, Any]
) -> None:
    plan = parse_plan(response)
    assert isinstance(plan, Plan)
    assert [task.id for task in plan.tasks] == [task["id"] for task in expected["tasks"]]
    assert [task.description for task in plan.tasks] == [
        task["description"] for task in expected["tasks"]
    ]
    assert all(task.status is TaskStatus.PENDING for task in plan.tasks)


def test_missing_rationale_falls_back_to_an_explicit_placeholder() -> None:
    plan = parse_plan('{"tasks": [{"id": "1", "description": "ship it"}]}')
    assert plan.rationale == "(planner provided no rationale)"


def test_top_level_array_of_tasks_is_accepted() -> None:
    """A top-level ``[{...}]`` response must yield tasks, not a silent empty plan.

    Regression guard: `extract_json` used to reach for the first inner ``{...}``
    before considering ``[...]``, so this response produced ``tasks == []``.
    """
    plan = parse_plan('[{"id": "1", "description": "a"}, {"id": "2", "description": "b"}]')
    assert [task.id for task in plan.tasks] == ["1", "2"]
    assert [task.description for task in plan.tasks] == ["a", "b"]


# --------------------------------------------------------------------------
# Mapping-keyed-by-id task shapes
# --------------------------------------------------------------------------

#: Deliberately includes "10" so that lexicographic and natural ordering differ.
_MAPPING_JSON = json.dumps(
    {
        "tasks": {
            "1": {"description": "first"},
            "10": {"description": "tenth"},
            "2": {"description": "second"},
        },
        "rationale": "r",
    }
)


def test_mapping_shaped_tasks_are_expanded_into_a_list() -> None:
    """A ``{"1": {...}, "2": {...}}`` ``tasks`` value becomes a list of tasks.

    KNOWN DEFECT (asserted, not fixed -- see the strict xfail below): the mapping is
    ordered with ``sorted(..., key=str)``, i.e. *lexicographically*, so "10" lands
    between "1" and "2". The tasks are then renumbered ``1, 2, 3`` by position, so
    the model never sees the ids it assigned and dependency wiring by id is lost.
    """
    plan = parse_plan(_MAPPING_JSON)
    assert isinstance(plan.tasks, list)
    assert len(plan.tasks) == 3
    # "tenth" (key "10") is placed second, between "first" (key "1") and "second".
    assert [task.description for task in plan.tasks] == ["first", "tenth", "second"]
    # ids are re-derived from position, so they are 1, 2, 3 -- not 1, 10, 2.
    assert [task.id for task in plan.tasks] == ["1", "2", "3"]


@pytest.mark.xfail(
    reason="source defect: furrow/agents/planner.py:175 orders mapping keys with "
    "sorted(raw_tasks, key=str), which is lexicographic, so key '10' sorts between "
    "'1' and '2'; a natural sort (or preserving the model's order) is needed",
    strict=True,
)
def test_mapping_shaped_tasks_are_naturally_ordered() -> None:
    plan = parse_plan(_MAPPING_JSON)
    assert [task.description for task in plan.tasks] == ["first", "second", "tenth"]


# --------------------------------------------------------------------------
# Normalisation: unknown keys, coercion, clamping
# --------------------------------------------------------------------------


def test_unknown_keys_and_a_hallucinated_status_are_dropped() -> None:
    """A model that volunteers ``status: "completed"`` must not pre-complete work."""
    plan = parse_plan(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "1",
                        "description": "d",
                        "status": "completed",
                        "result": "already done",
                        "status_message": "trust me",
                        "mystery": 1,
                    }
                ],
                "rationale": "r",
                "confidence": 0.9,
            }
        )
    )
    task = plan.tasks[0]
    assert task.status is TaskStatus.PENDING
    assert task.result is None
    assert task.status_message is None
    assert not hasattr(task, "mystery")
    assert set(plan.model_dump()) == {"tasks", "rationale"}


def test_files_as_a_comma_separated_string_become_a_list() -> None:
    plan = parse_plan(
        '{"tasks": [{"id": "1", "description": "d", "files": "a.py, b.py ,, c.py"}],'
        ' "rationale": "r"}'
    )
    assert plan.tasks[0].files == ["a.py", "b.py", "c.py"]


def test_a_missing_description_becomes_a_visible_placeholder() -> None:
    plan = parse_plan('{"tasks": [{"id": "7"}], "rationale": "r"}')
    assert plan.tasks[0].id == "7"
    assert plan.tasks[0].description == "Task 7: infer intent from the goal"


def test_more_tasks_than_max_tasks_are_clamped_to_the_first_ten() -> None:
    payload = json.dumps(
        {"tasks": [{"id": str(i), "description": f"d{i}"} for i in range(1, 15)], "rationale": "r"}
    )
    plan = parse_plan(payload)
    assert len(plan.tasks) == MAX_TASKS
    assert [task.id for task in plan.tasks] == [str(i) for i in range(1, 11)]


def test_non_object_task_entries_are_skipped() -> None:
    """Regression guard: ``"N/A"`` used to become a task whose description was ``"N/A"``."""
    plan = parse_plan(
        '{"tasks": ["N/A", {"id": "1", "description": "real"}, 42], "rationale": "r"}'
    )
    assert [task.description for task in plan.tasks] == ["real"]


# --------------------------------------------------------------------------
# Unparsable responses
# --------------------------------------------------------------------------

_BAD_RESPONSES = [
    ("empty", ""),
    ("whitespace", "   "),
    ("broken_object", "{ broken json"),
    ("truncated_array", '[{"id": "1", "description": "a"'),
    ("prose_refusal", "I am sorry, I cannot help with that."),
    ("json_scalar", '"just a string"'),
]


@pytest.mark.parametrize(("label", "text"), _BAD_RESPONSES, ids=[c[0] for c in _BAD_RESPONSES])
def test_unparsable_responses_raise_value_error(label: str, text: str) -> None:
    """Never hang, never invent a plan: raise so the caller can request a repair."""
    started = time.monotonic()
    with pytest.raises(ValueError):
        parse_plan(text)
    assert time.monotonic() - started < 2.0


def test_a_json_decode_error_is_reachable_as_a_value_error() -> None:
    """``JSONDecodeError`` subclasses ``ValueError``, which is what ``plan()`` catches."""
    assert issubclass(json.JSONDecodeError, ValueError)
    with pytest.raises(json.JSONDecodeError):
        parse_plan("{ broken json")


def test_a_top_level_non_object_json_scalar_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected a JSON object at the top level"):
        parse_plan("42")


# --------------------------------------------------------------------------
# Repair retry
# --------------------------------------------------------------------------

#: Unbalanced on purpose, so ``extract_json`` cannot salvage an inner object and
#: ``json.loads`` reports the failure back to the model.
LONG_BAD_JSON = '{"tasks": [{"id": "1", "description": "' + "x" * 3_000 + "TAIL_SENTINEL"


async def test_a_malformed_reply_is_repaired_with_exactly_one_extra_call(
    tmp_workspace: Settings,
) -> None:
    llm = FakeLLM(tmp_workspace, responses=[LONG_BAD_JSON, VALID_PLAN_JSON])
    plan = await PlannerAgent(client=llm, settings=tmp_workspace).plan("ship the widget")

    assert [task.description for task in plan.tasks] == ["ship it"]
    assert llm.complete_call_count == 2

    original, repair = llm.complete_calls
    assert "Goal: ship the widget" in original["prompt"]
    assert "could not be parsed" in repair["prompt"]
    # The repair prompt quotes the bad body, truncated to REPAIR_SNIPPET_CHARS.
    assert "x" * 200 in repair["prompt"]
    assert "... [truncated]" in repair["prompt"]
    assert "TAIL_SENTINEL" not in repair["prompt"]


async def test_always_malformed_replies_raise_a_named_and_bounded_value_error(
    tmp_workspace: Settings,
) -> None:
    llm = FakeLLM(tmp_workspace, responses=["sorry, no JSON here", "still no JSON here"])
    with pytest.raises(ValueError) as excinfo:
        await PlannerAgent(client=llm, settings=tmp_workspace).plan("ship the widget")

    assert llm.complete_call_count == 2
    message = str(excinfo.value)
    assert "Planner failed to return valid plan JSON" in message
    assert "after one repair attempt" in message
    # Both the first and the second failure are reported.
    assert message.count("Expecting value") == 2

    quoted = message.split("chars): ", 1)[1]
    assert quoted.startswith("'") and quoted.endswith("'")
    assert len(quoted[1:-1]) <= ERROR_SNIPPET_CHARS + len("... [truncated]")

    cause = excinfo.value.__cause__
    assert isinstance(cause, ValueError)
    assert "Expecting value" in str(cause)


async def test_the_reported_bad_response_snippet_is_truncated(
    tmp_workspace: Settings,
) -> None:
    long_bad = "{" + "y" * 5_000
    llm = FakeLLM(tmp_workspace, responses=[long_bad, long_bad])
    with pytest.raises(ValueError) as excinfo:
        await PlannerAgent(client=llm, settings=tmp_workspace).plan("ship the widget")

    inner = str(excinfo.value).split("chars): ", 1)[1][1:-1]
    assert inner.endswith("... [truncated]")
    assert len(inner) == ERROR_SNIPPET_CHARS + len("... [truncated]")
    assert "y" * (ERROR_SNIPPET_CHARS + 1) not in str(excinfo.value)
    assert len(str(excinfo.value)) < 1_000


async def test_the_repair_prompt_keeps_a_short_bad_body_intact(
    tmp_workspace: Settings,
) -> None:
    bad = "{ not json at all"
    llm = FakeLLM(tmp_workspace, responses=[bad, VALID_PLAN_JSON])
    await PlannerAgent(client=llm, settings=tmp_workspace).plan("goal")
    repair_prompt = llm.complete_calls[1]["prompt"]
    assert bad in repair_prompt
    assert "... [truncated]" not in repair_prompt
    assert len(bad) < REPAIR_SNIPPET_CHARS


# --------------------------------------------------------------------------
# Argument shapes the orchestrator depends on
# --------------------------------------------------------------------------

_POSITIONAL = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


@pytest.mark.parametrize(
    "args",
    [("g",), ("g", "a string context"), ("g", ["a", "b"]), ("g", None)],
    ids=["goal_only", "goal_and_str", "goal_and_list", "goal_and_none"],
)
async def test_plan_accepts_every_context_shape(
    tmp_workspace: Settings, args: tuple[Any, ...]
) -> None:
    llm = FakeLLM(tmp_workspace, responses=[VALID_PLAN_JSON])
    plan = await PlannerAgent(client=llm, settings=tmp_workspace).plan(*args)
    assert [task.description for task in plan.tasks] == ["ship it"]
    assert "Goal: g" in llm.complete_calls[0]["prompt"]


async def test_a_list_failure_context_is_flattened_into_the_prompt(
    tmp_workspace: Settings,
) -> None:
    llm = FakeLLM(tmp_workspace, responses=[VALID_PLAN_JSON])
    await PlannerAgent(client=llm, settings=tmp_workspace).plan("g", ["first failure", "second"])
    prompt = llm.complete_calls[0]["prompt"]
    assert "Previous cycle context" in prompt
    assert "- first failure" in prompt
    assert "- second" in prompt


async def test_extra_keyword_context_is_labelled_in_the_prompt(
    tmp_workspace: Settings,
) -> None:
    llm = FakeLLM(tmp_workspace, responses=[VALID_PLAN_JSON])
    await PlannerAgent(client=llm, settings=tmp_workspace).plan(
        "g", None, prior_outcomes="wrote the module"
    )
    assert "prior outcomes:\nwrote the module" in llm.complete_calls[0]["prompt"]


def test_plan_keeps_its_three_positional_parameters() -> None:
    """Pinned because ``Orchestrator._planner_accepts_context`` probes this signature."""
    params = inspect.signature(PlannerAgent.plan).parameters
    positional = [name for name, param in params.items() if param.kind in _POSITIONAL]
    assert positional == ["self", "goal", "failure_context"]
    assert params["failure_context"].default is None
    assert params["extra_context"].kind is inspect.Parameter.VAR_KEYWORD


def test_the_bound_plan_signature_still_satisfies_the_orchestrator_probe() -> None:
    """``_planner_accepts_context()`` counts ``>= 2`` positional parameters on the
    *bound* method; dropping ``failure_context`` would silently stop the orchestrator
    forwarding test failures and start appending them to the goal instead."""
    bound = inspect.signature(PlannerAgent(client=object()).plan).parameters
    assert sum(1 for param in bound.values() if param.kind in _POSITIONAL) == 2


def test_planner_agent_with_no_arguments_does_not_raise() -> None:
    """Regression guard: this used to raise
    ``AttributeError: 'NoneType' object has no attribute 'planner_model'``."""
    agent = PlannerAgent()
    assert agent.client.settings is not None
    assert agent.client.settings.planner_model


# --------------------------------------------------------------------------
# Bounded project context
# --------------------------------------------------------------------------


class _ListingClient(FakeLLM):
    """A :class:`FakeLLM` whose ``list_files`` returns (or raises) a scripted listing."""

    def __init__(
        self,
        settings: Settings,
        *,
        listing: list[Any] | None = None,
        listing_error: BaseException | None = None,
    ) -> None:
        super().__init__(settings)
        self.listing = listing if listing is not None else []
        self.listing_error = listing_error
        self.listed_roots: list[str] = []

    def list_files(self, directory: str | Path = ".") -> list[Any]:
        self.listed_roots.append(str(directory))
        if self.listing_error is not None:
            raise self.listing_error
        return self.listing


def _listing_section(prompt: str) -> str:
    """Everything the planner appended after ``Existing files:``, trailing newline gone.

    The prompt itself ends in ``"\\n"``, so the raw tail is one char longer than the
    listing body the planner actually built and bounded.
    """
    marker = "Existing files:\n"
    assert marker in prompt, f"no project listing in prompt:\n{prompt}"
    return prompt.split(marker, 1)[1].strip()


def _bullets(section: str) -> list[str]:
    return [line for line in section.splitlines() if line.startswith("- ")]


async def test_project_context_coerces_path_objects_with_str(tmp_workspace: Settings) -> None:
    client = _ListingClient(tmp_workspace, listing=[Path("src") / "module.py", "README.md"])
    client.queue(VALID_PLAN_JSON)
    await PlannerAgent(client=client, settings=tmp_workspace).plan("goal")

    section = _listing_section(client.complete_calls[0]["prompt"])
    assert _bullets(section) == ["- src/module.py", "- README.md"]
    assert client.listed_roots == [str(tmp_workspace.workspace)]


async def test_project_listing_is_capped_at_200_files(tmp_workspace: Settings) -> None:
    client = _ListingClient(tmp_workspace, listing=[f"pkg/mod_{i}.py" for i in range(500)])
    client.queue(VALID_PLAN_JSON)
    await PlannerAgent(client=client, settings=tmp_workspace).plan("goal")

    section = _listing_section(client.complete_calls[0]["prompt"])
    assert len(_bullets(section)) == MAX_CONTEXT_FILES
    assert _bullets(section) == [f"- pkg/mod_{i}.py" for i in range(MAX_CONTEXT_FILES)]
    assert f"{500 - MAX_CONTEXT_FILES} more files omitted" in section


async def test_project_listing_is_capped_at_8000_chars(tmp_workspace: Settings) -> None:
    client = _ListingClient(tmp_workspace, listing=[f"{i:03d}-" + "d" * 200 for i in range(200)])
    client.queue(VALID_PLAN_JSON)
    await PlannerAgent(client=client, settings=tmp_workspace).plan("goal")

    section = _listing_section(client.complete_calls[0]["prompt"])
    assert "[listing truncated]" in section
    assert len(section) <= MAX_CONTEXT_CHARS + len("\n... [listing truncated]")


@pytest.mark.parametrize(
    ("label", "listing", "listing_error"),
    [
        ("list_files_raises_oserror", [], OSError("permission denied")),
        ("list_files_raises_runtime_error", [], RuntimeError("boom")),
        ("empty_listing", [], None),
    ],
)
async def test_an_unusable_project_listing_degrades_to_no_context(
    tmp_workspace: Settings,
    label: str,
    listing: list[Any],
    listing_error: BaseException | None,
) -> None:
    """A broken listing must never cost us the plan itself."""
    client = _ListingClient(tmp_workspace, listing=listing, listing_error=listing_error)
    client.queue(VALID_PLAN_JSON)
    plan = await PlannerAgent(client=client, settings=tmp_workspace).plan("ship it")

    assert [task.description for task in plan.tasks] == ["ship it"]
    prompt = client.complete_calls[0]["prompt"]
    assert "Existing files:" not in prompt
    assert "Goal: ship it" in prompt
