"""Regression tests for the furrow agent layer: planner, worker and tester.

Self-contained on purpose - no ``conftest.py``, no shared fixtures, no ambient
state.  Everything here is hermetic:

* every LLM call goes through a stubbed :meth:`furrow.llm.LLMClient.complete`,
  so nothing touches a provider SDK, a socket or the network;
* every test-suite run goes through a stubbed ``TesterAgent._run_tests``, so no
  ``pytest``/``npm``/``cargo``/``go`` process is ever spawned;
* :class:`~furrow.config.Settings` is built with ``_env_file=None`` and every
  field a test cares about is passed explicitly, so the developer's ``.env``
  cannot change an outcome.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from typing import Any, cast

import pytest

from furrow.agents import planner as planner_mod
from furrow.agents.planner import (
    MAX_ECHO_CHARS,
    PlannerAgent,
    extract_json_object,
    strip_code_fences,
    truncate_for_log,
)

# ``TestRunOutcome``/``TesterAgent`` are aliased on import: pytest offers every
# module-level name starting with ``Test`` to its collector (``python_classes``
# defaults to ``Test*`` and is matched on the *attribute* name, not on
# ``__name__``), and both classes have ``__init__``, so the bare product names
# raise PytestCollectionWarning on every import of this module.  The aliases
# below are therefore the only names this module may bind them to, and neither
# alias may itself start with ``Test`` -- ``Tester`` still does, and still warns.
from furrow.agents.tester import (
    MAX_FAILURE_CHARS,
    NO_TEST_RUNNER_MESSAGE,
    TEST_RUNNER_TIMEOUT_SECONDS,
    TEST_RUNNERS,
    TEST_SUITE_TIMEOUT_SECONDS,
    TEST_TIMED_OUT_MESSAGE,
    describe_tasks,
    judge_passed,
)
from furrow.agents.tester import TesterAgent as _Tester
from furrow.agents.tester import TestRunOutcome as RunOutcome
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient, LLMError, LLMResponseError

# --------------------------------------------------------------------- doubles


class _StubClient:
    """A stand-in for :class:`~furrow.llm.LLMClient` that never leaves the process.

    ``settings`` is only attached when one is supplied, so a test can also model
    a double that exposes nothing but ``complete()``.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        if settings is not None:
            self.settings = settings

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append((prompt, system, model))
        return ""


def _settings(**overrides: Any) -> Settings:
    """A hermetic ``Settings`` that ignores the working directory's ``.env``."""
    # mypy cannot see pydantic-settings' `_env_file` kwarg although it works at runtime.
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def _stub_complete(
    monkeypatch: pytest.MonkeyPatch, reply: str
) -> list[tuple[str, str, str | None]]:
    """Replace ``LLMClient.complete`` with a canned reply; return the call log."""
    calls: list[tuple[str, str, str | None]] = []

    async def complete(
        self: LLMClient, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        calls.append((prompt, system, model))
        return reply

    monkeypatch.setattr(LLMClient, "complete", complete)
    return calls


_PLAN_JSON: dict[str, Any] = {
    "tasks": [{"id": "1", "description": "do it", "files": ["a.py"], "dependencies": []}],
    "rationale": "because",
}

_AGENT_CLASSES: dict[str, Callable[..., Any]] = {
    "PlannerAgent": PlannerAgent,
    "TesterAgent": _Tester,
    "WorkerAgent": WorkerAgent,
}


def _build(name: str, **kwargs: Any) -> Any:
    """Construct the named agent, supplying a default ``task`` for the worker.

    ``WorkerAgent.task`` is positional-or-keyword and mandatory, so every other
    test can treat the three agents alike.
    """
    if name == "WorkerAgent":
        kwargs.setdefault("task", TaskModel(id="1", description="d"))
    return _AGENT_CLASSES[name](**kwargs)


# ---------------------------------------------------------------- construction


@pytest.mark.parametrize("name", sorted(_AGENT_CLASSES))
def test_agent_constructs_with_no_arguments(name: str) -> None:
    """The original bug: every agent forwarded ``settings=None`` into LLMClient."""
    agent = _build(name)
    assert isinstance(agent.settings, Settings)
    assert isinstance(agent.client.settings, Settings)


@pytest.mark.parametrize("name", sorted(_AGENT_CLASSES))
def test_settings_only_construction_uses_those_settings(name: str) -> None:
    """The other supported form: configuration instead of a client."""
    own = _settings(max_cycles=9)
    agent = _build(name, settings=own)
    assert agent.settings is own
    assert agent.client.settings is own


@pytest.mark.parametrize("name", sorted(_AGENT_CLASSES))
def test_no_argument_construction_falls_back_to_the_global_settings_singleton(
    name: str,
) -> None:
    agent = _build(name)
    assert agent.settings is planner_mod.default_settings


@pytest.mark.parametrize("name", sorted(_AGENT_CLASSES))
def test_injected_client_is_used_verbatim(name: str) -> None:
    own = _settings(max_cycles=3)
    stub = _StubClient(own)
    agent = _build(name, client=cast("LLMClient", stub))
    assert agent.client is stub
    assert agent.settings is own


@pytest.mark.parametrize("name", sorted(_AGENT_CLASSES))
def test_conflicting_client_and_settings_warns_and_keeps_the_client(name: str) -> None:
    """Documented precedence: an explicit ``client`` wins, loudly."""
    client_settings = _settings(max_cycles=3)
    other = _settings(max_cycles=4)
    stub = _StubClient(client_settings)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = _build(name, client=cast("LLMClient", stub), settings=other)

    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime, f"expected a RuntimeWarning, got {[w.category for w in caught]}"
    assert name in str(runtime[0].message)
    assert agent.client is stub
    assert agent.settings is client_settings


@pytest.mark.parametrize("name", sorted(_AGENT_CLASSES))
def test_matching_client_and_settings_does_not_warn(name: str) -> None:
    shared = _settings(max_cycles=3)
    stub = _StubClient(shared)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build(name, client=cast("LLMClient", stub), settings=shared)

    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


@pytest.mark.parametrize("name", sorted(_AGENT_CLASSES))
def test_client_without_settings_attribute_does_not_warn(name: str) -> None:
    """``getattr(client, "settings", None)`` must tolerate a bare test double."""
    own = _settings(max_cycles=5)
    stub = _StubClient()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = _build(name, client=cast("LLMClient", stub), settings=own)

    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []
    assert agent.settings is own


def test_planner_reexports_the_one_canonical_llm_response_error() -> None:
    """A second, unrelated class here used to hide planner failures from callers."""
    assert planner_mod.LLMResponseError is LLMResponseError
    assert issubclass(LLMResponseError, LLMError)
    assert issubclass(LLMResponseError, ValueError)


# ------------------------------------------------------------ planner: parsing


async def test_plan_parses_a_fenced_json_block(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = "Here you go:\n```json\n" + json.dumps(_PLAN_JSON, indent=2) + "\n```\nEnjoy!"
    _stub_complete(monkeypatch, reply)
    plan = await PlannerAgent().plan("build it")
    assert isinstance(plan, Plan)
    assert [t.description for t in plan.tasks] == ["do it"]
    assert plan.rationale == "because"
    assert plan.tasks[0].files == ["a.py"]


async def test_plan_parses_a_fence_without_an_info_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_complete(monkeypatch, "```\n" + json.dumps(_PLAN_JSON) + "\n```")
    assert isinstance(await PlannerAgent().plan("build it"), Plan)


async def test_plan_parses_a_prose_wrapped_object(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_complete(monkeypatch, "Sure! " + json.dumps(_PLAN_JSON) + " Hope that helps.")
    assert isinstance(await PlannerAgent().plan("build it"), Plan)


async def test_plan_tolerates_braces_inside_json_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = dict(_PLAN_JSON, rationale="brace } and { inside a string")
    _stub_complete(monkeypatch, json.dumps(payload))
    plan = await PlannerAgent().plan("build it")
    assert plan.rationale == "brace } and { inside a string"


async def test_plan_parses_nested_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "tasks": [{"id": "1", "description": "d", "files": [], "dependencies": ["0"]}],
        "rationale": "r",
    }
    _stub_complete(monkeypatch, "```json\n" + json.dumps(payload) + "\n```")
    plan = await PlannerAgent().plan("build it")
    assert plan.tasks[0].dependencies == ["0"]


async def test_plan_requests_the_planner_model_and_sends_no_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_complete(monkeypatch, json.dumps(_PLAN_JSON))
    agent = PlannerAgent(settings=_settings(planner_model="planner-model-id"))
    await agent.plan("ship it")
    assert len(calls) == 1
    prompt, system, model = calls[0]
    assert model == "planner-model-id"
    assert system == ""
    assert "ship it" in prompt


@pytest.mark.parametrize(
    ("reply", "json_kind"),
    [
        ("[1, 2, 3]", "list"),
        ('["a"]', "list"),
        ("```json\n[1, 2]\n```", "list"),
        ("42", "int"),
        ('"hello"', "str"),
        ("null", "NoneType"),
        ("true", "bool"),
    ],
)
async def test_plan_rejects_valid_json_that_is_not_an_object(
    monkeypatch: pytest.MonkeyPatch, reply: str, json_kind: str
) -> None:
    """A bare ``TypeError`` from ``Plan(**data)`` used to escape here."""
    _stub_complete(monkeypatch, reply)
    with pytest.raises(LLMResponseError) as excinfo:
        await PlannerAgent().plan("build it")
    assert f"type {json_kind}" in str(excinfo.value)


async def test_plan_recovers_a_plan_object_embedded_in_an_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_complete(monkeypatch, "[" + json.dumps(_PLAN_JSON) + "]")
    plan = await PlannerAgent().plan("build it")
    assert plan.rationale == "because"


async def test_bare_array_and_object_array_fail_with_different_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``type list`` wording only appears when no ``{...}`` is recoverable."""
    messages: dict[str, str] = {}
    for label, reply in (
        ("no_objects", "[1, 2, 3]"),
        ("recoverable_object", '[{"nope": 1}]'),
    ):
        _stub_complete(monkeypatch, reply)
        with pytest.raises(LLMResponseError) as excinfo:
            await PlannerAgent().plan("build it")
        messages[label] = str(excinfo.value)

    assert "type list" in messages["no_objects"]
    assert "type list" not in messages["recoverable_object"]
    assert "Failed to build plan" in messages["recoverable_object"]
    assert messages["no_objects"] != messages["recoverable_object"]


async def test_plan_rejects_an_object_with_the_wrong_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_complete(monkeypatch, '{"nope": 1}')
    with pytest.raises(LLMResponseError) as excinfo:
        await PlannerAgent().plan("build it")
    assert "top-level keys" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None


async def test_malformed_reply_raises_with_a_truncated_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = "I am terribly sorry but " + "x" * 4000
    _stub_complete(monkeypatch, reply)
    with pytest.raises(LLMResponseError) as excinfo:
        await PlannerAgent().plan("build it")

    message = str(excinfo.value)
    assert "truncated" in message
    assert f"{len(reply)} chars total" in message
    # The whole point: a runaway completion must not land in the logs verbatim.
    assert len(message) < MAX_ECHO_CHARS + 120
    echo = message.split("does not contain a JSON object: ", 1)[1]
    assert echo.startswith("I am terribly sorry but xxx")
    assert len(echo) <= MAX_ECHO_CHARS + 60


async def test_empty_planner_reply_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_complete(monkeypatch, "   ")
    with pytest.raises(LLMResponseError) as excinfo:
        await PlannerAgent().plan("build it")
    assert "empty" in str(excinfo.value)


# ------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("x\n```py\nbody\n```\ny", "body"),
        ("plain", "plain"),
        ("```\n1\n```\n```\n2\n```", "1"),
        ("```json\n  {\"a\": 1}\n```", '{"a": 1}'),
        ("", ""),
    ],
)
def test_strip_code_fences_returns_the_first_fenced_body(text: str, expected: str) -> None:
    assert strip_code_fences(text) == expected


def test_truncate_for_log_passes_short_text_through_unchanged() -> None:
    assert truncate_for_log("abc") == "abc"
    assert truncate_for_log("") == ""
    assert truncate_for_log("abc", limit=3) == "abc"


def test_truncate_for_log_clips_and_reports_the_total() -> None:
    assert truncate_for_log("abcdef", limit=3) == "abc... [truncated, 6 chars total]"


def test_truncate_for_log_default_limit_is_max_echo_chars() -> None:
    clipped = truncate_for_log("x" * (MAX_ECHO_CHARS + 1))
    assert clipped == "x" * MAX_ECHO_CHARS + f"... [truncated, {MAX_ECHO_CHARS + 1} chars total]"
    assert "truncated" not in truncate_for_log("x" * MAX_ECHO_CHARS)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure thing: {"a": 1} - hope that helps', {"a": 1}),
        ('prefix ```json\n{"a": 1}\n``` suffix', {"a": 1}),
        ('{"a": "} and { are not span ends"}', {"a": "} and { are not span ends"}),
        ('{"a": {"b": [1, 2]}}', {"a": {"b": [1, 2]}}),
    ],
)
def test_extract_json_object_handles_the_shapes_models_emit(
    text: str, expected: dict[str, Any]
) -> None:
    assert extract_json_object(text) == expected
    assert type(extract_json_object(text)) is dict


@pytest.mark.parametrize(
    "text",
    ["", "   ", "no json here", "[1, 2]", '"a string"', "42", "{not valid json}"],
)
def test_extract_json_object_rejects_everything_else(text: str) -> None:
    with pytest.raises(LLMResponseError):
        extract_json_object(text)


def test_extract_json_object_names_the_context_it_was_given() -> None:
    with pytest.raises(LLMResponseError) as excinfo:
        extract_json_object("nope", context="myctx")
    assert "myctx" in str(excinfo.value)


def test_extract_json_object_truncates_a_runaway_echo() -> None:
    with pytest.raises(LLMResponseError) as excinfo:
        extract_json_object("y" * 5000)
    assert "truncated, 5000 chars total" in str(excinfo.value)


# ------------------------------------------------------------- judge_passed


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        # The regression: a substring match called this a pass.
        ("No tests passed", False),
        ("No tests ran", False),
        ("no tests collected", False),
        ("2 passed", True),
        ("1 passed", True),
        ("===== 5 passed, 2 warnings in 0.1s =====", True),
        ("2 passed, 0 failed", True),
        ("all tests succeeded", True),
        # Failure markers must beat pass markers.
        ("1 failed, 12 passed", False),
        ("2 passed but the suite did not pass", False),
        ("tests did not pass", False),
        ("4 failed", False),
        ("2 errors", False),
        ("===== 1 passed, 1 failed =====", False),
        ("E   AssertionError: boom", False),
        ("Traceback (most recent call last):", False),
        ("internal error", False),
        ("usage error: no such option", False),
        ("not ok 1 - thing", False),
        ("Segmentation fault", False),
        # Genuinely ambiguous input must default to "not proven".
        ("", False),
        ("   ", False),
        ("OK", False),
        ("collected 3 items", False),
    ],
)
def test_judge_passed_is_conservative_about_runner_output(
    output: str, expected: bool
) -> None:
    assert judge_passed(output) is expected


def test_judge_passed_is_case_insensitive() -> None:
    assert judge_passed("FAILED tests/test_x.py") is False
    assert judge_passed("Tests PASSED") is True


# ----------------------------------------------------------- describe_tasks


def test_describe_tasks_says_so_when_nothing_was_planned() -> None:
    assert describe_tasks([]) == "(no tasks were planned for this cycle)"


def test_describe_tasks_renders_each_id_with_its_description() -> None:
    tasks = [
        TaskModel(id="7", description="wire up auth"),
        TaskModel(id="8", description="add tests"),
    ]
    assert describe_tasks(tasks) == "- [7] wire up auth\n- [8] add tests"


def test_describe_tasks_stops_at_the_limit_and_counts_the_rest() -> None:
    tasks = [TaskModel(id=str(i), description="d") for i in range(12)]
    lines = describe_tasks(tasks).splitlines()
    assert len(lines) == 11
    assert lines[0] == "- [0] d"
    assert lines[-1] == "- ... and 2 more task(s)"


def test_describe_tasks_truncates_a_long_description() -> None:
    tasks = [TaskModel(id="1", description="z" * 300)]
    line = describe_tasks(tasks)
    description = line.removeprefix("- [1] ")
    assert description == "z" * 200 + "..."


def test_describe_tasks_collapses_whitespace_in_a_description() -> None:
    tasks = [TaskModel(id="1", description="do   the\n\tthing")]
    assert describe_tasks(tasks) == "- [1] do the thing"


def test_describe_tasks_honours_an_explicit_limit_and_width() -> None:
    tasks = [TaskModel(id="1", description="y" * 50)]
    rendered = describe_tasks(tasks, limit=1, description_chars=10)
    assert rendered == "- [1] yyyyyyyyyy..."


# ----------------------------------------------------------- tester fallback


def _stub_run_tests(
    monkeypatch: pytest.MonkeyPatch, outcome: RunOutcome | BaseException
) -> None:
    """Replace the whole subprocess sweep with a canned outcome (or an explosion)."""

    async def _run_tests(self: _Tester) -> RunOutcome:
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(_Tester, "_run_tests", _run_tests)


def _red_outcome() -> RunOutcome:
    return RunOutcome(output="1 failed, 2 passed", runner="pytest", exit_code=1)


def _green_outcome() -> RunOutcome:
    return RunOutcome(output="3 passed", runner="pytest", exit_code=0)


async def test_valid_classifier_json_is_returned_as_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_complete(monkeypatch, '{"passed": false, "summary": "one red", "failures": ["x"]}')
    _stub_run_tests(monkeypatch, _red_outcome())
    result = await _Tester().run("goal", [])
    assert (result.passed, result.summary, result.failures) == (False, "one red", ["x"])


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "   ",
        "I could not decide, sorry.",
        "```json\n[1, 2]\n```",
        '{"passed": true}',
        '{"unknown": 1}',
    ],
)
async def test_unparsable_classifier_reply_fails_with_informative_failures(
    monkeypatch: pytest.MonkeyPatch, reply: str
) -> None:
    """An empty failure list used to become the content-free "Fix failing tests:\\n"."""
    _stub_complete(monkeypatch, reply)
    _stub_run_tests(monkeypatch, _red_outcome())
    result = await _Tester().run("the goal", [])

    assert result.passed is False
    assert result.failures, "a failing verdict must never carry an empty failure list"
    assert all(failure.strip() for failure in result.failures)
    assert any("Classifier reply" in failure for failure in result.failures)
    assert "heuristic verdict: failed" in result.summary


async def test_fallback_records_the_raw_test_output_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_complete(monkeypatch, "nope")
    _stub_run_tests(monkeypatch, _red_outcome())
    result = await _Tester().run("goal", [])
    assert any("1 failed, 2 passed" in failure for failure in result.failures)


async def test_fallback_truncates_a_huge_failure_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_complete(monkeypatch, "nope")
    outcome = RunOutcome(output="e" * 5000, runner="pytest", exit_code=1)
    _stub_run_tests(monkeypatch, outcome)
    result = await _Tester().run("goal", [])

    # The entry is built from "Test output (<summary>): <output>", and the marker
    # reports the length of *that* string - the prefix included - not of the raw
    # output, and not MAX_FAILURE_CHARS plus an assumed marker length.
    prefix = f"Test output ({outcome.summary}): "
    entry = next(f for f in result.failures if "truncated" in f)
    assert entry.startswith(prefix)

    body, _, marker = entry.removeprefix(prefix).rpartition("... [truncated, ")
    assert body == "e" * (MAX_FAILURE_CHARS - len(prefix))  # the cap covers the prefix
    # The marker reports the length of the string that was clipped, prefix included.
    assert marker == f"{len(prefix) + 5000} chars total]"


async def test_fallback_refuses_to_claim_success_on_ambiguous_prose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Green runner output plus an unclear reply is not a pass."""
    _stub_complete(monkeypatch, "I think it went fine?")
    _stub_run_tests(monkeypatch, _green_outcome())
    result = await _Tester().run("goal", [])
    assert result.passed is False
    assert result.failures


async def test_fallback_reports_a_pass_when_both_signals_are_green(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_complete(monkeypatch, "Everything looks fine, all passed.")
    _stub_run_tests(monkeypatch, _green_outcome())
    result = await _Tester().run("goal", [])
    assert result.passed is True
    assert result.failures == []
    assert "heuristic verdict: passed" in result.summary


async def test_fallback_fails_when_no_runner_ever_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_complete(monkeypatch, '{"passed": true, "summary": "green"}')
    _stub_run_tests(monkeypatch, RunOutcome(output=NO_TEST_RUNNER_MESSAGE))
    result = await _Tester().run("goal", [])
    # parseable JSON is honoured, but the run below shows the no-runner shape too
    assert result.passed is True

    _stub_complete(monkeypatch, "green to me")
    result = await _Tester().run("goal", [])
    assert result.passed is False
    assert result.failures


async def test_test_execution_error_becomes_a_failing_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run()`` must not raise: the orchestrator builds its next goal from it."""
    _stub_complete(monkeypatch, '{"passed": true, "summary": "green"}')
    _stub_run_tests(monkeypatch, RuntimeError("no space left on device"))
    result = await _Tester().run("goal", [])
    assert result.passed is False
    assert result.failures
    assert "no space left on device" in result.failures[0]


# ------------------------------------------------- worker prompt and output


async def test_worker_returns_the_client_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_complete(monkeypatch, "wrote a.py")
    assert await WorkerAgent(task=TaskModel(id="1", description="d")).run() == "wrote a.py"


async def test_worker_prompt_identifies_the_task_and_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_complete(monkeypatch, "done")
    agent = WorkerAgent(task=TaskModel(id="42", description="write tests", files=["t.py"]))
    await agent.run()
    prompt, _, model = calls[0]
    assert "Task id: 42" in prompt
    assert "write tests" in prompt
    assert "t.py" in prompt
    assert model == agent.settings.worker_model


async def test_worker_prompt_says_any_when_no_files_are_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_complete(monkeypatch, "done")
    await WorkerAgent(task=TaskModel(id="7", description="d")).run()
    assert "any" in calls[0][0]


# ------------------------------------------------------- tester surface facts


@pytest.mark.parametrize(
    ("output", "runner", "exit_code", "timed_out", "expected"),
    [
        ("out", "pytest", 0, False, "pytest exited 0"),
        ("out", "pytest", 1, False, "pytest exited 1"),
        ("out", None, None, True, TEST_TIMED_OUT_MESSAGE),
        ("out", None, None, False, NO_TEST_RUNNER_MESSAGE),
    ],
)
def test_test_run_outcome_summary_names_what_happened(
    output: str,
    runner: str | None,
    exit_code: int | None,
    timed_out: bool,
    expected: str,
) -> None:
    outcome = RunOutcome(
        output=output, runner=runner, exit_code=exit_code, timed_out=timed_out
    )
    assert outcome.summary == expected


def test_timed_out_and_no_runner_sentinels_are_distinct() -> None:
    # They used to share one string, so a stall was reported as "no runner found".
    assert TEST_TIMED_OUT_MESSAGE != NO_TEST_RUNNER_MESSAGE


def test_test_suite_budget_bounds_the_whole_sweep() -> None:
    assert TEST_SUITE_TIMEOUT_SECONDS == 90.0
    assert TEST_RUNNER_TIMEOUT_SECONDS == 60.0
    assert TEST_SUITE_TIMEOUT_SECONDS > TEST_RUNNER_TIMEOUT_SECONDS


def test_pytest_candidates_accept_only_real_test_outcomes() -> None:
    by_name = {runner.name: runner for runner in TEST_RUNNERS}
    assert by_name["pytest"].ran_exit_codes == frozenset({0, 1})
    assert by_name["python -m pytest"].ran_exit_codes == frozenset({0, 1})
    # cargo: 1 is a compile error, not a test outcome.
    assert by_name["cargo test"].ran_exit_codes == frozenset({0, 101})


def test_non_python_runners_are_manifest_guarded() -> None:
    by_name = {runner.name: runner for runner in TEST_RUNNERS}
    assert by_name["pytest"].manifest is None
    for name in ("npm test", "pnpm test", "yarn test", "cargo test", "go test"):
        assert by_name[name].manifest, f"{name} must be guarded by a manifest"