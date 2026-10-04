"""The three agents, driven through a fake client.

``PlannerAgent``, ``WorkerAgent`` and ``TesterAgent`` are real objects here; only their
LLM collaborator is faked. The behavioural guarantees under test are the ones the loop
depends on: a fenced plan parses, malformed output is retried rather than fatal, a
silent worker counts as a failure, and the non-JSON tester fallback must not mistake
"0 passed, 12 failed" for a pass.
"""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from furrow.agents.planner import PlannerAgent

# ``TesterAgent`` is imported as ``Grader``: a module-level name starting with ``Test``
# makes pytest try to collect the class as a test class and warn about its ``__init__``.
from furrow.agents.tester import MAX_OUTPUT_CHARS, detect_candidates, truncate_output
from furrow.agents.tester import TesterAgent as Grader
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel

from .conftest import FAILING_SUITE, PASSING_SUITE, FakeLLM, SettingsFactory, run_coroutine

MALFORMED = "I'm afraid I can't produce JSON today."


def plan_payload(*task_ids: str) -> str:
    """A valid plan document for *task_ids*, as JSON text."""
    return json.dumps(
        {
            "tasks": [
                {"id": task_id, "description": f"work on {task_id}", "dependencies": []}
                for task_id in task_ids
            ],
            "rationale": "one step at a time",
        }
    )


def fenced(document: str) -> str:
    return f"```json\n{document}\n```"


@pytest.fixture
def client_for(
    make_settings: SettingsFactory, make_client: Callable[..., FakeLLM]
) -> Callable[..., FakeLLM]:
    """Build a :class:`FakeLLM` that shares its settings with the agent under test."""

    def _make(responses: list[str], settings: Any = None, **overrides: Any) -> FakeLLM:
        resolved = settings if settings is not None else make_settings(**overrides)
        return make_client(resolved, responses=responses)

    return _make


# --------------------------------------------------------------------- planner


def test_fenced_json_becomes_a_plan(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for([fenced(plan_payload("1", "2"))])
    plan = run_coroutine(PlannerAgent(client=client.as_client()).plan("do the thing"))
    assert [task.id for task in plan.tasks] == ["1", "2"]
    assert plan.rationale == "one step at a time"


def test_planner_asks_for_the_planner_model(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for([plan_payload("1")], planner_model="planner-model")
    run_coroutine(PlannerAgent(client=client.as_client()).plan("goal"))
    assert client.models == ["planner-model"]


def test_malformed_output_is_retried_and_recovers(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for([MALFORMED, fenced(plan_payload("1"))])
    plan = run_coroutine(PlannerAgent(client=client.as_client()).plan("do the thing"))
    assert [task.id for task in plan.tasks] == ["1"]
    assert len(client.calls) == 2


def test_retries_are_exhausted_into_a_value_error(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for([MALFORMED] * 5, llm_max_retries=2)
    with pytest.raises(ValueError, match="after 3 attempts"):
        run_coroutine(PlannerAgent(client=client.as_client()).plan("do the thing"))
    assert len(client.calls) == 3


def test_zero_retries_means_exactly_one_attempt(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for([MALFORMED], llm_max_retries=0)
    with pytest.raises(ValueError, match="after 1 attempts"):
        run_coroutine(PlannerAgent(client=client.as_client()).plan("do the thing"))
    assert len(client.calls) == 1


def test_retry_prompt_quotes_the_error_and_the_bad_response(
    client_for: Callable[..., FakeLLM],
) -> None:
    client = client_for([MALFORMED, plan_payload("1")])
    run_coroutine(PlannerAgent(client=client.as_client()).plan("do the thing"))
    retry_prompt = client.prompts[1]
    assert retry_prompt.startswith(client.prompts[0])
    assert MALFORMED in retry_prompt
    assert "failed to parse" in retry_prompt
    assert "only valid JSON" in retry_prompt


def test_planner_feedback_is_included_in_the_prompt_when_supplied(
    client_for: Callable[..., FakeLLM],
) -> None:
    client = client_for([plan_payload("1")])
    run_coroutine(
        PlannerAgent(client=client.as_client()).plan("goal", feedback="tests/test_x.py failed")
    )
    assert "tests/test_x.py failed" in client.prompts[0]


def test_planner_feedback_is_omitted_when_empty(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for([plan_payload("1")])
    run_coroutine(PlannerAgent(client=client.as_client()).plan("goal"))
    assert "Feedback from the previous cycle" not in client.prompts[0]


def test_context_is_included_in_the_prompt_when_supplied(
    client_for: Callable[..., FakeLLM],
) -> None:
    client = client_for([plan_payload("1")])
    run_coroutine(PlannerAgent(client=client.as_client()).plan("goal", context="src/app.py"))
    assert "Workspace context:\nsrc/app.py" in client.prompts[0]


def test_context_is_omitted_when_empty(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for([plan_payload("1")])
    run_coroutine(PlannerAgent(client=client.as_client()).plan("goal"))
    assert "Workspace context" not in client.prompts[0]


def test_goal_always_reaches_the_prompt(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for([plan_payload("1")])
    run_coroutine(PlannerAgent(client=client.as_client()).plan("make the widget"))
    assert "Goal: make the widget" in client.prompts[0]


# ---------------------------------------------------------------------- worker


def test_non_empty_result_is_returned(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for(["In src/app.py, add a Widget class."])
    task = TaskModel(id="1", description="add a widget")
    result = run_coroutine(WorkerAgent(task, client=client.as_client()).run())
    assert result == "In src/app.py, add a Widget class."


def test_result_is_returned_verbatim(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for(["  indented report\n"])
    task = TaskModel(id="1", description="add a widget")
    assert (
        run_coroutine(WorkerAgent(task, client=client.as_client()).run()) == "  indented report\n"
    )


def test_empty_result_is_a_failure(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for([""])
    task = TaskModel(id="1", description="add a widget")
    with pytest.raises(ValueError, match="empty result"):
        run_coroutine(WorkerAgent(task, client=client.as_client()).run())


@pytest.mark.parametrize("blank", ["   ", "\n\n", "\t \n"])
def test_whitespace_only_result_is_a_failure(
    client_for: Callable[..., FakeLLM], blank: str
) -> None:
    client = client_for([blank])
    task = TaskModel(id="1", description="add a widget")
    with pytest.raises(ValueError, match="empty result"):
        run_coroutine(WorkerAgent(task, client=client.as_client()).run())


def test_worker_prompt_carries_the_task_files_and_workspace(
    client_for: Callable[..., FakeLLM],
) -> None:
    client = client_for(["done"])
    task = TaskModel(id="1", description="add a widget", files=["src/app.py"])
    run_coroutine(WorkerAgent(task, client=client.as_client()).run())
    prompt = client.prompts[0]
    assert "Task: add a widget" in prompt
    assert "Files to touch: src/app.py" in prompt
    assert str(client.settings.workspace) in prompt


def test_files_default_to_any(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for(["done"])
    run_coroutine(WorkerAgent(TaskModel(id="1", description="x"), client=client.as_client()).run())
    assert "Files to touch: any" in client.prompts[0]


def test_worker_feedback_is_included_when_supplied(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for(["done"])
    task = TaskModel(id="1", description="add a widget")
    run_coroutine(
        WorkerAgent(task, client=client.as_client()).run(feedback="AssertionError: got 2")
    )
    assert "AssertionError: got 2" in client.prompts[0]


def test_worker_feedback_is_omitted_when_empty(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for(["done"])
    task = TaskModel(id="1", description="add a widget")
    run_coroutine(WorkerAgent(task, client=client.as_client()).run(feedback=""))
    assert "Failure output from the previous cycle" not in client.prompts[0]


def test_worker_asks_for_the_worker_model(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for(["done"], worker_model="worker-model")
    run_coroutine(WorkerAgent(TaskModel(id="1", description="x"), client=client.as_client()).run())
    assert client.models == ["worker-model"]


# ------------------------------------------------------- tester: model verdicts


def test_fenced_json_becomes_a_test_result(
    client_for: Callable[..., FakeLLM], suite_command: Callable[..., str]
) -> None:
    client = client_for(
        [fenced('{"passed": true, "summary": "3 passed"}')], test_command=suite_command()
    )
    result = run_coroutine(
        Grader(client=client.as_client()).run("goal", [TaskModel(id="1", description="x")])
    )
    assert (result.passed, result.summary, result.failures) == (True, "3 passed", [])


def test_tester_prompt_reports_the_command_and_the_output(
    client_for: Callable[..., FakeLLM], suite_command: Callable[..., str]
) -> None:
    client = client_for(['{"passed": true, "summary": "ok"}'], test_command=suite_command())
    run_coroutine(Grader(client=client.as_client()).run("goal", []))
    prompt = client.prompts[0]
    assert "Goal: goal" in prompt
    assert "Command: " in prompt
    assert "Exit status: 0" in prompt
    assert "1 passed" in prompt


def test_tester_asks_for_the_tester_model(
    client_for: Callable[..., FakeLLM], suite_command: Callable[..., str]
) -> None:
    client = client_for(
        ['{"passed": true, "summary": "ok"}'],
        test_command=suite_command(),
        tester_model="tester-model",
    )
    run_coroutine(Grader(client=client.as_client()).run("goal", []))
    assert client.models == ["tester-model"]


# -------------------------------------------------- tester: non-JSON fallback


@pytest.mark.parametrize(
    ("response", "expected"),
    [("0 passed, 12 failed", False), ("All tests passed", True), ("the suite is done", False)],
)
def test_non_json_verdict_is_graded_conservatively(
    client_for: Callable[..., FakeLLM],
    suite_command: Callable[..., str],
    response: str,
    expected: bool,
) -> None:
    client = client_for([response], test_command=suite_command())
    result = run_coroutine(Grader(client=client.as_client()).run("goal", []))
    assert result.passed is expected


@pytest.mark.parametrize(
    "response",
    ["FAILED tests/test_x.py::test_y", "Traceback (most recent call last)", "cannot import x"],
)
def test_failure_language_is_never_scored_as_a_pass(
    client_for: Callable[..., FakeLLM], suite_command: Callable[..., str], response: str
) -> None:
    client = client_for([response], test_command=suite_command())
    assert run_coroutine(Grader(client=client.as_client()).run("goal", [])).passed is False


def test_a_failure_marker_beats_a_success_marker(
    client_for: Callable[..., FakeLLM], suite_command: Callable[..., str]
) -> None:
    client = client_for(
        ["All tests passed, though the import could not fail"], suite_command=suite_command()
    )
    assert run_coroutine(Grader(client=client.as_client()).run("goal", [])).passed is False


def test_empty_non_json_verdict_is_a_failure_with_a_reason(
    client_for: Callable[..., FakeLLM], suite_command: Callable[..., str]
) -> None:
    client = client_for(["   "], test_command=suite_command())
    result = run_coroutine(Grader(client=client.as_client()).run("goal", []))
    assert result.passed is False
    assert "unparseable" in result.summary


def test_negated_pass_phrase_is_not_a_pass(
    client_for: Callable[..., FakeLLM], suite_command: Callable[..., str]
) -> None:
    client = client_for(["not passed"], test_command=suite_command())
    assert run_coroutine(Grader(client=client.as_client()).run("goal", [])).passed is False


def test_failing_non_json_verdict_keeps_the_response_as_the_failure(
    client_for: Callable[..., FakeLLM], suite_command: Callable[..., str]
) -> None:
    client = client_for(["tests failed: 2 errors in tests/test_x.py"], test_command=suite_command())
    result = run_coroutine(Grader(client=client.as_client()).run("goal", []))
    assert result.failures == ["tests failed: 2 errors in tests/test_x.py"]


def test_missing_runner_is_not_a_pass(client_for: Callable[..., FakeLLM]) -> None:
    client = client_for(
        ['{"passed": true, "summary": "sure"}'], test_command="/nonexistent/runner --flag"
    )
    result = run_coroutine(Grader(client=client.as_client()).run("goal", []))
    assert result.passed is False
    assert "not on PATH" in result.summary


# ----------------------------------------------------------- tester: the runner


def test_configured_test_command_overrides_detection(
    client_for: Callable[..., FakeLLM], suite_command: Callable[..., str]
) -> None:
    client = client_for(
        ['{"passed": true, "summary": "ok"}'],
        test_command=suite_command(PASSING_SUITE, "explicit.py"),
    )
    run_coroutine(Grader(client=client.as_client()).run("goal", []))
    assert "explicit.py" in client.prompts[0]


def test_configured_test_command_is_split_without_a_shell(
    client_for: Callable[..., FakeLLM], workspace: Path
) -> None:
    script = workspace / "argv.py"
    script.write_text("import sys\nprint(sys.argv[1])\nprint('1 passed')\n", encoding="utf-8")
    argument = "$HOME two words; rm -rf /"
    command = " ".join(
        [shlex.quote(sys.executable), shlex.quote(str(script)), shlex.quote(argument)]
    )
    client = client_for(['{"passed": true, "summary": "ok"}'], test_command=command)
    run_coroutine(Grader(client=client.as_client()).run("goal", []))
    assert argument in client.prompts[0]


def test_failing_run_is_reported_to_the_model_with_its_output(
    client_for: Callable[..., FakeLLM], suite_command: Callable[..., str]
) -> None:
    client = client_for(
        ['{"passed": false, "summary": "1 failed"}'],
        test_command=suite_command(FAILING_SUITE, "red.py"),
    )
    run_coroutine(Grader(client=client.as_client()).run("goal", []))
    assert "Exit status: 1" in client.prompts[0]
    assert "1 failed, 2 passed" in client.prompts[0]


def test_a_blown_up_test_sweep_is_a_failure_that_keeps_its_output(
    client_for: Callable[..., FakeLLM], workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from furrow.agents import tester as tester_module

    (workspace / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    collected_nothing = tester_module.RunAttempt(
        command=["pytest", "-q"], returncode=5, output="collected no tests"
    )
    invoked: list[list[str]] = []

    async def exploding_run_command(
        self: Grader, command: Sequence[str], cwd: Path, timeout: float
    ) -> tester_module.RunAttempt:
        invoked.append(list(command))
        if invoked == [["pytest", "-q"]]:
            return collected_nothing
        raise RuntimeError("spawn exploded")

    monkeypatch.setattr(
        tester_module, "shutil", SimpleNamespace(which=lambda name: f"/usr/bin/{name}")
    )
    monkeypatch.setattr(tester_module.TesterAgent, "_run_command", exploding_run_command)
    client = client_for(["2 failed"])

    result = run_coroutine(Grader(client=client.as_client()).run("goal", []))

    assert result.passed is False
    assert len(invoked) >= 2
    assert "collected no tests" in client.prompts[0]
    assert "spawn exploded" in client.prompts[0]
    assert "Exit status: 5" in client.prompts[0]
    assert result.failures == ["2 failed"]


# --------------------------------------------------------- tester: truncation


def test_short_output_is_left_alone() -> None:
    assert truncate_output("all good", 100) == "all good"


def test_truncation_keeps_the_tail() -> None:
    truncated = truncate_output("HEAD" + "x" * 200 + "TAIL", 4)
    assert truncated.endswith("TAIL")
    assert "HEAD" not in truncated


def test_truncation_states_how_much_was_dropped() -> None:
    assert "90 earlier characters omitted" in truncate_output("x" * 100, 10)


def test_a_non_positive_limit_disables_truncation() -> None:
    assert truncate_output("x" * 100, 0) == "x" * 100
    assert truncate_output("x" * 100, -1) == "x" * 100


def test_default_truncation_limit_is_the_documented_one() -> None:
    truncated = truncate_output("x" * (MAX_OUTPUT_CHARS + 5000))
    assert "5000 earlier characters omitted" in truncated
    assert len(truncated) < MAX_OUTPUT_CHARS + 200


# ----------------------------------------------------------- tester: detection


def test_pyproject_only_workspace_prefers_pytest(workspace: Path) -> None:
    (workspace / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    assert detect_candidates(workspace)[0] == ["pytest", "-q"]


def test_package_json_only_workspace_does_not_choose_pytest(workspace: Path) -> None:
    (workspace / "package.json").write_text("{}\n", encoding="utf-8")
    assert detect_candidates(workspace)[0] == ["npm", "test", "--", "--silent"]


def test_pnpm_lockfile_selects_pnpm(workspace: Path) -> None:
    (workspace / "package.json").write_text("{}\n", encoding="utf-8")
    (workspace / "pnpm-lock.yaml").write_text("lockfileVersion: 9\n", encoding="utf-8")
    assert detect_candidates(workspace)[0] == ["pnpm", "test", "--", "--silent"]


def test_yarn_lockfile_selects_yarn(workspace: Path) -> None:
    (workspace / "package.json").write_text("{}\n", encoding="utf-8")
    (workspace / "yarn.lock").write_text("# yarn\n", encoding="utf-8")
    assert detect_candidates(workspace)[0] == ["yarn", "test", "--silent"]


def test_cargo_workspace_prefers_cargo(workspace: Path) -> None:
    (workspace / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    assert detect_candidates(workspace)[0] == ["cargo", "test", "-q"]


def test_go_workspace_prefers_go(workspace: Path) -> None:
    (workspace / "go.mod").write_text("module x\n", encoding="utf-8")
    assert detect_candidates(workspace)[0] == ["go", "test", "./..."]


def test_every_detected_runner_is_listed_once(workspace: Path) -> None:
    (workspace / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (workspace / "package.json").write_text("{}\n", encoding="utf-8")
    candidates = detect_candidates(workspace)
    assert len(candidates) == len({tuple(candidate) for candidate in candidates})


def test_an_unmarked_workspace_still_offers_a_runner(workspace: Path) -> None:
    candidates = detect_candidates(workspace)
    assert ["pytest", "-q"] in candidates
    assert ["npm", "test", "--", "--silent"] in candidates


def test_detection_of_a_missing_directory_still_offers_the_fallbacks(
    workspace: Path,
) -> None:
    candidates = detect_candidates(workspace / "missing")
    assert [candidate[0] for candidate in candidates][:2] == ["pytest", "python"]
