"""Regression tests for :class:`furrow.agents.tester.TesterAgent`.

Three responsibilities are covered: running a test runner with three deliberately distinct
outcomes (not installed / timed out / finished, including a real failure), turning the model's
answer into a :class:`TestResult` without ever letting a malformed verdict kill a cycle, and
honouring ``test_timeout_seconds``.

No test here shells out to this repository's own pytest -- the runners are throwaway
``sys.executable -c`` scripts under ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Sequence

import pytest

from furrow.agents import tester as tester_module
from furrow.agents.prompts import TESTER_PROMPT
from furrow.agents.tester import DEFAULT_TEST_TIMEOUT_SECONDS
from furrow.config import Settings, TaskModel

from .conftest import PASSING_VERDICT, REAL_TEST_COMMANDS, StubLLMClient

# Aliased: pytest collects any module-level name starting with ``Test`` as a test class and
# warns about ``TesterAgent`` because its ``__init__`` takes arguments.
Agent = tester_module.TesterAgent

PYTHON = sys.executable
MISSING_RUNNER = "furrow-definitely-not-installed-xyz"

PASSING_SCRIPT = (PYTHON, "-c", "print('all good')")
FAILING_SCRIPT = (PYTHON, "-c", "import sys; sys.stderr.write('FAILED boom\\n'); sys.exit(3)")
HANGING_SCRIPT = (PYTHON, "-c", "import time; time.sleep(30)")


def make_agent(
    verdict: str = PASSING_VERDICT,
    commands: Sequence[tuple[str, ...]] = (PASSING_SCRIPT,),
    **settings_kwargs: object,
) -> tuple[Agent, StubLLMClient]:
    # ``Settings`` validates on construction and ``test_timeout_seconds`` is an ``int``, so
    # sub-second timeouts have to be assigned afterwards. Assignment is deliberately not
    # validated (``model_config`` sets no ``validate_assignment``), which is what makes
    # sub-second -- and deliberately invalid -- values testable at all.
    settings = Settings()
    for name, value in settings_kwargs.items():
        setattr(settings, name, value)
    client = StubLLMClient(settings=settings, verdicts=[verdict])
    agent = Agent(client=client)
    agent.test_commands = tuple(commands)
    return agent, client


def run(agent: Agent, goal: str = "GOAL") -> object:
    async def scenario() -> object:
        return await agent.run(goal, [TaskModel(id="1", description="do it")])

    return asyncio.run(scenario())


# ------------------------------------------------------------------ runner selection


def test_a_missing_runner_falls_through_to_the_next_candidate() -> None:
    agent, client = make_agent(
        commands=((MISSING_RUNNER,), (PYTHON, "-c", "print('SECOND CANDIDATE RAN')"))
    )

    result = run(agent)

    assert result.passed is True
    assert "SECOND CANDIDATE RAN" in client.tester_prompts[0]


def test_the_first_installed_runner_wins() -> None:
    agent, client = make_agent(
        commands=((MISSING_RUNNER,), (PYTHON, "-c", "print('chosen')"), FAILING_SCRIPT)
    )

    run(agent)

    assert "chosen" in client.tester_prompts[0]
    assert "FAILED boom" not in client.tester_prompts[0]


def test_no_installed_runner_reports_that_to_the_model() -> None:
    agent, client = make_agent(
        verdict='{"passed": false, "summary": "cannot check"}', commands=((MISSING_RUNNER,),)
    )

    result = run(agent)

    assert "No test runner found." in client.tester_prompts[0]
    assert result.passed is False


def test_the_test_output_is_put_in_front_of_the_model() -> None:
    agent, client = make_agent(commands=((PYTHON, "-c", "print('runner spoke')"),))

    run(agent, goal="Add a widget")

    prompt = client.tester_prompts[0]
    assert prompt.startswith(f"{TESTER_PROMPT}\n\nGoal: Add a widget\n\nTest output:\n")
    assert "runner spoke" in prompt


def test_the_configured_tester_model_is_requested() -> None:
    agent, client = make_agent(tester_model="tester-model-x")

    run(agent)

    assert client.calls[0].model == "tester-model-x"


# ------------------------------------------------------------------- timeout handling


def test_a_timing_out_runner_is_killed_and_reported() -> None:
    agent, _ = make_agent(commands=(HANGING_SCRIPT,), test_timeout_seconds=0.3)

    started = time.monotonic()
    output = asyncio.run(agent._run_tests())
    elapsed = time.monotonic() - started

    assert "timed out after 0.3s" in output
    assert "was killed" in output
    assert elapsed < 10, "the configured timeout must actually cut the run short"


def test_a_timeout_does_not_fall_through_to_further_candidates() -> None:
    agent, client = make_agent(
        commands=(
            HANGING_SCRIPT,
            (PYTHON, "-c", "print('SECOND CANDIDATE MUST NOT RUN')"),
        ),
        test_timeout_seconds=0.3,
    )

    result = run(agent)

    assert "timed out after 0.3s" in client.tester_prompts[0]
    assert "SECOND CANDIDATE MUST NOT RUN" not in client.tester_prompts[0]
    assert result.passed is True  # the canned verdict is honoured; the timeout is in the output


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (0.25, 0.25),
        (5, 5.0),
        (0, float(DEFAULT_TEST_TIMEOUT_SECONDS)),
        (-5, float(DEFAULT_TEST_TIMEOUT_SECONDS)),
        ("oops", float(DEFAULT_TEST_TIMEOUT_SECONDS)),
        (None, float(DEFAULT_TEST_TIMEOUT_SECONDS)),
    ],
)
def test_the_test_timeout_is_read_from_settings(configured: object, expected: float) -> None:
    agent, _ = make_agent(test_timeout_seconds=configured)

    assert agent.test_timeout_seconds == pytest.approx(expected)


def test_the_timeout_defaults_when_the_setting_is_absent() -> None:
    class Bare:
        pass

    agent, _ = make_agent()
    agent.client.settings = Bare()  # type: ignore[assignment]

    assert agent.test_timeout_seconds == float(DEFAULT_TEST_TIMEOUT_SECONDS)


def test_the_timeout_fallback_never_blocks_a_fast_run() -> None:
    agent, client = make_agent(test_timeout_seconds=0)

    run(agent)

    assert "all good" in client.tester_prompts[0]


# --------------------------------------------------------------------- real failures


def test_a_failing_suites_output_is_returned_to_the_model() -> None:
    agent, client = make_agent(
        verdict='{"passed": false, "summary": "1 failed", "failures": ["boom"]}',
        commands=(FAILING_SCRIPT,),
    )

    result = run(agent)

    assert "FAILED boom" in client.tester_prompts[0]
    assert result.passed is False
    assert result.failures == ["boom"]


def test_a_non_zero_exit_with_no_output_is_reported() -> None:
    agent, _ = make_agent(
        verdict='{"summary": "silent", "failures": []}',
        commands=((PYTHON, "-c", "raise SystemExit(7)"),),
    )

    output = asyncio.run(agent._run_tests())

    assert "exited with code 7" in output


def test_the_runner_runs_in_the_configured_workspace(tmp_path: object) -> None:
    agent, client = make_agent(
        commands=((PYTHON, "-c", "import os; print('cwd', os.getcwd())"),),
        workspace=tmp_path,
    )

    run(agent)

    assert "cwd" in client.tester_prompts[0]
    assert str(tmp_path) in client.tester_prompts[0]


def test_an_unusable_workspace_keeps_the_process_directory(tmp_path: object) -> None:
    agent, _ = make_agent(workspace=tmp_path / "not-a-directory")

    assert agent.workspace is None


# ------------------------------------------------------------------- verdict parsing


def test_a_well_formed_verdict_is_used_as_is() -> None:
    agent, _ = make_agent(
        verdict='{"passed": false, "summary": "2 failed", "failures": ["a", "b"]}'
    )

    result = run(agent)

    assert result.passed is False
    assert result.summary == "2 failed"
    assert result.failures == ["a", "b"]


@pytest.mark.parametrize(
    "verdict",
    [
        "I am not JSON at all",
        "[1, 2, 3]",
        '"just a string"',
        "",
    ],
)
def test_a_malformed_verdict_degrades_to_a_failure_with_reasons(verdict: str) -> None:
    agent, _ = make_agent(
        verdict=verdict, commands=((PYTHON, "-c", "print('FAILED tests/test_x.py::test_y')"),)
    )

    result = run(agent)

    assert result.passed is False, verdict
    assert result.failures, verdict
    assert "tests/test_x.py::test_y" in " ".join(result.failures), verdict


def test_an_empty_verdict_still_reports_a_failure() -> None:
    agent, _ = make_agent(verdict="", commands=((PYTHON, "-c", "print('all good')"),))

    result = run(agent)

    assert result.passed is False
    assert result.summary
    assert result.failures


def test_a_failed_verdict_never_has_an_empty_failure_list() -> None:
    agent, _ = make_agent(verdict='{"passed": false, "summary": "nope", "failures": []}')

    result = run(agent)

    assert result.passed is False
    assert result.failures == ["nope"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('{"passed": true, "summary": "s", "failures": []}', True),
        ('{"passed": false, "summary": "s", "failures": ["x"]}', False),
        ('{"passed": "yes", "summary": "s", "failures": []}', True),
        ('{"passed": "no", "summary": "s", "failures": ["x"]}', False),
        ('{"passed": 1, "summary": "s", "failures": []}', True),
        ('{"passed": 0, "summary": "s", "failures": ["x"]}', False),
    ],
)
def test_the_passed_field_is_coerced(value: str, expected: bool) -> None:
    agent, _ = make_agent(verdict=value)

    result = run(agent)

    assert result.passed is expected


def test_an_unusable_passed_field_decides_from_the_run_not_the_text() -> None:
    # The old heuristic keyed on the substring "passed", which is present in every parsed
    # verdict because it *is* the key -- so a verdict with no usable `passed` must be decided
    # by whether the suite itself looked clean.
    clean, _ = make_agent(
        verdict='{"summary": "?", "failures": []}',
        commands=((PYTHON, "-c", "print('clean output')"),),
    )
    assert run(clean).passed is True

    dirty, _ = make_agent(
        verdict='{"summary": "?", "failures": []}',
        commands=((PYTHON, "-c", "print('ERROR: something broke')"),),
    )
    result = run(dirty)
    assert result.passed is False
    assert result.failures


def test_a_suite_that_never_ran_is_not_evidence_of_success() -> None:
    agent, _ = make_agent(verdict='{"summary": "?", "failures": []}', commands=((MISSING_RUNNER,),))

    assert run(agent).passed is False


def test_a_summary_is_filled_in_from_the_run_output_when_absent() -> None:
    agent, _ = make_agent(
        verdict='{"passed": true, "failures": []}',
        commands=((PYTHON, "-c", "print('3 passed in 0.02s')"),),
    )

    result = run(agent)

    assert "3 passed in 0.02s" in result.summary


def test_failures_are_capped_and_truncated() -> None:
    noisy = "\n".join(f"FAILED test_{index}.py::test_x" for index in range(80))
    agent, _ = make_agent(
        verdict='{"passed": false, "summary": "many"}',
        commands=((PYTHON, "-c", f"print({noisy!r})"),),
    )

    result = run(agent)

    assert result.passed is False
    assert 0 < len(result.failures) <= 20
    assert all(len(line) <= 520 for line in result.failures)


# ---------------------------------------------------------------------- cancellation


def test_cancellation_propagates_and_the_child_is_reaped() -> None:
    # A runner that would sit there for 30s: cancelling mid-``communicate()`` is the case
    # that must not leak the child process and must not be mistaken for a missing runner.
    agent, _ = make_agent(commands=(HANGING_SCRIPT,), test_timeout_seconds=30)

    async def scenario() -> None:
        task = asyncio.create_task(agent.run("GOAL", []))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


# ------------------------------------------------------------------------ defaults


def test_the_default_candidates_are_the_documented_runner_list() -> None:
    candidates = REAL_TEST_COMMANDS

    # The interpreter actually running furrow leads: `sys.executable -m pytest` is guaranteed
    # to use the environment pytest is installed in, whereas a bare `python` may not exist.
    assert candidates[:2] == (
        (sys.executable, "-m", "pytest", "-q"),
        ("pytest", "-q"),
    )
    assert [command[0] for command in candidates] == [
        sys.executable,
        "pytest",
        "python",
        "npm",
        "pnpm",
        "yarn",
        "cargo",
        "go",
    ]


def test_an_agent_built_without_a_client_gets_one() -> None:
    agent = Agent()

    assert agent.client is not None
    assert agent.client.settings is not None
    assert agent.test_commands == tester_module.TEST_COMMANDS
