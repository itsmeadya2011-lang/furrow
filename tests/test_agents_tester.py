"""Tests for :mod:`furrow.agents.tester`.

The tester turns one real subprocess run into the loop's termination signal, so
these tests pin down the two bugs that cost this project the most:

1. **A failing verdict with an empty ``failures`` list.** The orchestrator feeds
   ``"\\n".join(failures)`` into the next cycle's goal (see
   :meth:`furrow.core.orchestrator.Orchestrator._append_repair_context`), so an
   empty list does not merely lose detail: it reduces the goal to whitespace and
   the next cycle starts inventing work. ``passed is False`` therefore implies at
   least one non-blank failure on *every* path out of
   :meth:`~furrow.agents.tester.TesterAgent.run` -- including one made only of
   blank strings, which is non-empty by ``len()`` and useless in practice.
2. **A model talking the loop out of a failed run.** The verdict used to be
   ``"passed" in response.lower()``, so a run whose output literally says "ALL
   TESTS PASSED" inside a failure message was reported green and the loop
   stopped. The subprocess return code is the ground truth: an unusable response
   falls back to that exit code, never to substring matching, and a parseable
   ``passed: true`` cannot overrule a command that exited non-zero.

The override is deliberately ONE-DIRECTIONAL -- a green run the model calls a
failure stays failing, because tests can pass while the goal is unmet -- so that
is pinned down here as well.

No test reaches a provider. The verdict comes from the ``fake_client`` stand-in
and the subprocess side from real, sub-second shell commands (``true``,
``false``, ``sh -c 'echo boom; exit 3'``) configured through
``settings.test_command``. The single exception is the timeout test, whose child
is killed after ``test_timeout`` rather than being allowed to finish.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import sys
import time
from typing import Any

import pytest

from furrow.agents.prompts import TESTER_PROMPT, TESTER_SYSTEM
from furrow.agents.tester import NO_RUNNER_SUMMARY, CommandResult, TesterAgent
from furrow.config import TaskModel, TestResult

# ``TesterAgent`` starts with "Test", so pytest tries to collect it as a test
# class and warns about its ``__init__``. Marking it non-collectable here (the
# same trick conftest plays on ``furrow.config.TestResult``) silences that
# process-wide; pytest's ``PyCollector.isnosetest()`` checks exactly this
# attribute. This mutates the imported class object, not furrow/agents/tester.py.
TesterAgent.__test__ = False

# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

GOAL = "make the retry loop in furrow/llm.py back off exponentially"

#: A runner that exists everywhere and exits 0 immediately.
EXIT_ZERO = "true"

#: A runner that exists everywhere and exits 1 immediately.
EXIT_ONE = "false"

#: A runner that prints one line and exits 3, so the reportable output and the
#: exit code disagree about nothing but can both be asserted on.
EXIT_THREE = "sh -c 'echo boom; exit 3'"

#: The old, broken heuristic, quoted verbatim so the test that guards against it
#: cannot be rewritten into the same mistake.
LEGACY_SUBSTRING_RULE = "passed" in "ALL TESTS PASSED".lower()

#: An unparseable response that trips that heuristic. There is no JSON in it at
#: all, so the only way to read it as a pass is substring matching.
CLAIMS_ALL_PASSED = (
    "ALL TESTS PASSED. Every test in the suite is green and the run exited cleanly."
)


def _verdict_text(
    *,
    passed: bool,
    failures: list[str] | None = None,
    summary: str = "the model's own summary",
) -> str:
    """A response that parses as :class:`~furrow.config.TestResult`."""
    payload: dict[str, Any] = {
        "passed": passed,
        "summary": summary,
        "failures": failures if failures is not None else [],
    }
    return json.dumps(payload)


def _tester(client: Any, settings: Any) -> TesterAgent:
    """A :class:`TesterAgent` bound to ``client`` and ``settings``."""
    return TesterAgent(client, settings=settings)


def _usable_failures(verdict: TestResult) -> list[str]:
    """The failures that survive a ``"\\n".join(...)`` into the next cycle's goal."""
    return [failure for failure in verdict.failures if failure.strip()]


async def _pid_is_gone(pid: int, *, timeout: float = 5.0) -> bool:
    """Whether ``pid`` no longer exists (poll, so a reap race cannot fail a test)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        await asyncio.sleep(0.02)
    return False


# --------------------------------------------------------------------------
# 1. A failing run must carry failures -- the next cycle's goal depends on it
# --------------------------------------------------------------------------


async def test_failing_exit_reports_failures_because_they_become_the_next_cycles_goal(
    fake_client: Any, make_settings: Any
) -> None:
    """``test_command="false"`` gives ``passed is False`` AND usable failures.

    The orchestrator joins ``failures`` into the next cycle's goal. An empty list
    on a failing run therefore silently destroys the goal, so the non-empty
    assertion is the point of this test, not a detail.
    """
    settings = make_settings(test_command=EXIT_ONE)
    client = fake_client(settings, text=_verdict_text(passed=True))
    tester = _tester(client, settings)

    verdict = await tester.run(GOAL, [])

    assert verdict.passed is False
    assert _usable_failures(verdict), "a failing verdict with no usable failure empties the next goal"
    # The synthesised entry has to name the run, or the next cycle cannot even
    # tell which command failed.
    assert "false" in _usable_failures(verdict)[0]
    assert any("1" in failure for failure in _usable_failures(verdict))
    # The model really was consulted; the verdict was corrected afterwards.
    assert len(client.calls) == 1


async def test_passing_exit_reports_a_passing_verdict(fake_client: Any, make_settings: Any) -> None:
    settings = make_settings(test_command=EXIT_ZERO)
    client = fake_client(settings, text=_verdict_text(passed=True))
    tester = _tester(client, settings)

    verdict = await tester.run(GOAL, [])

    assert verdict.passed is True
    assert verdict.failures == []


# --------------------------------------------------------------------------
# 2. The exit code outranks a reported pass ...
# --------------------------------------------------------------------------


async def test_model_reported_pass_is_forced_to_fail_when_the_command_exited_non_zero(
    fake_client: Any, make_settings: Any
) -> None:
    """``{"passed": true}`` cannot survive a non-zero exit code.

    A model that wrongly reports success makes the orchestrator stop believing
    the goal is met, so the subprocess is the ground truth we actually observed.
    """
    settings = make_settings(test_command=EXIT_THREE)
    client = fake_client(settings, text=_verdict_text(passed=True))
    tester = _tester(client, settings)

    verdict = await tester.run(GOAL, [])

    assert verdict.passed is False
    assert _usable_failures(verdict)
    assert any("3" in failure for failure in _usable_failures(verdict))
    assert len(client.calls) == 1, "the veto runs after parsing, not instead of it"


async def test_failing_exit_veto_keeps_the_failures_the_model_reported(
    fake_client: Any, make_settings: Any
) -> None:
    """Only ``passed`` is rewritten, so a model's real failures survive the veto."""
    reported = ["test_retry in tests/test_llm.py gave up after 1 attempt"]
    settings = make_settings(test_command=EXIT_THREE)
    client = fake_client(settings, text=_verdict_text(passed=True, failures=reported))
    tester = _tester(client, settings)

    verdict = await tester.run(GOAL, [])

    assert verdict.passed is False
    assert verdict.failures == reported


# --------------------------------------------------------------------------
# 3. ... but only that way: exit 0 with a reported failure stays failing
# --------------------------------------------------------------------------


async def test_green_run_the_model_calls_a_failure_stays_failing(
    fake_client: Any, make_settings: Any
) -> None:
    """The override is one-directional.

    The command can exit 0 while the goal is unmet -- tests green, feature
    missing -- and that failing verdict is a legitimate judgement, not a bug.
    """
    reported = ["the backoff loop still returns without ever sleeping"]
    settings = make_settings(test_command=EXIT_ZERO)
    client = fake_client(settings, text=_verdict_text(passed=False, failures=reported))
    tester = _tester(client, settings)

    verdict = await tester.run(GOAL, [])

    assert verdict.passed is False
    assert verdict.failures == reported, "the model's verdict is preserved verbatim"


# --------------------------------------------------------------------------
# 4. A timeout is reported as one, promptly, and leaves no orphan
# --------------------------------------------------------------------------


#: Writes its own pid, then *replaces* itself with sleep, so the pid in the file
#: is guaranteed to be the process the tester spawned (and therefore the one it
#: has to kill). 30s is far longer than any assertion below will wait.
_HANGING_COMMAND = "sh -c {inner}"
_HANGING_INNER = "echo $$ > {pidfile}; exec sleep 30"


def _hanging_test_command(pidfile: Any) -> str:
    """A test command that records its pid then hangs until it is killed."""
    inner = _HANGING_INNER.format(pidfile=shlex.quote(str(pidfile)))
    return _HANGING_COMMAND.format(inner=shlex.quote(inner))


async def test_command_exceeding_test_timeout_is_killed_and_reported_as_timed_out(
    make_settings: Any, fake_client: Any, tmp_workspace: Any
) -> None:
    pidfile = tmp_workspace / "hanging-child.pid"
    settings = make_settings(test_command=_hanging_test_command(pidfile), test_timeout=1)
    client = fake_client(settings, text=_verdict_text(passed=True))
    tester = _tester(client, settings)

    started = time.monotonic()
    verdict = await tester.run(GOAL, [])
    elapsed = time.monotonic() - started

    # The command would have taken 30s; returning at all proves the timeout fired.
    assert elapsed < 10.0, f"run() waited {elapsed:.1f}s for a 30s command"
    assert verdict.passed is False
    assert any("timed out" in failure for failure in _usable_failures(verdict))

    pid = int(pidfile.read_text().strip())
    assert await _pid_is_gone(pid), f"the killed child {pid} was left behind"


async def test_timeout_command_result_is_not_ok_and_names_the_timeout(
    make_settings: Any, fake_client: Any, tmp_workspace: Any
) -> None:
    pidfile = tmp_workspace / "hanging-child.pid"
    settings = make_settings(test_command=_hanging_test_command(pidfile), test_timeout=1)
    tester = _tester(fake_client(settings), settings)

    result: CommandResult = await tester._run_command(
        shlex.split(settings.test_command or ""),
    )

    assert result.timed_out is True
    assert result.ok is False, "a timeout is not a passing run"
    assert result.error is not None
    assert "timed out" in result.error
    assert "exceeding the 1s timeout" in result.output
    assert result.exit_code == -signal.SIGKILL, "the child was killed, not allowed to finish"
    assert await _pid_is_gone(int(pidfile.read_text().strip()))


# --------------------------------------------------------------------------
# 5. No discoverable runner is a failing result, not an exception
# --------------------------------------------------------------------------


async def test_no_discoverable_runner_in_an_empty_directory_fails_without_raising(
    fake_client: Any, make_settings: Any
) -> None:
    settings = make_settings()  # no test_command, and tmp_workspace is empty
    client = fake_client(settings, text=_verdict_text(passed=True))
    tester = _tester(client, settings)

    verdict = await tester.run(GOAL, [])

    assert verdict.passed is False
    assert verdict.summary == NO_RUNNER_SUMMARY
    assert _usable_failures(verdict)
    assert "no test runner found" in verdict.failures[0]
    # Nothing ran, so there is nothing for the model to classify.
    assert client.calls == []


async def test_a_runner_that_cannot_be_executed_is_a_failure_not_a_crash(
    fake_client: Any, make_settings: Any
) -> None:
    settings = make_settings(test_command="furrow-no-such-test-runner-xyz")
    client = fake_client(settings, text=_verdict_text(passed=True))
    tester = _tester(client, settings)

    verdict = await tester.run(GOAL, [])

    assert verdict.passed is False
    assert any("could not be executed" in failure for failure in _usable_failures(verdict))
    assert any("FileNotFoundError" in failure for failure in _usable_failures(verdict))
    assert client.calls == []


async def test_an_unreachable_model_still_leaves_a_usable_deterministic_verdict(
    fake_client: Any, make_settings: Any
) -> None:
    """``run()`` never raises: a broken LLM call falls back to the exit code."""
    settings = make_settings(test_command=EXIT_THREE)
    client = fake_client(settings, raises=RuntimeError("connection reset"))
    tester = _tester(client, settings)

    verdict = await tester.run(GOAL, [])

    assert verdict.passed is False
    assert _usable_failures(verdict)
    assert len(client.calls) == 1


# --------------------------------------------------------------------------
# 6. An unparseable response falls back to the exit code -- never to substrings
# --------------------------------------------------------------------------


async def test_unparseable_response_claiming_all_tests_passed_falls_back_to_exit_code(
    fake_client: Any, make_settings: Any
) -> None:
    """The regression test for ``"passed" in response.lower()``.

    There is no JSON in the response, so the only way to read it as a pass is
    substring matching -- and the command exited 3.
    """
    assert LEGACY_SUBSTRING_RULE is True, "the old heuristic really did accept this text"
    settings = make_settings(test_command=EXIT_THREE)
    client = fake_client(settings, text=CLAIMS_ALL_PASSED)
    tester = _tester(client, settings)

    verdict = await tester.run(GOAL, [])

    assert verdict.passed is False
    assert _usable_failures(verdict)
    assert any("3" in failure for failure in _usable_failures(verdict))
    assert "reporting the exit code" in verdict.summary


async def test_unparseable_response_on_a_green_run_reports_the_pass(
    fake_client: Any, make_settings: Any
) -> None:
    """The fallback is symmetric: exit 0 with unusable output is still a pass."""
    settings = make_settings(test_command=EXIT_ZERO)
    client = fake_client(settings, text=CLAIMS_ALL_PASSED)
    tester = _tester(client, settings)

    verdict = await tester.run(GOAL, [])

    assert verdict.passed is True
    assert verdict.failures == []
    assert "reporting the exit code" in verdict.summary


async def test_response_that_validates_but_is_not_a_test_result_falls_back(
    fake_client: Any, make_settings: Any
) -> None:
    """A JSON payload that is valid JSON but not a ``TestResult`` is unusable too."""
    settings = make_settings(test_command=EXIT_ONE)
    client = fake_client(settings, text='{"tasks": [], "rationale": "I planned instead."}')
    tester = _tester(client, settings)

    verdict = await tester.run(GOAL, [])

    assert verdict.passed is False
    assert _usable_failures(verdict)


# --------------------------------------------------------------------------
# 7. INVARIANT SWEEP: passed is False  =>  a usable failure exists
# --------------------------------------------------------------------------

_RUNS = [
    pytest.param(EXIT_ZERO, id="exit-0"),
    pytest.param(EXIT_ONE, id="exit-1"),
    pytest.param(None, id="no-runner"),
]
_MODEL_PASSED = [pytest.param(True, id="model-passed"), pytest.param(False, id="model-failed")]
_MODEL_FAILURES = [
    pytest.param([], id="empty-failures"),
    pytest.param(["", "   "], id="blank-failures"),
]


@pytest.mark.parametrize("test_command", _RUNS)
@pytest.mark.parametrize("model_passed", _MODEL_PASSED)
@pytest.mark.parametrize("model_failures", _MODEL_FAILURES)
async def test_failing_verdict_always_implies_a_usable_failure(
    test_command: str | None,
    model_passed: bool,
    model_failures: list[str],
    fake_client: Any,
    make_settings: Any,
) -> None:
    """The sweep behind the two bugs this module exists for.

    Every combination of run outcome, model verdict and failure list must obey
    ``passed is False`` => some failure carries text the next cycle can read.
    """
    settings = make_settings(test_command=test_command)
    client = fake_client(
        settings, text=_verdict_text(passed=model_passed, failures=model_failures)
    )
    tester = _tester(client, settings)

    verdict = await tester.run(GOAL, [])

    assert isinstance(verdict, TestResult)
    if verdict.passed is False:
        assert _usable_failures(verdict), (
            "'\\n'.join(failures) becomes the next cycle's goal; "
            f"an unusable list silently destroys it (command={test_command!r}, "
            f"model_passed={model_passed}, model_failures={model_failures!r})"
        )


# --------------------------------------------------------------------------
# 8. Command discovery: settings first, then project markers
# --------------------------------------------------------------------------


async def test_discover_command_shlex_splits_the_configured_test_command(
    make_settings: Any, fake_client: Any
) -> None:
    settings = make_settings(test_command=EXIT_THREE)
    tester = _tester(fake_client(settings), settings)

    command = await tester._discover_command()

    assert command == ["sh", "-c", "echo boom; exit 3"]


async def test_discover_command_prefers_the_configured_command_over_markers(
    make_settings: Any, fake_client: Any, tmp_workspace: Any
) -> None:
    (tmp_workspace / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    settings = make_settings(test_command=EXIT_ZERO)
    tester = _tester(fake_client(settings), settings)

    command = await tester._discover_command()

    assert command == ["true"]


async def test_discover_command_uses_pytest_for_a_python_project(
    make_settings: Any, fake_client: Any, tmp_workspace: Any
) -> None:
    (tmp_workspace / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    settings = make_settings(test_command=None)
    tester = _tester(fake_client(settings), settings)

    command = await tester._discover_command()

    assert command in (["pytest", "-q"], [sys.executable, "-m", "pytest", "-q"])


async def test_discover_command_returns_none_in_an_empty_directory(
    make_settings: Any, fake_client: Any
) -> None:
    settings = make_settings()
    tester = _tester(fake_client(settings), settings)

    assert await tester._discover_command() is None


async def test_whitespace_only_test_command_falls_back_to_discovery(
    make_settings: Any, fake_client: Any
) -> None:
    settings = make_settings(test_command="   ")
    tester = _tester(fake_client(settings), settings)

    assert await tester._discover_command() is None


# --------------------------------------------------------------------------
# 9. What the tester actually sends, and what it did not fabricate
# --------------------------------------------------------------------------


async def test_prompt_carries_the_observed_exit_code_and_output_not_the_hope(
    fake_client: Any, make_settings: Any
) -> None:
    """The ground truth has to reach the model, or its verdict is worth nothing."""
    settings = make_settings(test_command=EXIT_THREE, tester_model="test-tester-model")
    client = fake_client(settings, text=_verdict_text(passed=False))
    tester = _tester(client, settings)
    tasks = [TaskModel(id="1", description="add backoff", files=["furrow/llm.py"])]

    await tester.run(GOAL, tasks)

    (call,) = client.calls
    assert call.system == TESTER_SYSTEM
    assert call.model == settings.model_for("tester") == "test-tester-model"
    assert call.prompt == TESTER_PROMPT.format(
        goal=GOAL,
        test_output="boom\n",
        exit_code=3,
        command="sh -c echo boom; exit 3",
        files="furrow/llm.py",
    )
    assert "Exit code:\n3" in call.prompt
    assert "boom" in call.prompt


async def test_run_never_invents_failures_for_a_clean_run(
    fake_client: Any, make_settings: Any
) -> None:
    """The tester does not invent failures either -- it reports what happened."""
    settings = make_settings(test_command=EXIT_ZERO)
    client = fake_client(
        settings, text=_verdict_text(passed=False, failures=["the feature is still missing"])
    )
    tester = _tester(client, settings)

    verdict = await tester.run(GOAL, [])

    # A green command with a reported failure keeps that failure verbatim; it is
    # never replaced by a synthesised "exited with code 0" complaint.
    assert verdict.passed is False
    assert verdict.failures == ["the feature is still missing"]
