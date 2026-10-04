"""Contracts for :mod:`furrow.agents.tester`.

Three layers, in increasing order of machinery:

1. the process execution layer -- ``run_command()`` against a real (tiny, fast)
   command: timeouts, exit codes, the workspace used as ``cwd``, output truncation
   and the "nothing ran" shapes;
2. the verdict layer -- ``run()`` against a scripted ``FakeLLM``: the exit code is
   authoritative over the model's opinion, a missing runner is *not* evidence that
   the code is broken, and unparseable prose is read conservatively;
3. the constructor -- ``TesterAgent()`` with no arguments must not raise.

Everything here runs real subprocesses, so the wall-clock assertions matter: a
timeout must kill the child, not wait it out. A regression that let the hang
through used to burn ~30s per candidate runner (7 of them) for every single test.
"""

from __future__ import annotations

import shlex
import shutil
import sys
import time
from typing import Any

import pytest

from furrow.agents.tester import MAX_OUTPUT_CHARS
from furrow.agents.tester import TesterAgent as Tester  # "TesterAgent" would be collected
from furrow.config import Settings, TaskModel
from tests.conftest import FakeLLM

#: A test command that prints one marker line and then hangs forever. ``exec``
#: matters: without it the forked ``sleep`` keeps the inherited stdout pipe open,
#: so the pipe never reaches EOF and the partial output is lost.
HANGING_COMMAND = 'sh -c "echo MARK; exec sleep 30"'


def _agent(settings: Settings, **kwargs: Any) -> tuple[Tester, FakeLLM]:
    """A tester whose LLM entry point is a scripted :class:`FakeLLM`."""
    llm = FakeLLM(settings, **kwargs)
    return Tester(client=llm, settings=settings), llm


# --------------------------------------------------------------------------
# Process execution: timeouts, exit codes, cwd, truncation
# --------------------------------------------------------------------------


async def test_a_hanging_test_command_is_killed_and_its_partial_output_is_kept(
    tmp_workspace: Settings,
) -> None:
    """Regression guard: the runner must be killed at the timeout, not waited out.

    The old code awaited the (already timed-out) ``communicate()`` task, so this
    call blocked for the full ``sleep 30``.
    """
    tmp_workspace.test_command = HANGING_COMMAND
    tmp_workspace.test_timeout = 1

    agent, _llm = _agent(tmp_workspace)
    started = time.monotonic()
    run = await agent.run_command()
    elapsed = time.monotonic() - started

    assert run.timed_out is True
    assert run.available is True
    assert run.ok is False
    assert run.exit_code == -9, "the child was SIGKILLed, so the exit code is -9"
    assert run.output == "[TIMED OUT -- partial output follows]\nMARK\n"
    # Generous, but nowhere near the 30s the command would otherwise take.
    assert elapsed < 10.0, f"the timeout did not take effect ({elapsed:.1f}s elapsed)"


async def test_auto_detection_prefers_pytest_on_a_passing_suite(tmp_workspace: Settings) -> None:
    (tmp_workspace.workspace / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )

    agent, _llm = _agent(tmp_workspace)
    run = await agent.run_command()

    assert run.runner == "pytest -q"
    assert run.available is True
    assert run.exit_code == 0
    assert run.timed_out is False
    assert run.ok is True
    assert "1 passed" in run.output


async def test_auto_detection_reports_a_failing_suite_without_trying_another_runner(
    tmp_workspace: Settings,
) -> None:
    """A failing suite is a *result*: detection must stop there, not fall through."""
    (tmp_workspace.workspace / "test_bad.py").write_text(
        "def test_bad():\n    assert 0 == 1\n", encoding="utf-8"
    )

    agent, _llm = _agent(tmp_workspace)
    run = await agent.run_command()

    assert run.runner == "pytest -q"
    assert run.available is True
    assert run.exit_code == 1
    assert run.ok is False
    assert "assert 0 == 1" in run.output


async def test_the_test_command_runs_in_the_configured_workspace(tmp_workspace: Settings) -> None:
    tmp_workspace.test_command = 'sh -c "pwd"'

    agent, _llm = _agent(tmp_workspace)
    run = await agent.run_command()

    assert run.exit_code == 0
    assert run.output.strip() == str(tmp_workspace.workspace.resolve())


async def test_enormous_output_is_truncated_to_the_tail(tmp_workspace: Settings) -> None:
    """Failures live at the end of a log, so the head is what gets dropped."""
    payload = "HEAD_MARKER" + "A" * 25_000 + "TAIL_MARKER"
    noisy = tmp_workspace.workspace / "noisy.py"
    noisy.write_text(f"import sys\nsys.stdout.write({payload!r})\n", encoding="utf-8")
    tmp_workspace.test_command = f"{shlex.quote(sys.executable)} {shlex.quote(str(noisy))}"

    agent, _llm = _agent(tmp_workspace)
    run = await agent.run_command()

    assert run.exit_code == 0
    header, _, body = run.output.partition("\n")
    assert body == payload[-MAX_OUTPUT_CHARS:]
    dropped = int(header.removeprefix("... [").split(" ", 1)[0])
    assert dropped == len(payload) - MAX_OUTPUT_CHARS
    assert "TAIL_MARKER" in run.output
    assert "HEAD_MARKER" not in run.output
    assert len(run.output) < len(payload)


# --------------------------------------------------------------------------
# "Nothing ran" -- available is False, so no exit code and no runner
# --------------------------------------------------------------------------


async def test_a_missing_configured_command_reports_that_nothing_ran(
    tmp_workspace: Settings,
) -> None:
    tmp_workspace.test_command = "definitely-not-a-real-test-runner --fast"

    agent, _llm = _agent(tmp_workspace)
    run = await agent.run_command()

    assert run.available is False
    assert run.exit_code is None
    assert run.runner is None, "runner names the command that actually executed"
    assert run.ok is False
    assert "definitely-not-a-real-test-runner --fast" in run.output


async def test_auto_detection_with_nothing_on_path_names_every_candidate_it_tried(
    tmp_workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: None)

    agent, _llm = _agent(tmp_workspace)
    run = await agent.run_command()

    assert run.available is False
    assert run.exit_code is None
    assert run.runner is None
    assert run.ok is False
    assert "pytest -q" in run.output
    assert "No test runner found" in run.output


# --------------------------------------------------------------------------
# The verdict: the exit code outranks the model
# --------------------------------------------------------------------------


async def test_a_nonzero_exit_overrides_a_model_that_claims_success(
    tmp_workspace: Settings,
) -> None:
    tmp_workspace.test_command = "sh -c 'exit 3'"
    agent, llm = _agent(
        tmp_workspace,
        responses=['{"passed": true, "summary": "looks green to me", "failures": []}'],
    )

    result = await agent.run("add /health", [TaskModel(id="1", description="add /health")])

    assert result.passed is False
    assert result.failures == ["sh -c exit 3 exited with code 3"]
    assert llm.complete_call_count == 1


async def test_a_zero_exit_and_a_positive_verdict_pass(tmp_workspace: Settings) -> None:
    tmp_workspace.test_command = "sh -c 'exit 0'"
    agent, _llm = _agent(tmp_workspace, responses=['{"passed": true, "summary": "all green"}'])

    result = await agent.run("add /health", [])

    assert result.passed is True
    assert result.failures == []
    assert "all green" in result.summary


async def test_a_zero_exit_with_a_negative_verdict_is_a_failure(tmp_workspace: Settings) -> None:
    tmp_workspace.test_command = "sh -c 'exit 0'"
    agent, _llm = _agent(tmp_workspace, responses=['{"passed": false, "summary": "it broke"}'])

    result = await agent.run("add /health", [])

    assert result.passed is False
    assert "it broke" in result.summary


async def test_a_timed_out_run_is_reported_as_a_failure_even_if_the_model_says_otherwise(
    tmp_workspace: Settings,
) -> None:
    tmp_workspace.test_command = HANGING_COMMAND
    tmp_workspace.test_timeout = 1
    agent, _llm = _agent(tmp_workspace, responses=['{"passed": true, "summary": "all green"}'])

    result = await agent.run("add /health", [])

    assert result.passed is False
    assert "timed out after 1s" in result.summary


async def test_no_detected_runner_is_not_reported_as_broken_code(
    tmp_workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loop must not burn cycles on an unverifiable goal, but it must not panic."""
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: None)
    agent, llm = _agent(tmp_workspace)

    result = await agent.run("add /health", [])

    assert result.passed is False
    assert "No test runner was detected" in result.summary
    assert "not evidence that the code is broken" in result.summary
    assert result.failures, "the caller still gets something actionable"
    # Nothing ran, so there is nothing for a model to interpret.
    assert llm.complete_call_count == 0


# --------------------------------------------------------------------------
# Unparseable prose verdicts
# --------------------------------------------------------------------------


async def test_a_negated_pass_phrase_is_not_read_as_a_pass(tmp_workspace: Settings) -> None:
    tmp_workspace.test_command = "sh -c 'exit 0'"
    agent, _llm = _agent(tmp_workspace, responses=["All tests did not pass."])

    result = await agent.run("add /health", [])

    assert result.passed is False
    assert "All tests did not pass." in result.summary


async def test_an_unambiguous_pass_phrase_is_read_as_a_pass(tmp_workspace: Settings) -> None:
    tmp_workspace.test_command = "sh -c 'exit 0'"
    agent, _llm = _agent(tmp_workspace, responses=["All tests passed."])

    result = await agent.run("add /health", [])

    assert result.passed is True
    assert "All tests passed." in result.summary


# --------------------------------------------------------------------------
# Small contracts
# --------------------------------------------------------------------------


async def test_the_legacy_run_tests_helper_still_returns_a_plain_string(
    tmp_workspace: Settings,
) -> None:
    tmp_workspace.test_command = 'sh -c "echo hello"'

    agent, _llm = _agent(tmp_workspace)
    output = await agent._run_tests()

    assert type(output) is str
    assert output == "hello\n"


def test_tester_agent_with_no_arguments_does_not_raise() -> None:
    """Regression guard: the constructor used to require a client/settings pair."""
    agent = Tester()
    assert agent.client.settings is not None
    assert agent.client.settings.tester_model
