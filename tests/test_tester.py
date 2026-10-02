"""Regression tests for ``furrow.agents.tester``.

The historical bug this locks in is the plain-text fallback
``passed = "passed" in response.lower()``: "tests did not pass" and "no failures"
both read as a passing run, so a red suite could be reported green and the
orchestrator would stop asking for fixes.

``run()`` shells out to a test runner; the tests below stub ``_run_tests`` for
the parsing/heuristic cases and drive the real subprocess path only where the
behaviour under test *is* the subprocess (cwd, timeout, missing runner).

``_run_tests`` takes the first candidate runner it can execute, and which of
``pytest``/``npm``/``cargo``/... are on PATH is machine-specific, so the tests
that need a real runner pin one via :func:`with_pytest_available` instead of
depending on the environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from furrow.agents import tester as tester_module
from furrow.agents.tester import MISSING_RUNNER, TesterAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

TASKS = [TaskModel(id="a", description="do a")]


def _canned_complete(body: str, calls: list[dict[str, Any]]):
    """Return a stand-in for ``LLMClient.complete`` returning ``body``.

    The parameter names must match ``LLMClient.complete(self, prompt,
    system="", model=None)`` exactly: ``TesterAgent.run`` forwards ``model`` as
    a keyword, so renaming these (e.g. to silence ARG001) breaks the call with
    ``TypeError: unexpected keyword argument 'model'``. Every forwarded argument
    is therefore *recorded* on ``calls`` rather than ignored, which both keeps
    the call honest and lets a test assert what was forwarded.
    """

    async def fake_complete(prompt: str, system: str = "", model: str | None = None) -> str:
        calls.append({"prompt": prompt, "system": system, "model": model})
        return body

    return fake_complete


def make_agent(response: str, **settings_kwargs: Any) -> TesterAgent:
    """A real TesterAgent whose ``complete`` returns ``response`` verbatim.

    ``_run_tests`` is replaced so no subprocess runs unless a test asks for it.
    """
    client = LLMClient(settings=Settings(**settings_kwargs))

    async def fake_run_tests() -> str:
        return "RESULT: PASSED\n1 passed"

    client.complete = _canned_complete(response, [])  # type: ignore[method-assign]
    agent = TesterAgent(client=client)
    # `run()` calls `self._run_tests()`, so the instance binding is the one that
    # matters; `client._run_tests` is never consulted and is deliberately absent.
    agent._run_tests = fake_run_tests  # type: ignore[method-assign]
    return agent


def with_pytest_available(agent: TesterAgent, seen: dict[str, Any]):
    """Build a ``_run_candidate`` that pins the first candidate to ``sys.executable``.

    Returns the callable to hand to ``monkeypatch.setattr``; every call is
    recorded in ``seen`` as ``cmd``/``cwd``/``timeout``. Substituting the
    interpreter keeps the real subprocess behaviour under test (cwd, exit code,
    formatted output, heuristic) while removing the PATH dependency -- on this
    machine ``pytest`` is absent, so ``npm test`` would otherwise win the race.
    """
    real_run = agent._run_candidate

    async def substitute(cmd, cwd, timeout):
        seen["cmd"] = list(cmd)
        seen["cwd"] = cwd
        seen["timeout"] = timeout
        if Path(cmd[0]).name == "pytest":
            cmd = [sys.executable, "-m", "pytest", *cmd[1:]]
        return await real_run(cmd, cwd, timeout)

    return substitute


async def verdict_for(response: str) -> bool:
    result = await make_agent(response).run("the goal", TASKS)
    return result.passed


# ------------------------------------------------- plain-text fallback heuristic


@pytest.mark.parametrize(
    "response",
    [
        "3 failures",
        "FAILED",
        "2 passed, 1 failed",
        "no test runner found",
        "tests did not pass",
        "1 error during collection",
        "Traceback (most recent call last)",
        "pytest exited with code 1",
        "the suite did not pass",
        "tests were not run",
    ],
)
async def test_plain_text_reporting_problems_is_a_failure(response: str):
    # Bug: "tests did not pass" contains "pass", so the old substring test
    # reported it as a PASSING run.
    assert await verdict_for(response) is False


@pytest.mark.parametrize(
    "response",
    [
        "All tests passed",
        "0 failed, 12 passed",
        "0 failures",
        "no failures",
        "everything passed",
        "the run was green",
    ],
)
async def test_plain_text_reporting_success_is_a_pass(response: str):
    assert await verdict_for(response) is True


async def test_an_explicit_result_marker_beats_the_surrounding_prose():
    # Bug: "RESULT: FAILED" with a "2 passed" mention used to read as a pass.
    assert await verdict_for("RESULT: FAILED\n2 passed, 1 failed") is False
    assert await verdict_for("RESULT: PASSED\n0 failed, 12 passed") is True


async def test_a_result_marker_mid_text_is_still_honoured():
    assert await verdict_for("some preamble\nRESULT: FAILED\ndone") is False


async def test_a_missing_runner_report_is_never_read_as_a_pass():
    # Bug: a missing runner used to be reported as a pass, stopping the loop.
    assert await verdict_for(MISSING_RUNNER) is False
    assert tester_module._text_indicates_pass(MISSING_RUNNER) is False


@pytest.mark.parametrize("response", ["All tests passed", "0 failed, 12 passed"])
def test_the_fallback_heuristic_agrees_on_success(response: str):
    assert tester_module._text_indicates_pass(response) is True


# ------------------------------------------------- the ordering is load bearing


# Bug (must never regress): the verdict used to be
# `passed = "passed" in response.lower()`, a coin flip. "tests did not pass" and
# "no failures" both read as PASSING, and "2 passed, 1 failed" read as a pass
# because "passed" appears before "failed" in the string -- a red suite was
# reported green and the orchestrator stopped asking for fixes.
#
# `_text_indicates_pass` is only correct because of the ORDER of its checks:
#   1. `_RESULT_RE`   -- an explicit marker wins outright
#   2. `_ZERO_PROBLEM_RE` -- "0 failed" must be tested first; it contains "fail"
#   3. `_CLEAN_SUCCESS`  -- must beat `_FAILURE` ("no failures" contains "fail")
#   4. `_FAILURE`
#   5. `_PASS_HINT`
# Swap 2/3 with 4/5 and every PASSED row below flips, so both the helper and the
# end-to-end `run()` verdict are pinned for each string.
@pytest.mark.parametrize(
    ("response", "expected_pass"),
    [
        ("tests did not pass", False),
        ("no failures", True),
        ("0 failed, 12 passed", True),
        ("3 failures", False),
        ("2 passed, 1 failed", False),
        ("All tests passed", True),
        ("no test runner found", False),
        ("FAILED", False),
    ],
)
async def test_the_heuristic_check_order_decides_the_verdict(response: str, expected_pass: bool):
    assert tester_module._text_indicates_pass(response) is expected_pass
    assert await verdict_for(response) is expected_pass


# --------------------------------------------------------------- response parsing


async def test_a_fenced_json_response_parses_into_a_test_result():
    result = await make_agent(
        '```json\n{"passed": false, "summary": "1 failed", "failures": ["b.py::t"]}\n```'
    ).run("the goal", TASKS)
    assert result.passed is False
    assert result.summary == "1 failed"
    assert result.failures == ["b.py::t"]


async def test_a_prose_wrapped_json_response_parses():
    result = await make_agent(
        'Here is my verdict: {"passed": true, "summary": "all green"} enjoy'
    ).run("the goal", TASKS)
    assert result.passed is True
    assert result.summary == "all green"
    assert result.failures == []


async def test_a_json_array_response_is_rejected_and_the_text_is_analysed():
    result = await make_agent("[1, 2, 3] tests did not pass").run("the goal", TASKS)
    assert result.passed is False


async def test_a_fenced_json_body_missing_optional_keys_still_builds():
    result = await make_agent('```json\n{"passed": true, "summary": "ok"}\n```').run(
        "the goal", TASKS
    )
    assert result.failures == []


async def test_an_unparseable_failing_response_populates_failures():
    # The orchestrator's next-cycle directive is built from `failures`; an empty
    # list left the directive blank and the next cycle had nothing to fix.
    response = "3 failures, the parser is broken"
    result = await make_agent(response).run("the goal", TASKS)

    assert result.passed is False
    assert result.failures
    assert response in "\n".join(result.failures)


async def test_an_unparseable_passing_response_leaves_failures_empty():
    result = await make_agent("All tests passed").run("the goal", TASKS)
    assert result.passed is True
    assert result.failures == []


async def test_a_failing_verdict_always_reaches_the_llm_with_the_test_output():
    calls: list[dict[str, Any]] = []
    client = LLMClient(settings=Settings())

    async def fake_run_tests() -> str:
        return "RESULT: FAILED\nassert 1 == 2"

    client.complete = _canned_complete(  # type: ignore[method-assign]
        '{"passed": false, "summary": "red", "failures": ["x"]}', calls
    )
    agent = TesterAgent(client=client)
    agent._run_tests = fake_run_tests  # type: ignore[method-assign]

    await agent.run("the goal", TASKS)

    assert len(calls) == 1
    forwarded = calls[0]
    assert "assert 1 == 2" in forwarded["prompt"]
    assert "the goal" in forwarded["prompt"]
    # `run` forwards the tester model as a keyword; a signature drift here is
    # exactly what the recorded call is here to catch.
    assert forwarded["model"] == client.settings.tester_model
    assert forwarded["system"] == ""


async def test_a_broken_test_run_is_reported_as_a_failure_not_skipped():
    # Bug: an exception from the runner could be swallowed into a silent pass.
    client = LLMClient(settings=Settings())
    agent = TesterAgent(client=client)

    async def exploding_run_tests() -> str:
        raise OSError("no disk space")

    agent._run_tests = exploding_run_tests  # type: ignore[method-assign]

    result = await agent.run("the goal", TASKS)

    assert result.passed is False
    assert "OSError" in result.summary
    assert result.failures == [result.summary]


# -------------------------------------------------------------------- _run_tests


def test_missing_runner_message_states_the_tests_were_not_run():
    assert "NOT run" in MISSING_RUNNER
    assert "must not be reported as a passing run" in MISSING_RUNNER
    assert tester_module._text_indicates_pass(MISSING_RUNNER) is False


async def test_run_tests_reports_a_missing_runner_when_no_candidate_executes(tmp_path, monkeypatch):
    # `pytest -q` in an empty temp dir actually *runs* (exit code 5), so the
    # missing-runner branch is forced by making every candidate unavailable.
    agent = TesterAgent(client=LLMClient(settings=Settings(workspace=tmp_path)))

    attempts: list[tuple[list[str], str, int]] = []

    async def no_such_runner(cmd, cwd, timeout):
        attempts.append((list(cmd), cwd, timeout))
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(agent, "_run_candidate", no_such_runner)

    output = await agent._run_tests()

    assert output == MISSING_RUNNER
    assert "NOT run" in output
    assert tester_module._text_indicates_pass(output) is False
    # Every candidate is tried with the workspace and the configured timeout.
    assert len(attempts) == 7
    assert all(cwd == str(tmp_path) for _, cwd, _ in attempts)
    assert {timeout for _, _, timeout in attempts} == {300}


async def test_run_tests_falls_through_to_the_next_candidate_when_one_is_missing(
    tmp_path, monkeypatch
):
    agent = TesterAgent(client=LLMClient(settings=Settings(workspace=tmp_path)))
    tried: list[list[str]] = []
    timeouts: list[int] = []

    async def selective_runner(cmd, cwd, timeout):
        tried.append(list(cmd))
        timeouts.append(timeout)
        if len(tried) < 3:
            raise FileNotFoundError(cmd[0])
        return f"ran {cmd[0]} in {cwd}"

    monkeypatch.setattr(agent, "_run_candidate", selective_runner)

    output = await agent._run_tests()

    assert len(tried) == 3
    assert output == f"ran {tried[2][0]} in {tmp_path}"
    assert set(timeouts) == {300}  # Settings.test_timeout, not the old 120


async def test_run_tests_uses_settings_workspace_as_the_subprocess_cwd(tmp_path, monkeypatch):
    # Bug: the runner used the process cwd instead of Settings.workspace, so the
    # suite was executed against the orchestrator's directory instead of the repo.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_sample.py").write_text("def test_ok():\n    assert True\n")
    agent = TesterAgent(client=LLMClient(settings=Settings(workspace=workspace)))

    seen: dict[str, Any] = {}
    monkeypatch.setattr(agent, "_run_candidate", with_pytest_available(agent, seen))

    output = await agent._run_tests()

    assert seen["cwd"] == str(workspace)
    assert seen["cwd"] != str(Path.cwd())
    assert seen["timeout"] == 300  # Settings.test_timeout, not the old hardcoded 120
    assert f"Working directory: {workspace}" in output
    assert "RESULT: PASSED" in output
    assert "1 passed" in output


async def test_run_tests_reports_a_failing_suite_with_its_output(tmp_path, monkeypatch):
    workspace = tmp_path / "red"
    workspace.mkdir()
    (workspace / "test_sample.py").write_text("def test_ok():\n    assert 1 == 2, 'boom'\n")
    agent = TesterAgent(client=LLMClient(settings=Settings(workspace=workspace, test_timeout=60)))

    monkeypatch.setattr(agent, "_run_candidate", with_pytest_available(agent, {}))

    output = await agent._run_tests()

    assert "RESULT: FAILED" in output
    assert "Exit code: 1 (FAILED)" in output
    assert "boom" in output
    # The formatted report must not be re-read as a pass by the fallback heuristic.
    assert tester_module._text_indicates_pass(output) is False


async def test_run_tests_raises_for_a_missing_workspace_directory(tmp_path):
    # Bug: a missing workspace was reported as "no test runner found", which lies.
    missing = tmp_path / "does-not-exist"
    agent = TesterAgent(client=LLMClient(settings=Settings(workspace=missing)))

    with pytest.raises(FileNotFoundError, match="Workspace directory does not exist"):
        await agent._run_tests()


async def test_a_hanging_run_times_out_using_settings_test_timeout(tmp_path):
    # Bug (twice over): the timeout was hardcoded to 120s, and a timeout used to
    # `continue` to the next candidate, so a wedged suite ended up reported as
    # "No test runner found." -- a lie the orchestrator would act on. The hang is
    # a real failure with a known cause. Merged from three cases to keep the file
    # fast: one ~1s hang asserts every part of the timeout path.
    agent = TesterAgent(client=LLMClient(settings=Settings(workspace=tmp_path, test_timeout=1)))

    output = await agent._run_candidate(
        [sys.executable, "-c", "import time; time.sleep(30)"], str(tmp_path), 1
    )

    assert output.startswith("Test command: ")
    assert f"Working directory: {tmp_path}" in output
    assert "RESULT: FAILED" in output
    assert "Timed out after 1s" in output  # the configured timeout, not 120
    assert "120" not in output
    assert MISSING_RUNNER not in output
    assert "No test runner found" not in output
    assert tester_module._text_indicates_pass(output) is False


async def test_a_quick_run_within_the_timeout_reports_a_real_verdict(tmp_path):
    agent = TesterAgent(client=LLMClient(settings=Settings(workspace=tmp_path, test_timeout=60)))

    output = await agent._run_candidate(
        [sys.executable, "-c", "print('hello from the tests')"], str(tmp_path), 60
    )

    assert "RESULT: PASSED" in output
    assert "hello from the tests" in output
    assert "Exit code: 0 (PASSED)" in output


def test_format_result_labels_the_exit_code():
    text = tester_module._format_result(["pytest", "-q"], 1, "2 failed", "/tmp/ws")
    assert "RESULT: FAILED" in text
    assert "Test command: pytest -q" in text
    assert "Working directory: /tmp/ws" in text
    assert "2 failed" in text
    # The marker is last, so it is the sentence the LLM sees as the verdict.
    assert text.index("Exit code: 1 (FAILED)") < text.index("RESULT: FAILED")


def test_format_result_omits_the_output_section_when_there_is_none():
    text = tester_module._format_result(["pytest"], 0, "", "/tmp/ws")
    assert "RESULT: PASSED" in text
    # Header only: the report ends at the verdict marker, with no trailing
    # blank-line-plus-empty-output section.
    assert text.endswith("Exit code: 0 (PASSED)\nRESULT: PASSED")
    assert "\n\n" not in text


def test_snippet_truncates_long_text():
    assert tester_module._snippet("y" * 900).endswith("... [truncated]")
    assert tester_module._snippet("short") == "short"


def test_workspace_from_settings_is_used_not_the_process_cwd(tmp_path):
    # Guards the wiring itself: the agent must read Settings.workspace.
    client = LLMClient(settings=Settings(workspace=tmp_path))
    agent = TesterAgent(client=client)
    assert Path(client.settings.workspace) == tmp_path
    assert agent.client.settings.workspace == tmp_path
