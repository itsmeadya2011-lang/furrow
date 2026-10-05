"""Tests for the tester agent.

Covers three things:

* running a test command and detecting a runner, including the real subprocess
  path in ``_run_tests`` and the ``CommandResult`` it produces;
* exit-code authority in ``_combine``: the process exit status decides pass or
  fail, and neither a non-zero code nor a timeout is ever reported as a pass,
  whatever the model claims;
* tolerant verdict parsing in ``_verdict`` / ``_text_indicates_pass``, where a
  JSON reply is read into a ``TestResult`` and anything else falls back to a
  deliberately conservative reading of the text.
"""

from __future__ import annotations

from typing import Optional

from furrow.agents.tester import (
    TEST_TIMEOUT_SECONDS,
    CommandResult,
    TesterAgent,
    _failure,
    _text_indicates_pass,
    _verdict,
)
from furrow.config import Settings, TestResult, get_settings

CLEAN_JSON = '{"passed": true, "summary": "All tests passed", "failures": []}'
FAILING_JSON = (
    '{"passed": false, "summary": "2 failed", "failures": ["test_a", "test_b"]}'
)


class _StubClient:
    """Stand-in for ``LLMClient``: records prompts and returns a canned reply."""

    def __init__(self, reply: str = "", settings: Optional[Settings] = None) -> None:
        self.reply = reply
        self.settings = settings if settings is not None else get_settings()
        self.calls: list[str] = []

    async def complete(
        self, prompt: str, system: str = "", model: Optional[str] = None
    ) -> str:
        self.calls.append(prompt)
        return self.reply


# --------------------------------------------------------------------------
# _verdict: JSON replies
# --------------------------------------------------------------------------


def test_verdict_parses_clean_json() -> None:
    verdict = _verdict(CLEAN_JSON)
    assert verdict.passed is True
    assert verdict.summary == "All tests passed"
    assert verdict.failures == []


def test_verdict_parses_fenced_json() -> None:
    response = (
        "Here is my review of the run:\n"
        "```json\n"
        '{"passed": true, "summary": "Looks good", "failures": []}\n'
        "```\n"
        "Let me know if you want more detail."
    )
    verdict = _verdict(response)
    assert verdict.passed is True
    assert verdict.summary == "Looks good"
    assert verdict.failures == []


def test_verdict_parses_json_failure_report() -> None:
    verdict = _verdict(FAILING_JSON)
    assert verdict.passed is False
    assert verdict.summary == "2 failed"
    assert verdict.failures == ["test_a", "test_b"]


# --------------------------------------------------------------------------
# _verdict: conservative fallback for non-JSON replies
# --------------------------------------------------------------------------


def test_verdict_fallback_detects_a_pass_statement() -> None:
    verdict = _verdict("All tests passed successfully.")
    assert verdict.passed is True
    assert verdict.failures == []


def test_verdict_fallback_detects_a_failure_statement() -> None:
    verdict = _verdict("2 failed, 10 passed")
    assert verdict.passed is False


def test_verdict_fallback_is_conservative_for_neutral_text() -> None:
    verdict = _verdict("The runner printed nothing useful.")
    assert verdict.passed is False


def test_verdict_fallback_is_false_for_an_empty_reply() -> None:
    verdict = _verdict("")
    assert verdict.passed is False


def test_text_indicates_pass_neutralises_zero_failure_counts() -> None:
    assert _text_indicates_pass("0 failed, 12 passed") is True
    assert _text_indicates_pass("0 errors") is False


# --------------------------------------------------------------------------
# CommandResult
# --------------------------------------------------------------------------


def test_command_result_succeeds_on_zero_exit_code() -> None:
    result = CommandResult("pytest -q", 0, "", True)
    assert result.succeeded is True


def test_command_result_fails_on_nonzero_exit_code() -> None:
    result = CommandResult("pytest -q", 1, "boom", True)
    assert result.succeeded is False


def test_command_result_fails_when_it_timed_out() -> None:
    result = CommandResult("pytest -q", 0, "", True, timed_out=True)
    assert result.succeeded is False


def test_command_result_fails_when_no_runner_was_found() -> None:
    result = CommandResult("", None, "No test runner found.", False)
    assert result.succeeded is False


def test_render_includes_the_command_and_the_exit_code() -> None:
    rendered = CommandResult("pytest -q", 1, "boom", True).render()
    assert "pytest -q" in rendered
    assert "exit code: 1" in rendered
    assert "boom" in rendered


def test_render_mentions_a_timeout() -> None:
    rendered = CommandResult("pytest -q", None, "too slow", True, timed_out=True).render()
    assert "timed out" in rendered
    assert str(TEST_TIMEOUT_SECONDS) in rendered


def test_render_for_a_missing_runner_is_just_the_message() -> None:
    result = CommandResult("", None, "No test runner found.", False)
    assert result.render() == "No test runner found."


# --------------------------------------------------------------------------
# _combine: the exit code overrides the model's verdict
# --------------------------------------------------------------------------


def test_combine_rejects_a_nonzero_exit_code() -> None:
    agent = TesterAgent(client=_StubClient())
    result = CommandResult(command="pytest -q", returncode=1, output="2 failed", found=True)
    verdict = TestResult(passed=True, summary="model says green", failures=[])

    combined = agent._combine(result, verdict)

    assert combined.passed is False
    assert "exited with code 1" in combined.summary
    assert "pytest -q" in combined.summary
    assert len(combined.failures) >= 1
    assert "exited with code 1" in combined.failures[0]


def test_combine_keeps_the_model_failure_list() -> None:
    agent = TesterAgent(client=_StubClient())
    result = CommandResult(command="pytest -q", returncode=2, output="boom", found=True)
    verdict = TestResult(passed=False, summary="broken", failures=["test_a"])

    combined = agent._combine(result, verdict)

    assert combined.passed is False
    assert "test_a" in combined.failures
    assert "exited with code 2" in combined.failures[-1]


def test_combine_reports_a_timeout_as_a_failure() -> None:
    agent = TesterAgent(client=_StubClient())
    # ``returncode=None`` together with ``found=True`` is a defensive branch:
    # ``_collect`` reports the real ``proc.returncode`` even on the timeout path,
    # so the production code cannot currently produce this combination. It is
    # kept here because ``_combine`` must reject a timeout regardless of the
    # exit status, and this is the shape that exercises that branch cleanly.
    result = CommandResult("pytest -q", None, "too slow", True, timed_out=True)
    verdict = TestResult(passed=True, summary="model says green", failures=[])

    combined = agent._combine(result, verdict)

    assert combined.passed is False
    assert "timed out" in combined.summary
    # ``_combine`` calls ``_failure_reason``, which for ``timed_out`` yields
    # f"Test command timed out after 120s: pytest -q". The verdict contributed no
    # failures of its own, so the reason is appended to an empty list: exactly one.
    assert combined.summary == (
        f"model says green\nTest command timed out after {TEST_TIMEOUT_SECONDS}s: pytest -q"
    )
    assert combined.failures == [
        f"Test command timed out after {TEST_TIMEOUT_SECONDS}s: pytest -q"
    ]
    assert len(combined.failures) == 1


def test_combine_passes_a_zero_exit_verdict_through_unchanged() -> None:
    agent = TesterAgent(client=_StubClient())
    result = CommandResult("pytest -q", 0, "", True)
    verdict = TestResult(passed=True, summary="model says green", failures=[])

    combined = agent._combine(result, verdict)

    assert combined is verdict
    assert combined.passed is True
    assert combined.summary == "model says green"
    assert combined.failures == []


# --------------------------------------------------------------------------
# _run_tests: the real subprocess path
# --------------------------------------------------------------------------


def _agent_for(tmp_path, test_command: str) -> TesterAgent:
    settings = get_settings().model_copy(
        update={"workspace": tmp_path, "test_command": test_command}
    )
    stub = _StubClient(reply=CLEAN_JSON, settings=settings)
    return TesterAgent(client=stub, settings=settings)


async def test_run_tests_reports_zero_exit_as_success(tmp_path) -> None:
    agent = _agent_for(tmp_path, "exit 0")
    result = await agent._run_tests()
    assert result.found is True
    assert result.returncode == 0
    assert result.succeeded is True


async def test_run_tests_reports_nonzero_exit_as_failure(tmp_path) -> None:
    """The key regression: a red exit code must not look like success."""
    agent = _agent_for(tmp_path, "exit 3")
    result = await agent._run_tests()
    assert result.found is True
    assert result.returncode == 3
    assert result.succeeded is False


async def test_run_tests_captures_stdout(tmp_path) -> None:
    agent = _agent_for(tmp_path, "echo hello")
    result = await agent._run_tests()
    assert result.found is True
    assert result.returncode == 0
    assert "hello" in result.output


async def test_run_tests_prefers_the_configured_command_over_detection(
    tmp_path, monkeypatch
) -> None:
    """A configured command short-circuits auto-detection entirely."""
    settings = get_settings().model_copy(
        update={"workspace": tmp_path, "test_command": "exit 0"}
    )
    # ``_run_tests`` returns the configured command's result as soon as it runs,
    # without ever looping over ``_TEST_CANDIDATES``. The only candidate here
    # exists and exits 1, so if detection did run, the result would be a failure.
    monkeypatch.setattr("furrow.agents.tester._TEST_CANDIDATES", ("/bin/false",))
    agent = TesterAgent(client=_StubClient(settings=settings), settings=settings)

    result = await agent._run_tests()

    assert result.succeeded is True
    assert result.command == "exit 0"
    assert result.returncode == 0
    assert result.found is True


async def test_run_reports_failure_for_a_nonzero_exit_code(tmp_path) -> None:
    settings = get_settings().model_copy(
        update={"workspace": tmp_path, "test_command": "exit 3"}
    )
    # The model wrongly claims success; the exit code must win.
    stub = _StubClient(reply=CLEAN_JSON, settings=settings)
    agent = TesterAgent(client=stub, settings=settings)

    result = await agent.run("goal", [])

    assert result.passed is False
    assert "exited with code 3" in result.summary
    assert len(stub.calls) == 1


async def test_run_reports_success_for_a_zero_exit_code(tmp_path) -> None:
    settings = get_settings().model_copy(
        update={"workspace": tmp_path, "test_command": "exit 0"}
    )
    stub = _StubClient(reply=CLEAN_JSON, settings=settings)
    agent = TesterAgent(client=stub, settings=settings)

    result = await agent.run("goal", [])

    assert result.passed is True
    assert len(stub.calls) == 1


# --------------------------------------------------------------------------
# End to end: a red suite can never be reported as a pass
# --------------------------------------------------------------------------


async def test_run_never_reports_a_pass_against_a_red_suite(tmp_path) -> None:
    """Even a JSON verdict that claims success cannot beat the exit code."""
    settings = get_settings().model_copy(
        update={"workspace": tmp_path, "test_command": "exit 3"}
    )
    stub = _StubClient(
        reply='{"passed": true, "summary": "looks fine to me", "failures": []}',
        settings=settings,
    )
    agent = TesterAgent(client=stub, settings=settings)

    result = await agent.run("goal", [])

    assert result.passed is False
    # ``_failure_reason`` for a non-timeout run is
    # f"Test command exited with code {result.returncode}: {result.command}", and
    # ``_combine`` appends that reason to the summary after the model's own.
    assert "exited with code 3" in result.summary
    assert result.summary == (
        "looks fine to me\nTest command exited with code 3: exit 3"
    )
    assert result.failures == ["Test command exited with code 3: exit 3"]
    assert len(stub.calls) == 1


async def test_run_never_reports_a_pass_from_the_prose_fallback(tmp_path) -> None:
    """The same holds when the reply is prose instead of JSON."""
    prose = "All tests passed."
    # Proves the test is meaningful: on its own this reply is not JSON, so
    # ``_verdict`` falls back to ``_text_indicates_pass``, which does read it as a
    # pass. Only the exit code can override that.
    assert _text_indicates_pass(prose) is True
    assert _verdict(prose).passed is True

    settings = get_settings().model_copy(
        update={"workspace": tmp_path, "test_command": "exit 3"}
    )
    stub = _StubClient(reply=prose, settings=settings)
    agent = TesterAgent(client=stub, settings=settings)

    result = await agent.run("goal", [])

    assert result.passed is False
    assert "exited with code 3" in result.summary
    assert result.summary == (
        "All tests passed.\nTest command exited with code 3: exit 3"
    )
    assert result.failures == ["Test command exited with code 3: exit 3"]
    assert len(stub.calls) == 1


# --------------------------------------------------------------------------
# No runner available
# --------------------------------------------------------------------------


async def test_run_tests_auto_detect_gives_up_when_nothing_can_spawn(
    tmp_path, monkeypatch
) -> None:
    settings = get_settings().model_copy(
        update={"workspace": tmp_path, "test_command": None}
    )
    monkeypatch.setattr(
        "furrow.agents.tester._TEST_CANDIDATES",
        ("definitely-not-a-real-binary-xyz --version",),
    )
    agent = TesterAgent(client=_StubClient(settings=settings), settings=settings)

    result = await agent._run_tests()

    assert result.found is False
    assert result.returncode is None
    assert result.output == "No test runner found."


async def test_run_without_a_runner_skips_the_llm(tmp_path, monkeypatch) -> None:
    settings = get_settings().model_copy(
        update={"workspace": tmp_path, "test_command": None}
    )
    monkeypatch.setattr(
        "furrow.agents.tester._TEST_CANDIDATES",
        ("definitely-not-a-real-binary-xyz --version",),
    )
    stub = _StubClient(reply=CLEAN_JSON, settings=settings)
    agent = TesterAgent(client=stub, settings=settings)

    result = await agent.run("goal", [])

    assert result.passed is False
    assert "No test runner" in result.summary
    assert len(result.failures) >= 1
    assert stub.calls == []


# --------------------------------------------------------------------------
# _failure helper
# --------------------------------------------------------------------------


def test_failure_helper_is_always_a_failure() -> None:
    result = _failure("it broke")
    assert result.passed is False
    assert result.summary == "it broke"
    assert result.failures == ["it broke"]


def test_test_timeout_seconds_is_pinned_to_the_documented_value() -> None:
    # Coupling point: ``TEST_TIMEOUT_SECONDS`` in furrow/agents/tester.py. README
    # documents that a run is killed after 120 seconds, so the constant is
    # asserted as a literal rather than merely "positive": a silent change to the
    # timeout must be a deliberate one, made in both places at once.
    assert TEST_TIMEOUT_SECONDS == 120
