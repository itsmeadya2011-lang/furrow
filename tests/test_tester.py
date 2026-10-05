"""Tests for :class:`furrow.agents.tester.TesterAgent`.

The tester is the last gate before furrow declares a goal verified, so its two
jobs are tested hard:

* the verdict fails **closed**.  The tester once decided the result with
  ``"passed" in response.lower()``, which reported a response truncated mid-object
  (and any prose mentioning "passed") as a green build.
* :meth:`TesterAgent._run_tests` walks a list of candidate runners, and only
  ``USABLE_EXIT_CODES`` may be read as a real result.  Everything else - a missing
  binary, a usage error, a timeout - is a diagnostic that moves on to the next
  candidate.  The subprocess layer is faked: no real runner is ever executed and
  no test ever sleeps.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from furrow.agents.prompts import TESTER_PROMPT
from furrow.agents.tester import (
    MAX_TASK_RESULT_CHARS,
    TEST_TIMEOUT_SECONDS,
    USABLE_EXIT_CODES,
    TesterAgent,
    _test_runners,
)
from furrow.config import Settings, TaskModel, TestResult
from tests.conftest import FakeLLM

PASSING_RESPONSE = '{"passed": true, "summary": "all green"}'
FAILING_RESPONSE = '{"passed": false, "summary": "boom", "failures": ["a", "b"]}'
TRUNCATED_RESPONSE = '{"passed": false, "summary": "3 of 12 tests failed'
UNUSABLE_EXIT = "No usable test runner found in the workspace."


class FakeProcess:
    """Minimal stand-in for :class:`asyncio.subprocess.Process`.

    With ``hang=True`` :meth:`communicate` blocks until it is cancelled, so the
    caller's timeout is the only thing that can end it.  That exercises the
    kill/reap path without a real child process and without sleeping.
    """

    def __init__(
        self,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
        hang: bool = False,
    ) -> None:
        self.returncode = returncode
        self.killed = False
        self.reaped = False
        self._stdout = stdout
        self._stderr = stderr
        self._hang = hang

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            await asyncio.Event().wait()  # nobody ever sets it
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        self.reaped = True
        return self.returncode


class RecordingExec:
    """Stand-in for ``asyncio.create_subprocess_exec``, driven one call at a time.

    ``behaviors`` is consumed in order: an exception is raised for that call, a
    :class:`FakeProcess` is returned for it.  Once it is empty ``default`` answers
    every later call, so "the first candidate is the only one that responds" stays
    explicit in the test that cares.
    """

    def __init__(self, behaviors: list[Any], default: Any) -> None:
        self._behaviors = list(behaviors)
        self._default = default
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    async def __call__(self, *cmd: str, **kwargs: Any) -> FakeProcess:
        self.calls.append((cmd, kwargs))
        behavior = self._behaviors.pop(0) if self._behaviors else self._default
        if isinstance(behavior, BaseException):
            raise behavior
        if isinstance(behavior, type) and issubclass(behavior, BaseException):
            raise behavior
        return behavior


def _patch_exec(
    monkeypatch: pytest.MonkeyPatch,
    behaviors: list[Any],
    default: Any = FileNotFoundError,
) -> RecordingExec:
    """Point ``asyncio.create_subprocess_exec`` at a recorder for one test."""
    fake = RecordingExec(behaviors, default)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake)
    return fake


def _tester(settings: Settings, response: str = PASSING_RESPONSE) -> TesterAgent:
    return TesterAgent(client=FakeLLM(response=response), settings=settings)


def _tasks() -> list[TaskModel]:
    return [
        TaskModel(
            id="1",
            description="Implement the JWT helpers",
            files=["src/auth.py"],
            status="completed",
            result="x" * 800,
        ),
        TaskModel(
            id="2",
            description="Expose the login route",
            files=["src/api/routes.py"],
            dependencies=["1"],
            status="pending",
            result=None,
        ),
    ]


def test_module_constants() -> None:
    assert TEST_TIMEOUT_SECONDS == 120
    assert USABLE_EXIT_CODES == (0, 1)
    assert MAX_TASK_RESULT_CHARS == 500


async def test_truncated_tester_json_is_not_a_pass(settings: Settings) -> None:
    """The headline regression: a cut-off response used to be read as green.

    A truncated response always contains the key name, so the old substring check
    (``"passed" in response.lower()``) turned "3 of 12 tests failed" into a pass.
    """
    result = _tester(settings)._verdict(TRUNCATED_RESPONSE)

    assert result.passed is False
    assert result.summary == "Could not parse tester JSON"
    assert result.failures == [TRUNCATED_RESPONSE]


@pytest.mark.parametrize(
    "response",
    [
        pytest.param("1 failed, 2 passed", id="counts-mention-passed"),
        pytest.param("the suite passed but the build is red", id="prose-mentions-passed"),
        pytest.param("passed", id="bare-word"),
    ],
)
async def test_prose_mentioning_passed_is_not_a_pass(settings: Settings, response: str) -> None:
    result = _tester(settings)._verdict(response)

    assert result.passed is False
    assert result.failures == [response]


async def test_passing_tester_json_is_a_pass(settings: Settings) -> None:
    result = _tester(settings)._verdict(PASSING_RESPONSE)

    assert result.passed is True
    assert result.summary == "all green"
    assert result.failures == []


async def test_failing_tester_json_preserves_the_failure_list(settings: Settings) -> None:
    result = _tester(settings)._verdict(FAILING_RESPONSE)

    assert result.passed is False
    assert result.summary == "boom"
    assert result.failures == ["a", "b"]


async def test_fenced_tester_json_parses(settings: Settings) -> None:
    response = '```json\n{"passed": true, "summary": "all green", "failures": []}\n```'

    result = _tester(settings)._verdict(response)

    assert result.passed is True
    assert result.summary == "all green"
    assert result.failures == []


async def test_prose_wrapped_tester_json_parses(settings: Settings) -> None:
    response = f"Here is my judgement:\n{FAILING_RESPONSE}\nAnything else to check?"

    result = _tester(settings)._verdict(response)

    assert result.passed is False
    assert result.failures == ["a", "b"]


async def test_tester_json_missing_summary_fails_closed(settings: Settings) -> None:
    """A verdict with no summary is unusable and must not raise out of the agent."""
    response = '{"passed": true}'

    result = _tester(settings)._verdict(response)

    assert result.passed is False
    assert result.summary == "Could not parse tester JSON"
    assert result.failures == [response]


@pytest.mark.parametrize(
    "response",
    [
        pytest.param("", id="empty"),
        pytest.param("   \n\t ", id="whitespace"),
    ],
)
async def test_empty_response_fails_closed(settings: Settings, response: str) -> None:
    result = _tester(settings)._verdict(response)

    assert result.passed is False
    assert result.summary == "Could not parse tester JSON"
    assert result.failures == [response]


@pytest.mark.parametrize(
    "response",
    [
        pytest.param("null", id="json-null"),
        pytest.param("[]", id="json-array"),
        pytest.param("42", id="json-number"),
        pytest.param("true", id="json-bool"),
        pytest.param('"a verdict"', id="json-string"),
        pytest.param("{unclosed", id="unterminated-object"),
        pytest.param('{"passed":', id="cut-off-object"),
        pytest.param("not json at all", id="prose"),
        pytest.param("```", id="lone-fence"),
        pytest.param("```json\n{oops\n```", id="fenced-garbage"),
        pytest.param(
            '{"passed": true, "summary": "ok", "failures": "not a list"}',
            id="wrong-field-type",
        ),
        pytest.param('{"passed": null, "summary": "ok"}', id="null-passed"),
        pytest.param("\u00ff\u00fe not text", id="non-ascii-garbage"),
    ],
)
async def test_verdict_never_raises_and_never_passes(settings: Settings, response: str) -> None:
    result = _tester(settings)._verdict(response)

    assert isinstance(result, TestResult)
    assert result.passed is False
    assert result.summary == "Could not parse tester JSON"
    assert result.failures == [response]


async def test_run_tests_uses_the_workspace_as_cwd(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _patch_exec(monkeypatch, [FakeProcess(returncode=0, stdout=b"ok")])

    exit_code, output = await _tester(settings)._run_tests()

    assert (exit_code, output) == (0, "ok")
    assert len(fake.calls) == 1
    cmd, kwargs = fake.calls[0]
    assert cmd[0] == "pytest"
    assert kwargs["cwd"] == settings.workspace
    assert kwargs["cwd"] == Path(settings.workspace)
    assert kwargs["stdout"] == asyncio.subprocess.PIPE
    assert kwargs["stderr"] == asyncio.subprocess.PIPE


async def test_run_tests_falls_through_to_the_next_candidate(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing first runner is a diagnostic, not a failure."""
    fake = _patch_exec(
        monkeypatch,
        [FileNotFoundError(2, "No such file or directory")],
        FakeProcess(returncode=0, stdout=b"second runner answered"),
    )

    exit_code, output = await _tester(settings)._run_tests()

    assert exit_code == 0
    assert output == "second runner answered"
    assert len(fake.calls) == 2
    assert fake.calls[0][0][0] == "pytest"
    assert fake.calls[1][0][0] != "pytest"


async def test_run_tests_uses_the_first_usable_exit_code(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unusable first runner is skipped; the next usable one wins."""
    fake = _patch_exec(
        monkeypatch,
        [FakeProcess(returncode=4), FakeProcess(returncode=0, stdout=b"later runner answered")],
    )

    exit_code, output = await _tester(settings)._run_tests()

    assert exit_code == 0
    assert output == "later runner answered"
    assert len(fake.calls) == 2
    assert fake.calls[0][0][0] == "pytest"
    assert fake.calls[1][0][0] != "pytest"


@pytest.mark.parametrize("code", [4, 5], ids=["usage-error", "nothing-collected"])
async def test_run_tests_skips_unusable_exit_codes(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    fake = _patch_exec(monkeypatch, [FakeProcess(returncode=code)])

    exit_code, report = await _tester(settings)._run_tests()

    assert exit_code is None
    assert f"unusable exit code {code}" in report
    assert len(fake.calls) == len(_test_runners())
    for cmd, _kwargs in fake.calls:
        assert " ".join(cmd) in report


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(0, id="runner-succeeded"),
        pytest.param(1, id="tests-ran-and-failed"),
    ],
)
async def test_run_tests_accepts_usable_exit_codes(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    fake = _patch_exec(monkeypatch, [FakeProcess(returncode=code, stdout=b"out", stderr=b"err")])

    exit_code, output = await _tester(settings)._run_tests()

    assert exit_code == code
    assert output == "outerr"
    assert len(fake.calls) == 1


async def test_run_tests_replaces_invalid_utf8(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test output is bytes; undecodable bytes must not blow up the agent."""
    bad = b"\xff\xfe"
    _patch_exec(
        monkeypatch,
        [FakeProcess(returncode=1, stdout=b"before" + bad, stderr=bad + b"after")],
    )

    exit_code, output = await _tester(settings)._run_tests()

    assert exit_code == 1
    assert isinstance(output, str)
    assert output.startswith("before")
    assert output.endswith("after")
    assert "\ufffd" in output


async def test_run_tests_kills_and_reaps_a_timed_out_child(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hung runner must be killed *and* awaited, or it is left as a zombie."""
    timeout = 0.01
    monkeypatch.setattr("furrow.agents.tester.TEST_TIMEOUT_SECONDS", timeout)
    hangy = FakeProcess(hang=True)
    _patch_exec(monkeypatch, [hangy])

    exit_code, report = await _tester(settings)._run_tests()

    assert hangy.killed is True
    assert hangy.reaped is True
    assert exit_code is None
    assert f"timed out after {timeout}s" in report


async def test_run_tests_reports_diagnostics_when_nothing_runs(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _patch_exec(monkeypatch, [FileNotFoundError(2, "No such file or directory")])

    exit_code, report = await _tester(settings)._run_tests()

    assert exit_code is None
    assert report.startswith(UNUSABLE_EXIT)
    assert "Diagnostics:" in report
    assert fake.calls
    for cmd, _kwargs in fake.calls:
        assert f"- {' '.join(cmd)}: unusable (" in report


async def test_run_tests_reports_a_missing_workspace(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "never-created"
    agent = _tester(settings.model_copy(update={"workspace": missing}))
    fake = _patch_exec(monkeypatch, [], FakeProcess(returncode=0))

    exit_code, report = await agent._run_tests()

    assert exit_code is None
    assert f"Workspace directory does not exist: {missing}" in report
    assert fake.calls == []


async def test_run_prompt_carries_the_exit_code_and_task_digest(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_exec(monkeypatch, [FakeProcess(returncode=1, stdout=b"1 failed, 3 passed")])
    fake_llm = FakeLLM(response=PASSING_RESPONSE)
    agent = TesterAgent(client=fake_llm, settings=settings)

    result = await agent.run("Add JWT login", _tasks())

    prompt, system, model = fake_llm.calls[0]
    assert len(fake_llm.calls) == 1
    assert system == TESTER_PROMPT
    assert model == settings.tester_model
    assert "Goal: Add JWT login" in prompt
    assert "Test runner exit code: 1 - tests ran and at least one failed" in prompt
    assert "1 failed, 3 passed" in prompt
    assert "id=1" in prompt
    assert "status=completed" in prompt
    assert "depends_on=none" in prompt
    assert "Implement the JWT helpers" in prompt
    assert "id=2" in prompt
    assert "status=pending" in prompt
    assert "depends_on=1" in prompt
    assert "Expose the login route" in prompt
    # Long results are truncated, and the marker is what a reader looks for.
    assert "x" * MAX_TASK_RESULT_CHARS in prompt
    assert "x" * (MAX_TASK_RESULT_CHARS + 1) not in prompt
    assert "... [truncated]" in prompt
    assert "(no result reported)" in prompt
    assert result.passed is True


async def test_run_prompt_reports_an_unavailable_exit_code(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_exec(monkeypatch, [FileNotFoundError(2, "No such file or directory")])
    fake_llm = FakeLLM(response=FAILING_RESPONSE)
    agent = TesterAgent(client=fake_llm, settings=settings)

    result = await agent.run("Add JWT login", _tasks())

    prompt = fake_llm.calls[0][0]
    assert "unavailable - no test runner could be executed" in prompt
    assert UNUSABLE_EXIT in prompt
    assert result.passed is False
    assert result.failures == ["a", "b"]


async def test_run_with_no_tasks_uses_a_placeholder(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_exec(monkeypatch, [FakeProcess(returncode=0, stdout=b"")])
    fake_llm = FakeLLM(response=PASSING_RESPONSE)
    agent = TesterAgent(client=fake_llm, settings=settings)

    result = await agent.run("Add JWT login", [])

    prompt = fake_llm.calls[0][0]
    assert "(no tasks reported)" in prompt
    assert "id=" not in prompt
    assert result.passed is True


async def test_run_fails_closed_on_a_truncated_tester_response(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression, end to end: a cut-off verdict must not reach the caller as a pass."""
    _patch_exec(monkeypatch, [FakeProcess(returncode=1, stdout=b"3 of 12 tests failed")])
    agent = TesterAgent(client=FakeLLM(response=TRUNCATED_RESPONSE), settings=settings)

    result = await agent.run("Add JWT login", _tasks())

    assert result.passed is False
    assert result.failures == [TRUNCATED_RESPONSE]


async def test_run_returns_the_validated_failure_list(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_exec(monkeypatch, [FakeProcess(returncode=1, stdout=b"boom")])
    response = json.dumps(
        {"passed": False, "summary": "1 of 4 tests failed", "failures": ["tests/test_a.py::test_x"]}
    )
    agent = TesterAgent(client=FakeLLM(response=response), settings=settings)

    result = await agent.run("Add JWT login", _tasks())

    assert result.passed is False
    assert result.summary == "1 of 4 tests failed"
    assert result.failures == ["tests/test_a.py::test_x"]


async def test_run_fails_closed_when_the_suite_cannot_be_run(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error escaping the runner loop fails closed and never asks for a verdict."""
    _patch_exec(monkeypatch, [asyncio.TimeoutError()])
    fake_llm = FakeLLM(response=PASSING_RESPONSE)
    agent = TesterAgent(client=fake_llm, settings=settings)

    result = await agent.run("Add JWT login", _tasks())

    assert result.passed is False
    assert result.summary.startswith("Could not run the test suite: ")
    assert result.failures == [result.summary]
    assert fake_llm.calls == []
