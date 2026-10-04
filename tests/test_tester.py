from __future__ import annotations

import asyncio
from typing import Any

import pytest

from furrow.agents import tester as tester_module
from furrow.agents.tester import TesterAgent as TesterAgentUnderTest
from furrow.config import TaskModel


class FakeSettings:
    tester_model = "fake-tester-model"


class FakeClient:
    """Minimal stand-in for ``LLMClient``."""

    def __init__(self, response: str) -> None:
        self.settings = FakeSettings()
        self.response = response
        self.prompts: list[str] = []
        self.models: list[str | None] = []
        self.calls = 0

    async def complete(self, prompt: str, model: str | None = None) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        self.models.append(model)
        return self.response


class FakeProcess:
    """Fake ``asyncio.subprocess.Process`` with controllable completion."""

    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        hang: bool = False,
        partial: bytes = b"",
        hang_calls: int = 1,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self._hang_calls = hang_calls
        self.communicate_calls = 0
        self._partial = partial
        self.killed = 0
        self.waited = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicate_calls += 1
        if self._hang and self.communicate_calls <= self._hang_calls:
            await asyncio.Event().wait()  # cancelled by the agent's wait_for timeout
            raise AssertionError("unreachable")
        if self._partial:
            return self._partial, b""
        return self._stdout, self._stderr

    async def wait(self) -> int:
        self.waited += 1
        if self.returncode is None:
            self.returncode = -9
        return self.returncode

    def kill(self) -> None:
        self.killed += 1


def install_process(monkeypatch: pytest.MonkeyPatch, proc: FakeProcess) -> FakeProcess:
    async def fake_create(*cmd: str, **kwargs: Any) -> FakeProcess:
        proc.command = list(cmd)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    return proc


def install_missing_runner(monkeypatch: pytest.MonkeyPatch, missing: set[str]) -> list[str]:
    attempted: list[str] = []

    async def fake_create(*cmd: str, **kwargs: Any) -> FakeProcess:
        attempted.append(cmd[0])
        if cmd[0] in missing or len(cmd) > 1:
            raise FileNotFoundError(2, "No such file or directory", cmd[0])
        raise AssertionError(f"unexpected runner probe: {cmd}")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    return attempted


def make_tasks() -> list[TaskModel]:
    return [
        TaskModel(
            id="1",
            description="Add the parser",
            files=["app/parse.py"],
            dependencies=[],
            status="completed",
            result=None,
        ),
        TaskModel(
            id="2",
            description="Wire up the CLI",
            files=["app/cli.py"],
            dependencies=["1"],
            status="completed",
            result=None,
        ),
    ]


async def test_nonzero_returncode_fails_and_surfaces_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = install_process(
        monkeypatch,
        FakeProcess(stdout=b"3 failed, 12 passed\n", stderr=b"AssertionError\n", returncode=1),
    )
    client = FakeClient('{"passed": false, "summary": "3 tests failed", "failures": ["boom"]}')

    result = await TesterAgentUnderTest(client=client).run("do the thing", make_tasks())

    assert result.passed is False
    blob = " ".join([result.summary, *result.failures])
    assert "3 failed" in blob or "boom" in blob
    assert proc.killed == 0


async def test_zero_returncode_takes_passed_path(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = install_process(monkeypatch, FakeProcess(stdout=b"12 passed\n", returncode=0))
    client = FakeClient('{"passed": true, "summary": "All green", "failures": []}')

    result = await TesterAgentUnderTest(client=client).run("do the thing", make_tasks())

    assert result.passed is True
    assert result.summary == "All green"
    assert result.failures == []
    assert proc.waited == 0
    # the run status and the planned tasks reach the judge prompt
    assert "Test runner status: passed (returncode=0)" in client.prompts[0]
    assert "- 1: Add the parser" in client.prompts[0]
    assert "- 2: Wire up the CLI" in client.prompts[0]


async def test_timeout_kills_waits_and_surfaces_partial_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = install_process(
        monkeypatch, FakeProcess(hang=True, partial=b"collected 5 items ... hung here\n")
    )
    client = FakeClient("not json at all")

    agent = TesterAgentUnderTest(client=client)
    run = await agent._run_tests(timeout=0.01)

    assert run.status == "timeout"
    assert run.passed is False
    assert proc.killed == 1, "timeout must kill the process"
    assert proc.waited == 1, "timeout must await proc.wait() so no zombie survives"
    assert "hung here" in run.output


async def test_timeout_verdict_reaches_result_and_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    install_process(monkeypatch, FakeProcess(hang=True, partial=b"stuck on flaky test\n"))
    client = FakeClient('{"passed": true, "summary": "looks good", "failures": []}')

    agent = TesterAgentUnderTest(client=client, timeout=0.01)
    result = await agent.run("do the thing", make_tasks())

    assert result.passed is False, "a timed-out runner can never be a pass"
    assert "timeout" in client.prompts[0]
    assert "stuck on flaky test" in result.failures[0]


async def test_no_runner_skips_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    attempted = install_missing_runner(monkeypatch, missing={"pytest", "npm", "pnpm", "yarn"})
    client = FakeClient('{"passed": true, "summary": "should not be used", "failures": []}')

    result = await TesterAgentUnderTest(client=client).run("do the thing", make_tasks())

    assert result.passed is False
    assert "No test runner found" in result.summary
    assert "pytest -q" in result.summary
    assert "cargo test -q" in result.summary
    assert result.failures
    assert client.calls == 0, "LLM must not be called when no runner exists"
    assert len(attempted) == 7


async def test_markdown_fenced_json_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    install_process(monkeypatch, FakeProcess(stdout=b"ok\n", returncode=0))
    client = FakeClient(
        '```json\n{"passed": true, "summary": "Fenced but valid", "failures": []}\n```'
    )

    result = await TesterAgentUnderTest(client=client).run("do the thing", make_tasks())

    assert result.passed is True
    assert result.summary == "Fenced but valid"
    assert result.failures == []


async def test_ambiguous_output_never_false_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    install_process(monkeypatch, FakeProcess(stdout=b"2 passed, 1 failed\n", returncode=1))
    client = FakeClient(
        "I could not tell whether the suite passed: the report says 1 test did not pass."
    )

    result = await TesterAgentUnderTest(client=client).run("do the thing", make_tasks())

    assert result.passed is False
    assert "I could not tell" in result.summary
    assert result.failures


async def test_unambiguous_passed_token_in_prose(monkeypatch: pytest.MonkeyPatch) -> None:
    install_process(monkeypatch, FakeProcess(stdout=b"12 passed\n", returncode=0))
    client = FakeClient("Verdict: PASSED, everything is green.")

    result = await TesterAgentUnderTest(client=client).run("do the thing", make_tasks())

    assert result.passed is True
    assert "Verdict: PASSED" in result.summary


async def test_tasks_are_rendered_into_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    install_process(monkeypatch, FakeProcess(stdout=b"ok\n", returncode=0))
    client = FakeClient('{"passed": true, "summary": "ok", "failures": []}')

    await TesterAgentUnderTest(client=client).run("ship it", [])

    assert "- (no tasks were planned)" in client.prompts[0]
    assert client.models == ["fake-tester-model"]


def test_test_run_dataclass_fields() -> None:
    run = tester_module.TestRun(status="passed", passed=True, output="ok", returncode=0)
    assert run.status == "passed"
    assert run.tried == []
    assert run.command is None
    assert tester_module.TestRun(status="no_runner").tried == []
