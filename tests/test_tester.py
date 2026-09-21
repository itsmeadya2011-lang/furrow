import json

import pytest

from furrow.agents import tester as tester_module
from furrow.agents.tester import TesterAgent, TestExecution
from furrow.config import Settings


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    async def complete(self, prompt, model=None):
        self.prompts.append(prompt)
        return self.response


class FakeProcess:
    def __init__(self, returncode=0, stdout=b"", stderr=b"", timeout=False):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timeout = timeout
        self.killed = False
        self.waited = False

    async def communicate(self):
        if self.timeout:
            self.timeout = False
            raise TimeoutError
        return self.stdout, self.stderr

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        self.waited = True
        self.returncode = -9


def settings(workspace, timeout=1.0):
    return Settings(workspace=workspace, test_timeout=timeout)


@pytest.mark.asyncio
async def test_nonzero_return_code_is_authoritative(monkeypatch, tmp_path):
    process = FakeProcess(returncode=1, stdout=b"boom")
    client = FakeClient(
        json.dumps({"passed": True, "summary": "llm says pass", "failures": []})
    )

    async def create_process(*command, cwd, stdout, stderr):
        assert cwd == tmp_path
        return process

    monkeypatch.setattr(tester_module.asyncio, "create_subprocess_exec", create_process)
    tester = TesterAgent(client=client, settings=settings(tmp_path))

    result = await tester.run("goal", [])

    assert result.passed is False
    assert result.summary == "llm says pass"
    assert any("boom" in failure for failure in result.failures)


@pytest.mark.asyncio
async def test_zero_return_code_is_authoritative(monkeypatch, tmp_path):
    process = FakeProcess(returncode=0, stdout=b"ok")
    client = FakeClient(
        json.dumps({"passed": False, "summary": "llm says fail", "failures": ["wrong"]})
    )

    async def create_process(*command, cwd, stdout, stderr):
        return process

    monkeypatch.setattr(tester_module.asyncio, "create_subprocess_exec", create_process)
    tester = TesterAgent(client=client, settings=settings(tmp_path))

    result = await tester.run("goal", [])

    assert result.passed is True
    assert result.summary == "llm says fail"
    assert result.failures == []


@pytest.mark.asyncio
async def test_timeout_kills_process_and_reports_failure(monkeypatch, tmp_path):
    process = FakeProcess(timeout=True, stdout=b"partial")
    client = FakeClient(
        json.dumps({"passed": True, "summary": "llm says pass", "failures": []})
    )

    async def create_process(*command, cwd, stdout, stderr):
        return process

    monkeypatch.setattr(tester_module.asyncio, "create_subprocess_exec", create_process)
    tester = TesterAgent(client=client, settings=settings(tmp_path, timeout=0.01))

    result = await tester.run("goal", [])

    assert result.passed is False
    assert process.killed is True
    assert process.waited is True
    assert any("timed out" in failure.lower() for failure in result.failures)


@pytest.mark.asyncio
async def test_missing_runner_reports_failure(monkeypatch, tmp_path):
    client = FakeClient(
        json.dumps({"passed": True, "summary": "llm says pass", "failures": []})
    )

    async def create_process(*command, cwd, stdout, stderr):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(tester_module.asyncio, "create_subprocess_exec", create_process)
    tester = TesterAgent(client=client, settings=settings(tmp_path))

    result = await tester.run("goal", [])

    assert result.passed is False
    assert "No test runner found" in result.summary
    assert result.failures


@pytest.mark.asyncio
async def test_process_cleanup_errors_are_safe(monkeypatch, tmp_path):
    class CleanupFailureProcess(FakeProcess):
        def kill(self):
            self.killed = True
            raise ProcessLookupError

    process = CleanupFailureProcess(timeout=True)
    client = FakeClient(json.dumps({"summary": "timeout", "failures": []}))

    async def create_process(*command, cwd, stdout, stderr):
        return process

    monkeypatch.setattr(tester_module.asyncio, "create_subprocess_exec", create_process)
    tester = TesterAgent(client=client, settings=settings(tmp_path, timeout=0.01))

    result = await tester.run("goal", [])

    assert result.passed is False
    assert process.killed is True


def test_test_execution_remains_output_compatible():
    execution = TestExecution(returncode=1, output="failure output")

    assert str(execution) == "failure output"
