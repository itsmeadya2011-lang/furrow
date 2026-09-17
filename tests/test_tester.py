import asyncio

import pytest

from furrow.agents import tester as tester_module
from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TaskModel


class FakeProcess:
    def __init__(
        self,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
        delay: bool = False,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.delay = delay
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self.delay:
            await asyncio.sleep(1)
        return self.stdout, self.stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return self.returncode


def make_subprocess_factory(results):
    calls = []
    processes = []

    async def create_subprocess_exec(*args, **kwargs):
        calls.append((args, kwargs))
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        process = FakeProcess(**result)
        processes.append(process)
        return process

    return create_subprocess_exec, calls, processes


def task() -> TaskModel:
    return TaskModel(id="task", description="task")


@pytest.mark.asyncio
async def test_runs_validators_in_workspace_and_uses_return_codes(
    tmp_path, monkeypatch
) -> None:
    factory, calls, _processes = make_subprocess_factory(
        [
            {"returncode": 0, "stdout": b"pytest ok\n"},
            {"returncode": 0, "stdout": b"ruff ok\n"},
            {"returncode": 0, "stdout": b"mypy ok\n"},
        ]
    )
    monkeypatch.setattr(tester_module.asyncio, "create_subprocess_exec", factory)
    agent = TesterAgent(settings=Settings(workspace=tmp_path))

    result = await agent.run("goal", [task()])

    assert result.passed is True
    assert result.failures == []
    assert "pytest -q passed" in result.summary
    assert [call[0] for call in calls] == [
        ("pytest", "-q"),
        ("ruff", "check", "."),
        ("mypy", "."),
    ]
    assert all(call[1]["cwd"] == tmp_path for call in calls)


@pytest.mark.asyncio
async def test_nonzero_return_code_is_a_failure(tmp_path, monkeypatch) -> None:
    factory, _, _processes = make_subprocess_factory(
        [
            {"returncode": 1, "stdout": b"assertion failed\n", "stderr": b"details\n"},
            {"returncode": 0},
            {"returncode": 0},
        ]
    )
    monkeypatch.setattr(tester_module.asyncio, "create_subprocess_exec", factory)
    agent = TesterAgent(settings=Settings(workspace=tmp_path))

    result = await agent.run("goal", [task()])

    assert result.passed is False
    assert "pytest -q failed with return code 1" in result.failures[0]
    assert "assertion failed" in result.failures[0]


@pytest.mark.asyncio
async def test_missing_required_runner_fails_but_optional_tools_are_neutral(
    tmp_path, monkeypatch
) -> None:
    factory, _, _processes = make_subprocess_factory(
        [
            FileNotFoundError("pytest"),
            FileNotFoundError("ruff"),
            FileNotFoundError("mypy"),
        ]
    )
    monkeypatch.setattr(tester_module.asyncio, "create_subprocess_exec", factory)
    agent = TesterAgent(settings=Settings(workspace=tmp_path))

    result = await agent.run("goal", [task()])

    assert result.passed is False
    assert "pytest -q executable was not found" in result.failures[0]
    assert "ruff" not in " ".join(result.failures)
    assert "mypy" not in " ".join(result.failures)
    assert "Skipped unavailable optional tools" in result.summary


@pytest.mark.asyncio
async def test_timeout_is_a_failure_and_process_is_killed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tester_module, "TEST_TIMEOUT_SECONDS", 0.01)
    factory, _, processes = make_subprocess_factory(
        [
            {"delay": True},
            {"returncode": 0},
            {"returncode": 0},
        ]
    )
    monkeypatch.setattr(tester_module.asyncio, "create_subprocess_exec", factory)
    agent = TesterAgent(settings=Settings(workspace=tmp_path))

    result = await agent.run("goal", [task()])

    assert result.passed is False
    assert "pytest -q timed out" in result.failures[0]
    assert processes[0].killed is True


@pytest.mark.asyncio
async def test_output_is_capped_and_truncation_is_reported(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tester_module, "MAX_OUTPUT_BYTES", 64)
    monkeypatch.setattr(tester_module, "OUTPUT_SNIPPET_BYTES", 32)
    factory, _, processes = make_subprocess_factory(
        [
            {"returncode": 1, "stdout": b"x" * 100},
            {"returncode": 0},
            {"returncode": 0},
        ]
    )
    monkeypatch.setattr(tester_module.asyncio, "create_subprocess_exec", factory)
    agent = TesterAgent(settings=Settings(workspace=tmp_path))

    result = await agent.run("goal", [task()])

    assert result.passed is False
    assert "[output truncated]" in result.failures[0]
    assert "Output was truncated" in result.summary
