"""Tests for :class:`~furrow.agents.tester.TesterAgent` that need no LLM.

``_run_tests`` shells out to a real test runner, so these tests build throwaway
pytest suites in ``tmp_path`` and inspect the return code. The LLM-judging path
(``_summarize_failures``) is covered with a stub client that returns canned JSON.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from furrow.agents import tester as tester_module
from furrow.agents.tester import NO_TEST_RUNNER
from furrow.config import Settings, TaskModel

PASSING_SUITE = """\
def test_addition():
    assert 1 + 1 == 2


def test_truthiness():
    assert not []
"""

FAILING_SUITE = """\
def test_addition():
    assert 1 + 1 == 3
"""


class FakeClient:
    """Minimal client: only ``settings.workspace`` and canned ``complete`` output."""

    def __init__(self, workspace: Path, responses: list[str] | None = None) -> None:
        self.settings = Settings(_env_file=None, workspace=workspace, tester_model="fake-tester")
        self.responses = list(responses or [])
        self.prompts: list[str] = []
        self.models: list[str | None] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        self.models.append(model)
        if not self.responses:
            raise AssertionError("the tester called the LLM unexpectedly")
        return self.responses.pop(0)


class BrokenClient(FakeClient):
    """Client whose completions always fail, e.g. a missing API key."""

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        raise RuntimeError("no API key")


class HangingProcess:
    """Stand-in for an ``asyncio.subprocess.Process`` that never finishes."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return -9


def write_suite(workspace: Path, body: str) -> Path:
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    suite = tests_dir / "test_generated.py"
    suite.write_text(body, encoding="utf-8")
    # Pin the rootdir so the child pytest run never walks up into this repository.
    (workspace / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    return suite


def make_agent(
    workspace: Path, responses: list[str] | None = None
) -> tuple[tester_module.TesterAgent, FakeClient]:
    client = FakeClient(workspace, responses)
    return tester_module.TesterAgent(client=client), client  # type: ignore[arg-type]


async def test_no_runner_returns_passed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Nothing in TEST_COMMANDS can be spawned, so every candidate raises
    # FileNotFoundError and the agent falls back to the sentinel return code.
    monkeypatch.setattr(tester_module, "TEST_COMMANDS", [["furrow-no-such-test-runner"]])
    agent, client = make_agent(tmp_path)

    returncode, output = await agent._run_tests()
    assert returncode == NO_TEST_RUNNER
    assert "No test runner" in output

    result = await agent.run("build a thing", [])
    assert result.passed is True
    assert "no test runner" in result.summary.lower()
    assert result.failures == []
    assert client.prompts == []


async def test_passing_returncode(tmp_path: Path) -> None:
    write_suite(tmp_path, PASSING_SUITE)
    agent, client = make_agent(tmp_path)

    returncode, output = await agent._run_tests()
    assert returncode == 0, output
    assert "2 passed" in output

    result = await agent.run("build a thing", [])
    assert result.passed is True
    assert result.failures == []
    assert client.prompts == []


async def test_failing_returncode(tmp_path: Path) -> None:
    write_suite(tmp_path, FAILING_SUITE)
    canned = json.dumps({"summary": "1 test failed", "failures": ["tests/test_generated.py::x"]})
    agent, client = make_agent(tmp_path, responses=[canned])

    returncode, output = await agent._run_tests()
    assert returncode == 1, output
    assert "1 failed" in output

    result = await agent.run("build a thing", [TaskModel(id="1", description="d", files=["a.py"])])
    assert result.passed is False
    assert result.summary == "1 test failed"
    assert result.failures == ["tests/test_generated.py::x"]
    # The judging prompt carries the goal, the exit code, the task files and the output.
    assert client.models == ["fake-tester"]
    assert "1 (non-zero means the suite failed)" in client.prompts[0]
    assert "build a thing" in client.prompts[0]
    assert "a.py" in client.prompts[0]
    assert "1 failed" in client.prompts[0]


async def test_failing_suite_without_llm_judgement(tmp_path: Path) -> None:
    write_suite(tmp_path, FAILING_SUITE)
    agent = tester_module.TesterAgent(client=BrokenClient(tmp_path))  # type: ignore[arg-type]

    result = await agent.run("build a thing", [])
    assert result.passed is False
    assert "1 failed" in result.summary
    assert result.failures == []


async def test_failing_suite_with_unparsable_llm_reply(tmp_path: Path) -> None:
    write_suite(tmp_path, FAILING_SUITE)
    agent, _ = make_agent(tmp_path, responses=["not json at all"])

    result = await agent.run("build a thing", [])
    assert result.passed is False
    assert result.summary == "not json at all"
    assert result.failures == []


async def test_runner_errors_are_reported_as_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def explode(*cmd: str, **kwargs: Any) -> Any:
        raise OSError("workspace is gone")

    monkeypatch.setattr(tester_module, "TEST_COMMANDS", [["pytest"]])
    monkeypatch.setattr(tester_module.asyncio, "create_subprocess_exec", explode)
    agent, _ = make_agent(tmp_path)

    result = await agent.run("build a thing", [])
    assert result.passed is False
    assert result.summary == "workspace is gone"
    assert result.failures == ["workspace is gone"]


async def test_runner_timeout_is_reported_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[HangingProcess] = []

    async def hangs_forever(*cmd: str, **kwargs: Any) -> HangingProcess:
        process = HangingProcess()
        created.append(process)
        return process

    monkeypatch.setattr(tester_module, "TEST_COMMANDS", [["pytest"]])
    monkeypatch.setattr(tester_module, "TEST_TIMEOUT", 0)
    monkeypatch.setattr(tester_module.asyncio, "create_subprocess_exec", hangs_forever)
    canned = json.dumps({"summary": "timed out", "failures": ["pytest -q"]})
    agent, _ = make_agent(tmp_path, responses=[canned])

    returncode, output = await agent._run_tests()
    assert returncode == 1
    assert "pytest" in output
    assert "timed out after 0s" in output
    assert created and created[0].killed is True

    result = await agent.run("build a thing", [])
    assert result.passed is False
    assert result.failures == ["pytest -q"]


def test_task_files_deduplicates_across_tasks() -> None:
    agent, _ = make_agent(Path("/tmp"))
    tasks = [
        TaskModel(id="1", description="a", files=["a.py", "b.py"]),
        TaskModel(id="2", description="b", files=["b.py"]),
        TaskModel(id="3", description="c"),
    ]

    assert sorted(agent._task_files(tasks)) == ["a.py", "b.py"]
