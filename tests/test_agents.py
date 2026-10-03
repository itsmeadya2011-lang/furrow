from __future__ import annotations

import asyncio
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent, TestRun
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel


@dataclass
class FakeClient:
    """Fake LLM client for deterministic testing."""
    settings: Settings
    responses: list[str]
    call_count: int = 0
    last_prompt: str = ""
    last_system: str = ""
    last_model: str | None = None

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system = system
        self.last_model = model
        if self.responses:
            return self.responses.pop(0)
        return "{}"


def make_settings(workspace: Path | str | None = None, **overrides: Any) -> Settings:
    """Create a Settings instance with test defaults.

    ``workspace`` becomes the subprocess ``cwd`` of ``TesterAgent._exec``, so it
    has to be an existing directory: with a missing one the spawn raises
    FileNotFoundError, the TestRun comes back with ``returncode=None`` and every
    exit-code assertion silently degrades. It defaults to the current working
    directory; tests that really execute a command pass the ``tmp_path`` fixture.
    """
    resolved = Path.cwd() if workspace is None else Path(workspace)
    assert resolved.is_dir(), f"workspace must be an existing directory: {resolved}"

    defaults: dict[str, Any] = {
        "provider": "anthropic",
        "planner_model": "test-planner-model",
        "worker_model": "test-worker-model",
        "tester_model": "test-tester-model",
        "test_command": None,
        "test_timeout": 10,
        "workspace": resolved,
    }
    defaults.update(overrides)
    return Settings(**defaults)


# Commands that work as actual executables (not shell builtins). They run this
# interpreter, so nothing beyond CPython has to be installed, and they exit
# immediately, which keeps the tests fast and deterministic. sys.executable is
# quoted because _run_tests splits test_command with shlex.split.
PYTHON_PASS = f"{shlex.quote(sys.executable)} -c \"print('all good')\""
PYTHON_FAIL = f"{shlex.quote(sys.executable)} -c \"import sys; sys.exit(1)\""
PYTHON_HELLO = f"{shlex.quote(sys.executable)} -c \"print('hello world')\""
PYTHON_OUTPUT = f"{shlex.quote(sys.executable)} -c \"print('test output here')\""


class _HangingProcess:
    """Stand-in for ``asyncio.subprocess.Process`` that never finishes on its own.

    Only the spawn is faked, so ``TesterAgent._exec`` still runs its real timeout
    path (wait_for on communicate, then kill and drain) without the test waiting
    on a genuinely slow process.
    """

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        while not self.killed:
            await asyncio.sleep(0)
        self.returncode = -9
        return b"partial test output", b""

    def kill(self) -> None:
        self.killed = True


class TestPlannerAgent:
    """Tests for PlannerAgent."""

    @pytest.mark.asyncio
    async def test_returns_valid_plan_from_json(self) -> None:
        """PlannerAgent returns a valid Plan when client returns well-formed JSON."""
        settings = make_settings()
        plan_response = (
            '{"tasks": [{"id": "1", "description": "Test task", "files": ["a.py"],'
            ' "dependencies": [], "status": "pending"}], "rationale": "Test plan"}'
        )
        client = FakeClient(settings=settings, responses=[plan_response])
        agent = PlannerAgent(client=client, settings=settings)

        plan = await agent.plan("Build a thing")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "1"
        assert plan.tasks[0].description == "Test task"
        assert plan.rationale == "Test plan"

    @pytest.mark.asyncio
    async def test_handles_markdown_fenced_json(self) -> None:
        """PlannerAgent succeeds when JSON is wrapped in markdown fence."""
        settings = make_settings()
        fenced_response = """```json
{
  "tasks": [
    {"id": "1", "description": "Fenced task", "files": [],
     "dependencies": [], "status": "pending"}
  ],
  "rationale": "Fenced plan"
}
```"""
        client = FakeClient(settings=settings, responses=[fenced_response])
        agent = PlannerAgent(client=client, settings=settings)

        plan = await agent.plan("Fenced goal")

        assert isinstance(plan, Plan)
        assert plan.tasks[0].description == "Fenced task"

    @pytest.mark.asyncio
    async def test_raises_value_error_on_unparseable_response(self) -> None:
        """PlannerAgent raises ValueError with raw response when no parseable JSON."""
        settings = make_settings()
        bad_response = "This is not JSON at all, just prose."
        client = FakeClient(settings=settings, responses=[bad_response])
        agent = PlannerAgent(client=client, settings=settings)

        with pytest.raises(ValueError) as exc_info:
            await agent.plan("Bad goal")

        assert "Failed to parse plan from LLM" in str(exc_info.value)
        assert bad_response in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_uses_planner_model_from_settings(self) -> None:
        """PlannerAgent uses settings.planner_model when calling complete."""
        settings = make_settings(planner_model="custom-planner-model")
        client = FakeClient(settings=settings, responses=['{"tasks": [], "rationale": "empty"}'])
        agent = PlannerAgent(client=client, settings=settings)

        await agent.plan("Test goal")

        assert client.last_model == "custom-planner-model"

    @pytest.mark.asyncio
    async def test_goal_text_reaches_prompt(self) -> None:
        """The goal text is included in the prompt sent to the LLM."""
        settings = make_settings()
        client = FakeClient(settings=settings, responses=['{"tasks": [], "rationale": "done"}'])
        agent = PlannerAgent(client=client, settings=settings)

        await agent.plan("My specific goal text")

        assert "My specific goal text" in client.last_prompt
        assert "Goal: My specific goal text" in client.last_prompt


class TestWorkerAgent:
    """Tests for WorkerAgent."""

    @pytest.mark.asyncio
    async def test_returns_text_from_client_complete(self) -> None:
        """WorkerAgent returns the text from client.complete."""
        settings = make_settings()
        expected = "Implementation summary: modify foo.py to add bar"
        client = FakeClient(settings=settings, responses=[expected])
        task = TaskModel(id="1", description="Add bar to foo", files=["foo.py"])
        agent = WorkerAgent(task=task, client=client, settings=settings)

        result = await agent.run()

        assert result == expected

    @pytest.mark.asyncio
    async def test_task_description_reaches_prompt(self) -> None:
        """The task description is included in the prompt."""
        settings = make_settings()
        client = FakeClient(settings=settings, responses=["summary"])
        task = TaskModel(id="1", description="Specific task description", files=[])
        agent = WorkerAgent(task=task, client=client, settings=settings)

        await agent.run()

        assert "Specific task description" in client.last_prompt
        assert "Task: Specific task description" in client.last_prompt

    @pytest.mark.asyncio
    async def test_files_list_included_when_present(self) -> None:
        """The files list is included in the prompt when present."""
        settings = make_settings()
        client = FakeClient(settings=settings, responses=["summary"])
        task = TaskModel(id="1", description="Task", files=["a.py", "b.py"])
        agent = WorkerAgent(task=task, client=client, settings=settings)

        await agent.run()

        assert "a.py" in client.last_prompt
        assert "b.py" in client.last_prompt
        assert "Files to touch: a.py, b.py" in client.last_prompt

    @pytest.mark.asyncio
    async def test_files_handled_sanely_when_empty(self) -> None:
        """Empty files list is handled sanely (shows 'any')."""
        settings = make_settings()
        client = FakeClient(settings=settings, responses=["summary"])
        task = TaskModel(id="1", description="Task", files=[])
        agent = WorkerAgent(task=task, client=client, settings=settings)

        await agent.run()

        assert "Files to touch: any" in client.last_prompt

    @pytest.mark.asyncio
    async def test_uses_worker_model_from_settings(self) -> None:
        """WorkerAgent uses settings.worker_model when calling complete."""
        settings = make_settings(worker_model="custom-worker-model")
        client = FakeClient(settings=settings, responses=["summary"])
        task = TaskModel(id="1", description="Task", files=[])
        agent = WorkerAgent(task=task, client=client, settings=settings)

        await agent.run()

        assert client.last_model == "custom-worker-model"


class TestTesterAgent:
    """Tests for TesterAgent."""

    @pytest.mark.asyncio
    async def test_passing_run_exit_code_authoritative(self, tmp_path: Path) -> None:
        """Exit code 0 -> TestResult.passed is True regardless of LLM response."""
        settings = make_settings(tmp_path, test_command=PYTHON_PASS)
        # LLM claims failure but exit code is 0
        llm_response = '{"passed": false, "summary": "LLM thinks it failed", "failures": ["fake"]}'
        client = FakeClient(settings=settings, responses=[llm_response])
        agent = TesterAgent(client=client, settings=settings)

        # Harness check: the command really launches (workspace exists) and really
        # exits 0. Without this, an unspawnable command would pass vacuously.
        probe = await agent._run_tests()
        assert probe.returncode == 0

        result = await agent.run("Goal", [])

        assert result.passed is True
        assert result.summary == "LLM thinks it failed"
        assert result.failures == ["fake"]

    @pytest.mark.asyncio
    async def test_failing_run_exit_code_authoritative(self, tmp_path: Path) -> None:
        """Nonzero exit code -> TestResult.passed is False even if LLM claims success."""
        settings = make_settings(tmp_path, test_command=PYTHON_FAIL)
        # LLM claims success but exit code is 1
        llm_response = '{"passed": true, "summary": "LLM thinks it passed", "failures": []}'
        client = FakeClient(settings=settings, responses=[llm_response])
        agent = TesterAgent(client=client, settings=settings)

        probe = await agent._run_tests()
        assert probe.returncode == 1  # nonzero, so the assertions below are meaningful

        result = await agent.run("Goal", [])

        assert result.passed is False
        assert result.summary == "LLM thinks it passed"

    @pytest.mark.asyncio
    async def test_captured_output_and_command_reflect_test_command(self, tmp_path: Path) -> None:
        """Output and command fields reflect the executed test_command."""
        settings = make_settings(tmp_path, test_command=PYTHON_HELLO)
        llm_response = '{"passed": true, "summary": "ok", "failures": []}'
        client = FakeClient(settings=settings, responses=[llm_response])
        agent = TesterAgent(client=client, settings=settings)

        result = await agent.run("Goal", [])

        assert result.passed is True
        # The captured stdout is handed to the LLM prompt
        assert "hello world" in client.last_prompt
        assert "Command: " in client.last_prompt
        assert "Exit code: 0" in client.last_prompt

    @pytest.mark.asyncio
    async def test_no_runner_found_returns_failed(self) -> None:
        """When no runner is found, returncode is None and passed is False."""
        settings = make_settings(test_command="this_command_does_not_exist_12345")
        llm_response = '{"passed": true, "summary": "ok", "failures": []}'
        client = FakeClient(settings=settings, responses=[llm_response])
        agent = TesterAgent(client=client, settings=settings)

        result = await agent.run("Goal", [])

        # When runner not found, _run_tests returns returncode=None
        # The agent should treat this as inconclusive -> passed=False
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_failures_list_from_json_response_surfaced(self, tmp_path: Path) -> None:
        """Failures list from JSON LLM response is surfaced on TestResult."""
        settings = make_settings(tmp_path, test_command=PYTHON_PASS)
        llm_response = (
            '{"passed": false, "summary": "Two failures",'
            ' "failures": ["test_foo failed", "test_bar error"]}'
        )
        client = FakeClient(settings=settings, responses=[llm_response])
        agent = TesterAgent(client=client, settings=settings)

        result = await agent.run("Goal", [])

        assert result.failures == ["test_foo failed", "test_bar error"]

    @pytest.mark.asyncio
    async def test_non_json_fallback_produces_test_result(self, tmp_path: Path) -> None:
        """Non-JSON LLM response falls back to prose summary without exception."""
        settings = make_settings(tmp_path, test_command=PYTHON_PASS)
        prose_response = "Tests passed! Everything looks good."
        client = FakeClient(settings=settings, responses=[prose_response])
        agent = TesterAgent(client=client, settings=settings)

        result = await agent.run("Goal", [])

        assert isinstance(result, type(result))  # TestResult instance
        assert result.passed is True  # exit code 0
        assert result.summary == prose_response
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_uses_tester_model_from_settings(self, tmp_path: Path) -> None:
        """TesterAgent uses settings.tester_model when calling complete."""
        settings = make_settings(
            tmp_path, tester_model="custom-tester-model", test_command=PYTHON_PASS
        )
        llm_response = '{"passed": true, "summary": "ok", "failures": []}'
        client = FakeClient(settings=settings, responses=[llm_response])
        agent = TesterAgent(client=client, settings=settings)

        await agent.run("Goal", [])

        assert client.last_model == "custom-tester-model"

    @pytest.mark.asyncio
    async def test_goal_and_test_output_reach_llm_prompt(self, tmp_path: Path) -> None:
        """Goal and test output are included in the prompt to the LLM."""
        settings = make_settings(tmp_path, test_command=PYTHON_OUTPUT)
        llm_response = '{"passed": true, "summary": "ok", "failures": []}'
        client = FakeClient(settings=settings, responses=[llm_response])
        agent = TesterAgent(client=client, settings=settings)

        await agent.run("My testing goal", [])

        assert "My testing goal" in client.last_prompt
        assert "test output here" in client.last_prompt
        assert "Exit code: 0" in client.last_prompt

    @pytest.mark.asyncio
    async def test_monkeypatch_run_tests_for_deterministic_control(self) -> None:
        """Monkeypatching _run_tests allows full control over output/returncode."""
        settings = make_settings()
        llm_response = '{"passed": false, "summary": "LLM summary", "failures": ["llm failure"]}'
        client = FakeClient(settings=settings, responses=[llm_response])
        agent = TesterAgent(client=client, settings=settings)

        # Monkeypatch _run_tests to return a controlled TestRun
        controlled_run = TestRun(
            output="controlled output\nline 2",
            returncode=0,
            command="controlled-command",
        )
        with patch.object(agent, "_run_tests", new=AsyncMock(return_value=controlled_run)):
            result = await agent.run("Goal", [])

        assert result.passed is True  # exit code 0 wins
        assert "controlled output" in client.last_prompt
        assert "controlled-command" in client.last_prompt
        assert "Exit code: 0" in client.last_prompt
        # LLM still provides summary and failures
        assert result.summary == "LLM summary"
        assert result.failures == ["llm failure"]

    @pytest.mark.asyncio
    async def test_timeout_handling_in_run_tests(self, tmp_path: Path) -> None:
        """A hung test command yields returncode=None plus a timeout notice.

        The spawn is faked (not ``_exec``: a patched-out ``_exec`` would just
        propagate the TimeoutError, which ``_run_tests`` does not catch), so the
        production timeout branch still runs against a 1s budget.
        """
        settings = make_settings(tmp_path, test_command=PYTHON_PASS, test_timeout=1)
        llm_response = '{"passed": true, "summary": "ok", "failures": []}'
        client = FakeClient(settings=settings, responses=[llm_response])
        agent = TesterAgent(client=client, settings=settings)
        hanging = _HangingProcess()

        async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> _HangingProcess:
            return hanging

        with patch("asyncio.create_subprocess_exec", new=fake_create_subprocess_exec):
            test_run = await agent._run_tests()

        assert hanging.killed is True  # the hung process really was killed
        assert test_run.returncode is None
        assert "timed out" in test_run.output.lower()
        assert test_run.command is not None

    @pytest.mark.asyncio
    async def test_runner_not_found_via_which(self) -> None:
        """When shutil.which finds no runner, _run_tests returns returncode=None, command=None."""
        settings = make_settings(test_command=None)  # triggers auto-detection
        client = FakeClient(settings=settings, responses=[])
        agent = TesterAgent(client=client, settings=settings)

        # Patch shutil.which to return None for all candidates
        with patch("shutil.which", return_value=None):
            test_run = await agent._run_tests()

        assert test_run.returncode is None
        assert test_run.command is None
        assert "No test runner found" in test_run.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
