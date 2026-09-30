import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class MockLLMClient:
    """Mock LLM client that returns canned responses for testing."""

    def __init__(self, response: str = "{}", responses: list | None = None):
        if responses is not None:
            self._responses = list(responses)
        else:
            self._responses = [response]
        self._response_index = 0
        self.settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
        )
        self.call_count = 0
        self.calls: list[dict] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.call_count += 1
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
        elif self._responses:
            response = self._responses[-1]
        else:
            response = "{}"
        self._response_index += 1
        return response

    async def read_file(self, path: str | Path) -> str:
        return ""

    async def write_file(self, path: str | Path, content: str) -> None:
        pass

    def list_files(self, directory: str | Path) -> list[str]:
        return []


_plan_json = json.dumps(
    {
        "tasks": [
            {"id": "1", "description": "task1", "files": [], "dependencies": []},
        ],
        "rationale": "test plan",
    }
)

_pass_test_json = json.dumps({"passed": True, "summary": "All passed", "failures": []})
_fail_test_json = json.dumps({"passed": False, "summary": "Tests failed", "failures": ["failure1"]})


class TestOrchestratorGoal:
    def test_stores_original_goal(self):
        client = MockLLMClient()
        orch = Orchestrator(goal="build a rocket", client=client)
        assert orch.original_goal == "build a rocket"

    def test_stores_goal_attribute(self):
        client = MockLLMClient()
        orch = Orchestrator(goal="my goal", client=client)
        assert orch.goal == "my goal"


class TestOrchestratorMaxCycles:
    async def test_respects_max_cycles_and_stops(self):
        settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
            max_cycles=1,
        )
        client = MockLLMClient(
            responses=[
                _plan_json,
                "task result",
                _fail_test_json,
            ]
        )
        client.settings = settings
        orch = Orchestrator(goal="test", client=client, on_log=lambda msg: None)
        orch.tester._run_tests = AsyncMock(return_value="test output")

        await orch.run()

        assert orch.cycles == 1
        assert orch.max_cycles == 1

    async def test_max_cycles_zero_runs_until_done(self):
        settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
            max_cycles=0,
        )
        client = MockLLMClient(
            responses=[
                _plan_json,
                "task result",
                _pass_test_json,
            ]
        )
        client.settings = settings
        orch = Orchestrator(goal="test", client=client, on_log=lambda msg: None)
        orch.tester._run_tests = AsyncMock(return_value="test output")

        await orch.run()

        assert orch.cycles == 1  # _is_done() returns True after first cycle
        assert orch._is_done() is True

    async def test_max_cycles_two(self):
        settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
            max_cycles=2,
        )
        responses = []
        for i in range(6):
            responses.extend([_plan_json, "task result", _fail_test_json])
        client = MockLLMClient(responses=responses)
        client.settings = settings
        orch = Orchestrator(goal="test", client=client, on_log=lambda msg: None)
        orch.tester._run_tests = AsyncMock(return_value="test output")

        await orch.run()

        assert orch.cycles == 2


class TestOrchestratorSemaphore:
    def test_reads_max_parallel_tasks_from_settings(self):
        settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
            max_parallel_tasks=3,
        )
        client = MockLLMClient()
        client.settings = settings
        orch = Orchestrator(goal="test", client=client)
        assert orch.max_parallel_tasks == 3

    async def test_enforces_max_parallel_tasks_via_semaphore(self):
        settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
            max_parallel_tasks=2,
        )
        client = MockLLMClient()
        client.settings = settings
        orch = Orchestrator(goal="test", client=client, on_log=lambda msg: None)
        orch.planner.plan = AsyncMock(
            return_value=Plan(
                tasks=[
                    TaskModel(id=str(i), description=f"task{i}", status="pending")
                    for i in range(1, 7)
                ],
                rationale="test",
            )
        )

        current = 0
        max_concurrent = 0

        async def mock_worker_run():
            nonlocal current, max_concurrent
            current += 1
            max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0.05)
            current -= 1
            return "done"

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            MockWorker.return_value.run = AsyncMock(side_effect=mock_worker_run)
            orch.tester.run = AsyncMock(
                return_value=TestResult(passed=True, summary="ok", failures=[])
            )
            await orch._cycle()

        assert max_concurrent <= 2
        assert max_concurrent > 0

    async def test_semaphore_allows_sequential_when_one(self):
        settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
            max_parallel_tasks=1,
        )
        client = MockLLMClient()
        client.settings = settings
        orch = Orchestrator(goal="test", client=client, on_log=lambda msg: None)
        orch.planner.plan = AsyncMock(
            return_value=Plan(
                tasks=[
                    TaskModel(id="1", description="t1", status="pending"),
                    TaskModel(id="2", description="t2", status="pending"),
                    TaskModel(id="3", description="t3", status="pending"),
                ],
                rationale="test",
            )
        )

        current = 0
        max_concurrent = 0

        async def mock_worker_run():
            nonlocal current, max_concurrent
            current += 1
            max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0.05)
            current -= 1
            return "done"

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            MockWorker.return_value.run = AsyncMock(side_effect=mock_worker_run)
            orch.tester.run = AsyncMock(
                return_value=TestResult(passed=True, summary="ok", failures=[])
            )
            await orch._cycle()

        assert max_concurrent == 1


class TestOrchestratorIsDone:
    def test_is_done_empty_tasks(self):
        client = MockLLMClient()
        orch = Orchestrator(goal="test", client=client)
        assert orch._is_done() is True

    def test_is_done_all_completed_with_passing_tests(self):
        client = MockLLMClient()
        orch = Orchestrator(goal="test", client=client)
        orch.current_tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="completed"),
        ]
        orch.test_result = TestResult(passed=True, summary="ok", failures=[])
        assert orch._is_done() is True

    def test_is_done_all_completed_no_test_result(self):
        client = MockLLMClient()
        orch = Orchestrator(goal="test", client=client)
        orch.current_tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="completed"),
        ]
        assert orch._is_done() is True

    def test_is_done_with_failed_tasks(self):
        client = MockLLMClient()
        orch = Orchestrator(goal="test", client=client)
        orch.current_tasks = [
            TaskModel(id="1", description="t1", status="failed", result="error"),
            TaskModel(id="2", description="t2", status="completed"),
        ]
        orch.test_result = TestResult(passed=True, summary="ok", failures=[])
        assert orch._is_done() is False

    def test_is_done_with_pending_tasks(self):
        client = MockLLMClient()
        orch = Orchestrator(goal="test", client=client)
        orch.current_tasks = [
            TaskModel(id="1", description="t1", status="pending"),
            TaskModel(id="2", description="t2", status="completed"),
        ]
        orch.test_result = TestResult(passed=True, summary="ok", failures=[])
        assert orch._is_done() is False

    def test_is_done_all_completed_but_tests_failed(self):
        client = MockLLMClient()
        orch = Orchestrator(goal="test", client=client)
        orch.current_tasks = [
            TaskModel(id="1", description="t1", status="completed"),
        ]
        orch.test_result = TestResult(passed=False, summary="fail", failures=["err"])
        assert orch._is_done() is False


class TestOrchestratorOnLog:
    async def test_calls_on_log_callback(self):
        logs: list[str] = []
        on_log = lambda msg: logs.append(msg)

        client = MockLLMClient(
            responses=[
                _plan_json,
                "task result",
                _pass_test_json,
            ]
        )
        client.settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
        )
        orch = Orchestrator(goal="test", client=client, on_log=on_log)
        orch.tester._run_tests = AsyncMock(return_value="test output")

        await orch._cycle()

        assert len(logs) > 0
        assert any("completed" in msg.lower() for msg in logs)
        assert any("passed" in msg.lower() for msg in logs)

    async def test_on_log_receives_plain_messages(self):
        logs: list[str] = []
        on_log = lambda msg: logs.append(msg)

        client = MockLLMClient(
            responses=[
                _plan_json,
                "task result",
                _pass_test_json,
            ]
        )
        client.settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
        )
        orch = Orchestrator(goal="test", client=client, on_log=on_log)
        orch.tester._run_tests = AsyncMock(return_value="test output")

        await orch._cycle()

        assert any("Plan:" in msg for msg in logs)
        assert any("Tests passed" in msg for msg in logs)

    async def test_no_on_log_uses_console(self, monkeypatch):
        import io

        from rich.console import Console

        client = MockLLMClient(
            responses=[
                _plan_json,
                "task result",
                _pass_test_json,
            ]
        )
        client.settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
        )
        orch = Orchestrator(goal="test", client=client)
        orch.tester._run_tests = AsyncMock(return_value="test output")

        output = io.StringIO()
        test_console = Console(file=output, force_terminal=False)
        monkeypatch.setattr("furrow.core.orchestrator.console", test_console)

        await orch._cycle()

        output_text = output.getvalue()
        assert output_text


class TestOrchestratorTaskFailures:
    async def test_handles_task_failures_gracefully(self):
        client = MockLLMClient()
        client.settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
        )
        orch = Orchestrator(goal="test", client=client, on_log=lambda msg: None)
        orch.planner.plan = AsyncMock(
            return_value=Plan(
                tasks=[
                    TaskModel(id="1", description="task1", status="pending"),
                ],
                rationale="test",
            )
        )

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            MockWorker.return_value.run = AsyncMock(side_effect=RuntimeError("worker crashed"))
            orch.tester.run = AsyncMock(
                return_value=TestResult(passed=True, summary="ok", failures=[])
            )
            await orch._cycle()

        assert orch.current_tasks[0].status == "failed"
        assert orch.current_tasks[0].result == "worker crashed"

    async def test_mixed_success_and_failure(self):
        client = MockLLMClient()
        client.settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
        )
        orch = Orchestrator(goal="test", client=client, on_log=lambda msg: None)
        orch.planner.plan = AsyncMock(
            return_value=Plan(
                tasks=[
                    TaskModel(id="1", description="t1", status="pending"),
                    TaskModel(id="2", description="t2", status="pending"),
                ],
                rationale="test",
            )
        )

        call_count = 0

        async def mock_worker_run():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "success"
            raise ValueError("task 2 failed")

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            MockWorker.return_value.run = AsyncMock(side_effect=mock_worker_run)
            orch.tester.run = AsyncMock(
                return_value=TestResult(passed=True, summary="ok", failures=[])
            )
            await orch._cycle()

        statuses = [t.status for t in orch.current_tasks]
        assert "completed" in statuses
        assert "failed" in statuses

    async def test_all_workers_fail(self):
        client = MockLLMClient()
        client.settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
        )
        orch = Orchestrator(goal="test", client=client, on_log=lambda msg: None)
        orch.planner.plan = AsyncMock(
            return_value=Plan(
                tasks=[
                    TaskModel(id="1", description="t1", status="pending"),
                    TaskModel(id="2", description="t2", status="pending"),
                ],
                rationale="test",
            )
        )

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            MockWorker.return_value.run = AsyncMock(side_effect=RuntimeError("all failed"))
            orch.tester.run = AsyncMock(
                return_value=TestResult(passed=True, summary="ok", failures=[])
            )
            await orch._cycle()

        assert all(t.status == "failed" for t in orch.current_tasks)
        assert not orch._is_done()
