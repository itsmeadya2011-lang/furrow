from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    """A Settings instance pointing at a temp workspace with no API keys."""
    return Settings(
        workspace=tmp_path,
        max_parallel_tasks=2,
        max_cycles=5,
        log_level="INFO",
    )


@pytest.fixture
def mock_settings_ollama(tmp_path: Path) -> Settings:
    return Settings(
        workspace=tmp_path,
        provider=Provider.OLLAMA,
        model="llama3",
        planner_model="llama3",
        worker_model="llama3",
        tester_model="llama3",
        ollama_base_url="http://localhost:11434",
        log_level="INFO",
    )


@pytest.fixture
def mock_client(mock_settings: Settings) -> LLMClient:
    """An LLMClient with a mocked complete method."""
    client = LLMClient(settings=mock_settings)
    client.complete = AsyncMock(return_value="mocked response")  # type: ignore[assignment]
    return client


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_provider_is_anthropic(self):
        assert Settings().provider == Provider.ANTHROPIC

    def test_ollama_provider_exists(self):
        assert Provider.OLLAMA == "ollama"

    def test_max_parallel_tasks_default(self):
        assert Settings().max_parallel_tasks == 5

    def test_max_cycles_default(self):
        assert Settings().max_cycles == 0

    def test_workspace_defaults_to_cwd(self):
        assert Settings().workspace == Path.cwd()


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_plan_parse(self):
        p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
        assert p.tasks[0].description == "do thing"

    def test_task_model_defaults(self):
        t = TaskModel(id="1", description="test")
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None

    def test_test_result(self):
        t = TestResult(passed=True, summary="ok", failures=[])
        assert t.passed is True

    def test_test_result_defaults(self):
        t = TestResult(passed=False, summary="fail")
        assert t.failures == []


# ---------------------------------------------------------------------------
# LLM Client tests
# ---------------------------------------------------------------------------

class TestLLMClient:
    def test_ollama_complete_calls_http(self, mock_settings_ollama: Settings):
        client = LLMClient(settings=mock_settings_ollama)

        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ollama response"}}
        mock_http.post = AsyncMock(return_value=mock_response)
        client._http = mock_http

        result = asyncio.run(client.complete("hello"))
        assert result == "ollama response"

    def test_complete_routes_to_provider(self, mock_settings: Settings):
        client = LLMClient(settings=mock_settings)

        with patch.object(client, "_complete_anthropic", new=AsyncMock(return_value="anthropic ok")):
            result = asyncio.run(client.complete("hi"))
        assert result == "anthropic ok"

    def test_complete_routes_openai(self, tmp_path: Path):
        settings = Settings(workspace=tmp_path, provider=Provider.OPENAI, openai_api_key="test-key")
        client = LLMClient(settings=settings)

        with patch.object(client, "_complete_openai", new=AsyncMock(return_value="openai ok")):
            result = asyncio.run(client.complete("hi"))
        assert result == "openai ok"

    def test_write_file_creates_parent_dirs(self, mock_settings: Settings, tmp_path: Path):
        client = LLMClient(settings=mock_settings)
        target = tmp_path / "subdir" / "file.txt"

        async def do_write():
            await client.write_file(target, "content")

        asyncio.run(do_write())
        assert target.read_text() == "content"

    def test_list_files_returns_relative(self, mock_settings: Settings, tmp_path: Path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y")
        client = LLMClient(settings=mock_settings)
        files = client.list_files(tmp_path)
        assert "a.py" in files
        assert "b.py" in files

    def test_list_files_missing_dir_returns_empty(self, tmp_path: Path):
        client = LLMClient(settings=Settings(workspace=tmp_path))
        assert client.list_files(tmp_path / "nonexistent") == []

    def test_retry_on_failure(self, mock_settings: Settings):
        """The @retry decorator should retry on exceptions and eventually succeed."""
        client = LLMClient(settings=mock_settings)
        call_count = 0

        async def failing_complete(prompt: str, system: str, model: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary failure")
            return "success after retry"

        with patch.object(client, "_complete_anthropic", side_effect=failing_complete):
            result = asyncio.run(client.complete("test"))
        assert result == "success after retry"
        assert call_count == 3


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------

class TestPlannerAgent:
    def test_plan_parses_json(self, mock_client: LLMClient):
        mock_client.complete = AsyncMock(  # type: ignore[assignment]
            return_value=json.dumps({
                "tasks": [
                    {"id": "1", "description": "Add auth", "files": ["auth.py"], "dependencies": []}
                ],
                "rationale": "Need auth",
            })
        )
        planner = PlannerAgent(client=mock_client)
        plan = asyncio.run(planner.plan("Add JWT auth"))
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "Add JWT auth"
        assert plan.rationale == "Need auth"

    def test_plan_raises_on_bad_json(self, mock_client: LLMClient):
        mock_client.complete = AsyncMock(return_value="not json at all")  # type: ignore[assignment]
        planner = PlannerAgent(client=mock_client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            asyncio.run(planner.plan("some goal"))


# ---------------------------------------------------------------------------
# Worker tests
# ---------------------------------------------------------------------------

class TestWorkerAgent:
    def test_worker_returns_summary(self, mock_client: LLMClient):
        mock_client.complete = AsyncMock(return_value="Implemented feature X")  # type: ignore[assignment]
        task = TaskModel(id="1", description="Add feature X", files=["a.py"])
        worker = WorkerAgent(task=task, client=mock_client)
        result = asyncio.run(worker.run())
        assert result == "Implemented feature X"

    def test_worker_gathers_context(self, mock_client: LLMClient, tmp_path: Path):
        (tmp_path / "main.py").write_text("print('hello')")
        mock_client.complete = AsyncMock(return_value="done")  # type: ignore[assignment]
        task = TaskModel(id="1", description="test", files=["main.py"])
        worker = WorkerAgent(task=task, client=mock_client, workspace=tmp_path)
        asyncio.run(worker.run())
        call_args = mock_client.complete.call_args  # type: ignore[attr-defined]
        assert call_args is not None
        prompt = call_args.kwargs.get("prompt") or call_args.args[0]
        assert "main.py" in prompt


# ---------------------------------------------------------------------------
# Tester tests
# ---------------------------------------------------------------------------

class TestTesterAgent:
    def test_tester_passes_on_valid_json(self, mock_client: LLMClient):
        mock_client.complete = AsyncMock(  # type: ignore[assignment]
            return_value=json.dumps({"passed": True, "summary": "all good", "failures": []})
        )
        tester = TesterAgent(client=mock_client)
        result = asyncio.run(tester.run("some goal", []))
        assert result.passed is True
        assert result.summary == "all good"

    def test_tester_falls_back_on_bad_json(self, mock_client: LLMClient):
        mock_client.complete = AsyncMock(return_value="tests passed successfully")  # type: ignore[assignment]
        tester = TesterAgent(client=mock_client)
        result = asyncio.run(tester.run("some goal", []))
        assert result.passed is True

    def test_tester_returns_error_when_no_runner(self, mock_client: LLMClient):
        mock_client.complete = AsyncMock(return_value="{}")  # type: ignore[assignment]
        tester = TesterAgent(client=mock_client)
        with patch.object(tester, "_run_tests", side_effect=RuntimeError("no runner")):
            result = asyncio.run(tester.run("some goal", []))
        assert result.passed is False


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def test_is_done_empty_tasks(self):
        orch = Orchestrator(goal="test", client=LLMClient(settings=Settings(workspace=Path("/tmp"))))
        assert orch._is_done() is False

    def test_is_done_all_completed_tests_pass(self):
        ts = Settings(workspace=Path("/tmp"))
        orch = Orchestrator(goal="test", client=LLMClient(settings=ts))
        orch._last_tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ]
        orch._last_test_passed = True
        assert orch._is_done() is True

    def test_is_done_all_completed_tests_fail(self):
        """If all tasks completed but tests failed, should not be done."""
        ts = Settings(workspace=Path("/tmp"))
        orch = Orchestrator(goal="test", client=LLMClient(settings=ts))
        orch._last_tasks = [
            TaskModel(id="1", description="a", status="completed"),
        ]
        orch._last_test_passed = False
        assert orch._is_done() is False

    def test_is_done_has_failed(self):
        ts = Settings(workspace=Path("/tmp"))
        orch = Orchestrator(goal="test", client=LLMClient(settings=ts))
        orch._last_tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ]
        assert orch._is_done() is False

    def test_is_done_some_completed(self):
        ts = Settings(workspace=Path("/tmp"))
        orch = Orchestrator(goal="test", client=LLMClient(settings=ts))
        orch._last_tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="pending"),
        ]
        assert orch._is_done() is False

    def test_max_cycles_stops(self, mock_client: LLMClient):
        """max_cycles=1 should stop after one cycle."""
        mock_client.complete = AsyncMock(side_effect=[  # type: ignore[assignment]
            json.dumps({"tasks": [], "rationale": "nothing to do"}),
        ])
        orch = Orchestrator(goal="test", client=mock_client, max_cycles=1)
        asyncio.run(orch.run())
        assert orch.cycles == 1

    def test_max_cycles_zero_means_unbounded(self, mock_client: LLMClient):
        """max_cycles=0 means no limit (stopped by empty plans or completion)."""
        call_count = 0

        async def fake_complete(prompt: str, **kwargs) -> str:
            nonlocal call_count
            call_count += 1
            # Always return empty tasks to test the 3-empty-plan stop mechanism
            return json.dumps({"tasks": [], "rationale": "nothing else to do"})

        mock_client.complete = fake_complete  # type: ignore[assignment]
        orch = Orchestrator(goal="test", client=mock_client, max_cycles=0)
        asyncio.run(orch.run())
        # Should stop after 3 empty plans
        assert orch.cycles == 3

    def test_progress_callback_registered(self, mock_client: LLMClient):
        events = []
        orch = Orchestrator(goal="test", client=mock_client, max_cycles=1)
        orch.add_progress_callback(lambda e, d: events.append(e))
        assert len(orch._callbacks) == 1

    def test_empty_plan_count_triggers_completion(self):
        ts = Settings(workspace=Path("/tmp"))
        orch = Orchestrator(goal="test", client=LLMClient(settings=ts))
        orch._empty_plan_count = 3
        assert orch._is_done() is True

    def test_get_tasks_returns_last_tasks(self):
        ts = Settings(workspace=Path("/tmp"))
        orch = Orchestrator(goal="test", client=LLMClient(settings=ts))
        task = TaskModel(id="1", description="a")
        orch._last_tasks = [task]
        assert orch._get_tasks() == [task]


# ---------------------------------------------------------------------------
# Integration: Orchestrator with mocked LLM
# ---------------------------------------------------------------------------

class TestOrchestratorIntegration:
    def test_single_cycle_completion(self, mock_client: LLMClient):
        """Orchestrator runs one cycle: plan with tasks, all complete, tests pass → done."""
        mock_client.complete = AsyncMock(  # type: ignore[assignment]
            side_effect=[
                # Planner response
                json.dumps({
                    "tasks": [
                        {"id": "1", "description": "task A", "files": ["a.py"], "dependencies": []},
                        {"id": "2", "description": "task B", "files": ["b.py"], "dependencies": []},
                    ],
                    "rationale": "initial tasks",
                }),
                # Worker A response
                "Implemented A",
                # Worker B response
                "Implemented B",
                # Tester response
                json.dumps({"passed": True, "summary": "all tests passed", "failures": []}),
            ]
        )
        ts = Settings(workspace=Path("/tmp"), max_parallel_tasks=5, max_cycles=0)
        orch = Orchestrator(goal="implement features", client=mock_client, max_cycles=0)
        asyncio.run(orch.run())

        assert orch.cycles == 1
        assert orch._last_tasks[0].status == "completed"
        assert orch._last_tasks[1].status == "completed"

    def test_failed_test_continues_to_next_cycle(self, mock_client: LLMClient):
        """When tests fail, the orchestrator should continue to the next cycle."""
        mock_client.complete = AsyncMock(side_effect=[  # type: ignore[assignment]
            # Cycle 1: Planner → 1 task
            json.dumps({
                "tasks": [{"id": "1", "description": "task A", "files": [], "dependencies": []}],
                "rationale": "tasks",
            }),
            # Cycle 1: Worker → done
            "Implemented A",
            # Cycle 1: Tester → fail
            json.dumps({"passed": False, "summary": "fail", "failures": ["assertion error"]}),
            # Cycle 2: Planner → empty (giving up after 3 empty plans)
            json.dumps({"tasks": [], "rationale": "all done"}),
            # Cycle 3: Planner → empty
            json.dumps({"tasks": [], "rationale": "all done"}),
            # Cycle 4: Planner → empty (triggers stop)
            json.dumps({"tasks": [], "rationale": "all done"}),
        ])
        ts = Settings(workspace=Path("/tmp"), max_parallel_tasks=5, max_cycles=10)
        orch = Orchestrator(goal="fix tests", client=mock_client, max_cycles=10)
        asyncio.run(orch.run())
        # Should have run at least 2 cycles (first with failing tests, second attempting fix)
        assert orch.cycles >= 2

    def test_concurrency_limited_by_semaphore(self, mock_client: LLMClient):
        """Verify that the semaphore is initialized with max_parallel_tasks."""
        ts = Settings(workspace=Path("/tmp"), max_parallel_tasks=2, max_cycles=0)
        orch = Orchestrator(goal="test", client=LLMClient(settings=ts), max_cycles=0)

        async def check():
            sem = orch.semaphore
            assert sem._value == 2  # noqa: SLF001

        asyncio.run(check())
        assert orch.client.settings.max_parallel_tasks == 2


# ---------------------------------------------------------------------------
# Planner context tests
# ---------------------------------------------------------------------------

class TestPlannerContext:
    def test_gather_context_reads_files(self, mock_client: LLMClient, tmp_path: Path):
        (tmp_path / "app.py").write_text("def hello(): pass")
        (tmp_path / "README.md").write_text("# Project")
        planner = PlannerAgent(client=mock_client, workspace=tmp_path)
        context = asyncio.run(planner._gather_context())
        assert "app.py" in context
        assert "hello" in context

    def test_gather_context_skips_binary(self, mock_client: LLMClient, tmp_path: Path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "code.py").write_text("x = 1")
        planner = PlannerAgent(client=mock_client, workspace=tmp_path)
        context = asyncio.run(planner._gather_context())
        assert "image.png" not in context
        assert "code.py" in context

    def test_gather_context_missing_workspace(self, mock_client: LLMClient, tmp_path: Path):
        planner = PlannerAgent(client=mock_client, workspace=tmp_path / "doesn't exist")
        context = asyncio.run(planner._gather_context())
        assert context == ""


# ---------------------------------------------------------------------------
# Web server tests
# ---------------------------------------------------------------------------

class TestWebServer:
    def test_start_request_model(self):
        from furrow.web.server import StartRequest
        req = StartRequest(goal="test goal", model="gpt-4")
        assert req.goal == "test goal"
        assert req.model == "gpt-4"

    def test_start_request_optional_model(self):
        from furrow.web.server import StartRequest
        req = StartRequest(goal="test goal")
        assert req.model is None
