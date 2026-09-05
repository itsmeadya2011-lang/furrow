from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from furrow.agents.prompts import WORKER_PROMPT
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


# ---------------------------------------------------------------------------
# Plan / TaskModel serialization
# ---------------------------------------------------------------------------


class TestPlanSerialization:
    def test_task_model_defaults(self):
        task = TaskModel(id="1", description="do thing")
        assert task.files == []
        assert task.dependencies == []
        assert task.status == "pending"
        assert task.result is None

    def test_plan_model_dump_round_trip(self):
        plan = Plan(
            tasks=[
                TaskModel(
                    id="1",
                    description="Implement auth",
                    files=["src/auth.py"],
                    dependencies=[],
                    status="pending",
                ),
                TaskModel(
                    id="2",
                    description="Write tests",
                    files=["tests/test_auth.py"],
                    dependencies=["1"],
                ),
            ],
            rationale="Two-step plan",
        )

        dumped = plan.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["rationale"] == "Two-step plan"
        assert len(dumped["tasks"]) == 2
        assert dumped["tasks"][0]["id"] == "1"
        assert dumped["tasks"][1]["dependencies"] == ["1"]

        restored = Plan.model_validate(dumped)
        assert restored == plan

    def test_task_model_validate_from_dict(self):
        raw = {
            "id": "9",
            "description": "refactor X",
            "files": ["x.py", "y.py"],
            "dependencies": ["1", "2"],
            "status": "completed",
            "result": "done",
        }
        task = TaskModel.model_validate(raw)
        assert task.id == "9"
        assert task.files == ["x.py", "y.py"]
        assert task.status == "completed"
        assert task.result == "done"

    def test_plan_validates_missing_required_field(self):
        with pytest.raises(ValidationError):
            Plan.model_validate({"tasks": [], "rationale": ""})
        with pytest.raises(ValidationError):
            TaskModel.model_validate({"id": "1"})


# ---------------------------------------------------------------------------
# TestResult behavior
# ---------------------------------------------------------------------------


class TestTestResult:
    def test_passed_with_no_failures(self):
        result = TestResult(passed=True, summary="All good", failures=[])
        assert result.passed is True
        assert result.summary == "All good"
        assert result.failures == []

    def test_failed_with_failures(self):
        result = TestResult(
            passed=False,
            summary="2 failed",
            failures=["test_a: assert 1 == 2", "test_b: import error"],
        )
        assert result.passed is False
        assert len(result.failures) == 2
        assert "test_a" in result.failures[0]

    def test_default_failures_empty(self):
        result = TestResult(passed=True, summary="ok")
        assert result.failures == []

    def test_model_dump_and_validate(self):
        original = TestResult(passed=False, summary="nope", failures=["x"])
        again = TestResult.model_validate(original.model_dump())
        assert again == original


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------


class TestSettingsValidation:
    def test_workspace_validator_rejects_nonexistent_path(self):
        with pytest.raises(ValidationError) as excinfo:
            Settings(workspace="/this/path/definitely/does/not/exist/xyz123")
        assert "workspace must be an existing directory" in str(excinfo.value)

    def test_workspace_validator_rejects_file_path(self, tmp_path):
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("hello")
        with pytest.raises(ValidationError):
            Settings(workspace=str(file_path))

    def test_workspace_validator_accepts_existing_dir(self, tmp_path):
        s = Settings(workspace=str(tmp_path))
        assert s.workspace == tmp_path

    def test_workspace_default_is_cwd(self, monkeypatch):
        monkeypatch.delenv("FURROW_WORKSPACE", raising=False)
        s = Settings(_env_file=None)
        assert s.workspace == Path.cwd()

    def test_anthropic_provider_requires_api_key(self):
        with pytest.raises(ValidationError) as excinfo:
            Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
        assert "anthropic provider requires an API key" in str(excinfo.value)

    def test_openai_provider_requires_api_key(self):
        with pytest.raises(ValidationError) as excinfo:
            Settings(provider=Provider.OPENAI, openai_api_key=None)
        assert "openai provider requires an API key" in str(excinfo.value)

    def test_anthropic_provider_accepts_api_key(self):
        s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="sk-test")
        assert s.anthropic_api_key == "sk-test"

    def test_ollama_provider_does_not_require_api_key(self):
        # Ollama is local; no cloud API key required.
        s = Settings(provider=Provider.OLLAMA)
        assert s.provider == Provider.OLLAMA
        assert s.anthropic_api_key is None
        assert s.openai_api_key is None


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


def _make_settings(**overrides) -> Settings:
    base = dict(
        provider=Provider.ANTHROPIC,
        anthropic_api_key="sk-test",
        openai_api_key="sk-test",
        workspace=tempfile.gettempdir(),
    )
    base.update(overrides)
    return Settings(**base)


class TestLLMClient:
    def test_unsupported_provider_raises_value_error(self):
        # Build a settings-like object whose `provider` attribute will fail the
        # `if/elif` chain in `LLMClient.complete`.
        fake_settings = MagicMock()
        fake_settings.provider = "bogus-provider"
        fake_settings.model = "x"
        client = LLMClient(settings=fake_settings)
        with pytest.raises(ValueError, match="Unsupported provider"):
            # Bypass the tenacity retry wrapper via __func__.__wrapped__.
            client.complete.__func__.__wrapped__(client, "hi")

    @pytest.mark.asyncio
    async def test_ollama_completion_calls_httpx(self):
        client = LLMClient(settings=_make_settings(provider=Provider.OLLAMA))

        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.return_value = {
            "message": {"content": "hello from ollama"}
        }

        with patch("furrow.llm.httpx.AsyncClient") as mock_async_client_cls:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=fake_response)
            # Make `async with httpx.AsyncClient() as client:` work.
            mock_async_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_async_client_cls.return_value.__aexit__ = AsyncMock(
                return_value=None
            )

            result = await client._complete_ollama("prompt", "sys", "llama3")

        assert result == "hello from ollama"
        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.await_args
        assert call_args.args[0].endswith("/api/chat")
        payload = call_args.kwargs["json"]
        assert payload["model"] == "llama3"
        assert payload["stream"] is False
        assert payload["messages"][-1] == {"role": "user", "content": "prompt"}
        assert payload["messages"][0] == {"role": "system", "content": "sys"}

    @pytest.mark.asyncio
    async def test_ollama_completion_no_system_message(self):
        client = LLMClient(settings=_make_settings(provider=Provider.OLLAMA))

        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.return_value = {"message": {"content": "ok"}}

        with patch("furrow.llm.httpx.AsyncClient") as mock_async_client_cls:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=fake_response)
            mock_async_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_async_client_cls.return_value.__aexit__ = AsyncMock(
                return_value=None
            )

            await client._complete_ollama("prompt", "", "llama3")

        payload = mock_client.post.await_args.kwargs["json"]
        assert payload["messages"] == [{"role": "user", "content": "prompt"}]

    def test_complete_is_wrapped_with_tenacity_retry(self):
        client = LLMClient(settings=_make_settings())
        # Tenacity attaches retry metadata to the underlying function.
        # On an instance method, access it via __func__.
        assert hasattr(client.complete.__func__, "retry")
        assert hasattr(client.complete.__func__, "__wrapped__")
        # The wrapped function should be the original async `complete`.
        import asyncio

        assert asyncio.iscoroutinefunction(client.complete.__func__.__wrapped__)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _make_plan(task_statuses: list[str]) -> Plan:
    return Plan(
        tasks=[
            TaskModel(id=str(i), description=f"task {i}", status=s)
            for i, s in enumerate(task_statuses)
        ],
        rationale="test",
    )


class TestOrchestrator:
    def test_is_done_all_completed(self):
        orch = Orchestrator(goal="x", client=MagicMock(spec=LLMClient))
        orch.plan = _make_plan(["completed", "completed", "completed"])
        assert orch._is_done() is True

    def test_is_done_not_done_with_pending(self):
        orch = Orchestrator(goal="x", client=MagicMock(spec=LLMClient))
        orch.plan = _make_plan(["completed", "pending"])
        assert orch._is_done() is False

    def test_is_done_with_failed_task(self):
        orch = Orchestrator(goal="x", client=MagicMock(spec=LLMClient))
        orch.plan = _make_plan(["completed", "failed", "completed"])
        assert orch._is_done() is False

    def test_is_done_empty_plan(self):
        orch = Orchestrator(goal="x", client=MagicMock(spec=LLMClient))
        orch.plan = _make_plan([])
        # completed (0) >= len (0) -> True
        assert orch._is_done() is True

    def test_get_tasks_returns_plan_tasks(self):
        orch = Orchestrator(goal="x", client=MagicMock(spec=LLMClient))
        orch.plan = _make_plan(["pending", "completed"])
        tasks = orch._get_tasks()
        assert len(tasks) == 2
        assert tasks[0].id == "0"
        assert tasks[1].status == "completed"

    def test_get_tasks_no_plan(self):
        orch = Orchestrator(goal="x", client=MagicMock(spec=LLMClient))
        assert orch.plan is None
        assert orch._get_tasks() == []

    @pytest.mark.asyncio
    async def test_status_calls_callback_when_provided(self):
        calls: list[str | dict] = []

        async def cb(msg):
            calls.append(msg)

        orch = Orchestrator(goal="x", client=MagicMock(spec=LLMClient), status_callback=cb)
        await orch._status("hello")
        assert calls == ["hello"]
        await orch._status({"type": "plan", "data": {}})
        assert calls[-1] == {"type": "plan", "data": {}}

    @pytest.mark.asyncio
    async def test_status_swallows_callback_exceptions(self):
        async def bad_cb(msg):
            raise RuntimeError("boom")

        orch = Orchestrator(goal="x", client=MagicMock(spec=LLMClient), status_callback=bad_cb)
        # Should not raise.
        await orch._status("anything")

    @pytest.mark.asyncio
    async def test_status_no_callback_does_not_raise(self):
        orch = Orchestrator(goal="x", client=MagicMock(spec=LLMClient))
        # No callback; falls through to console.print path. We patch the
        # console to avoid noisy output.
        with patch("furrow.core.orchestrator.console") as mock_console:
            await orch._status("hi")
        mock_console.print.assert_called_with("hi")


# ---------------------------------------------------------------------------
# WorkerAgent prompt construction
# ---------------------------------------------------------------------------


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_prompt_contains_worker_prompt_and_task_description(self):
        task = TaskModel(id="1", description="Implement feature X", files=["a.py", "b.py"])
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = _make_settings()
        mock_client.complete = AsyncMock(return_value="summary text")

        agent = WorkerAgent(task=task, client=mock_client)
        result = await agent.run()

        assert result == "summary text"
        mock_client.complete.assert_awaited_once()
        args, kwargs = mock_client.complete.call_args
        prompt = args[0]
        assert WORKER_PROMPT in prompt
        assert "Implement feature X" in prompt
        assert "a.py, b.py" in prompt
        assert kwargs["model"] == mock_client.settings.worker_model

    @pytest.mark.asyncio
    async def test_prompt_with_no_files_says_any(self):
        task = TaskModel(id="2", description="Do Y", files=[])
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = _make_settings()
        mock_client.complete = AsyncMock(return_value="ok")

        agent = WorkerAgent(task=task, client=mock_client)
        await agent.run()

        prompt = mock_client.complete.await_args.args[0]
        assert "any" in prompt
        assert task.description in prompt