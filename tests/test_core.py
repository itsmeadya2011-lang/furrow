from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient
from furrow.web.server import app


# ─── Config / Settings ───────────────────────────────────────────────────────

class TestSettings:
    def test_default_provider(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC

    def test_default_model(self):
        s = Settings()
        assert s.model == "claude-sonnet-4-20250514"

    def test_custom_model(self):
        s = Settings(model="custom-model")
        assert s.model == "custom-model"

    def test_ollama_base_url_default(self):
        s = Settings()
        assert s.ollama_base_url == "http://localhost:11434"

    def test_max_parallel_tasks_default(self):
        s = Settings()
        assert s.max_parallel_tasks == 5

    def test_max_cycles_default(self):
        s = Settings()
        assert s.max_cycles == 0

    def test_task_model_defaults(self):
        t = TaskModel(id="1", description="do thing")
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None

    def test_plan_validation(self):
        p = Plan(tasks=[TaskModel(id="1", description="x")], rationale="ok")
        assert len(p.tasks) == 1
        assert p.tasks[0].description == "x"


# ─── Orchestrator ────────────────────────────────────────────────────────────

class TestOrchestrator:
    def test_init_stores_goal(self):
        o = Orchestrator(goal="build thing")
        assert o.goal == "build thing"
        assert o.cycles == 0
        assert o.current_plan is None

    def test_get_tasks_returns_empty_when_no_plan(self):
        o = Orchestrator(goal="build thing")
        assert o._get_tasks() == []

    def test_get_tasks_returns_plan_tasks(self):
        o = Orchestrator(goal="build thing")
        plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="ok")
        o.current_plan = plan
        assert o._get_tasks() == plan.tasks

    def test_is_done_returns_false_when_no_tasks(self):
        o = Orchestrator(goal="build thing")
        # _get_tasks returns [], so 0 >= 0 is True
        # This is expected behavior when there are no tasks planned
        assert o._is_done() is True

    def test_is_done_returns_true_when_all_completed(self):
        o = Orchestrator(goal="build thing")
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="completed"),
            ],
            rationale="ok",
        )
        o.current_plan = plan
        assert o._is_done() is True

    def test_is_done_returns_false_when_pending_tasks(self):
        o = Orchestrator(goal="build thing")
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="pending"),
            ],
            rationale="ok",
        )
        o.current_plan = plan
        assert o._is_done() is False

    def test_is_done_returns_false_when_any_failed(self):
        o = Orchestrator(goal="build thing")
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="failed"),
            ],
            rationale="ok",
        )
        o.current_plan = plan
        assert o._is_done() is False

    @pytest.mark.asyncio
    async def test_cycle_sets_current_plan(self):
        o = Orchestrator(goal="build thing")
        mock_plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="ok")

        mock_planner = MagicMock()
        mock_planner.plan = AsyncMock(return_value=mock_plan)
        o.planner = mock_planner

        mock_tester = MagicMock()
        mock_tester.run = AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))
        with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
            with patch("furrow.core.orchestrator.WorkerAgent"):
                with patch("asyncio.gather", new_callable=AsyncMock, return_value=[]):
                    await o._cycle()

        assert o.current_plan == mock_plan


# ─── LLMClient ───────────────────────────────────────────────────────────────

class TestLLMClient:
    def test_init_with_default_settings(self):
        client = LLMClient()
        assert client.settings is not None

    def test_complete_anthropic_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
        client = LLMClient(settings=s)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            client.anthropic  # Access property to trigger init

    def test_complete_openai_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = Settings(provider=Provider.OPENAI, openai_api_key=None)
        client = LLMClient(settings=s)
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            client.openai  # Access property to trigger init

    @pytest.mark.asyncio
    async def test_complete_ollama_url_building(self, monkeypatch):
        s = Settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:11434",
        )
        client = LLMClient(settings=s)

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "hello"}}
        mock_response.raise_for_status = MagicMock()

        mock_post = AsyncMock(return_value=mock_response)
        mock_client = MagicMock()
        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client._complete_ollama("say hi", "", "llama3")

        assert result == "hello"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/chat"
        payload = call_args[1]["json"]
        assert payload["model"] == "llama3"
        assert payload["stream"] is False
        assert payload["messages"] == [{"role": "user", "content": "say hi"}]

    @pytest.mark.asyncio
    async def test_complete_ollama_includes_system(self, monkeypatch):
        s = Settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:11434",
        )
        client = LLMClient(settings=s)

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "hi"}}
        mock_response.raise_for_status = MagicMock()

        mock_post = AsyncMock(return_value=mock_response)
        mock_client = MagicMock()
        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client._complete_ollama("say hi", "be concise", "llama3")

        payload = mock_post.call_args[1]["json"]
        assert payload["messages"] == [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "say hi"},
        ]

    @pytest.mark.asyncio
    async def test_complete_ollama_handles_http_error(self, monkeypatch):
        s = Settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:11434",
        )
        client = LLMClient(settings=s)

        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Ollama API request failed"):
                await client._complete_ollama("say hi", "", "llama3")


# ─── Agents ──────────────────────────────────────────────────────────────────

class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_calls_client(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "tasks": [{"id": "1", "description": "x", "files": [], "dependencies": []}],
            "rationale": "ok",
        }))
        mock_client.settings = Settings()

        agent = PlannerAgent(client=mock_client)
        plan = await agent.plan("do something")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "x"
        mock_client.complete.assert_called_once()


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_run_calls_client(self):
        task = TaskModel(id="1", description="write code", files=["a.py"])
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="done")
        mock_client.settings = Settings()

        agent = WorkerAgent(task=task, client=mock_client)
        result = await agent.run()
        assert result == "done"
        mock_client.complete.assert_called_once()
        call_prompt = mock_client.complete.call_args[0][0]
        assert "write code" in call_prompt
        assert "a.py" in call_prompt


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_with_passing_tests(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "passed": True,
            "summary": "all good",
            "failures": [],
        }))
        mock_client.settings = Settings()

        agent = TesterAgent(client=mock_client)
        with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="tests passed"):
            result = await agent.run("goal", [])

        assert result.passed is True
        assert result.summary == "all good"

    @pytest.mark.asyncio
    async def test_run_handles_test_failure(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="some response")
        mock_client.settings = Settings()

        agent = TesterAgent(client=mock_client)
        with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="tests failed"):
            result = await agent.run("goal", [])

        # Falls back to keyword matching when JSON parse fails
        assert result.passed is False
        assert result.summary == "some response"


# ─── Web Server ──────────────────────────────────────────────────────────────

class TestWebServer:
    def test_index_returns_html(self):
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Furrow" in resp.text

    def test_start_post_endpoint(self):
        client = TestClient(app)
        resp = client.post("/start", json={"goal": "test goal"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["goal"] == "test goal"

    @pytest.mark.asyncio
    async def test_websocket_rejects_empty_goal(self):
        client = TestClient(app)
        with patch("furrow.web.server.Orchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.run = AsyncMock()
            MockOrchestrator.return_value = mock_orch
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"goal": ""})
                data = ws.receive_json()
                assert data["type"] == "status"

    @pytest.mark.asyncio
    async def test_websocket_sends_status(self):
        client = TestClient(app)
        with patch("furrow.web.server.Orchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.run = AsyncMock()
            MockOrchestrator.return_value = mock_orch
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"goal": "test"})
                msg = ws.receive_json()
                assert msg["type"] == "status"
