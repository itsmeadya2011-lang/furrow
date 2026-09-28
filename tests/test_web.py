from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import furrow.web.server as server_module
from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import CompletionResult, LLMClient, ToolCall

PLAN_JSON = (
    '{"tasks":[{"id":"1","description":"Create result.py","files":["result.py"],'
    '"dependencies":[]}],"rationale":"ok"}'
)
TEST_PASS_JSON = '{"passed": true, "summary": "All good", "failures": []}'


class FakeLLMClient(LLMClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings=settings)
        self.cwt_calls: int = 0

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        if "Break the user's goal" in prompt:
            return PLAN_JSON
        if "determine if tests" in prompt:
            return TEST_PASS_JSON
        return "ok"

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        system: str = "",
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        self.cwt_calls += 1
        if any(m.get("role") == "tool" for m in messages):
            return CompletionResult(text="Created result.py.", tool_calls=[])
        return CompletionResult(
            text="I'll create the file.",
            tool_calls=[
                ToolCall(
                    id="t1", name="write_file", args={"path": "result.py", "content": "# done"}
                )
            ],
        )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        max_cycles=0,
        max_parallel_tasks=5,
        workspace=tmp_path,
        provider=Provider.ANTHROPIC,
        model="x",
        planner_model="x",
        worker_model="x",
        tester_model="x",
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = _settings(tmp_path)

    def make(goal, output=None):
        return Orchestrator(goal=goal, client=FakeLLMClient(settings), output=output)

    monkeypatch.setattr(server_module, "Orchestrator", make)

    async def fake_run_tests(self):
        return "no tests collected"

    monkeypatch.setattr("furrow.agents.tester.TesterAgent._run_tests", fake_run_tests)
    return TestClient(server_module.app, raise_server_exceptions=False)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Furrow" in r.text


def test_websocket_streams(client, tmp_path):
    received = []
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "create a placeholder file"})
        for _ in range(50):
            msg = ws.receive_text()
            if msg is None:
                break
            received.append(msg)
            if "closed" in msg:
                break
    blob = "\n".join(received)
    assert "Goal complete" in blob or "Cycle 1" in blob
    assert (tmp_path / "result.py").exists()
