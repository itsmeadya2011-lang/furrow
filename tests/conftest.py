"""Shared fixtures for the furrow test suite.

Two rules are enforced for every test in this directory:

* nothing may reach the network (real provider clients are replaced by a
  function that raises),
* nothing may shell out to a real test runner (the tester agent's runner is
  stubbed), and every write happens inside ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Provider, Settings
from tests.helpers import FakeLLMClient


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make it impossible for a test to build a real provider client."""

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("network access is not allowed in the furrow test suite")

    monkeypatch.setattr("furrow.llm.AsyncAnthropic", _boom)
    monkeypatch.setattr("furrow.llm.AsyncOpenAI", _boom)
    monkeypatch.setattr("furrow.llm.httpx.AsyncClient", _boom)


@pytest.fixture(autouse=True)
def stub_test_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never run a real test suite from inside a test."""

    async def _fake_run_tests(self: TesterAgent) -> str:
        return "stubbed test output"

    monkeypatch.setattr(TesterAgent, "_run_tests", _fake_run_tests)


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """A workspace directory inside ``tmp_path``; nothing is written outside it."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def settings(tmp_workspace: Path) -> Settings:
    """Settings with sane defaults, no API keys and a temporary workspace."""
    return Settings(
        provider=Provider.ANTHROPIC,
        model="test-model",
        planner_model="test-planner-model",
        worker_model="test-worker-model",
        tester_model="test-tester-model",
        anthropic_api_key=None,
        openai_api_key=None,
        ollama_base_url="http://localhost:11434",
        max_parallel_tasks=5,
        max_cycles=0,
        max_consecutive_failures=3,
        workspace=tmp_workspace,
        state_file=tmp_workspace / "state.json",
        log_level="INFO",
        apply_worker_edits=True,
    )


@pytest.fixture
def client(settings: Settings) -> FakeLLMClient:
    """A fake LLM client with no scripted responses yet."""
    return FakeLLMClient(settings)
