"""Tests for LLMClient and configuration models."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Optional

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.llm import LLMClient


def _make_settings(**overrides) -> Settings:
    kwargs = dict(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        planner_model="claude-3-5-haiku-20241022",
        worker_model="claude-3-5-sonnet-20241022",
        tester_model="claude-3-5-sonnet-20241022",
        anthropic_api_key="sk-test",
        max_parallel_tasks=5,
        max_cycles=0,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0
        assert s.log_level == "INFO"

    def test_settings_from_env(self, monkeypatch):
        monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "10")
        monkeypatch.setenv("FURROW_MAX_CYCLES", "5")
        s = Settings()
        assert s.max_parallel_tasks == 10
        assert s.max_cycles == 5

    def test_settings_model_override(self):
        s = Settings(model="gpt-4o")
        assert s.model == "gpt-4o"

    def test_provider_enum(self):
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OPENAI.value == "openai"
        assert Provider.OLLAMA.value == "ollama"


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_complete_anthropic(self):
        settings = _make_settings(provider="anthropic")
        client = LLMClient(settings=settings)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello from Anthropic")]
        client._anthropic = MagicMock()
        client._anthropic.messages.create = AsyncMock(return_value=mock_response)

        result = await client.complete("say hi", system="you are helpful")
        assert result == "Hello from Anthropic"

    @pytest.mark.asyncio
    async def test_complete_openai(self):
        settings = _make_settings(provider="openai", openai_api_key="sk-test")
        client = LLMClient(settings=settings)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello from OpenAI"))]
        client._openai = MagicMock()
        client._openai.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.complete("say hi")
        assert result == "Hello from OpenAI"

    @pytest.mark.asyncio
    async def test_complete_unsupported_provider(self):
        settings = _make_settings(provider="anthropic")
        # Monkey-patch to use an invalid provider
        settings.provider = "invalid"
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("say hi")

    def test_anthropic_requires_api_key(self):
        settings = Settings(provider="anthropic", anthropic_api_key=None)
        # Clear env var too
        client = LLMClient(settings=settings)
        import os
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                _ = client.anthropic
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_openai_requires_api_key(self):
        settings = Settings(provider="openai", openai_api_key=None)
        client = LLMClient(settings=settings)
        import os
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                _ = client.openai
        finally:
            if old:
                os.environ["OPENAI_API_KEY"] = old

    @pytest.mark.asyncio
    async def test_read_write_file(self, tmp_path):
        settings = _make_settings()
        client = LLMClient(settings=settings)

        test_file = tmp_path / "test.txt"
        await client.write_file(test_file, "hello world")
        content = await client.read_file(test_file)
        assert content == "hello world"
        # Parent dir should be created
        nested = tmp_path / "sub" / "deep" / "file.txt"
        await client.write_file(nested, "nested content")
        assert await client.read_file(nested) == "nested content"

    def test_list_files(self, tmp_path):
        settings = _make_settings()
        client = LLMClient(settings=settings)

        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.txt").write_text("c")

        files = client.list_files(tmp_path)
        assert "a.txt" in files
        assert "b.txt" in files
        assert str(Path("sub") / "c.txt") in files
        assert len(files) == 3

    def test_list_files_nonexistent(self):
        settings = _make_settings()
        client = LLMClient(settings=settings)
        assert client.list_files("/nonexistent/path") == []

    @pytest.mark.asyncio
    async def test_close_releases_http_client(self):
        from unittest.mock import MagicMock
        import httpx  # noqa: F401

        settings = _make_settings(provider="ollama", ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=settings)
        client._http_client = MagicMock()
        client._http_client.aclose = AsyncMock()
        await client.close()
        assert client._http_client is None


class TestConfigModels:
    def test_task_model_defaults(self):
        t = TaskModel(id="1", description="do something")
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None

    def test_plan_model(self):
        t = TaskModel(id="1", description="do something", dependencies=["0"])
        p = Plan(tasks=[t], rationale="test")
        assert len(p.tasks) == 1
        assert p.tasks[0].dependencies == ["0"]

    def test_test_result_model(self):
        t = TestResult(passed=False, summary="failures", failures=["error1"])
        assert t.passed is False
        assert "error1" in t.failures

    def test_plan_requires_rationale(self):
        with pytest.raises(ValidationError):
            Plan(tasks=[], rationale=None)  # type: ignore[arg-type]
