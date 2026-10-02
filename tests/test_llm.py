"""Tests for the LLMClient: provider routing, Ollama support, retry behavior."""

from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def _make_settings(**overrides):
    """Create a Settings instance with optional overrides for testing."""
    defaults = {
        "provider": Provider.ANTHROPIC,
        "model": "test-model",
        "planner_model": "test-planner",
        "worker_model": "test-worker",
        "tester_model": "test-tester",
        "anthropic_api_key": "test-key",
        "openai_api_key": "test-key",
        "ollama_base_url": "http://localhost:11434",
        "max_parallel_tasks": 5,
        "max_cycles": 0,
        "workspace": pytest.importorskip("pathlib").Path.cwd(),
        "log_level": "INFO",
        "llm_timeout": 60,
        "max_retries": 3,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestLLMClientInit:
    def test_default_settings(self):
        client = LLMClient()
        assert client.settings is not None

    def test_custom_settings(self):
        custom = _make_settings(provider=Provider.OPENAI)
        client = LLMClient(settings=custom)
        assert client.settings.provider == Provider.OPENAI


class TestOllamaProvider:
    @pytest.mark.asyncio
    async def test_ollama_client_uses_correct_base_url(self):
        settings = _make_settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=settings)
        # Accessing the property triggers client creation
        ollama_client = client.ollama
        assert "ollama" == ollama_client.api_key
        assert str(ollama_client.base_url).startswith("http://localhost:11434")

    @pytest.mark.asyncio
    async def test_ollama_complete_calls_ollama_method(self):
        settings = _make_settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)

        with patch.object(client, "_complete_ollama", new=AsyncMock(return_value="ollama response")):
            result = await client.complete("test prompt")
            assert result == "ollama response"
            client._complete_ollama.assert_called_once()


class TestProviderRouting:
    @pytest.mark.asyncio
    async def test_unsupported_provider_raises(self):
        settings = _make_settings(provider=Provider.ANTHROPIC)
        # Monkey-patch to bypass enum
        settings.provider = "unknown"
        client = LLMClient(settings=settings)

        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("test")

    @pytest.mark.asyncio
    async def test_anthropic_routing(self):
        settings = _make_settings(provider=Provider.ANTHROPIC)
        client = LLMClient(settings=settings)

        with patch.object(client, "_complete_anthropic", new=AsyncMock(return_value="anthropic response")):
            result = await client.complete("test")
            assert result == "anthropic response"

    @pytest.mark.asyncio
    async def test_openai_routing(self):
        settings = _make_settings(provider=Provider.OPENAI)
        client = LLMClient(settings=settings)

        with patch.object(client, "_complete_openai", new=AsyncMock(return_value="openai response")):
            result = await client.complete("test")
            assert result == "openai response"


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self):
        """LLMClient should retry on transient exceptions."""
        settings = _make_settings(max_retries=3)
        client = LLMClient(settings=settings)

        call_count = {"n": 0}

        async def flaky_complete(prompt, system="", model=None, timeout=None):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("transient network error")
            return "success on third try"

        # We need to patch _complete_anthropic to use our flanky function
        with patch.object(client, "_complete_anthropic", side_effect=flaky_complete):
            result = await client.complete("test")

        assert result == "success on third try"
        assert call_count["n"] == 3

    @pytest.mark.asyncio
    async def test_stops_retrying_after_max_attempts(self):
        """Should stop retrying after max_retries and raise the last exception."""
        from openai import APIError

        settings = _make_settings(max_retries=2)
        client = LLMClient(settings=settings)

        call_count = {"n": 0}

        async def always_fail(prompt, system="", model=None, timeout=None):
            call_count["n"] += 1
            raise APIError(message="permanent error", request=None, body=None)

        with (
            patch.object(client, "_complete_anthropic", side_effect=always_fail),
            pytest.raises(APIError),
        ):
            await client.complete("test")

        assert call_count["n"] == 2


class TestFileOperations:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        client = LLMClient(settings=_make_settings())
        content = await client.read_file(str(test_file))
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_write_file_creates_dirs(self, tmp_path):
        nested_path = tmp_path / "subdir" / "deep" / "file.txt"

        client = LLMClient(settings=_make_settings())
        await client.write_file(str(nested_path), "test content")

        assert nested_path.exists()
        assert nested_path.read_text() == "test content"

    @pytest.mark.asyncio
    async def test_list_files(self, tmp_path):
        (tmp_path / "file1.py").write_text("x")
        (tmp_path / "file2.py").write_text("y")
        (tmp_path / "subdir").mkdir()

        client = LLMClient(settings=_make_settings())
        files = client.list_files(str(tmp_path))

        assert "file1.py" in files
        assert "file2.py" in files
        assert len(files) == 2  # should not include the subdir itself

    def test_list_files_nonexistent_dir(self):
        client = LLMClient(settings=_make_settings())
        files = client.list_files("/nonexistent/path/12345")
        assert files == []
