from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def settings_anthropic():
    return Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")


@pytest.fixture
def settings_openai():
    return Settings(provider=Provider.OPENAI, openai_api_key="test-key")


@pytest.fixture
def settings_ollama():
    return Settings(provider=Provider.OLLAMA, ollama_base_url="http://custom:11434")


@pytest.fixture
def settings_unsupported():
    class MockSettings:
        provider = "unsupported"
        model = "test-model"
        anthropic_api_key = None
        openai_api_key = None
        ollama_base_url = "http://localhost:11434"
    return MockSettings()


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_provider_routing_anthropic(self, settings_anthropic):
        client = LLMClient(settings=settings_anthropic)
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "anthropic response"
            result = await client.complete("test prompt")
            assert result == "anthropic response"
            mock_complete.assert_called_once_with("test prompt", "", settings_anthropic.model)

    @pytest.mark.asyncio
    async def test_provider_routing_openai(self, settings_openai):
        client = LLMClient(settings=settings_openai)
        with patch.object(client, "_complete_openai", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "openai response"
            result = await client.complete("test prompt")
            assert result == "openai response"
            mock_complete.assert_called_once_with("test prompt", "", settings_openai.model)

    @pytest.mark.asyncio
    async def test_provider_routing_ollama(self, settings_ollama):
        client = LLMClient(settings=settings_ollama)
        with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "ollama response"
            result = await client.complete("test prompt")
            assert result == "ollama response"
            mock_complete.assert_called_once_with("test prompt", "", settings_ollama.model)

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises_value_error(self, settings_unsupported):
        client = LLMClient(settings=settings_unsupported)
        with pytest.raises(ValueError, match="Unsupported provider: unsupported"):
            await client.complete("test prompt")

    def test_ollama_client_uses_correct_base_url(self, settings_ollama):
        client = LLMClient(settings=settings_ollama)
        ollama_client = client.ollama
        assert str(ollama_client.base_url) == "http://custom:11434"
        assert ollama_client.api_key == "ollama"

    def test_ollama_client_default_base_url(self):
        settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)
        ollama_client = client.ollama
        assert str(ollama_client.base_url) == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_complete_passes_system_and_model(self, settings_anthropic):
        client = LLMClient(settings=settings_anthropic)
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "response"
            await client.complete("prompt", system="system prompt", model="custom-model")
            mock_complete.assert_called_once_with("prompt", "system prompt", "custom-model")

    @pytest.mark.asyncio
    async def test_retry_behavior_on_transient_failure(self, settings_anthropic):
        client = LLMClient(settings=settings_anthropic)
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                from openai import APITimeoutError
                raise APITimeoutError("timeout")
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="success after retries")]
            return mock_response

        with patch.object(client.anthropic.messages, "create", side_effect=mock_create):
            result = await client.complete("test prompt")
            assert result == "success after retries"
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_exception(self, settings_anthropic):
        client = LLMClient(settings=settings_anthropic)

        async def always_fail(*args, **kwargs):
            from openai import APITimeoutError
            raise APITimeoutError("timeout")

        with patch.object(client.anthropic.messages, "create", side_effect=always_fail):
            from openai import APITimeoutError
            with pytest.raises(APITimeoutError):
                await client.complete("test prompt")

    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        settings = Settings()
        client = LLMClient(settings=settings)
        test_file = tmp_path / "test.txt"
        test_file.write_text("file content")
        content = await client.read_file(test_file)
        assert content == "file content"

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path):
        settings = Settings()
        client = LLMClient(settings=settings)
        test_file = tmp_path / "subdir" / "test.txt"
        await client.write_file(test_file, "new content")
        assert test_file.read_text() == "new content"

    def test_list_files(self, tmp_path):
        settings = Settings()
        client = LLMClient(settings=settings)
        (tmp_path / "file1.txt").write_text("1")
        (tmp_path / "file2.txt").write_text("2")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.txt").write_text("3")
        files = client.list_files(tmp_path)
        assert set(files) == {"file1.txt", "file2.txt", "subdir/file3.txt"}

    def test_list_files_nonexistent_directory(self):
        settings = Settings()
        client = LLMClient(settings=settings)
        files = client.list_files("/nonexistent/path")
        assert files == []


class TestLLMClientProperties:
    def test_anthropic_property_creates_client_once(self, settings_anthropic):
        client = LLMClient(settings=settings_anthropic)
        with patch("furrow.llm.AsyncAnthropic") as mock_anthropic:
            mock_instance = MagicMock()
            mock_anthropic.return_value = mock_instance
            c1 = client.anthropic
            c2 = client.anthropic
            assert c1 is c2
            mock_anthropic.assert_called_once()

    def test_openai_property_creates_client_once(self, settings_openai):
        client = LLMClient(settings=settings_openai)
        with patch("furrow.llm.AsyncOpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            c1 = client.openai
            c2 = client.openai
            assert c1 is c2
            mock_openai.assert_called_once()

    def test_anthropic_property_raises_without_api_key(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
        client = LLMClient(settings=settings)
        with patch.dict("os.environ", {}, clear=True), pytest.raises(
            ValueError, match="ANTHROPIC_API_KEY is not set"
        ):
            _ = client.anthropic

    def test_openai_property_raises_without_api_key(self):
        settings = Settings(provider=Provider.OPENAI, openai_api_key=None)
        client = LLMClient(settings=settings)
        with patch.dict("os.environ", {}, clear=True), pytest.raises(
            ValueError, match="OPENAI_API_KEY is not set"
        ):
            _ = client.openai