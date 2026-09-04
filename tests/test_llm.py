from unittest.mock import AsyncMock, patch

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


async def test_ollama_provider_dispatch():
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)
    with patch.object(
        client, '_complete_ollama', new_callable=AsyncMock, return_value="ollama response"
    ):
        result = await client.complete("test prompt")
        assert result == "ollama response"


def test_retry_decorator_applied():
    assert hasattr(LLMClient.complete, 'retry')


def test_ollama_missing_url_uses_default():
    settings = Settings()
    assert settings.ollama_base_url == "http://localhost:11434"
