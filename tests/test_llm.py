from __future__ import annotations

import pytest
from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_provider_validation():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


async def test_unsupported_provider_raises():
    custom_settings = Settings()
    custom_settings.provider = "invalid_provider"
    client = LLMClient(settings=custom_settings)
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("test prompt")
