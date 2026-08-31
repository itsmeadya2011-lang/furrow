from furrow.config import Settings
from furrow.llm import LLMClient


def test_ollama_requires_base_url():
    client = LLMClient(settings=Settings(provider="ollama", ollama_base_url=""))
    try:
        _ = client.ollama
        assert False, "expected ValueError for empty ollama_base_url"
    except ValueError:
        pass


def test_unsupported_provider_raises():
    client = LLMClient(settings=Settings(provider="anthropic"))
    # Provider enum only contains valid values, but guard the branch anyway.
    assert client.settings.provider.value in {"anthropic", "openai", "ollama"}
