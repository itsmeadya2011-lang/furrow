import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def _clean_settings(monkeypatch: pytest.MonkeyPatch, **kwargs) -> Settings:
    for key in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "FURROW_ANTHROPIC_API_KEY",
        "FURROW_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    return Settings(**kwargs)


async def test_complete_anthropic_without_api_key_raises(monkeypatch):
    settings = _clean_settings(
        monkeypatch, provider=Provider.ANTHROPIC, anthropic_api_key=None
    )
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        await client.complete("hello")


async def test_complete_openai_without_api_key_raises(monkeypatch):
    settings = _clean_settings(
        monkeypatch, provider=Provider.OPENAI, openai_api_key=None
    )
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        await client.complete("hello")


async def test_anthropic_property_raises_without_api_key(monkeypatch):
    settings = _clean_settings(
        monkeypatch, provider=Provider.ANTHROPIC, anthropic_api_key=None
    )
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        _ = client.anthropic


async def test_openai_property_raises_without_api_key(monkeypatch):
    settings = _clean_settings(
        monkeypatch, provider=Provider.OPENAI, openai_api_key=None
    )
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        _ = client.openai
