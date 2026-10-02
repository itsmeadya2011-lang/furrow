"""Tests for the LLMClient."""

import json
import os

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def make_settings(provider: Provider = Provider.ANTHROPIC, **kwargs) -> Settings:
    """Create settings with a specific provider."""
    defaults = {
        "provider": provider,
        "anthropic_api_key": "test-anthropic-key",
        "openai_api_key": "test-openai-key",
        "ollama_base_url": "http://localhost:11434",
        "workspace": ".",
    }
    defaults.update(kwargs)
    return Settings(**defaults)


class FakeAnthropic:
    """Fake Anthropic client for testing."""

    class Messages:
        def __init__(self, text="test response"):
            self._text = text

        async def create(self, **kwargs):
            class Response:
                def __init__(self, text):
                    class Content:
                        def __init__(self, content_text):
                            self.text = content_text

                    self.content = [Content(text)]

            return Response(self._text)

    @property
    def messages(self):
        return self.Messages()


class FakeOpenAI:
    """Fake OpenAI client for testing."""

    class Chat:
        class Completions:
            def __init__(self, text="test response"):
                self._text = text

            async def create(self, **kwargs):
                class Message:
                    def __init__(self, msg_text):
                        self.content = msg_text

                class Choice:
                    def __init__(self, choice_text):
                        self.message = Message(choice_text)

                class Response:
                    def __init__(self, resp_text):
                        self.choices = [Choice(resp_text)]

                return Response(self._text)

        def __init__(self):
            self.completions = self.Completions()

    chat = Chat()


@pytest.fixture
def anthropic_client():
    client = LLMClient(settings=make_settings(Provider.ANTHROPIC))
    client._anthropic = FakeAnthropic()
    return client


@pytest.fixture
def openai_client():
    client = LLMClient(settings=make_settings(Provider.OPENAI))
    client._openai = FakeOpenAI()
    return client


def test_llm_client_init():
    """LLMClient initializes with default settings."""
    client = LLMClient()
    assert client.settings is not None
    assert client._anthropic is None
    assert client._openai is None


def test_llm_client_custom_settings():
    """LLMClient accepts custom Settings."""
    custom = make_settings(Provider.OPENAI, model="gpt-4")
    client = LLMClient(settings=custom)
    assert client.settings.provider == Provider.OPENAI
    assert client.settings.model == "gpt-4"


@pytest.mark.asyncio
async def test_anthropic_complete(anthropic_client):
    """_complete_anthropic returns text from Anthropic API."""
    result = await anthropic_client.complete("Hello", system="You are helpful")
    assert result == "test response"


@pytest.mark.asyncio
async def test_openai_complete(openai_client):
    """_complete_openai returns text from OpenAI API."""
    result = await openai_client.complete("Hello", system="You are helpful")
    assert result == "test response"


@pytest.mark.asyncio
async def test_complete_with_model_override(anthropic_client):
    """Complete method uses model override when provided."""
    result = await anthropic_client.complete("Hello", model="claude-3-opus")
    assert result == "test response"


@pytest.mark.asyncio
async def test_converse_anthropic(anthropic_client):
    """Converse method works with multi-turn messages."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
    ]
    result = await anthropic_client.converse(messages)
    assert result == "test response"


@pytest.mark.asyncio
async def test_converse_openai(openai_client):
    """Converse method works with OpenAI multi-turn."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]
    result = await openai_client.converse(messages)
    assert result == "test response"


@pytest.mark.asyncio
async def test_complete_json_success():
    """Complete_json parses valid JSON response."""
    client = LLMClient(settings=make_settings(Provider.ANTHROPIC))
    client._anthropic = FakeAnthropic()

    async def fake_complete(prompt, system="", model=None, temperature=0.7, max_tokens=4096):
        return json.dumps({"key": "value", "num": 42})

    client.complete = fake_complete
    result = await client.complete_json("test")
    assert result == {"key": "value", "num": 42}


@pytest.mark.asyncio
async def test_read_write_file(tmp_path):
    """LLMClient can read and write files."""
    client = LLMClient(settings=make_settings())
    test_file = tmp_path / "test.txt"
    content = "Hello, Furrow!"
    await client.write_file(test_file, content)
    result = await client.read_file(test_file)
    assert result == content


@pytest.mark.asyncio
async def test_read_file_safe_returns_none():
    """read_file_safe returns None for nonexistent files."""
    client = LLMClient(settings=make_settings())
    result = await client.read_file_safe("/nonexistent/path/file.txt")
    assert result is None


@pytest.mark.asyncio
async def test_read_file_safe_returns_content(tmp_path):
    """read_file_safe returns content for existing files."""
    client = LLMClient(settings=make_settings())
    test_file = tmp_path / "exists.txt"
    await client.write_file(test_file, "content here")
    result = await client.read_file_safe(test_file)
    assert result == "content here"


@pytest.mark.asyncio
async def test_list_files(tmp_path):
    """list_files returns relative file paths."""
    client = LLMClient(settings=make_settings())
    (tmp_path / "a.py").write_text("content")
    (tmp_path / "b.py").write_text("content")
    result = await client.list_files(tmp_path)
    assert "a.py" in result
    assert "b.py" in result


@pytest.mark.asyncio
async def test_list_files_empty_dir(tmp_path):
    """list_files returns empty list for empty directory."""
    client = LLMClient(settings=make_settings())
    result = await client.list_files(tmp_path)
    assert result == []


@pytest.mark.asyncio
async def test_list_files_nonexistent_dir():
    """list_files returns empty list for nonexistent directory."""
    client = LLMClient(settings=make_settings())
    result = await client.list_files("/nonexistent/dir")
    assert result == []


def test_provider_enum():
    """Provider enum has all expected values."""
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_unsupported_provider_raises():
    """Complete raises ValueError for unsupported providers."""
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="key")
    # Temporarily set provider to a value that won't match
    client = LLMClient(settings=settings)

    async def _raise_on_complete():
        # Simulate unsupported provider by using object()
        client.settings.provider = "unknown"
        try:
            await client.complete("test")
            assert False, "Should have raised"
        except ValueError as e:
            assert "Unsupported provider" in str(e)
        finally:
            client.settings.provider = Provider.ANTHROPIC

    import asyncio

    asyncio.run(_raise_on_complete())


@pytest.mark.asyncio
async def test_anthropic_api_key_missing():
    """Anthropic property raises if no API key."""
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    # Clear env var too
    old = os.environ.pop("ANTHROPIC_API_KEY", None)
    client = LLMClient(settings=settings)
    try:
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            _ = client.anthropic
    finally:
        if old:
            os.environ["ANTHROPIC_API_KEY"] = old


@pytest.mark.asyncio
async def test_openai_api_key_missing():
    """OpenAI property raises if no API key."""
    settings = Settings(provider=Provider.OPENAI, openai_api_key=None)
    old = os.environ.pop("OPENAI_API_KEY", None)
    client = LLMClient(settings=settings)
    try:
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            _ = client.openai
    finally:
        if old:
            os.environ["OPENAI_API_KEY"] = old
