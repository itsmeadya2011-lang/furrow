"""Tests for furrow.llm — LLMClient settings handling, path resolution, and file I/O."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient

# ---------------------------------------------------------------------------
#  Settings handling
# ---------------------------------------------------------------------------

def test_client_uses_defaults_when_settings_none():
    """LLMClient(settings=None) should not crash — must fall back to Settings()."""
    client = LLMClient(settings=None)
    assert client.settings is not None
    assert isinstance(client.settings, Settings)
    assert client.settings.provider == Provider.ANTHROPIC


def test_client_uses_provided_settings():
    custom = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=custom)
    assert client.settings is custom
    assert client.settings.provider == Provider.OPENAI


# ---------------------------------------------------------------------------
#  Path resolution
# ---------------------------------------------------------------------------

def test_resolve_path_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    client = LLMClient()
    resolved = client.resolve_path("src/file.py")
    assert resolved == tmp_path / "src" / "file.py"


def test_resolve_path_absolute():
    client = LLMClient()
    abs_path = Path("/tmp/test_file.txt")
    resolved = client.resolve_path(abs_path)
    assert resolved == abs_path


def test_resolve_path_with_custom_workspace():
    ws = Path("/custom/workspace")
    client = LLMClient(settings=Settings(workspace=ws))
    resolved = client.resolve_path("src/main.py")
    assert resolved == ws / "src" / "main.py"


# ---------------------------------------------------------------------------
#  File I/O
# ---------------------------------------------------------------------------

async def test_read_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    client = LLMClient()
    test_file = tmp_path / "data.txt"
    test_file.write_text("hello world")
    content = await client.read_file("data.txt")
    assert content == "hello world"


async def test_write_file_creates_parents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    client = LLMClient()
    await client.write_file("deep/nested/file.py", "print('hi')")
    result = (tmp_path / "deep" / "nested" / "file.py").read_text()
    assert result == "print('hi')"


async def test_list_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    client = LLMClient()
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("")
    files = client.list_files(tmp_path)
    files_set = set(files)
    assert "a.py" in files_set
    assert "b.txt" in files_set
    assert "sub/c.py" in files_set


async def test_list_files_nonexistent_dir():
    client = LLMClient()
    assert client.list_files("/nonexistent/directory") == []


# ---------------------------------------------------------------------------
#  Provider dispatch
# ---------------------------------------------------------------------------

async def test_complete_anthropic():
    """Test that Anthropic provider is called correctly."""
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="key"))
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello from Anthropic")]
    client._anthropic = MagicMock()
    client._anthropic.messages.create = AsyncMock(return_value=mock_response)
    result = await client.complete("hi", system="sys", model="claude-3")
    assert result == "Hello from Anthropic"
    client._anthropic.messages.create.assert_awaited_once()


async def test_complete_openai():
    """Test that OpenAI provider is called correctly."""
    client = LLMClient(settings=Settings(provider=Provider.OPENAI, openai_api_key="key"))
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello from OpenAI"))]
    client._openai = MagicMock()
    client._openai.chat.completions.create = AsyncMock(return_value=mock_response)
    result = await client.complete("hi", model="gpt-4")
    assert result == "Hello from OpenAI"


async def test_complete_unsupported_provider():
    """Test that an unknown provider raises ValueError."""
    # We can't easily create an invalid Provider, but we can test the else branch
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="key"))
    # Monkey-patch settings to use an invalid provider value
    with (
        patch.object(client.settings, "provider", "invalid"),
        pytest.raises(ValueError, match="Unsupported provider"),
    ):
        await client.complete("hi")


async def test_ollama_not_called_for_anthropic():
    """Ensure Ollama HTTP client is not created when using Anthropic."""
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="key"))
    client._anthropic = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ok")]
    client._anthropic.messages.create = AsyncMock(return_value=mock_response)
    await client.complete("hi", model="claude-3")
    assert client._http is None
