"""Tests for the Furrow LLM client."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import DEFAULT_TOOLS, LLMClient

# --------------------------------------------------------------------------- #
# Tool definitions
# --------------------------------------------------------------------------- #


def test_default_tools_defined():
    """Ensure all required default tools are present."""
    tool_names = {t["name"] for t in DEFAULT_TOOLS}
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "list_files" in tool_names
    assert "execute_command" in tool_names
    assert "glob_files" in tool_names
    assert "grep_search" in tool_names
    assert len(DEFAULT_TOOLS) == 6


def test_default_tools_have_required_fields():
    for tool in DEFAULT_TOOLS:
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool
        assert tool["parameters"]["type"] == "object"
        assert "properties" in tool["parameters"]


# --------------------------------------------------------------------------- #
# LLMClient basic tests
# --------------------------------------------------------------------------- #


def test_llm_client_init_lazy():
    """Clients should not be created until accessed (lazy init)."""
    client = LLMClient()
    assert client._anthropic is None
    assert client._openai is None


def test_llm_client_provider_openai():
    """Test that OpenAI provider can be configured."""
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)
    assert client.settings.provider == Provider.OPENAI


def test_llm_client_provider_ollama():
    """Test that Ollama provider can be configured."""
    settings = Settings(
        provider=Provider.OLLAMA,
        ollama_base_url="http://localhost:11434",
    )
    client = LLMClient(settings=settings)
    assert client.settings.provider == Provider.OLLAMA
    assert client.settings.ollama_base_url == "http://localhost:11434"


def test_llm_client_ollama_property():
    """Ollama should configure an OpenAI client with the Ollama base URL."""
    settings = Settings(
        provider=Provider.OLLAMA,
        openai_api_key="ollama",
        ollama_base_url="http://localhost:11434",
    )
    client = LLMClient(settings=settings)
    ollama_client = client.ollama
    assert ollama_client is not None


def test_llm_client_anthropic_missing_key():
    """Should raise ValueError when no API key is set for Anthropic."""
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    client = LLMClient(settings=settings)
    env = {}
    with patch.dict("os.environ", env, clear=True), pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _ = client.anthropic


def test_llm_client_openai_missing_key():
    """Should raise ValueError when no API key is set for OpenAI."""
    settings = Settings(provider=Provider.OPENAI, openai_api_key=None)
    client = LLMClient(settings=settings)
    env = {}
    with patch.dict("os.environ", env, clear=True), pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _ = client.openai


# --------------------------------------------------------------------------- #
# Tool execution tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tool_read_file(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)

    result = await client._execute_tool("read_file", {"path": str(test_file)})
    assert result == "hello world"


@pytest.mark.asyncio
async def test_tool_write_file(tmp_path):
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)

    test_file = tmp_path / "output.txt"
    result = await client._execute_tool(
        "write_file", {"path": str(test_file), "content": "test content"}
    )
    assert "File written" in result
    assert test_file.read_text() == "test content"


@pytest.mark.asyncio
async def test_tool_list_files(tmp_path):
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)

    (tmp_path / "a.py").write_text("print('a')")
    (tmp_path / "b.py").write_text("print('b')")

    result = await client._execute_tool("list_files", {"directory": str(tmp_path)})
    assert "a.py" in result
    assert "b.py" in result


@pytest.mark.asyncio
async def test_tool_glob_files(tmp_path):
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)

    (tmp_path / "test_foo.py").write_text("# test")
    (tmp_path / "app.py").write_text("# app")

    result = await client._execute_tool(
        "glob_files", {"pattern": "test_*.py", "directory": str(tmp_path)}
    )
    assert "test_foo.py" in result
    assert "app.py" not in result


@pytest.mark.asyncio
async def test_tool_grep_search(tmp_path):
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)

    (tmp_path / "file.py").write_text("import os\nimport sys\n")
    result = await client._execute_tool(
        "grep_search", {"query": "import", "directory": str(tmp_path)}
    )
    assert "import" in result


@pytest.mark.asyncio
async def test_tool_grep_search_case_insensitive(tmp_path):
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)

    (tmp_path / "file.py").write_text("Import os\nIMPORT sys\n")
    result = await client._execute_tool(
        "grep_search",
        {"query": "import", "directory": str(tmp_path), "case_insensitive": True},
    )
    assert "Import" in result
    assert "IMPORT" in result


@pytest.mark.asyncio
async def test_tool_unknown_tool():
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)
    result = await client._execute_tool("nonexistent_tool", {})
    assert "Unknown tool" in result


# --------------------------------------------------------------------------- #
# Retry behavior tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_complete_has_retry_decorator():
    """Verify that complete() uses tenacity retry decorator."""
    client = LLMClient()
    assert hasattr(client.complete, "__wrapped__")


@pytest.mark.asyncio
async def test_complete_retries_on_failure():
    """complete() should retry on retryable exceptions."""
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)

    call_count = 0

    async def mock_complete(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise TimeoutError("network error")

    # The retry decorator wraps the function, so we test the underlying behavior
    # by patching the provider-specific method
    with (
        patch.object(client, "_complete_openai", side_effect=mock_complete),
        patch("os.environ", {"OPENAI_API_KEY": "test-key"}),
        pytest.raises(TimeoutError),
    ):
        # This should retry 3 times then fail
        await client.complete("test prompt")

    assert call_count == 3  # tenacity default stop_after_attempt(3)
