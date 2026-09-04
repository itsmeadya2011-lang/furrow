from pathlib import Path

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from furrow.llm import LLMClient
from furrow.config import Provider, Settings


@pytest.fixture
def llm_client(tmp_path: Path) -> LLMClient:
    return LLMClient(settings=Settings(workspace=tmp_path))


def test_llm_client_init_defaults():
    client = LLMClient()
    assert client._anthropic is None
    assert client._openai is None


def test_llm_client_init_with_settings(tmp_path: Path):
    s = Settings(workspace=tmp_path)
    client = LLMClient(settings=s)
    assert client.settings is s


def test_llm_client_missing_anthropic_key(tmp_path: Path):
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None, workspace=tmp_path))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        _ = client.anthropic


def test_llm_client_missing_openai_key(tmp_path: Path):
    client = LLMClient(settings=Settings(provider=Provider.OPENAI, openai_api_key=None, workspace=tmp_path))
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        _ = client.openai


@pytest.mark.asyncio
async def test_llm_read_write_delete_file(tmp_path: Path):
    client = LLMClient(settings=Settings(workspace=tmp_path))
    target = tmp_path / "test.txt"
    await client.write_file(target, "hello world")
    assert target.exists()
    content = await client.read_file(target)
    assert content == "hello world"
    await client.delete_file(target)
    assert not target.exists()


@pytest.mark.asyncio
async def test_llm_read_missing_file(tmp_path: Path):
    client = LLMClient(settings=Settings(workspace=tmp_path))
    with pytest.raises(FileNotFoundError):
        await client.read_file(tmp_path / "nonexistent.txt")


def test_llm_list_files(tmp_path: Path):
    client = LLMClient(settings=Settings(workspace=tmp_path))
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.py").write_text("b")
    files = client.list_files(tmp_path)
    assert set(files) == {"a.py", "b.py"}


def test_llm_list_files_empty(tmp_path: Path):
    client = LLMClient(settings=Settings(workspace=tmp_path))
    assert client.list_files(tmp_path) == []


@pytest.mark.asyncio
async def test_llm_complete_anthropic(tmp_path: Path):
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test", workspace=tmp_path))
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="response text")]
    mock_response.usage = MagicMock()
    with patch.object(client, "anthropic") as mock_anthropic:
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        result = await client.complete("prompt", model="claude-test")
    assert result == "response text"


@pytest.mark.asyncio
async def test_llm_complete_openai(tmp_path: Path):
    client = LLMClient(settings=Settings(provider=Provider.OPENAI, openai_api_key="test", workspace=tmp_path))
    mock_choice = MagicMock()
    mock_choice.message.content = "openai response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    with patch.object(client, "openai") as mock_openai:
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        result = await client.complete("prompt", model="gpt-test")
    assert result == "openai response"


@pytest.mark.asyncio
async def test_llm_complete_openai_empty_content(tmp_path: Path):
    client = LLMClient(settings=Settings(provider=Provider.OPENAI, openai_api_key="test", workspace=tmp_path))
    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    with patch.object(client, "openai") as mock_openai:
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        result = await client.complete("prompt", model="gpt-test")
    assert result == ""


@pytest.mark.asyncio
async def test_llm_complete_ollama(tmp_path: Path):
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434", workspace=tmp_path))
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "ollama response"}}
    mock_response.raise_for_status = MagicMock()
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await client.complete("prompt", model="llama3")
    assert result == "ollama response"
