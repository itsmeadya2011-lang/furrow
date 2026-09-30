from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


async def test_complete_unsupported_provider():
    fake_settings = Settings.model_construct(provider="invalid_provider", model="test-model")
    client = LLMClient(settings=fake_settings)

    with patch("asyncio.sleep", new=AsyncMock()), pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("prompt")


async def test_read_write_file(tmp_path):
    client = LLMClient(settings=Mock())

    test_file = tmp_path / "subdir" / "test.txt"
    content = "hello world"
    await client.write_file(str(test_file), content)

    read_content = await client.read_file(str(test_file))
    assert read_content == content


def test_list_files(tmp_path):
    client = LLMClient(settings=Mock())

    (tmp_path / "file1.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "file2.txt").write_text("b")

    files = client.list_files(str(tmp_path))
    assert set(files) == {"file1.txt", str(Path("sub") / "file2.txt")}


async def test_ollama_provider_posts_to_correct_url():
    fake_settings = Settings.model_construct(
        provider=Provider.OLLAMA,
        model="test-model",
        ollama_base_url="http://test-ollama:1234",
    )
    client = LLMClient(settings=fake_settings)

    mock_response = Mock()
    mock_response.json.return_value = {"response": "test response"}

    mock_http_client = MagicMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=None)
    mock_http_client.post = AsyncMock(return_value=mock_response)

    with patch("furrow.llm.httpx.AsyncClient", return_value=mock_http_client):
        result = await client.complete("my prompt", system="my system")

    assert result == "test response"
    mock_http_client.post.assert_called_once()
    call_args = mock_http_client.post.call_args
    assert call_args[0][0] == "http://test-ollama:1234/api/generate"
    assert call_args[1]["json"] == {
        "model": "test-model",
        "prompt": "my prompt",
        "system": "my system",
        "stream": False,
    }
