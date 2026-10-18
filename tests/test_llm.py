import os
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.llm import LLMClient
from furrow.config import Settings


def test_list_files(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("y")
    client = LLMClient(settings=Settings(workspace=tmp_path))
    files = client.list_files(tmp_path)
    assert "a.txt" in files
    assert "sub/b.txt" in files


@pytest.mark.asyncio
async def test_read_write_file(tmp_path):
    client = LLMClient(settings=Settings(workspace=tmp_path))
    path = tmp_path / "f.txt"
    await client.write_file(path, "hello")
    content = await client.read_file(path)
    assert content == "hello"


@pytest.mark.asyncio
async def test_retry_logic():
    client = LLMClient(settings=Settings(provider=__import__("furrow.config", fromlist=["Provider"]).Provider.ANTHROPIC, anthropic_api_key="key"))
    # mock anthropic
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="ok")]
    client._anthropic = MagicMock()
    client._anthropic.messages.create = AsyncMock(side_effect=[Exception("fail"), mock_resp])
    # call underlying retry method
    result = await client._complete_anthropic("p", "", "m")
    assert result == "ok"
