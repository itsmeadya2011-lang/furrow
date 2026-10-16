from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def _anthropic_response(text):
    mock_content = Mock()
    mock_content.text = text
    mock_response = Mock()
    mock_response.content = [mock_content]
    return mock_response


def _openai_response(text):
    mock_message = Mock()
    mock_message.content = text
    mock_choice = Mock()
    mock_choice.message = mock_message
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    return mock_response


async def test_anthropic_provider_works():
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=_anthropic_response("hello anthropic"))
    with patch("furrow.llm.AsyncAnthropic", return_value=mock_client):
        client = LLMClient(
            settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
        )
        result = await client.complete("prompt", system="sys", model="claude")
    assert result == "hello anthropic"
    mock_client.messages.create.assert_called_once_with(
        model="claude",
        max_tokens=4096,
        system="sys",
        messages=[{"role": "user", "content": "prompt"}],
    )


async def test_openai_provider_works():
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_openai_response("hello openai"))
    with patch("furrow.llm.AsyncOpenAI", return_value=mock_client):
        client = LLMClient(settings=Settings(provider=Provider.OPENAI, openai_api_key="test-key"))
        result = await client.complete("prompt", system="sys", model="gpt")
    assert result == "hello openai"
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt"
    assert kwargs["messages"][0]["role"] == "system"


async def test_ollama_provider_works():
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_openai_response("hello ollama"))
    with patch("furrow.llm.AsyncOpenAI", return_value=mock_client) as m_openai:
        client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
        result = await client.complete("prompt")
    assert result == "hello ollama"
    m_openai.assert_called_once()
    _, kwargs = m_openai.call_args
    assert kwargs["base_url"] == "http://localhost:11434"
    assert kwargs["api_key"] == "ollama"


async def test_unsupported_provider_raises_value_error():
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="k"))
    client.settings.provider = "bogus"
    try:
        await client.complete("prompt")
    except ValueError as e:
        assert "Unsupported provider" in str(e)
    else:
        raise AssertionError("expected ValueError")


async def test_retry_behavior_transient_failure():
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(
        side_effect=[Exception("transient"), Exception("transient"), _anthropic_response("ok")]
    )
    with patch("furrow.llm.AsyncAnthropic", return_value=mock_client):
        client = LLMClient(
            settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
        )
        result = await client.complete("prompt")
    assert result == "ok"
    assert mock_client.messages.create.call_count == 3


async def test_read_file(tmp_path):
    target = tmp_path / "sub" / "file.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("file content")

    mock_file = AsyncMock()
    mock_file.read = AsyncMock(return_value="file content")
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_file)
    cm.__aexit__ = AsyncMock(return_value=False)

    client = LLMClient(settings=Settings())
    with patch("furrow.llm.aiofiles.open", return_value=cm):
        content = await client.read_file(target)
    assert content == "file content"


async def test_write_file(tmp_path):
    target = tmp_path / "sub" / "file.txt"
    captured = {}

    def capture_open(path, mode):
        p = Path(path)
        captured["path"] = str(p)
        captured["mode"] = mode
        f = AsyncMock()

        async def write(content):
            captured["content"] = content

        f.write = write
        fcm = AsyncMock()
        fcm.__aenter__ = AsyncMock(return_value=f)
        fcm.__aexit__ = AsyncMock(return_value=False)
        return fcm

    client = LLMClient(settings=Settings())
    with patch("furrow.llm.aiofiles.open", side_effect=capture_open):
        await client.write_file(target, "write content")
    assert captured["content"] == "write content"
    assert Path(captured["path"]).name == "file.txt"
    assert captured["mode"] == "w"


def test_list_files(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")

    client = LLMClient(settings=Settings())
    files = client.list_files(tmp_path)
    assert "a.txt" in files
    assert "sub/b.txt" in files
