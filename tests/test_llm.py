import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def settings() -> Settings:
    return Settings(
        provider=Provider.ANTHROPIC,
        anthropic_api_key="test-key",
        max_cycles=5,
    )


def make_client(settings: Settings) -> LLMClient:
    return LLMClient(settings=settings)


class TestLLMClientComplete:
    @pytest.mark.asyncio
    async def test_unsupported_provider_raises(self):
        s = Settings(provider=Provider.OLLAMA)
        client = make_client(s)
        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("hello")

    @pytest.mark.asyncio
    async def test_complete_anthropic(self, settings: Settings):
        client = make_client(settings)
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="anthropic response")]
        client._anthropic = AsyncMock()
        client._anthropic.messages.create = AsyncMock(return_value=mock_response)
        result = await client.complete("hello")
        assert result == "anthropic response"

    @pytest.mark.asyncio
    async def test_complete_openai(self, settings: Settings):
        s = Settings(provider=Provider.OPENAI, openai_api_key="test-openai-key")
        client = make_client(s)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="openai response"))]
        client._openai = AsyncMock()
        client._openai.chat.completions.create = AsyncMock(return_value=mock_response)
        result = await client.complete("hello")
        assert result == "openai response"


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_file(self, settings: Settings, tmp_path):
        client = make_client(settings)
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        content = await client.read_file(str(f))
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_read_file_missing(self, settings: Settings):
        client = make_client(settings)
        with pytest.raises(FileNotFoundError):
            await client.read_file("/nonexistent/path/file.txt")


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_write_file_with_mkdir(self, settings: Settings, tmp_path):
        client = make_client(settings)
        f = tmp_path / "subdir" / "out.txt"
        await client.write_file(str(f), "content here")
        assert f.read_text() == "content here"

    @pytest.mark.asyncio
    async def test_write_file_overwrites(self, settings: Settings, tmp_path):
        client = make_client(settings)
        f = tmp_path / "out.txt"
        f.write_text("old")
        await client.write_file(str(f), "new")
        assert f.read_text() == "new"


class TestListFiles:
    def test_list_files_in_directory(self, settings: Settings, tmp_path):
        client = make_client(settings)
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y")
        files = client.list_files(tmp_path)
        assert sorted(files) == ["a.py", "b.py"]

    def test_list_files_empty_dir(self, settings: Settings, tmp_path):
        client = make_client(settings)
        files = client.list_files(tmp_path)
        assert files == []

    def test_list_files_nonexistent(self, settings: Settings):
        client = make_client(settings)
        files = client.list_files("/nonexistent/dir/xyz")
        assert files == []


class TestCompleteOllama:
    @pytest.mark.asyncio
    async def test_complete_ollama(self):
        s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        client = make_client(s)
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ollama answer"}
        mock_response.raise_for_status.return_value = None
        client._httpx_client = AsyncMock()
        client._httpx_client.post = AsyncMock(return_value=mock_response)
        result = await client.complete("hello")
        assert result == "ollama answer"

    @pytest.mark.asyncio
    async def test_complete_ollama_missing_response_key(self):
        s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        client = make_client(s)
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        client._httpx_client = AsyncMock()
        client._httpx_client.post = AsyncMock(return_value=mock_response)
        result = await client.complete("hello")
        assert result == ""
