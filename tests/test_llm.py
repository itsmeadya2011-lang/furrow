import pytest
from pathlib import Path

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


class TestLLMClientInstantiation:
    def test_instantiation_default(self):
        client = LLMClient()
        assert client.settings is not None
        assert client._anthropic is None
        assert client._openai is None

    def test_instantiation_with_settings(self):
        settings = Settings()
        client = LLMClient(settings=settings)
        assert client.settings is settings


class TestLLMClientComplete:
    @pytest.mark.asyncio
    async def test_complete_unsupported_provider_raises(self):
        settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("test prompt")


class TestLLMClientReadWriteFile:
    @pytest.mark.asyncio
    async def test_read_file_success(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        client = LLMClient()
        content = await client.read_file(str(test_file))
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_write_file_creates_content(self, tmp_path):
        client = LLMClient()
        target = tmp_path / "subdir" / "test.txt"
        await client.write_file(str(target), "file content")
        assert target.read_text() == "file content"
        assert target.parent.exists()

    @pytest.mark.asyncio
    async def test_write_file_overwrites_existing(self, tmp_path):
        target = tmp_path / "test.txt"
        target.write_text("old content")
        client = LLMClient()
        await client.write_file(str(target), "new content")
        assert target.read_text() == "new content"


class TestLLMClientListFiles:
    def test_list_files_empty_directory(self, tmp_path):
        client = LLMClient()
        files = client.list_files(str(tmp_path))
        assert files == []

    def test_list_files_with_files(self, tmp_path):
        (tmp_path / "a.py").write_text("print(1)")
        (tmp_path / "b.txt").write_text("hello")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").write_text("print(2)")
        client = LLMClient()
        files = client.list_files(str(tmp_path))
        assert "a.py" in files
        assert "b.txt" in files
        assert "sub" in files or "sub/c.py" in files

    def test_list_files_nonexistent_directory(self, tmp_path):
        client = LLMClient()
        missing = tmp_path / "does_not_exist"
        files = client.list_files(str(missing))
        assert files == []
