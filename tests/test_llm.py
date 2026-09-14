import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


class TestLLMClientInstantiation:
    def test_instantiation_with_default_settings(self):
        settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)
        assert client.settings is settings
        assert client._anthropic is None
        assert client._openai is None

    def test_instantiation_with_custom_settings(self):
        settings = MagicMock()
        settings.provider = Provider.ANTHROPIC
        settings.model = "test-model"
        settings.anthropic_api_key = "key"
        settings.openai_api_key = None
        settings.ollama_base_url = "http://localhost:11434"
        client = LLMClient(settings=settings)
        assert client.settings == settings


class TestLLMClientReadFile:
    @pytest.mark.asyncio
    async def test_read_file_returns_content(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello world")

        client = LLMClient.__new__(LLMClient)
        client.settings = MagicMock()
        content = await client.read_file(str(file_path))
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_read_file_json_content(self, tmp_path):
        file_path = tmp_path / "data.json"
        file_path.write_text(json.dumps({"key": "value"}))

        client = LLMClient.__new__(LLMClient)
        client.settings = MagicMock()
        content = await client.read_file(str(file_path))
        assert json.loads(content) == {"key": "value"}


class TestLLMClientWriteFile:
    @pytest.mark.asyncio
    async def test_write_file_creates_file(self, tmp_path):
        file_path = tmp_path / "output.txt"

        client = LLMClient.__new__(LLMClient)
        client.settings = MagicMock()
        await client.write_file(str(file_path), "test content")

        assert file_path.exists()
        assert file_path.read_text() == "test content"

    @pytest.mark.asyncio
    async def test_write_file_creates_parent_dirs(self, tmp_path):
        file_path = tmp_path / "subdir" / "nested" / "output.txt"

        client = LLMClient.__new__(LLMClient)
        client.settings = MagicMock()
        await client.write_file(str(file_path), "deep content")

        assert file_path.exists()
        assert file_path.read_text() == "deep content"

    @pytest.mark.asyncio
    async def test_write_file_overwrites(self, tmp_path):
        file_path = tmp_path / "overwrite.txt"
        file_path.write_text("old")

        client = LLMClient.__new__(LLMClient)
        client.settings = MagicMock()
        await client.write_file(str(file_path), "new")

        assert file_path.read_text() == "new"


class TestLLMClientListFiles:
    def test_list_files_returns_files(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "c.py").write_text("c")

        client = LLMClient.__new__(LLMClient)
        client.settings = MagicMock()
        files = client.list_files(str(tmp_path))

        assert len(files) == 3
        assert "a.py" in files
        assert "b.py" in files
        assert str(Path("sub") / "c.py") in files

    def test_list_files_empty_directory(self, tmp_path):
        client = LLMClient.__new__(LLMClient)
        client.settings = MagicMock()
        files = client.list_files(str(tmp_path))
        assert files == []

    def test_list_files_nonexistent_directory(self):
        client = LLMClient.__new__(LLMClient)
        client.settings = MagicMock()
        files = client.list_files("/nonexistent/path/xyz")
        assert files == []


class TestLLMClientComplete:
    @pytest.mark.asyncio
    async def test_complete_raises_for_unsupported_provider(self):
        settings = MagicMock()
        settings.provider = "unsupported_provider"
        settings.model = "test-model"
        settings.planner_model = "test-planner"
        settings.worker_model = "test-worker"
        settings.tester_model = "test-tester"
        settings.anthropic_api_key = None
        settings.openai_api_key = None
        settings.ollama_base_url = "http://localhost:11434"

        client = LLMClient(settings=settings)

        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("prompt")

    @pytest.mark.asyncio
    async def test_complete_raises_for_none_provider(self):
        settings = MagicMock()
        settings.provider = None
        settings.model = "test-model"
        settings.planner_model = "test-planner"
        settings.worker_model = "test-worker"
        settings.tester_model = "test-tester"
        settings.anthropic_api_key = None
        settings.openai_api_key = None
        settings.ollama_base_url = "http://localhost:11434"

        client = LLMClient(settings=settings)

        with pytest.raises(ValueError):
            await client.complete("prompt")

    @pytest.mark.asyncio
    async def test_complete_anthropic(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text="response")]
        ))

        settings = MagicMock()
        settings.provider = Provider.ANTHROPIC
        settings.model = "test-model"
        settings.planner_model = "test-planner"
        settings.worker_model = "test-worker"
        settings.tester_model = "test-tester"
        settings.anthropic_api_key = "key"
        settings.openai_api_key = None
        settings.ollama_base_url = "http://localhost:11434"

        client = LLMClient(settings=settings)
        client._anthropic = mock_client

        result = await client.complete("test prompt")
        assert result == "response"
        mock_client.messages.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_complete_openai(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="response"))]
        ))

        settings = MagicMock()
        settings.provider = Provider.OPENAI
        settings.model = "test-model"
        settings.planner_model = "test-planner"
        settings.worker_model = "test-worker"
        settings.tester_model = "test-tester"
        settings.anthropic_api_key = None
        settings.openai_api_key = "key"
        settings.ollama_base_url = "http://localhost:11434"

        client = LLMClient(settings=settings)
        client._openai = mock_client

        result = await client.complete("test prompt")
        assert result == "response"
        mock_client.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_complete_ollama(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="ollama response"))]
        ))

        settings = MagicMock()
        settings.provider = Provider.OLLAMA
        settings.model = "test-model"
        settings.planner_model = "test-planner"
        settings.worker_model = "test-worker"
        settings.tester_model = "test-tester"
        settings.anthropic_api_key = None
        settings.openai_api_key = None
        settings.ollama_base_url = "http://localhost:11434"

        client = LLMClient(settings=settings)
        client._openai = mock_client

        result = await client.complete("test prompt")
        assert result == "ollama response"
        mock_client.chat.completions.create.assert_awaited_once()
