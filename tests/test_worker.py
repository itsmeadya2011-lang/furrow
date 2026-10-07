from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


@pytest.fixture
def settings():
    return Settings(
        provider="anthropic",
        anthropic_api_key="test-key",
        worker_model="test-model",
    )


@pytest.fixture
def mock_client(settings):
    client = MagicMock(spec=LLMClient)
    client.settings = settings
    client.complete = AsyncMock()
    client.write_file = AsyncMock()
    return client


@pytest.fixture
def task():
    return TaskModel(id="1", description="Create a hello world file", files=["hello.py"])


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_worker_parses_file_edits_and_writes_to_disk(self, task, mock_client, tmp_path):
        task.files = [str(tmp_path / "hello.py")]
        mock_client.complete.return_value = f'''```{tmp_path / "hello.py"}
print("Hello, World!")
```'''
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        mock_client.write_file.assert_called_once()
        call_args = mock_client.write_file.call_args
        assert call_args[0][0] == str(tmp_path / "hello.py")
        assert call_args[0][1] == 'print("Hello, World!")'
        assert "Modified 1 file(s)" in result
        assert "hello.py" in result

    @pytest.mark.asyncio
    async def test_worker_multiple_file_edits(self, task, mock_client, tmp_path):
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        task.files = [str(file1), str(file2)]
        mock_client.complete.return_value = f'''```{file1}
content1
```

```{file2}
content2
```'''
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        assert mock_client.write_file.call_count == 2
        calls = mock_client.write_file.call_args_list
        paths = [call[0][0] for call in calls]
        assert str(file1) in paths
        assert str(file2) in paths
        assert "Modified 2 file(s)" in result

    @pytest.mark.asyncio
    async def test_worker_returns_summary_when_no_file_edits(self, task, mock_client):
        mock_client.complete.return_value = "No changes needed. The code is already correct."
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        assert result == "No changes needed. The code is already correct."
        mock_client.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_worker_handles_empty_llm_response(self, task, mock_client):
        mock_client.complete.return_value = ""
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        assert result == "No changes were needed."
        mock_client.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_worker_handles_whitespace_only_response(self, task, mock_client):
        mock_client.complete.return_value = "   \n\t  \n  "
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        assert result == "No changes were needed."
        mock_client.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_worker_with_language_tagged_blocks_ignored(self, task, mock_client, tmp_path):
        file_path = tmp_path / "main.py"
        task.files = [str(file_path)]
        mock_client.complete.return_value = f'''```python
def helper():
    pass
```

```{file_path}
print("real code")
```'''
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        mock_client.write_file.assert_called_once()
        call_args = mock_client.write_file.call_args
        assert call_args[0][1] == 'print("real code")'
        assert "Modified 1 file(s)" in result

    @pytest.mark.asyncio
    async def test_worker_with_tilde_fences(self, task, mock_client, tmp_path):
        file_path = tmp_path / "test.py"
        task.files = [str(file_path)]
        mock_client.complete.return_value = f'''~~~{file_path}
print("tilde fence")
~~~'''
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()
        mock_client.write_file.assert_called_once()
        call_args = mock_client.write_file.call_args
        assert call_args[0][1] == 'print("tilde fence")'

    @pytest.mark.asyncio
    async def test_worker_uses_worker_model_from_settings(self, task, mock_client, settings):
        mock_client.complete.return_value = "done"
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()
        mock_client.complete.assert_called_once()
        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs.get("model") == settings.worker_model

    @pytest.mark.asyncio
    async def test_worker_includes_task_description_in_prompt(self, task, mock_client):
        mock_client.complete.return_value = "done"
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()
        mock_client.complete.assert_called_once()
        call_args = mock_client.complete.call_args
        prompt = call_args[0][0]
        assert "Create a hello world file" in prompt
        assert "hello.py" in prompt

    @pytest.mark.asyncio
    async def test_worker_returns_summary_with_unique_paths(self, task, mock_client, tmp_path):
        file_path = tmp_path / "duplicate.py"
        task.files = [str(file_path)]
        mock_client.complete.return_value = f'''```{file_path}
first edit
```

```{file_path}
second edit
```'''
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        assert mock_client.write_file.call_count == 2
        assert "Modified 1 file(s)" in result


class TestWorkerAgentInitialization:
    def test_initializes_with_default_client(self, task):
        with patch("furrow.agents.worker.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.settings = Settings()
            mock_client_class.return_value = mock_client
            worker = WorkerAgent(task=task)
            assert worker.client is mock_client
            mock_client_class.assert_called_once()

    def test_initializes_with_provided_client(self, task, mock_client):
        worker = WorkerAgent(task=task, client=mock_client)
        assert worker.client is mock_client

    def test_initializes_with_provided_settings(self, task):
        settings = Settings(provider="anthropic", anthropic_api_key="test")
        with patch("furrow.agents.worker.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.settings = settings
            mock_client_class.return_value = mock_client
            worker = WorkerAgent(task=task, settings=settings)
            assert worker.client is mock_client
            mock_client_class.assert_called_once_with(settings=settings)