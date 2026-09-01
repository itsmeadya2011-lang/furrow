import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel

class TestWorkerInit:
    def test_worker_accepts_workspace(self):
        client = MagicMock()
        worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=client, workspace="/tmp/work")
        assert worker.workspace == Path("/tmp/work")

    def test_worker_uses_client_workspace_by_default(self):
        client = MagicMock()
        client.settings.workspace = Path("/default/work")
        worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=client)
        assert worker.workspace == Path("/default/work")

class TestWorkerApply:
    @pytest.mark.asyncio
    async def test_apply_writes_files_from_valid_json(self, tmp_path):
        client = MagicMock()
        client.write_file = AsyncMock()
        worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=client, workspace=tmp_path)

        response = json.dumps({
            "files": [{"path": "test.py", "content": "print('hello')"}],
            "summary": "Created test.py"
        })

        result = await worker._apply(response)
        result_data = json.loads(result)

        assert result_data["files"] == ["test.py"]
        assert result_data["summary"] == "Created test.py"
        client.write_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_falls_back_to_summary_when_not_json(self, tmp_path):
        client = MagicMock()
        worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=client, workspace=tmp_path)

        response = "I completed the task successfully"
        result = await worker._apply(response)
        result_data = json.loads(result)

        assert result_data["files"] == []
        assert result_data["summary"] == response

    @pytest.mark.asyncio
    async def test_apply_handles_write_errors_gracefully(self, tmp_path):
        client = MagicMock()
        client.write_file = AsyncMock(side_effect=IOError("disk full"))
        worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=client, workspace=tmp_path)

        response = json.dumps({
            "files": [{"path": "test.py", "content": "print('hello')"}],
            "summary": "Created test.py"
        })

        # Should not raise - falls back to summary on error
        result = await worker._apply(response)
        result_data = json.loads(result)
        # Files that failed to write shouldn't be in the list
        assert result_data["summary"] == "Created test.py"