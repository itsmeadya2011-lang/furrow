import json
from unittest.mock import AsyncMock, Mock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel


def test_parse_operations_with_valid_json_array():
    operations = [
        {"action": "read", "path": "src/existing.py"},
        {"action": "write", "path": "src/new.py", "content": "file contents"},
    ]

    assert WorkerAgent._parse_operations(json.dumps(operations)) == operations


def test_parse_operations_with_markdown_code_block_wrapping():
    operations = [{"action": "write", "path": "src/new.py", "content": "file contents"}]
    response = "```json\n" + json.dumps(operations) + "\n```"

    assert WorkerAgent._parse_operations(response) == operations


def test_parse_operations_with_prose_and_embedded_json_array():
    operations = [{"action": "write", "path": "src/new.py", "content": "file contents"}]
    response = "Please apply these operations:\n" + json.dumps(operations) + "\nThank you."

    assert WorkerAgent._parse_operations(response) == operations


def test_parse_operations_with_invalid_input_returns_none():
    assert WorkerAgent._parse_operations("this is not an operation list") is None


def test_parse_operations_with_empty_array_returns_empty_list():
    assert WorkerAgent._parse_operations("[]") == []


@pytest.mark.parametrize(
    "operation",
    [
        {"path": "src/new.py"},
        {"action": "write"},
        {"action": "delete", "path": "src/new.py"},
    ],
)
def test_parse_operations_with_invalid_operation_returns_none(operation):
    assert WorkerAgent._parse_operations(json.dumps([operation])) is None


def test_resolve_path_with_valid_relative_path(tmp_path):
    agent = WorkerAgent(
        TaskModel(id="1", description="test task"),
        settings=Settings(workspace=tmp_path),
    )

    assert agent._resolve_path("src/new.py") == (tmp_path / "src/new.py").resolve()


def test_resolve_path_rejects_absolute_path(tmp_path):
    agent = WorkerAgent(
        TaskModel(id="1", description="test task"),
        settings=Settings(workspace=tmp_path),
    )

    with pytest.raises(ValueError, match="relative to the workspace"):
        agent._resolve_path(str(tmp_path / "new.py"))


def test_resolve_path_rejects_path_traversal(tmp_path):
    agent = WorkerAgent(
        TaskModel(id="1", description="test task"),
        settings=Settings(workspace=tmp_path),
    )

    with pytest.raises(ValueError, match="escapes the workspace"):
        agent._resolve_path("../outside.py")


@pytest.mark.asyncio
async def test_run_applies_valid_operations_and_writes_file(tmp_path):
    task = TaskModel(id="1", description="create a file", files=[])
    operations = [
        {
            "action": "write",
            "path": "src/new.py",
            "content": "file contents",
        }
    ]
    client = Mock(settings=Settings(workspace=tmp_path))
    client.complete = AsyncMock(return_value=json.dumps(operations))
    client.write_file = AsyncMock()
    agent = WorkerAgent(task, client=client)

    result = await agent.run()

    client.write_file.assert_awaited_once_with(
        (tmp_path / "src/new.py").resolve(),
        "file contents",
    )
    assert "Changed: src/new.py" in result


@pytest.mark.asyncio
async def test_run_falls_back_to_text_response_for_invalid_json(tmp_path):
    task = TaskModel(id="1", description="invalid response", files=[])
    response = "The worker could not produce JSON."
    client = Mock(settings=Settings(workspace=tmp_path))
    client.complete = AsyncMock(return_value=response)
    client.write_file = AsyncMock()
    agent = WorkerAgent(task, client=client)

    result = await agent.run()

    assert result == response.strip()
    client.write_file.assert_not_awaited()
