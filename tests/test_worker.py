from unittest.mock import AsyncMock, MagicMock

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


def make_client(
    read_return: str | Exception | None = None,
    write_return=None,
    complete_return: str = "",
):
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.worker_model = "test-worker-model"

    async def read_file(path):
        if isinstance(read_return, Exception):
            raise read_return
        return read_return or ""

    client.read_file = AsyncMock(side_effect=read_file)
    client.write_file = AsyncMock(return_value=write_return)
    client.complete = AsyncMock(return_value=complete_return)
    return client


async def test_worker_reads_existing_files():
    client = make_client(read_return="file contents here")
    task = TaskModel(id="1", description="do work", files=["a.py", "b.py"])
    worker = WorkerAgent(task=task, client=client)

    response_json = '{"changes": [{"file": "a.py", "content": "new"}], "summary": "done"}'
    client.complete = AsyncMock(return_value=response_json)

    result = await worker.run()

    assert client.read_file.await_count == 2
    assert client.read_file.await_args_list[0].args[0] == "a.py"
    assert client.write_file.await_count == 1
    assert client.write_file.await_args.kwargs == {"path": "a.py", "content": "new"} or \
        client.write_file.await_args.args == ("a.py", "new")
    assert result == "done"


async def test_worker_writes_files_from_json_response():
    client = make_client(read_return="")
    task = TaskModel(id="1", description="implement feature", files=[])
    worker = WorkerAgent(task=task, client=client)

    response_json = (
        '{"changes": ['
        '{"file": "src/x.py", "content": "print(1)"}, '
        '{"file": "src/y.py", "content": "print(2)"}'
        '], "summary": "wrote 2 files"}'
    )
    client.complete = AsyncMock(return_value=response_json)

    result = await worker.run()

    assert client.write_file.await_count == 2
    written_paths = [c.args[0] for c in client.write_file.await_args_list]
    assert "src/x.py" in written_paths
    assert "src/y.py" in written_paths
    assert result == "wrote 2 files"


async def test_worker_handles_json_parse_failure():
    client = make_client(read_return="")
    task = TaskModel(id="1", description="anything", files=[])
    worker = WorkerAgent(task=task, client=client)

    raw = "this is not json { broken"
    client.complete = AsyncMock(return_value=raw)

    result = await worker.run()

    assert result == raw
    client.write_file.assert_not_awaited()


async def test_worker_handles_nonexistent_files(tmp_path):
    nonexistent = str(tmp_path / "does_not_exist.py")
    client = make_client(read_return=FileNotFoundError(nonexistent))
    task = TaskModel(id="1", description="create file", files=[nonexistent])
    worker = WorkerAgent(task=task, client=client)

    response_json = (
        '{"changes": [{"file": "' + nonexistent + '", "content": "x = 1"}], '
        '"summary": "created"}'
    )
    client.complete = AsyncMock(return_value=response_json)

    result = await worker.run()
    assert result == "created"
    assert client.write_file.await_count == 1