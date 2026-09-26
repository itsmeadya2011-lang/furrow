import json

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from tests.fakes import FakeClient


@pytest.fixture
def fake():
    return FakeClient()


async def test_worker_writes_files(tmp_path, fake):
    content = json.dumps(
        {
            "summary": "wrote foo",
            "files": [{"path": str(tmp_path / "foo.py"), "content": "x = 1", "action": "create"}],
        }
    )
    fake.responses = [content]
    task = TaskModel(id="1", description="write foo")
    result = await WorkerAgent(task=task, client=fake).run()
    assert result == "wrote foo"
    assert (tmp_path / "foo.py").read_text() == "x = 1"


async def test_worker_deletes_files(tmp_path, fake):
    target = tmp_path / "bar.py"
    target.write_text("y = 2")
    content = json.dumps(
        {"summary": "removed bar", "files": [{"path": str(target), "action": "delete"}]}
    )
    fake.responses = [content]
    await WorkerAgent(task=TaskModel(id="2", description="remove"), client=fake).run()
    assert not target.exists()


async def test_worker_fallback_on_invalid_json(fake):
    fake.responses = ["this is not json"]
    task = TaskModel(id="3", description="d")
    result = await WorkerAgent(task=task, client=fake).run()
    assert result == "this is not json"


async def test_worker_no_files_key(fake):
    fake.responses = [json.dumps({"summary": "nothing done"})]
    task = TaskModel(id="4", description="d")
    result = await WorkerAgent(task=task, client=fake).run()
    assert result == "nothing done"
