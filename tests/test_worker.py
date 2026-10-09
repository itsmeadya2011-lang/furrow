import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


def test_parse_file_blocks_extracts_path_and_content():
    response = '''Some intro text.

```file:src/main.py
def hello():
    print("hi")
```

```file:tests/test_main.py
def test_hello():
    assert True
```
'''
    blocks = WorkerAgent.parse_file_blocks(response)
    assert len(blocks) == 2
    assert blocks[0][0] == "src/main.py"
    assert "def hello()" in blocks[0][1]
    assert blocks[1][0] == "tests/test_main.py"


def test_parse_file_blocks_no_files():
    blocks = WorkerAgent.parse_file_blocks("Just some text, no file blocks.")
    assert blocks == []


def test_parse_file_blocks_handles_paths_with_dots():
    response = '''```file:src/app.py
x = 1
```'''
    blocks = WorkerAgent.parse_file_blocks(response)
    assert len(blocks) == 1
    assert blocks[0][0] == "src/app.py"


def test_extract_summary_finds_summary_line():
    response = '''```file:a.py
x = 1
```
SUMMARY: Added the thing'''
    assert WorkerAgent._extract_summary(response) == "Added the thing"


def test_extract_summary_returns_truncated_when_no_summary():
    response = "x" * 1000
    result = WorkerAgent._extract_summary(response)
    assert len(result) == 500


def test_extract_summary_strips_whitespace():
    response = "SUMMARY:   spaced summary  "
    assert WorkerAgent._extract_summary(response) == "spaced summary"


@pytest.mark.asyncio
async def test_worker_writes_files_via_callback():
    written = []

    async def fake_write(path, content):
        written.append((path, content))

    task = TaskModel(id="1", description="do thing", files=["a.py"])
    worker = WorkerAgent(task=task, write_file=fake_write)

    # Mock the client.complete to return file blocks
    class FakeClient:
        settings = type("S", (), {"worker_model": "test-model"})()
        async def complete(self, prompt, system="", model=None):
            return '''```file:a.py
x = 1
```
SUMMARY: wrote a.py'''

    worker.client = FakeClient()
    summary = await worker.run()
    assert len(written) == 1
    assert written[0][0] == "a.py"
    assert "x = 1" in written[0][1]
    assert summary == "wrote a.py"


@pytest.mark.asyncio
async def test_worker_propagates_write_errors():
    task = TaskModel(id="1", description="do thing")

    async def failing_write(path, content):
        raise OSError("disk full")

    worker = WorkerAgent(task=task, write_file=failing_write)

    class FakeClient:
        settings = type("S", (), {"worker_model": "test-model"})()
        async def complete(self, prompt, system="", model=None):
            return '''```file:a.py
x = 1
```'''

    worker.client = FakeClient()
    with pytest.raises(RuntimeError, match="Failed to write"):
        await worker.run()
