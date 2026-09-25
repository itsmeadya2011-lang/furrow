import pytest

from furrow.config import Plan, Provider, TaskModel, TestResult
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="1", description="test task")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_with_files():
    task = TaskModel(id="1", description="test", files=["src/a.py", "src/b.py"])
    assert len(task.files) == 2
    assert "src/a.py" in task.files


def test_plan_with_multiple_tasks():
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
    ]
    plan = Plan(tasks=tasks, rationale="test plan")
    assert len(plan.tasks) == 2


def test_test_result_with_failures():
    t = TestResult(
        passed=False,
        summary="tests failed",
        failures=["test_foo failed", "test_bar failed"]
    )
    assert t.passed is False
    assert len(t.failures) == 2


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_llm_client_init():
    client = LLMClient()
    assert client.settings is not None
    assert client._anthropic is None
    assert client._openai is None


@pytest.mark.asyncio
async def test_llm_client_list_files(tmp_path):
    # Create some test files
    (tmp_path / "file1.py").write_text("print('hello')")
    (tmp_path / "file2.py").write_text("print('world')")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file3.py").write_text("print('sub')")
    
    client = LLMClient()
    files = client.list_files(tmp_path)
    
    assert len(files) == 3
    assert "file1.py" in files
    assert "file2.py" in files
    assert "subdir/file3.py" in files


@pytest.mark.asyncio
async def test_llm_client_read_write_file(tmp_path):
    client = LLMClient()
    test_file = tmp_path / "test.txt"
    
    # Write file
    await client.write_file(test_file, "Hello, World!")
    
    # Read file
    content = await client.read_file(test_file)
    assert content == "Hello, World!"


@pytest.mark.asyncio
async def test_llm_client_write_file_creates_parents(tmp_path):
    client = LLMClient()
    test_file = tmp_path / "deep" / "nested" / "file.txt"
    
    await client.write_file(test_file, "nested content")
    
    assert test_file.exists()
    content = await client.read_file(test_file)
    assert content == "nested content"