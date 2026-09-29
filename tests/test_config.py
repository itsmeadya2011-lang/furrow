"""Unit tests for the Furrow config models."""

from pathlib import Path

from furrow.config import (
    FileEdit,
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
    WorkerResult,
)


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_fields():
    t = TaskModel(
        id="2",
        description="add feature",
        files=["src/main.py", "tests/test_main.py"],
        dependencies=["1"],
    )
    assert t.files == ["src/main.py", "tests/test_main.py"]
    assert t.dependencies == ["1"]


def test_plan_parse():
    p = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )
    assert p.tasks[0].description == "do thing"
    assert p.rationale == "ok"
    assert len(p.tasks) == 1


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_defaults():
    t = TestResult(passed=False, summary="broken")
    assert t.failures == []
    assert t.passed is False


def test_file_edit_model():
    fe = FileEdit(path="src/main.py", content="print('hello')")
    assert fe.path == "src/main.py"
    assert fe.content == "print('hello')"


def test_worker_result_defaults():
    wr = WorkerResult(summary="done")
    assert wr.files == []
    assert wr.summary == "done"


def test_worker_result_with_files():
    fe = FileEdit(path="a.py", content="x = 1")
    wr = WorkerResult(summary="created a.py", files=[fe])
    assert len(wr.files) == 1
    assert wr.files[0].path == "a.py"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.llm_timeout == 120
    assert s.workspace == Path.cwd()


def test_settings_ollama_provider():
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:9999")
    assert s.provider == Provider.OLLAMA
    assert s.ollama_base_url == "http://localhost:9999"
