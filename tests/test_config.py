from pathlib import Path

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_task_model_defaults():
    task = TaskModel(id="1", description="do thing")
    assert task.id == "1"
    assert task.description == "do thing"
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_with_files_and_deps():
    task = TaskModel(
        id="2",
        description="another thing",
        files=["src/a.py", "tests/test_a.py"],
        dependencies=["1"],
    )
    assert task.files == ["src/a.py", "tests/test_a.py"]
    assert task.dependencies == ["1"]


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"
    assert p.rationale == "ok"


def test_plan_multiple_tasks():
    p = Plan(
        tasks=[
            TaskModel(id="1", description="a"),
            TaskModel(id="2", description="b", dependencies=["1"]),
        ],
        rationale="split it",
    )
    assert len(p.tasks) == 2
    assert p.tasks[1].dependencies == ["1"]


def test_test_result_passed():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
    assert t.summary == "ok"
    assert t.failures == []


def test_test_result_with_failures():
    t = TestResult(
        passed=False,
        summary="2 failures",
        failures=["test_a.py::test_one failed", "test_b.py::test_two failed"],
    )
    assert t.passed is False
    assert len(t.failures) == 2
    assert "test_a.py" in t.failures[0]


def test_test_result_defaults():
    t = TestResult(passed=True, summary="all good")
    assert t.failures == []


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.model == "claude-sonnet-4-20250514"
    assert s.planner_model == "claude-3-5-haiku-20241022"
    assert s.worker_model == "claude-3-5-sonnet-20241022"
    assert s.tester_model == "claude-3-5-sonnet-20241022"
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"
    assert isinstance(s.workspace, Path)


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "ollama")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "3")
    s = Settings()
    assert s.provider == Provider.OLLAMA
    assert s.max_cycles == 3


def test_provider_enum_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"