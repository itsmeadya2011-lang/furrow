from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from furrow.config import (
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
    setup_logging,
)


def test_provider_enum():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"
    members = {member.value for member in Provider}
    assert members == {"anthropic", "openai", "ollama"}


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.id == "1"
    assert t.description == "do thing"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_custom():
    t = TaskModel(
        id="t-2",
        description="build module",
        files=["src/module.py", "src/__init__.py"],
        dependencies=["t-1"],
        status="completed",
        result="done",
    )
    assert t.id == "t-2"
    assert t.description == "build module"
    assert t.files == ["src/module.py", "src/__init__.py"]
    assert t.dependencies == ["t-1"]
    assert t.status == "completed"
    assert t.result == "done"


def test_plan_parse():
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="first task", files=["a.py"]),
            TaskModel(id="2", description="second task", dependencies=["1"]),
        ],
        rationale="This is a test plan.",
    )
    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "first task"
    assert plan.tasks[0].files == ["a.py"]
    assert plan.tasks[1].id == "2"
    assert plan.tasks[1].description == "second task"
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.rationale == "This is a test plan."


def test_test_result_defaults():
    t = TestResult(passed=True, summary="All tests passed")
    assert t.passed is True
    assert t.summary == "All tests passed"
    assert t.failures == []


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.model == "claude-sonnet-4-20250514"
    assert s.planner_model == "claude-3-5-haiku-20241022"
    assert s.worker_model == "claude-3-5-sonnet-20241022"
    assert s.tester_model == "claude-3-5-sonnet-20241022"
    assert s.anthropic_api_key is None
    assert s.openai_api_key is None
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.ollama_model == "llama3"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.workspace == Path.cwd()
    assert s.log_level == "INFO"


def test_setup_logging():
    with patch("furrow.config.structlog.configure") as mock_configure:
        setup_logging("DEBUG")
    assert mock_configure.called


def test_setup_logging_default_level():
    with patch("furrow.config.structlog.configure") as mock_configure:
        setup_logging()
    assert mock_configure.called


def test_setup_logging_invalid_level():
    with patch("furrow.config.structlog.configure"):
        setup_logging("NOTALEVEL")
    assert True
