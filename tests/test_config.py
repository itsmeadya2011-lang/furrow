import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.id == "1"
    assert t.description == "do thing"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_values():
    t = TaskModel(
        id="2",
        description="refactor auth",
        files=["src/auth.py"],
        dependencies=["1"],
        status="completed",
        result="done",
    )
    assert t.files == ["src/auth.py"]
    assert t.dependencies == ["1"]
    assert t.status == "completed"
    assert t.result == "done"


def test_plan_creation():
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "do thing"
    assert plan.rationale == "ok"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
    assert t.summary == "ok"
    assert t.failures == []


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="some tests failed", failures=["test_a failed"])
    assert t.passed is False
    assert len(t.failures) == 1


def test_settings_detect_project_type_python(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'foo'\n")
    s = Settings(workspace=tmp_path)
    assert s.detect_project_type() == "python"


def test_settings_detect_project_type_setup_py(tmp_path):
    (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()\n")
    s = Settings(workspace=tmp_path)
    assert s.detect_project_type() == "python"


def test_settings_detect_project_type_requirements(tmp_path):
    (tmp_path / "requirements.txt").write_text("pytest\n")
    s = Settings(workspace=tmp_path)
    assert s.detect_project_type() == "python"


def test_settings_detect_project_type_node(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    s = Settings(workspace=tmp_path)
    assert s.detect_project_type() == "node"


def test_settings_detect_project_type_rust(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'foo'\n")
    s = Settings(workspace=tmp_path)
    assert s.detect_project_type() == "rust"


def test_settings_detect_project_type_go(tmp_path):
    (tmp_path / "go.mod").write_text("module foo\n")
    s = Settings(workspace=tmp_path)
    assert s.detect_project_type() == "go"


def test_settings_detect_project_type_unknown(tmp_path):
    s = Settings(workspace=tmp_path)
    assert s.detect_project_type() == "unknown"


def test_settings_get_test_command_python(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'foo'\n")
    s = Settings(workspace=tmp_path)
    cmd = s.get_test_command()
    assert cmd == ["python", "-m", "pytest", "-q"]


def test_settings_get_test_command_node(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    s = Settings(workspace=tmp_path)
    cmd = s.get_test_command()
    assert cmd == ["npm", "test", "--", "--silent"]


def test_settings_get_test_command_rust(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'foo'\n")
    s = Settings(workspace=tmp_path)
    cmd = s.get_test_command()
    assert cmd == ["cargo", "test", "-q"]


def test_settings_get_test_command_go(tmp_path):
    (tmp_path / "go.mod").write_text("module foo\n")
    s = Settings(workspace=tmp_path)
    cmd = s.get_test_command()
    assert cmd == ["go", "test", "./..."]


def test_settings_get_test_command_unknown(tmp_path):
    s = Settings(workspace=tmp_path)
    cmd = s.get_test_command()
    assert cmd == ["pytest", "-q"]


def test_settings_max_parallel_tasks_valid():
    s = Settings(max_parallel_tasks=3)
    assert s.max_parallel_tasks == 3


def test_settings_max_parallel_tasks_invalid():
    with pytest.raises(ValidationError):
        Settings(max_parallel_tasks=0)
    with pytest.raises(ValidationError):
        Settings(max_parallel_tasks=11)


def test_settings_provider_enum():
    s = Settings(provider=Provider.OPENAI)
    assert s.provider == Provider.OPENAI