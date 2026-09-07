import pytest
from pathlib import Path

from furrow.config import Plan, Settings, TaskModel, TestResult, Provider


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_ollama_url_valid():
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    assert s.ollama_base_url == "http://localhost:11434"


def test_ollama_url_invalid_scheme():
    with pytest.raises(Exception):
        Settings(provider=Provider.OLLAMA, ollama_base_url="ftp://localhost")


def test_ollama_url_invalid_hostname():
    with pytest.raises(Exception):
        Settings(provider=Provider.OLLAMA, ollama_base_url="http://")


def test_workspace_validation(tmp_path):
    s = Settings(workspace=tmp_path)
    assert s.workspace == tmp_path


def test_workspace_invalid_path():
    with pytest.raises(Exception):
        Settings(workspace=Path("/nonexistent/path/12345"))


def test_workspace_not_a_directory(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("hello")
    with pytest.raises(Exception):
        Settings(workspace=file_path)
