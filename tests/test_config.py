from pathlib import Path

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_provider_enum_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_taskmodel_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_taskmodel_creation():
    t = TaskModel(
        id="2", description="x", files=["a.py"], dependencies=["1"], status="done", result="ok"
    )
    assert t.files == ["a.py"]
    assert t.dependencies == ["1"]
    assert t.status == "done"
    assert t.result == "ok"


def test_plan_creation():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.rationale == "ok"
    assert len(p.tasks) == 1


def test_testresult_creation():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
    assert t.failures == []


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.provider == Provider.ANTHROPIC
    assert s.model == "claude-sonnet-4-20250514"
    assert s.max_tokens == 4096
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.test_commands == ["pytest", "-q"]
    assert isinstance(s.workspace, Path)


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "ollama")
    monkeypatch.setenv("FURROW_MAX_TOKENS", "100")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "5")
    s = Settings(_env_file=None)
    assert s.provider == Provider.OLLAMA
    assert s.max_tokens == 100
    assert s.max_cycles == 5
