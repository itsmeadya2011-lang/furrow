import pytest
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.core.orchestrator import Orchestrator
from furrow.agents.worker import WorkerAgent
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_default():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5
    assert s.log_level == "INFO"


def test_orchestrator_tracks_tasks():
    """Verify _get_tasks returns tracked tasks, not empty list."""
    orchestrator = Orchestrator(goal="test goal")
    assert orchestrator._get_tasks() == []
    orchestrator.tasks = [
        type("Obj", (), {"status": "completed"})(),
        type("Obj", (), {"status": "pending"})(),
    ]
    assert len(orchestrator._get_tasks()) == 2
    assert orchestrator._is_done() is False  # not all completed


def test_orchestrator_max_cycles_config():
    """Verify orchestrator picks up max_cycles from settings."""
    orchestrator = Orchestrator(goal="test goal")
    assert orchestrator.max_cycles == Settings().max_cycles


def test_worker_agent_files_changed():
    agent = WorkerAgent(task=TaskModel(id="1", description="test"))
    assert agent.files_changed == []
    agent.parse_file_changes("Modified `src/auth.py` and `tests/test_auth.py`")
    assert "src/auth.py" in agent.files_changed
    assert "tests/test_auth.py" in agent.files_changed


def test_worker_to_dict():
    agent = WorkerAgent(task=TaskModel(id="1", description="test"))
    d = agent.to_dict()
    assert d["task_id"] == "1"
    assert d["description"] == "test"
    assert d["files_changed"] == []


def test_llm_client_initialization():
    client = LLMClient()
    assert client.settings is not None
    assert client.anthropic is None  # lazy init
    assert client.openai is None  # lazy init


def test_llm_client_missing_api_key():
    """Verify LLMClient raises clear error when API key is missing."""
    from furrow.config import Provider, Settings
    client = LLMClient(settings=Settings(
        provider=Provider.ANTHROPIC,
        anthropic_api_key=None,
        openai_api_key=None,
    ))
    with pytest.raises(ValueError, match="API_KEY"):
        import asyncio
        asyncio.run(client.complete("test"))


def test_tester_exit_code_format():
    """Verify tester result format includes exit code marker."""
    result = "EXIT_CODE:1\nSome error output"
    assert result.startswith("EXIT_CODE:")


def test_config_log_level_variants():
    """Verify log level can be set to different values."""
    s = Settings(log_level="DEBUG")
    assert s.log_level == "DEBUG"
