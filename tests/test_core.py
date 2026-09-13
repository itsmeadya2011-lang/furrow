import pytest
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.agents.worker import WorkerAgent
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_orchestrator_persists_tasks():
    """Orchestrator should store plan and tasks after planning so _is_done works."""
    orc = Orchestrator(goal="test goal")
    orc.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    assert orc._is_done() is True


def test_orchestrator_not_done_with_failures():
    orc = Orchestrator(goal="test goal")
    orc.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="failed"),
    ]
    assert orc._is_done() is False


def test_orchestrator_not_done_with_pending():
    orc = Orchestrator(goal="test goal")
    orc.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="pending"),
    ]
    assert orc._is_done() is False


def test_orchestrator_done_with_no_tasks():
    orc = Orchestrator(goal="test goal")
    assert orc._is_done() is True


def test_worker_extracts_and_writes_files(tmp_path):
    """Worker should extract file writes from LLM response markers."""
    from furrow.config import TaskModel
    from furrow.llm import LLMClient

    task = TaskModel(
        id="1",
        description="create a file",
        files=["src/example.py"],
    )
    client = LLMClient()
    worker = WorkerAgent(task=task, client=client)

    response = """Done. Here is the file:
---FILE: src/example.py---
def hello():
    return "world"
---END---"""

    written = worker._extract_and_write_files(response, tmp_path)
    assert "src/example.py" in written
    assert (tmp_path / "src" / "example.py").exists()
    content = (tmp_path / "src" / "example.py").read_text()
    assert 'def hello():' in content


def test_worker_no_files_to_write():
    client = LLMClient()
    task = TaskModel(id="1", description="no changes", files=[])
    worker = WorkerAgent(task=task, client=client)
    response = "No changes needed."
    written = worker._extract_and_write_files(response, "/tmp")
    assert written == []


def test_llm_client_raises_on_missing_key():
    from furrow.llm import LLMClient
    from furrow.config import Settings, Provider

    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    client = LLMClient(settings=s)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        import asyncio
        asyncio.run(client.complete("test"))


def test_worker_prompt_contains_file_markers():
    from furrow.agents.prompts import WORKER_PROMPT
    assert "---FILE:" in WORKER_PROMPT
    assert "---END---" in WORKER_PROMPT
