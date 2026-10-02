"""Core integration tests for furrow models and basic wiring."""


from furrow import LLMClient, Settings


def test_plan_parse():
    from furrow.config import Plan, TaskModel
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    from furrow.config import TestResult
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_llm_client_importable():
    """LLMClient and Settings are exported from the top-level package."""
    assert LLMClient is not None
    assert Settings is not None


def test_orchestrator_importable():
    """Orchestrator should be importable from core."""
    from furrow.core.orchestrator import Orchestrator
    assert Orchestrator is not None


def test_agents_importable():
    """Agent classes should be importable from the agents package."""
    from furrow.agents import PlannerAgent, TesterAgent, WorkerAgent
    assert PlannerAgent is not None
    assert TesterAgent is not None
    assert WorkerAgent is not None


def test_cli_importable():
    """CLI entry point should be importable."""
    from furrow.cli.main import main
    assert main is not None


def test_web_app_importable():
    """Web app should be importable."""
    from furrow.web.server import app, run
    assert app is not None
    assert run is not None


def test_task_status_enum():
    """TaskStatus values should be usable as task status strings."""
    from furrow.config import TaskModel, TaskStatus
    t = TaskModel(id="1", description="test", status=TaskStatus.COMPLETED)
    assert t.status == "completed"
