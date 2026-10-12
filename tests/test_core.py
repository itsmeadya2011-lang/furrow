import pytest
from pathlib import Path
from furrow.config import Plan, TaskModel, TestResult, Provider, Settings
from furrow.core.state import StateManager
from furrow.agents.tools import TOOLS, get_tool_descriptions, create_tool


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


def test_task_model_with_files_and_deps():
    task = TaskModel(
        id="2",
        description="complex task",
        files=["src/main.py", "tests/test_main.py"],
        dependencies=["1"],
        status="completed",
        result="Done!"
    )
    assert task.files == ["src/main.py", "tests/test_main.py"]
    assert task.dependencies == ["1"]
    assert task.status == "completed"
    assert task.result == "Done!"


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_settings_defaults():
    settings = Settings()
    assert settings.provider == Provider.ANTHROPIC
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0


def test_state_manager_save_load(tmp_path: Path):
    state = StateManager(tmp_path)
    tasks = [
        TaskModel(id="1", description="task 1", status="completed", result="done"),
        TaskModel(id="2", description="task 2", status="pending"),
    ]
    state.set_tasks(tasks)
    state.set_goal("Test goal")
    state.set_cycles(3)

    # Create new manager to test loading
    state2 = StateManager(tmp_path)
    loaded_tasks = state2.get_tasks()
    assert len(loaded_tasks) == 2
    assert loaded_tasks[0].id == "1"
    assert loaded_tasks[0].status == "completed"
    assert loaded_tasks[1].id == "2"
    assert loaded_tasks[1].status == "pending"
    assert state2.get_goal() == "Test goal"
    assert state2.get_cycles() == 3


def test_state_manager_plan(tmp_path: Path):
    state = StateManager(tmp_path)
    plan = Plan(
        tasks=[TaskModel(id="1", description="task 1")],
        rationale="Test rationale"
    )
    state.set_plan(plan)

    state2 = StateManager(tmp_path)
    loaded_plan = state2.get_plan()
    assert loaded_plan is not None
    assert loaded_plan.rationale == "Test rationale"
    assert len(loaded_plan.tasks) == 1


def test_state_manager_clear(tmp_path: Path):
    state = StateManager(tmp_path)
    state.set_goal("Test")
    state.save()

    state.clear()
    assert not state.state_file.exists()
    assert state.get_goal() == ""


def test_tools_registry():
    assert "read_file" in TOOLS
    assert "write_file" in TOOLS
    assert "list_files" in TOOLS
    assert "run_command" in TOOLS
    assert "edit_file" in TOOLS
    assert "glob" in TOOLS
    assert "grep" in TOOLS


def test_tool_descriptions():
    desc = get_tool_descriptions()
    assert "read_file" in desc
    assert "write_file" in desc
    assert "run_command" in desc


def test_create_tool():
    from furrow.llm import LLMClient
    from furrow.config import settings

    client = LLMClient(settings=settings)
    tool = create_tool("read_file", client)
    assert tool.name == "read_file"

    tool = create_tool("write_file", client)
    assert tool.name == "write_file"


def test_orchestrator_ready_tasks():
    from furrow.core.orchestrator import Orchestrator
    from furrow.llm import LLMClient
    from furrow.config import TaskModel, settings

    client = LLMClient(settings=settings)
    orchestrator = Orchestrator(goal="test", client=client)

    # Add some tasks
    orchestrator.all_tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="pending", dependencies=["1"]),
        TaskModel(id="3", description="task 3", status="pending", dependencies=["2"]),
        TaskModel(id="4", description="task 4", status="failed"),
    ]

    # Task 2 should be ready (dep 1 is completed)
    ready = orchestrator._get_ready_tasks([orchestrator.all_tasks[1]])
    assert len(ready) == 1
    assert ready[0].id == "2"

    # Task 3 should NOT be ready (dep 2 is not completed)
    ready = orchestrator._get_ready_tasks([orchestrator.all_tasks[2]])
    assert len(ready) == 0

    # Task 4 should be ready (failed tasks can be retried)
    ready = orchestrator._get_ready_tasks([orchestrator.all_tasks[3]])
    assert len(ready) == 1
    assert ready[0].id == "4"


def test_orchestrator_is_done():
    from furrow.core.orchestrator import Orchestrator
    from furrow.llm import LLMClient
    from furrow.config import TaskModel, settings

    client = LLMClient(settings=settings)
    orchestrator = Orchestrator(goal="test", client=client)

    # No tasks = not done
    assert orchestrator._is_done() is False

    # All completed = done
    orchestrator.all_tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    assert orchestrator._is_done() is True

    # Has failed = not done
    orchestrator.all_tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="failed"),
    ]
    assert orchestrator._is_done() is False

    # Has pending = not done
    orchestrator.all_tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="pending"),
    ]
    assert orchestrator._is_done() is False