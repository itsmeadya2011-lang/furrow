import pytest
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.core.orchestrator import OrchestratorState
from furrow.agents.prompts import PLANNER_PROMPT, WORKER_PROMPT, TESTER_PROMPT


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_with_dependencies():
    task = TaskModel(id="1", description="test", files=["a.py"], dependencies=["2"])
    assert task.files == ["a.py"]
    assert task.dependencies == ["2"]
    assert task.status == "pending"


def test_plan_with_multiple_tasks():
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2", dependencies=["1"]),
    ]
    p = Plan(tasks=tasks, rationale="test")
    assert len(p.tasks) == 2
    assert p.tasks[1].dependencies == ["1"]


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="failed", failures=["test1 failed", "test2 failed"])
    assert t.passed is False
    assert len(t.failures) == 2


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_validation():
    # Test max_parallel_tasks validation
    with pytest.raises(ValueError):
        Settings(max_parallel_tasks=0)
    with pytest.raises(ValueError):
        Settings(max_parallel_tasks=21)
    
    # Test log_level validation
    with pytest.raises(ValueError):
        Settings(log_level="INVALID")


def test_orchestrator_state_serialization():
    tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="pending"),
    ]
    state = OrchestratorState(
        goal="Test goal",
        all_tasks=tasks,
        completed_tasks=["1"],
        failed_tasks=[],
        cycles=3,
    )
    
    json_str = state.to_json()
    restored = OrchestratorState.from_json(json_str)
    
    assert restored.goal == "Test goal"
    assert restored.cycles == 3
    assert len(restored.all_tasks) == 2
    assert restored.all_tasks[0].id == "1"
    assert restored.all_tasks[0].status == "completed"
    assert restored.completed_tasks == ["1"]


def test_prompts_contain_tool_instructions():
    assert "read_file" in WORKER_PROMPT
    assert "write_file" in WORKER_PROMPT
    assert "list_files" in WORKER_PROMPT
    assert "run_command" in WORKER_PROMPT
    assert "```tool" in WORKER_PROMPT


def test_planner_prompt_structure():
    assert "tasks" in PLANNER_PROMPT
    assert "rationale" in PLANNER_PROMPT
    assert "dependencies" in PLANNER_PROMPT


def test_tester_prompt_structure():
    assert "passed" in TESTER_PROMPT
    assert "summary" in TESTER_PROMPT
    assert "failures" in TESTER_PROMPT