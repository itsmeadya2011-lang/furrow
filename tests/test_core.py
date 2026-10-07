import pytest
import tempfile
import os
from pathlib import Path

from furrow.config import Plan, TaskModel, TestResult, Settings
from furrow.core.orchestrator import OrchestratorState, CycleState


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_state_serialization():
    """Test that OrchestratorState can be serialized and deserialized."""
    state = OrchestratorState(goal="Test goal")
    state.completed_task_ids.add("1")
    state.failed_task_ids.add("2")

    task = TaskModel(id="1", description="Task 1", status="completed")
    state.all_tasks["1"] = task

    cycle = CycleState(cycle_num=1, goal="Test goal")
    state.cycles.append(cycle)

    # Serialize
    json_str = state.to_json()

    # Deserialize
    restored = OrchestratorState.from_json(json_str)

    assert restored.goal == "Test goal"
    assert "1" in restored.completed_task_ids
    assert "2" in restored.failed_task_ids
    assert "1" in restored.all_tasks
    assert restored.all_tasks["1"].description == "Task 1"
    assert len(restored.cycles) == 1
    assert restored.cycles[0].cycle_num == 1


def test_orchestrator_state_with_plan():
    """Test state serialization with plan and tasks."""
    state = OrchestratorState(goal="Build feature")

    plan = Plan(
        tasks=[
            TaskModel(id="1", description="Task 1", files=["file1.py"]),
            TaskModel(id="2", description="Task 2", files=["file2.py"]),
        ],
        rationale="Build feature in two tasks"
    )

    cycle = CycleState(cycle_num=1, goal="Build feature", plan=plan)
    cycle.tasks = [state.all_tasks.setdefault(t.id, t) for t in plan.tasks]
    state.cycles.append(cycle)

    json_str = state.to_json()
    restored = OrchestratorState.from_json(json_str)

    assert restored.cycles[0].plan is not None
    assert restored.cycles[0].plan.rationale == "Build feature in two tasks"
    assert len(restored.cycles[0].plan.tasks) == 2


def test_cycle_state():
    """Test CycleState creation and serialization."""
    cycle = CycleState(cycle_num=1, goal="Test")
    assert cycle.cycle_num == 1
    assert cycle.goal == "Test"
    assert cycle.plan is None
    assert cycle.tasks == []
    assert cycle.test_result is None
    assert cycle.start_time > 0


def test_settings_defaults():
    """Test Settings has expected defaults."""
    settings = Settings()
    assert settings.provider.value == "anthropic"
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0  # unlimited


def test_task_model_defaults():
    """Test TaskModel default values."""
    task = TaskModel(id="1", description="Test task")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_test_result_defaults():
    """Test TestResult default values."""
    result = TestResult(passed=False, summary="Failed")
    assert result.failures == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])