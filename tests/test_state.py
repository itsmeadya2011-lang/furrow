import tempfile
from pathlib import Path

from furrow.config import TaskModel
from furrow.state import CycleRecord, FurrowState, StateManager


def test_furrow_state_defaults():
    state = FurrowState()
    assert state.goal == ""
    assert state.cycles == []
    assert state.current_cycle == 0
    assert state.created_at is not None
    assert state.updated_at is not None


def test_furrow_state_properties():
    state = FurrowState()
    state.cycles = [
        CycleRecord(
            cycle=1,
            timestamp="2024-01-01T00:00:00",
            goal="test",
            plan_rationale="test",
            tasks=[
                {"id": "1", "status": "completed", "description": "task 1"},
                {"id": "2", "status": "failed", "description": "task 2"},
                {"id": "3", "status": "pending", "description": "task 3"},
            ],
            test_result={"passed": False, "summary": "failed", "failures": ["error"]}
        )
    ]
    
    assert len(state.completed_tasks) == 1
    assert len(state.failed_tasks) == 1
    assert len(state.pending_tasks) == 1
    assert state.is_complete is False


def test_furrow_state_is_complete():
    state = FurrowState()
    state.cycles = [
        CycleRecord(
            cycle=1,
            timestamp="2024-01-01T00:00:00",
            goal="test",
            plan_rationale="test",
            tasks=[
                {"id": "1", "status": "completed", "description": "task 1"},
                {"id": "2", "status": "completed", "description": "task 2"},
            ],
            test_result={"passed": True, "summary": "passed", "failures": []}
        )
    ]
    
    assert state.is_complete is True


def test_state_manager_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir) / "test_state.json"
        manager = StateManager(workspace=tmpdir, state_file="test_state.json")
        
        # Create initial state
        state = manager.load()
        assert state.goal == ""
        
        # Modify and save
        state.goal = "test goal"
        state.current_cycle = 1
        manager.save()
        
        # Load again
        manager2 = StateManager(workspace=tmpdir, state_file="test_state.json")
        loaded = manager2.load()
        assert loaded.goal == "test goal"
        assert loaded.current_cycle == 1


def test_state_manager_cycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StateManager(workspace=tmpdir, state_file="test_state.json")
        manager.load()
        
        tasks = [
            TaskModel(id="1", description="task 1", files=["a.py"]),
            TaskModel(id="2", description="task 2", files=["b.py"]),
        ]
        
        cycle = manager.start_cycle("test goal", "test rationale", tasks)
        assert cycle.cycle == 1
        assert cycle.goal == "test goal"
        assert cycle.plan_rationale == "test rationale"
        assert len(cycle.tasks) == 2
        
        # Update task result
        manager.update_task_result("1", "completed", "done")
        manager.update_task_result("2", "failed", "error")
        
        # Set test result
        manager.set_test_result(False, "tests failed", ["error in test"])
        
        state = manager.get_state()
        assert state.goal == "test goal"
        assert state.current_cycle == 1
        assert len(state.cycles) == 1
        assert state.cycles[0].tasks[0]["status"] == "completed"
        assert state.cycles[0].tasks[1]["status"] == "failed"
        assert state.cycles[0].test_result["passed"] is False


def test_state_manager_reset():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StateManager(workspace=tmpdir, state_file="test_state.json")
        manager.load()
        state = manager.get_state()
        state.goal = "test"
        manager.save()
        
        manager.reset()
        assert not (Path(tmpdir) / "test_state.json").exists()
        new_state = manager.load()
        assert new_state.goal == ""