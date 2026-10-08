import json
from pathlib import Path

from furrow.agents.prompts import PLANNER_PROMPT, TESTER_PROMPT, WORKER_PROMPT
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="1", description="test task")
    assert task.status == "pending"
    assert task.result is None
    assert task.files == []
    assert task.dependencies == []


def test_task_model_with_files_and_deps():
    task = TaskModel(
        id="1",
        description="test task",
        files=["src/file.py"],
        dependencies=["0"],
        status="completed",
        result="done"
    )
    assert task.files == ["src/file.py"]
    assert task.dependencies == ["0"]
    assert task.status == "completed"
    assert task.result == "done"


def test_plan_model():
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2", dependencies=["1"]),
        ],
        rationale="test plan"
    )
    assert len(plan.tasks) == 2
    assert plan.tasks[1].dependencies == ["1"]


def test_test_result_model():
    result = TestResult(passed=False, summary="failed", failures=["test1 failed", "test2 failed"])
    assert result.passed is False
    assert len(result.failures) == 2


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.workspace == Path.cwd()


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "3")
    s = Settings()
    assert s.max_cycles == 10
    assert s.max_parallel_tasks == 3


def test_planner_prompt_structure():
    assert "tasks" in PLANNER_PROMPT
    assert "rationale" in PLANNER_PROMPT
    assert "id" in PLANNER_PROMPT
    assert "description" in PLANNER_PROMPT
    assert "files" in PLANNER_PROMPT
    assert "dependencies" in PLANNER_PROMPT


def test_worker_prompt_structure():
    assert "files" in WORKER_PROMPT
    assert "summary" in WORKER_PROMPT
    assert "path" in WORKER_PROMPT
    assert "content" in WORKER_PROMPT


def test_tester_prompt_structure():
    assert "passed" in TESTER_PROMPT
    assert "summary" in TESTER_PROMPT
    assert "failures" in TESTER_PROMPT


def test_orchestrator_initialization():
    orchestrator = Orchestrator(goal="test goal")
    assert orchestrator.goal == "test goal"
    assert orchestrator.cycles == 0
    assert orchestrator._tasks == []


def test_orchestrator_get_tasks():
    orchestrator = Orchestrator(goal="test goal")
    tasks = orchestrator._get_tasks()
    assert tasks == []


def test_orchestrator_is_done_empty():
    orchestrator = Orchestrator(goal="test goal")
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_with_tasks():
    orchestrator = Orchestrator(goal="test goal")
    task1 = TaskModel(id="1", description="task 1", status="completed")
    task2 = TaskModel(id="2", description="task 2", status="completed")
    orchestrator._tasks = [task1, task2]
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_with_pending():
    orchestrator = Orchestrator(goal="test goal")
    task1 = TaskModel(id="1", description="task 1", status="completed")
    task2 = TaskModel(id="2", description="task 2", status="pending")
    orchestrator._tasks = [task1, task2]
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_with_failed():
    orchestrator = Orchestrator(goal="test goal")
    task1 = TaskModel(id="1", description="task 1", status="completed")
    task2 = TaskModel(id="2", description="task 2", status="failed")
    orchestrator._tasks = [task1, task2]
    assert orchestrator._is_done() is False


def test_state_persistence(tmp_path):
    state_file = tmp_path / "test_state.json"
    orchestrator = Orchestrator(goal="test goal", state_file=state_file)
    task = TaskModel(id="1", description="test task", status="completed", result="done")
    orchestrator._tasks = [task]
    orchestrator.cycles = 5
    orchestrator._save_state()
    
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["goal"] == "test goal"
    assert data["cycles"] == 5
    assert len(data["tasks"]) == 1
    assert data["tasks"][0]["id"] == "1"
    assert data["tasks"][0]["status"] == "completed"


def test_state_loading(tmp_path):
    state_file = tmp_path / "test_state.json"
    initial_data = {
        "goal": "loaded goal",
        "cycles": 3,
        "tasks": [
            {"id": "1", "description": "loaded task", "files": [], "dependencies": [], "status": "completed", "result": "done"}
        ]
    }
    state_file.write_text(json.dumps(initial_data))
    
    orchestrator = Orchestrator(goal="original goal", state_file=state_file)
    assert orchestrator.goal == "loaded goal"
    assert orchestrator.cycles == 3
    assert len(orchestrator._tasks) == 1
    assert orchestrator._tasks[0].id == "1"
    assert orchestrator._tasks[0].description == "loaded task"


def test_task_dependency_ordering():
    task_map = {
        "1": TaskModel(id="1", description="task 1", dependencies=[]),
        "2": TaskModel(id="2", description="task 2", dependencies=["1"]),
        "3": TaskModel(id="3", description="task 3", dependencies=["1"]),
        "4": TaskModel(id="4", description="task 4", dependencies=["2", "3"]),
    }
    
    ready = [t for t in task_map.values() if all(dep in set() for dep in t.dependencies)]
    assert len(ready) == 1
    assert ready[0].id == "1"
    
    completed = {"1"}
    ready = [t for t in task_map.values() if t.id not in completed and all(dep in completed for dep in t.dependencies)]
    assert len(ready) == 2
    assert {t.id for t in ready} == {"2", "3"}
    
    completed = {"1", "2", "3"}
    ready = [t for t in task_map.values() if t.id not in completed and all(dep in completed for dep in t.dependencies)]
    assert len(ready) == 1
    assert ready[0].id == "4"


def test_circular_dependency_detection():
    task_map = {
        "1": TaskModel(id="1", description="task 1", dependencies=["2"]),
        "2": TaskModel(id="2", description="task 2", dependencies=["1"]),
    }
    
    pending = {t.id: t for t in task_map.values()}
    completed = set()
    
    ready = [task for task in pending.values() if all(dep in completed for dep in task.dependencies)]
    assert len(ready) == 0
    assert len(pending) == 2