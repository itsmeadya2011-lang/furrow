from furrow.agents.utils import extract_json
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# Tests for extract_json from furrow.agents.utils
def test_extract_json_basic():
    """Test basic JSON extraction."""
    text = '{"key": "value"}'
    result = extract_json(text)
    assert result == '{"key": "value"}'


def test_extract_json_with_prose():
    """Test JSON extraction with surrounding prose."""
    text = 'Here is the result: {"key": "value"} and some more text'
    result = extract_json(text)
    assert result == '{"key": "value"}'


def test_extract_json_markdown_fence():
    """Test JSON extraction from markdown code fence."""
    text = '```json\n{"key": "value"}\n```'
    result = extract_json(text)
    assert result == '{"key": "value"}'


def test_extract_json_markdown_fence_no_json_tag():
    """Test JSON extraction from markdown code fence without json tag."""
    text = '```\n{"key": "value"}\n```'
    result = extract_json(text)
    assert result == '{"key": "value"}'


def test_extract_json_multiple_objects():
    """Test extraction takes first { to last } when multiple objects exist in prose."""
    text = 'First: {"a": 1} and second: {"b": 2}'
    result = extract_json(text)
    assert result == '{"a": 1} and second: {"b": 2}'  # extracts from first { to last }


def test_extract_json_empty_input():
    """Test extraction with empty input."""
    text = ''
    result = extract_json(text)
    assert result == ''


def test_extract_json_no_json():
    """Test extraction when no JSON is present - should return original text."""
    text = 'This is just plain text'
    result = extract_json(text)
    assert result == text


def test_extract_json_nested():
    """Test extraction with nested JSON."""
    text = '```json\n{"outer": {"inner": "value"}}\n```'
    result = extract_json(text)
    assert result == '{"outer": {"inner": "value"}}'


# Tests for Plan and TaskModel
def test_task_model_construction():
    """Test TaskModel basic construction."""
    task = TaskModel(id="task-1", description="Implement feature X")
    assert task.id == "task-1"
    assert task.description == "Implement feature X"


def test_task_model_defaults():
    """Test TaskModel default values."""
    task = TaskModel(id="1", description="do thing")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_with_values():
    """Test TaskModel with all values set."""
    task = TaskModel(
        id="2",
        description="Fix bug",
        files=["file1.py", "file2.py"],
        dependencies=["task-1"],
        status="completed",
        result="Done successfully"
    )
    assert task.id == "2"
    assert task.description == "Fix bug"
    assert task.files == ["file1.py", "file2.py"]
    assert task.dependencies == ["task-1"]
    assert task.status == "completed"
    assert task.result == "Done successfully"


def test_plan_construction():
    """Test Plan construction."""
    tasks = [
        TaskModel(id="1", description="Task 1"),
        TaskModel(id="2", description="Task 2")
    ]
    plan = Plan(tasks=tasks, rationale="Because it's needed")
    assert len(plan.tasks) == 2
    assert plan.rationale == "Because it's needed"
    assert plan.tasks[0].id == "1"
    assert plan.tasks[1].id == "2"


def test_plan_validation():
    """Test Plan validation with empty tasks."""
    plan = Plan(tasks=[], rationale="No tasks")
    assert plan.tasks == []
    assert plan.rationale == "No tasks"


# Tests for TestResult
def test_test_result_passed():
    """Test TestResult with passed state."""
    result = TestResult(passed=True, summary="All tests passed", failures=[])
    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


def test_test_result_failed():
    """Test TestResult with failed state."""
    failures = ["test_foo failed", "test_bar failed"]
    result = TestResult(passed=False, summary="2 tests failed", failures=failures)
    assert result.passed is False
    assert result.summary == "2 tests failed"
    assert result.failures == failures


def test_test_result_defaults():
    """Test TestResult default for failures."""
    result = TestResult(passed=False, summary="Failed")
    assert result.failures == []


# Tests for Orchestrator._is_done()
def test_orchestrator_is_done_all_completed():
    """Test _is_done when all tasks are completed."""
    orchestrator = Orchestrator(goal="test goal")
    orchestrator._tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="completed"),
    ]
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_some_failed():
    """Test _is_done when some tasks failed - should return False."""
    orchestrator = Orchestrator(goal="test goal")
    orchestrator._tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="failed"),
    ]
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_failed_and_pending():
    """Test _is_done with failed tasks - always returns False even if others pending."""
    orchestrator = Orchestrator(goal="test goal")
    orchestrator._tasks = [
        TaskModel(id="1", description="t1", status="pending"),
        TaskModel(id="2", description="t2", status="failed"),
    ]
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_mixed_pending_completed():
    """Test _is_done when some completed, some pending (no failures) - should return False."""
    orchestrator = Orchestrator(goal="test goal")
    orchestrator._tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="pending"),
    ]
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_empty_tasks():
    """Test _is_done with empty task list."""
    orchestrator = Orchestrator(goal="test goal")
    orchestrator._tasks = []
    # If no tasks, completed (0) >= len(tasks)=0 is true, so returns True
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_all_pending():
    """Test _is_done when all tasks are pending."""
    orchestrator = Orchestrator(goal="test goal")
    orchestrator._tasks = [
        TaskModel(id="1", description="t1", status="pending"),
        TaskModel(id="2", description="t2", status="pending"),
    ]
    assert orchestrator._is_done() is False


# Tests for Orchestrator._get_tasks()
def test_orchestrator_get_tasks_returns_tracked_tasks():
    """Test that _get_tasks returns tracked tasks."""
    orchestrator = Orchestrator(goal="test goal")
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
    ]
    orchestrator._tasks = tasks
    result = orchestrator._get_tasks()
    assert result is orchestrator._tasks
    assert len(result) == 2
    assert result[0].id == "1"
    assert result[1].id == "2"


def test_orchestrator_get_tasks_empty():
    """Test _get_tasks returns empty list when no tasks tracked."""
    orchestrator = Orchestrator(goal="test goal")
    result = orchestrator._get_tasks()
    assert result == []
    assert len(result) == 0
