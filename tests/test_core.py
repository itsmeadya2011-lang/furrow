import pytest
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.agents.tester import TesterAgent
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_task_model_with_all_fields():
    t = TaskModel(
        id="2",
        description="test with files",
        files=["src/a.py", "src/b.py"],
        dependencies=["1"],
        status="completed",
        result="done"
    )
    assert t.files == ["src/a.py", "src/b.py"]
    assert t.dependencies == ["1"]
    assert t.status == "completed"
    assert t.result == "done"


def test_test_result_failed():
    t = TestResult(passed=False, summary="failed", failures=["error 1", "error 2"])
    assert t.passed is False
    assert len(t.failures) == 2


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


def test_settings_custom():
    s = Settings(provider=Provider.OPENAI, max_cycles=10)
    assert s.provider == Provider.OPENAI
    assert s.max_cycles == 10


def test_tester_parse_pytest_output():
    """Test that tester correctly parses pytest output."""
    tester = TesterAgent()
    
    # Passing output
    passed_output = "5 passed in 0.5s"
    passed, failures = tester._parse_test_output(passed_output)
    assert passed is True
    assert failures == []
    
    # Failing output
    failed_output = "FAILED tests/test_auth.py::test_login - AssertionError: expected True"
    passed, failures = tester._parse_test_output(failed_output)
    assert passed is False
    assert len(failures) > 0
    assert "FAILED" in failures[0]
    
    # No test runner
    no_runner = "No test runner found."
    passed, failures = tester._parse_test_output(no_runner)
    assert passed is False
    assert "No test runner" in failures[0]


def test_llm_client_init():
    """Test LLMClient initialization."""
    client = LLMClient()
    assert client.settings is not None
    assert client._anthropic is None
    assert client._openai is None
