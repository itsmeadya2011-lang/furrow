import json
import tempfile
from pathlib import Path

import pytest

from furrow.config import Plan, TaskModel, TestResult, Settings
from furrow.agents.tester import TesterAgent
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="1", description="test")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_with_files():
    task = TaskModel(id="1", description="test", files=["a.py", "b.py"])
    assert task.files == ["a.py", "b.py"]


def test_settings_defaults():
    s = Settings()
    assert s.provider.value == "anthropic"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.workspace == Path.cwd()


def test_tester_parse_pytest_pass():
    tester = TesterAgent()
    output = """
============================= test session starts ==============================
collected 3 items

test_foo.py ...                                                           [100%]

============================== 3 passed in 0.12s ===============================
"""
    result = tester._parse_test_output(output)
    assert result.passed is True
    assert result.failures == []


def test_tester_parse_pytest_fail():
    tester = TesterAgent()
    output = """
============================= test session starts ==============================
collected 3 items

test_foo.py .F.                                                           [100%]

=================================== FAILURES ===================================
___________________ test_bar _________________________________________________

    def test_bar():
>       assert False
E       AssertionError: assert False

test_foo.py:5: AssertionError
=========================== short test summary info ============================
FAILED test_foo.py::test_bar - AssertionError: assert False
========================== 1 failed, 2 passed in 0.12s =========================
"""
    result = tester._parse_test_output(output)
    assert result.passed is False
    assert len(result.failures) > 0
    assert any("test_bar" in f for f in result.failures)


def test_tester_parse_no_runner():
    tester = TesterAgent()
    output = "No test runner found."
    result = tester._parse_test_output(output)
    assert result.passed is False
    assert "No test runner found" in result.summary


def test_orchestrator_state_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / ".furrow_state.json"
        
        # Create mock LLM client that returns a simple plan
        class MockClient:
            def __init__(self):
                self.settings = Settings()
            
            async def complete(self, prompt, system="", model=None):
                return json.dumps({
                    "tasks": [
                        {"id": "1", "description": "Task 1", "files": ["file1.py"], "dependencies": []},
                        {"id": "2", "description": "Task 2", "files": ["file2.py"], "dependencies": []}
                    ],
                    "rationale": "Test plan"
                })
            
            def list_files(self, directory):
                return ["file1.py", "file2.py"]
            
            async def read_file(self, path):
                return ""
            
            async def write_file(self, path, content):
                pass
        
        client = MockClient()
        
        # First orchestrator run - should create state
        orch = Orchestrator(goal="Test goal", client=client, state_file=state_file)
        orch.cycles = 1
        orch.tasks = [
            TaskModel(id="1", description="Task 1", files=["file1.py"]),
            TaskModel(id="2", description="Task 2", files=["file2.py"]),
        ]
        orch._save_state()
        
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["cycles"] == 1
        assert len(data["tasks"]) == 2
        
        # Second orchestrator - should load state
        orch2 = Orchestrator(goal="Test goal", client=client, state_file=state_file)
        assert orch2.cycles == 1
        assert len(orch2.tasks) == 2


def test_orchestrator_add_user_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / ".furrow_state.json"
        
        class MockClient:
            def __init__(self):
                self.settings = Settings()
            
            async def complete(self, prompt, system="", model=None):
                return json.dumps({"tasks": [], "rationale": "done"})
            
            def list_files(self, directory):
                return []
            
            async def read_file(self, path):
                return ""
            
            async def write_file(self, path, content):
                pass
        
        client = MockClient()
        orch = Orchestrator(goal="Original goal", client=client, state_file=state_file)
        orch.tasks = [
            TaskModel(id="1", description="Task 1", status="pending"),
            TaskModel(id="2", description="Task 2", status="completed"),
        ]
        
        orch.add_user_input("New requirement")
        
        assert "User input: New requirement" in orch.goal
        # Pending tasks should be cancelled
        assert orch.tasks[0].status == "cancelled"
        # Completed tasks should remain completed
        assert orch.tasks[1].status == "completed"


def test_orchestrator_is_done():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / ".furrow_state.json"
        
        class MockClient:
            def __init__(self):
                self.settings = Settings()
            
            async def complete(self, prompt, system="", model=None):
                return json.dumps({"tasks": [], "rationale": "done"})
            
            def list_files(self, directory):
                return []
            
            async def read_file(self, path):
                return ""
            
            async def write_file(self, path, content):
                pass
        
        client = MockClient()
        orch = Orchestrator(goal="Test", client=client, state_file=state_file)
        
        # No tasks - not done
        orch.tasks = []
        assert orch._is_done() is False
        
        # All completed - done
        orch.tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="completed"),
        ]
        assert orch._is_done() is True
        
        # Has pending - not done
        orch.tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="pending"),
        ]
        assert orch._is_done() is False
        
        # Has failed and no pending - not done
        orch.tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="failed"),
        ]
        assert orch._is_done() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
