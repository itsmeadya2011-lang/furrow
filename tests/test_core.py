import pytest
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.core.orchestrator import Orchestrator
from furrow.agents.worker import WorkerAgent
from furrow.agents.planner import PlannerAgent


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


class FakeClient:
    def __init__(self, complete_response: str | None = None):
        self.complete_response = complete_response
        self.prompts_received = []
        self.write_records = []

    async def complete(self, prompt: str, model: str | None = None, **kwargs):
        self.prompts_received.append(prompt)
        if self.complete_response is not None:
            return self.complete_response
        # Default valid plan JSON for orchestrator/planner
        return '{"tasks":[{"id":"1","description":"do thing","files":["a.py"],"dependencies":[]}],"rationale":"ok"}'

    def list_files(self, path: str | None = None):
        return []

    async def write_file(self, path: str, content: str):
        self.write_records.append((path, content))


@pytest.mark.asyncio
async def test_orchestrator_initial_state():
    """Test orchestrator has empty tasks and correct settings."""
    fake = FakeClient()
    orch = Orchestrator(goal="test", client=fake)
    assert orch.tasks == []
    assert orch._get_tasks() == []
    assert orch.settings.max_cycles is not None


@pytest.mark.asyncio
async def test_orchestrator_cycle_tracks_tasks():
    """Test orchestrator tracks tasks after running a cycle."""
    plan_json = '{"tasks":[{"id":"1","description":"do thing","files":["a.py"],"dependencies":[]}],"rationale":"ok"}'
    fake = FakeClient(complete_response=plan_json)
    orch = Orchestrator(goal="test", client=fake)
    await orch._cycle()
    tasks = orch._get_tasks()
    assert len(tasks) == 1
    assert tasks[0].id == "1"
    assert tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_orchestrator_is_done_empty():
    """Test _is_done returns True on empty task list."""
    fake = FakeClient()
    orch = Orchestrator(goal="test", client=fake)
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_worker_file_parsing():
    """Test WorkerAgent parses FILE: blocks and writes files."""
    worker_response = """FILE: /tmp/furrow_test_file.txt
hello world"""
    fake = FakeClient(complete_response=worker_response)
    from furrow.config import TaskModel
    task = TaskModel(id="1", description="x")
    result = await WorkerAgent(task=task, client=fake).run()
    # Check file was written
    assert len(fake.write_records) == 1
    assert fake.write_records[0][0] == "/tmp/furrow_test_file.txt"
    assert fake.write_records[0][1] == "hello world"
    assert "file" in result.lower()


@pytest.mark.asyncio
async def test_planner_includes_context():
    """Test PlannerAgent includes project files and goal in prompt."""
    plan_json = '{"tasks":[{"id":"1","description":"build thing","files":[],"dependencies":[]}],"rationale":"ok"}'
    fake = FakeClient(complete_response=plan_json)
    result = await PlannerAgent(client=fake).plan("build thing")
    # Check prompt captured context
    assert len(fake.prompts_received) == 1
    prompt = fake.prompts_received[0]
    assert "Current project files" in prompt
    assert "build thing" in prompt
    assert result.tasks[0].description == "build thing"
