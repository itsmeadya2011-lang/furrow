import json

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult


class MockLLMClient:
    def __init__(self, response="", workspace=None):
        self._response = response
        self.written_files = {}
        self.read_files = []
        self.listed_dirs = []
        self.workspace = workspace or "/tmp/furrow_test"

    @property
    def settings(self):
        class _Settings:
            workspace = "/tmp/furrow_test"
            planner_model = "test-planner"
            worker_model = "test-worker"
            tester_model = "test-tester"
        return _Settings()

    async def complete(self, prompt, system=None, model=None):
        return self._response

    async def read_file(self, path):
        self.read_files.append(str(path))
        return "mock content"

    async def write_file(self, path, content):
        self.written_files[str(path)] = content

    def list_files(self, directory):
        self.listed_dirs.append(str(directory))
        return []


class TestWorkerAgentParseJSONResponse:
    def test_valid_json(self):
        result = WorkerAgent._parse_json_response('{"files": {}, "summary": "ok"}')
        assert result == {"files": {}, "summary": "ok"}

    def test_json_in_markdown_code_block(self):
        text = '```json\n{"files": {"a.py": "print(1)"}, "summary": "done"}\n```'
        result = WorkerAgent._parse_json_response(text)
        assert result == {"files": {"a.py": "print(1)"}, "summary": "done"}

    def test_json_in_markdown_code_block_no_language(self):
        text = '```\n{"files": {}, "summary": "ok"}\n```'
        result = WorkerAgent._parse_json_response(text)
        assert result == {"files": {}, "summary": "ok"}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n{"files": {"x.py": "x"}}\nAnd some trailing text.'
        result = WorkerAgent._parse_json_response(text)
        assert result == {"files": {"x.py": "x"}}

    def test_invalid_text_returns_none(self):
        result = WorkerAgent._parse_json_response("this is not json at all")
        assert result is None

    def test_empty_string_returns_none(self):
        result = WorkerAgent._parse_json_response("")
        assert result is None


class TestWorkerAgentRun:
    @pytest.mark.asyncio
    async def test_run_with_json_response_writes_files(self):
        response = json.dumps({
            "files": {"src/test.py": "def test():\n    pass"},
            "summary": "created test file"
        })
        client = MockLLMClient(response=response)
        task = TaskModel(id="1", description="create test", files=[])
        worker = WorkerAgent(task=task, client=client)
        result = await worker.run()
        assert result == "created test file"
        assert "src/test.py" in client.written_files
        assert client.written_files["src/test.py"] == "def test():\n    pass"

    @pytest.mark.asyncio
    async def test_run_with_invalid_json_returns_response(self):
        client = MockLLMClient(response="no json here")
        task = TaskModel(id="1", description="test", files=[])
        worker = WorkerAgent(task=task, client=client)
        result = await worker.run()
        assert result == "no json here"


class TestPlannerAgentPlan:
    @pytest.mark.asyncio
    async def test_plan_valid_json(self):
        plan_json = json.dumps({
            "tasks": [
                {"id": "1", "description": "do thing", "files": [], "dependencies": []}
            ],
            "rationale": "test rationale"
        })
        client = MockLLMClient(response=plan_json)
        planner = PlannerAgent(client=client)
        plan = await planner.plan("test goal")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "1"
        assert plan.tasks[0].description == "do thing"
        assert plan.rationale == "test rationale"

    @pytest.mark.asyncio
    async def test_plan_invalid_json_raises_valueerror(self):
        client = MockLLMClient(response="not json at all")
        planner = PlannerAgent(client=client)
        with pytest.raises(ValueError):
            await planner.plan("test goal")


class TestTesterAgentRun:
    @pytest.mark.asyncio
    async def test_run_returns_test_result(self):
        response = json.dumps({
            "passed": True,
            "summary": "All tests passed",
            "failures": []
        })
        client = MockLLMClient(response=response)
        tester = TesterAgent(client=client)
        task = TaskModel(id="1", description="test", files=[])
        result = await tester.run("test goal", [task])
        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_failing_result(self):
        response = json.dumps({
            "passed": False,
            "summary": "1 test failed",
            "failures": ["test_a failed"]
        })
        client = MockLLMClient(response=response)
        tester = TesterAgent(client=client)
        task = TaskModel(id="1", description="test", files=[])
        result = await tester.run("test goal", [task])
        assert result.passed is False
        assert result.summary == "1 test failed"
        assert result.failures == ["test_a failed"]
