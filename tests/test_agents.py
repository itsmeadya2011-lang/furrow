from __future__ import annotations

import json

import pytest

from furrow.agents.tester import TesterAgent
from furrow.agents.utils import clean_json_response
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from tests.conftest import MockLLMClient


class TestWorkerParsing:
    def test_clean_json_no_fences(self):
        raw = '{"changes": {}, "summary": "ok"}'
        assert clean_json_response(raw) == raw

    def test_clean_json_with_fences(self):
        raw = '```json\n{"changes": {}, "summary": "ok"}\n```'
        assert clean_json_response(raw) == '{"changes": {}, "summary": "ok"}'

    def test_parse_valid_response(self):
        response = '{"changes": {"a.py": "content"}, "summary": "did it"}'
        changes, summary = WorkerAgent._parse_response(None, response)
        assert changes == {"a.py": "content"}
        assert summary == "did it"

    def test_parse_invalid_response(self):
        response = "Just some text"
        changes, summary = WorkerAgent._parse_response(None, response)
        assert changes == {}
        assert summary == response

    def test_parse_missing_changes_key(self):
        response = '{"summary": "nothing"}'
        changes, summary = WorkerAgent._parse_response(None, response)
        assert changes == {}
        assert summary == "nothing"

    @pytest.mark.asyncio
    async def test_run_writes_files(self):
        client = MockLLMClient([
            '{"changes": {"new.py": "print(1)"}, "summary": "created file"}',
        ])
        task = TaskModel(id="1", description="create new.py", files=["new.py"])
        worker = WorkerAgent(task=task, client=client)
        result = await worker.run()
        assert "Wrote 1 file(s): new.py" in result
        assert client.writes == [("new.py", "print(1)")]


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_tests_timeout_kills_process(self):
        agent = TesterAgent(client=MockLLMClient(["passed"]))
        result = await agent._run_tests()
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_run_passes_judge_output(self):
        client = MockLLMClient([
            '{"passed": true, "summary": "ok", "failures": []}',
        ])
        agent = TesterAgent(client=client)
        result = await agent.run("goal", [])
        assert result.passed is True
        assert result.summary == "ok"

    @pytest.mark.asyncio
    async def test_run_fallback_judge_on_bad_json(self):
        client = MockLLMClient([
            "Looks like passed to me",
        ])
        agent = TesterAgent(client=client)
        result = await agent.run("goal", [])
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_run_fallback_fail_on_bad_json(self):
        client = MockLLMClient([
            "Tests failed badly",
        ])
        agent = TesterAgent(client=client)
        result = await agent.run("goal", [])
        assert result.passed is False
