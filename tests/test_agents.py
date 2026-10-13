from __future__ import annotations

import json

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from tests.fakes import FakeLLMClient, make_settings

VALID_PLAN = {
    "tasks": [
        {
            "id": "1",
            "description": "Add a greeting function",
            "files": ["src/greet.py", "tests/test_greet.py"],
            "dependencies": ["0"],
        }
    ],
    "rationale": "First step to implement the greeting.",
}

VALID_TEST_RESULT = {
    "passed": True,
    "summary": "All tests passed",
    "failures": [],
}


# --------------------------------------------------------------------------- #
# PlannerAgent
# --------------------------------------------------------------------------- #
class TestPlannerAgent:
    async def test_valid_fenced_json_produces_plan(self, make_llm):
        fake = make_llm(responses=['```json\n' + json.dumps(VALID_PLAN) + '\n```'])
        plan = await PlannerAgent(client=fake).plan("say hello")
        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 1
        t = plan.tasks[0]
        assert isinstance(t, TaskModel)
        assert t.id == "1"
        assert t.description == "Add a greeting function"
        assert t.files == ["src/greet.py", "tests/test_greet.py"]
        assert t.dependencies == ["0"]
        assert t.status == "pending"
        assert t.result is None
        assert plan.rationale == "First step to implement the greeting."
        assert fake.calls[0][0] == "complete_json"

    async def test_invalid_json_retries_then_raises(self, make_llm):
        fake = make_llm(responses=[ValueError("nope"), ValueError("nope")])
        with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
            await PlannerAgent(client=fake).plan("say hello")
        methods = [c[0] for c in fake.calls]
        assert methods == ["complete_json", "complete_json"]

    async def test_valid_on_retry(self, make_llm):
        fake = make_llm(responses=[ValueError("nope"), VALID_PLAN])
        plan = await PlannerAgent(client=fake).plan("say hello")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "1"

    async def test_history_included_in_prompt(self, make_llm):
        fake = make_llm(responses=[VALID_PLAN])
        history = [{"cycle": 1, "goal": "say hello", "marker": "UNIQUEMARKER123"}]
        await PlannerAgent(client=fake).plan("say hello", history=history)
        prompt = fake.calls[0][1]
        assert "Previous cycle history" in prompt
        assert "UNIQUEMARKER123" in prompt

    async def test_no_history_does_not_include_history_block(self, make_llm):
        fake = make_llm(responses=[VALID_PLAN])
        await PlannerAgent(client=fake).plan("say hello")
        prompt = fake.calls[0][1]
        assert "Previous cycle history" not in prompt

    async def test_empty_tasks_when_goal_complete(self, make_llm):
        fake = make_llm(responses=[{"tasks": [], "rationale": "Goal already complete"}])
        plan = await PlannerAgent(client=fake).plan("say hello", history=[{"cycle": 1}])
        assert plan.tasks == []
        assert plan.rationale == "Goal already complete"

    async def test_malformed_task_entries_are_skipped(self, make_llm):
        data = {
            "tasks": [
                {"description": "no id"},
                {"id": "1"},
                {"id": "1", "description": "good"},
                "not a dict",
                42,
            ],
            "rationale": "ok",
        }
        fake = make_llm(responses=[data])
        plan = await PlannerAgent(client=fake).plan("say hello")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "1"
        assert plan.tasks[0].description == "good"

    async def test_tasks_not_list_becomes_empty(self, make_llm):
        fake = make_llm(responses=[{"tasks": "not a list", "rationale": "ok"}])
        plan = await PlannerAgent(client=fake).plan("say hello")
        assert plan.tasks == []


# --------------------------------------------------------------------------- #
# WorkerAgent
# --------------------------------------------------------------------------- #
class TestWorkerAgent:
    async def test_tool_calls_happen_in_order_and_file_written(self, tmp_path, make_llm):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "seed.txt").write_text("seed content")
        out_path = ws / "out.txt"

        actions = [
            {"action": "list_files", "directory": str(ws)},
            {"action": "read_file", "path": str(ws / "seed.txt")},
            {"action": "write_file", "path": str(out_path), "content": "hello world"},
            {"action": "finish", "summary": "wrote the file"},
        ]
        responses = [json.dumps(a) for a in actions]
        fake = make_llm(workspace=ws, responses=responses)

        summary = await WorkerAgent(task=TaskModel(id="1", description="write"), client=fake).run()

        assert summary == "wrote the file"
        assert out_path.read_text() == "hello world"

        tool_kinds = [tc[0] for tc in fake.tool_calls]
        assert tool_kinds == ["list_files", "read_file", "write_file"]
        assert fake.tool_calls[0] == ("list_files", str(ws))
        assert fake.tool_calls[1] == ("read_file", str(ws / "seed.txt"))
        assert fake.tool_calls[2] == ("write_file", str(out_path), "hello world")
        assert len(fake.calls) == 4  # 4 complete_messages round-trips

    async def test_finish_only(self, tmp_path, make_llm):
        fake = make_llm(
            workspace=tmp_path,
            responses=[json.dumps({"action": "finish", "summary": "done"})],
        )
        summary = await WorkerAgent(task=TaskModel(id="1", description="x"), client=fake).run()
        assert summary == "done"
        assert fake.tool_calls == []

    async def test_path_escape_is_blocked(self, tmp_path, make_llm):
        ws = tmp_path / "ws"
        ws.mkdir()
        actions = [
            {"action": "write_file", "path": "../escape.txt", "content": "pwned"},
            {"action": "finish", "summary": "finished despite bad path"},
        ]
        responses = [json.dumps(a) for a in actions]
        fake = make_llm(workspace=ws, responses=responses)

        summary = await WorkerAgent(task=TaskModel(id="1", description="x"), client=fake).run()

        assert summary == "finished despite bad path"
        assert not any(tc[0] == "write_file" for tc in fake.tool_calls)
        assert not (tmp_path / "escape.txt").exists()

    async def test_absolute_path_within_workspace_allowed(self, tmp_path, make_llm):
        ws = tmp_path / "ws"
        ws.mkdir()
        target = ws / "allowed.txt"
        actions = [
            {"action": "write_file", "path": str(target), "content": "ok"},
            {"action": "finish", "summary": "ok"},
        ]
        fake = make_llm(workspace=ws, responses=[json.dumps(a) for a in actions])
        await WorkerAgent(task=TaskModel(id="1", description="x"), client=fake).run()
        assert target.read_text() == "ok"

    async def test_invalid_action_format_retries(self, tmp_path, make_llm):
        fake = make_llm(
            workspace=tmp_path,
            responses=["not an action at all", json.dumps({"action": "finish", "summary": "ok"})],
        )
        summary = await WorkerAgent(task=TaskModel(id="1", description="x"), client=fake).run()
        assert summary == "ok"
        assert len(fake.calls) == 2

    async def test_round_limit_returned(self, tmp_path, make_llm):
        fake = make_llm(workspace=tmp_path, responses=["not an action"], worker_rounds=1)
        summary = await WorkerAgent(task=TaskModel(id="1", description="x"), client=fake).run()
        assert "round limit" in summary
        assert len(fake.calls) == 1

    async def test_llm_error_returns_worker_error(self, tmp_path, make_llm):
        fake = make_llm(workspace=tmp_path, responses=[ValueError("connection down")])
        summary = await WorkerAgent(task=TaskModel(id="1", description="x"), client=fake).run()
        assert "Worker error" in summary
        assert "connection down" in summary


# --------------------------------------------------------------------------- #
# TesterAgent
# --------------------------------------------------------------------------- #
class TestTesterAgent:
    async def test_valid_test_result_json_parsed(self, tmp_path, make_llm, dummy_test_cmd):
        fake = make_llm(
            workspace=tmp_path, test_command=dummy_test_cmd, responses=[VALID_TEST_RESULT]
        )
        result = await TesterAgent(client=fake).run(
            "goal", [TaskModel(id="1", description="d")]
        )
        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    async def test_malformed_json_falls_back_to_heuristic(self, tmp_path, make_llm, dummy_test_cmd):
        fake = make_llm(
            workspace=tmp_path,
            test_command=dummy_test_cmd,
            responses=[ValueError("bad json"), "The tests passed; all good"],
        )
        result = await TesterAgent(client=fake).run("goal", [TaskModel(id="1", description="d")])
        assert result.passed is True
        assert "passed" in result.summary.lower()
        assert result.failures == []
        methods = [c[0] for c in fake.calls]
        assert methods == ["complete_json", "complete"]

    async def test_run_tests_with_dummy_command(self, tmp_path, make_llm, dummy_test_cmd):
        fake = make_llm(workspace=tmp_path, test_command=dummy_test_cmd, responses=[])
        tester = TesterAgent(client=fake)
        output = await tester._run_tests()
        assert "exit code: 0" in output
        assert "dummy test output" in output

    async def test_no_test_runner_returns_no_runner_message(self, tmp_path):
        # point test_command at a binary that does not exist; _run_tests falls through
        settings = make_settings(workspace=tmp_path, test_command="nonexistent_testrunner_xyz")
        fake = FakeLLMClient(settings=settings, responses=[])
        tester = TesterAgent(client=fake)
        output = await tester._run_tests()
        assert "No test runner found" in output
