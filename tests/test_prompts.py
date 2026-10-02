"""Tests for furrow prompts."""
from furrow.agents.prompts import PLANNER_PROMPT, TESTER_PROMPT, WORKER_PROMPT


class TestPlannerPrompt:
    def test_contains_role(self):
        assert "planning agent" in PLANNER_PROMPT.lower()

    def test_contains_json_shape(self):
        assert '"tasks"' in PLANNER_PROMPT
        assert '"rationale"' in PLANNER_PROMPT

    def test_mentions_parallel(self):
        assert "parallel" in PLANNER_PROMPT.lower()


class TestWorkerPrompt:
    def test_contains_role(self):
        assert "worker agent" in WORKER_PROMPT.lower()

    def test_contains_changes_structure(self):
        assert '"changes"' in WORKER_PROMPT
        assert '"path"' in WORKER_PROMPT
        assert '"content"' in WORKER_PROMPT

    def test_mentions_json(self):
        assert "JSON" in WORKER_PROMPT


class TestTesterPrompt:
    def test_contains_role(self):
        assert "tester agent" in TESTER_PROMPT.lower()

    def test_contains_json_shape(self):
        assert '"passed"' in TESTER_PROMPT
        assert '"summary"' in TESTER_PROMPT
        assert '"failures"' in TESTER_PROMPT
