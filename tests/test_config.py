from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import (
    FileEdit,
    Plan,
    Provider,
    Settings,
    TaskModel,
    TaskResult,
    TestResult,
)


class TestProvider:
    def test_provider_values(self):
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OPENAI.value == "openai"
        assert Provider.OLLAMA.value == "ollama"

    def test_provider_is_str_enum(self):
        assert isinstance(Provider.ANTHROPIC, str)
        assert Provider("openai") == Provider.OPENAI


class TestTaskModel:
    def test_defaults(self):
        t = TaskModel(id="1", description="do thing")
        assert t.status == "pending"
        assert t.result is None
        assert t.files == []
        assert t.dependencies == []

    def test_custom_fields(self):
        t = TaskModel(
            id="2",
            description="build api",
            files=["a.py", "b.py"],
            dependencies=["1"],
            status="completed",
            result="done",
        )
        assert t.id == "2"
        assert t.files == ["a.py", "b.py"]
        assert t.dependencies == ["1"]
        assert t.status == "completed"
        assert t.result == "done"

    def test_status_mutable(self):
        t = TaskModel(id="1", description="do thing")
        t.status = "failed"
        t.result = "error"
        assert t.status == "failed"
        assert t.result == "error"


class TestPlan:
    def test_parse(self):
        p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
        assert len(p.tasks) == 1
        assert p.rationale == "ok"

    def test_empty_tasks_allowed(self):
        p = Plan(tasks=[], rationale="nothing")
        assert p.tasks == []

    def test_from_dict(self):
        p = Plan(
            tasks=[{"id": "1", "description": "x"}], rationale="ok"
        )
        assert p.tasks[0].id == "1"


class TestTestResult:
    def test_defaults(self):
        t = TestResult(passed=True, summary="ok")
        assert t.passed is True
        assert t.failures == []

    def test_failed(self):
        t = TestResult(passed=False, summary="bad", failures=["test_a failed"])
        assert t.passed is False
        assert len(t.failures) == 1


class TestFileEdit:
    @pytest.mark.parametrize("op", ["create", "edit", "delete"])
    def test_operations(self, op):
        e = FileEdit(file="path.py", operation=op)
        assert e.operation == op

    def test_create_has_content(self):
        e = FileEdit(file="x.py", operation="create", content="print(1)")
        assert e.content == "print(1)"

    def test_edit_has_old_and_new(self):
        e = FileEdit(file="x.py", operation="edit", old="a", new="b")
        assert e.old == "a"
        assert e.new == "b"

    def test_invalid_operation(self):
        with pytest.raises(ValidationError):
            FileEdit(file="x.py", operation="replace")


class TestTaskResult:
    def test_defaults(self):
        t = TaskResult(summary="did it")
        assert t.edits == []
        assert t.issues == []

    def test_with_edits(self):
        edit = FileEdit(file="x.py", operation="create", content="hi")
        t = TaskResult(summary="done", edits=[edit], issues=["warn"])
        assert len(t.edits) == 1
        assert t.edits[0].file == "x.py"
        assert t.issues == ["warn"]


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0
        assert s.max_retries == 3

    def test_required_api_keys(self):
        s = Settings()
        # Defaults are None unless env vars are set.
        assert isinstance(s.anthropic_api_key, (str, type(None))) or s.anthropic_api_key is None

    def test_provider_override(self):
        s = Settings(provider=Provider.OPENAI, openai_api_key="sk-test")
        assert s.provider == Provider.OPENAI
        assert s.openai_api_key == "sk-test"

    def test_env_prefix(self, monkeypatch):
        monkeypatch.setenv("FURROW_MAX_CYCLES", "7")
        s = Settings()
        assert s.max_cycles == 7

    def test_workspace_default(self):
        s = Settings()
        assert isinstance(s.workspace, Path)
