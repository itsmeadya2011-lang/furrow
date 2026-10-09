"""Tests for the PlannerAgent project context building."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from furrow.agents.planner import PlannerAgent, _strip_json_markers
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


class TestStripJsonMarkers:
    def test_strips_json_fence(self) -> None:
        text = '```json\n{"key": "val"}\n```'
        assert _strip_json_markers(text) == '{"key": "val"}'

    def test_strips_generic_fence(self) -> None:
        text = '```\n{"key": "val"}\n```'
        assert _strip_json_markers(text) == '{"key": "val"}'

    def test_no_fence(self) -> None:
        assert _strip_json_markers('{"key": "val"}').strip() == '{"key": "val"}'


class TestPlannerContext:
    """Tests for _build_project_context."""

    @pytest.mark.asyncio
    async def test_context_with_existing_files(self, tmp_path: Path) -> None:
        # Create some files
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "README.md").write_text("# Project")
        (tmp_path / "__pycache__").mkdir()

        settings = Settings(workspace=tmp_path)
        client = LLMClient(settings=settings)
        planner = PlannerAgent(client=client)

        context = await planner._build_project_context([])
        assert "main.py" in context
        assert "README.md" in context
        # __pycache__ should be filtered out
        assert "__pycache__" not in context

    @pytest.mark.asyncio
    async def test_context_with_prior_tasks(self, tmp_path: Path) -> None:
        settings = Settings(workspace=tmp_path)
        client = LLMClient(settings=settings)
        planner = PlannerAgent(client=client)

        prior = [
            TaskModel(id="1", description="first task", status="completed"),
            TaskModel(id="2", description="second task", status="failed"),
        ]
        context = await planner._build_project_context(prior)
        assert "first task" in context
        assert "second task" in context
        assert "completed" in context
        assert "failed" in context
