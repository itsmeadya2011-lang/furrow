from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel


def test_build_candidates_includes_env_var(monkeypatch):
    monkeypatch.setenv("FURROW_TEST_CMD", "python -m pytest -x")
    agent = TesterAgent()
    candidates = agent._build_candidates()
    assert ["python", "-m", "pytest", "-x"] in candidates


def test_build_candidates_has_defaults():
    agent = TesterAgent()
    candidates = agent._build_candidates()
    assert ["pytest", "-q"] in candidates
    assert ["npm", "test", "--", "--silent"] in candidates
    assert ["go", "test", "./..."] in candidates


def test_build_candidates_empty_env():
    agent = TesterAgent()
    candidates = agent._build_candidates()
    assert len(candidates) >= 7


@pytest.mark.asyncio
async def test_run_tests_returns_output():
    agent = TesterAgent()

    # Create a mock process with async communicate
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"test output", b""))
    mock_proc.kill = MagicMock()

    async def mock_create(*args, **kwargs):
        return mock_proc

    with patch("asyncio.create_subprocess_exec", side_effect=mock_create):
        output, attempted, succeeded = await agent._run_tests()
        assert "test output" in output
        assert succeeded