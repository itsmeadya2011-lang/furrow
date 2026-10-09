"""Tests for the Tester agent: runner detection, timeout config, exception handling."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Settings


def _make_agent(workspace: str | None = None) -> TesterAgent:
    """Create a TesterAgent with a mock client to avoid API calls."""
    settings = Settings(workspace=workspace or os.getcwd())
    return TesterAgent(client=Mock(), settings=settings)


# ---------------------------------------------------------------------------
# Test runner detection tests
# ---------------------------------------------------------------------------

def test_detect_python_project_runners():
    """Should return pytest runners for a Python project (with pyproject.toml)."""
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agent = _make_agent(workspace=workspace)
    runners = agent._detect_test_runners()
    assert runners[0] == ["pytest", "-q"]
    assert runners[1] == ["python", "-m", "pytest", "-q"]


def test_detect_empty_directory_runners():
    """Should return fallback runners when no project files detected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(workspace=tmpdir)
        runners = agent._detect_test_runners()
        # Should return the fallback list with all common runners
        assert len(runners) >= 5
        assert ["pytest", "-q"] in runners


def test_detect_nondefault_workspace():
    """_detect_test_runners should use the configured workspace path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a package.json to simulate a JS project
        import json
        with open(os.path.join(tmpdir, "package.json"), "w") as f:
            json.dump({"name": "test"}, f)
        agent = _make_agent(workspace=tmpdir)
        runners = agent._detect_test_runners()
        assert runners[0] == ["npm", "test", "--", "--silent"]


# ---------------------------------------------------------------------------
# Timeout tests
# ---------------------------------------------------------------------------

def test_test_timeout_from_settings():
    """test_timeout should be read from settings."""
    settings = Settings(test_timeout=60)
    agent = TesterAgent(client=Mock(), settings=settings)
    assert agent.settings.test_timeout == 60


def test_test_timeout_default():
    """Default test_timeout should be 120."""
    settings = Settings()
    assert settings.test_timeout == 120


# ---------------------------------------------------------------------------
# Exception handling tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_test_runner_returns_message():
    """When no test runner is available, should return a descriptive message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(workspace=tmpdir)
        # Mock subprocess so all runners appear missing (no system npm, etc.)
        with patch("furrow.agents.tester.asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await agent._run_tests()
        assert "No test runner found" in result


@pytest.mark.asyncio
async def test_test_runner_not_found_continues():
    """FileNotFoundError should be caught and continue to next runner."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(workspace=tmpdir)
        with patch("furrow.agents.tester.asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await agent._run_tests()
        # Should return "No test runner found" since none of the runners exist
        assert result == "No test runner found."


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

def test_tester_agent_accepts_settings():
    """TesterAgent should accept and store settings."""
    settings = Settings(test_timeout=30)
    agent = TesterAgent(client=Mock(), settings=settings)
    assert agent.settings is settings


def test_tester_agent_uses_client_settings():
    """TesterAgent should fall back to client settings when settings is None."""
    mock_client = Mock()
    mock_client.settings = Settings(test_timeout=45)
    agent = TesterAgent(client=mock_client, settings=None)
    assert agent.settings is mock_client.settings
    assert agent.settings.test_timeout == 45
