from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TestRunOutput


class TestTesterAgent:
    def test_detect_runners_pytest(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest]")
        settings = Settings(workspace=tmp_path)
        client = MagicMock()
        client.settings = settings
        agent = TesterAgent(client=client)
        runners = agent._detect_runners()
        assert any(name == "pytest" for name, _ in runners)

    def test_detect_runners_npm(self, tmp_path: Path):
        (tmp_path / "package.json").write_text("{}")
        settings = Settings(workspace=tmp_path)
        client = MagicMock()
        client.settings = settings
        agent = TesterAgent(client=client)
        runners = agent._detect_runners()
        assert any(name == "npm" for name, _ in runners)

    def test_detect_runners_cargo(self, tmp_path: Path):
        (tmp_path / "Cargo.toml").write_text("[package]")
        settings = Settings(workspace=tmp_path)
        client = MagicMock()
        client.settings = settings
        agent = TesterAgent(client=client)
        runners = agent._detect_runners()
        assert any(name == "cargo" for name, _ in runners)

    def test_detect_runners_go(self, tmp_path: Path):
        (tmp_path / "go.mod").write_text("module test")
        settings = Settings(workspace=tmp_path)
        client = MagicMock()
        client.settings = settings
        agent = TesterAgent(client=client)
        runners = agent._detect_runners()
        assert any(name == "go" for name, _ in runners)

    def test_format_test_output(self):
        agent = TesterAgent(client=MagicMock())
        run = TestRunOutput(
            stdout="pass",
            stderr="",
            returncode=0,
            command="pytest -q",
            runner="pytest",
            timed_out=False,
        )
        text = agent._format_test_output(run)
        assert "Runner: pytest" in text
        assert "Command: pytest -q" in text
        assert "Return code: 0" in text
        assert "Stdout:\npass" in text

    @pytest.mark.asyncio
    async def test_retry_json_prompts_for_valid_json(self):
        client = MagicMock()
        client.settings = Settings()
        client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')
        agent = TesterAgent(client=client)
        result = await agent._retry_json("Goal: test", "not json")
        assert json.loads(result)["passed"] is True
        assert "not json" in client.complete.call_args[0][0]
