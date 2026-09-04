from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent, extract_json
from furrow.agents.tester import TesterAgent, _detect_test_command, _run_tests
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


# --- extract_json ---


def test_extract_json_direct():
    text = '{"key": "value"}'
    assert extract_json(text) == {"key": "value"}


def test_extract_json_code_block():
    text = "Here is the plan:\n```json\n{\"key\": \"value\"}\n```"
    assert extract_json(text) == {"key": "value"}


def test_extract_json_code_block_no_language():
    text = "Here is the plan:\n```\n{\"key\": \"value\"}\n```"
    assert extract_json(text) == {"key": "value"}


def test_extract_json_brace_match():
    text = "Some text before {\"key\": \"value\"} and after"
    assert extract_json(text) == {"key": "value"}


def test_extract_json_invalid():
    with pytest.raises(ValueError):
        extract_json("no json here")


# --- PlannerAgent ---


def _settings(**kwargs) -> Settings:
    data = {"provider": Provider.ANTHROPIC, "anthropic_api_key": "sk-ant-test", **kwargs}
    return Settings(**data)


@pytest.mark.asyncio
async def test_planner_agent_plan():
    settings = _settings()
    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "done"}')
    agent = PlannerAgent(client=client)
    plan = await agent.plan("build a thing")
    assert isinstance(plan, Plan)
    assert plan.rationale == "done"


@pytest.mark.asyncio
async def test_planner_agent_retries_on_bad_json():
    settings = _settings()
    client = MagicMock()
    client.settings = settings
    # First call returns invalid JSON, second returns valid
    client.complete = AsyncMock(
        side_effect=[
            "not json",
            '{"tasks": [{"id": "1", "description": "x", "files": [], "dependencies": []}], "rationale": "ok"}',
        ]
    )
    agent = PlannerAgent(client=client)
    plan = await agent.plan("build a thing")
    assert len(plan.tasks) == 1
    assert client.complete.call_count == 2


@pytest.mark.asyncio
async def test_planner_agent_fails_after_max_retries():
    settings = _settings()
    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(return_value="always bad")
    agent = PlannerAgent(client=client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await agent.plan("build a thing")
    assert client.complete.call_count == 3  # initial + 2 retries


# --- WorkerAgent ---


@pytest.mark.asyncio
async def test_worker_agent_includes_context():
    settings = _settings()
    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(return_value="done")
    client.list_files = MagicMock(return_value=["main.py", "utils.py"])
    client.read_file = AsyncMock(return_value="print('hi')")

    task = TaskModel(id="1", description="add logging", files=["main.py"])
    agent = WorkerAgent(task=task, client=client)
    result = await agent.run(workspace=Path("/fake"))

    assert result == "done"
    prompt = client.complete.call_args[0][0]
    assert "Project structure:" in prompt
    assert "main.py" in prompt
    assert "utils.py" in prompt
    assert "Workspace context:" in prompt


@pytest.mark.asyncio
async def test_worker_agent_no_files():
    settings = _settings()
    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(return_value="done")
    client.list_files = MagicMock(return_value=["main.py"])

    task = TaskModel(id="1", description="do something")
    agent = WorkerAgent(task=task, client=client)
    result = await agent.run(workspace=Path("/fake"))

    prompt = client.complete.call_args[0][0]
    assert "Project structure:" in prompt
    assert "Relevant file contents:" not in prompt


# --- TesterAgent ---


def test_detect_test_command_cargo(tmp_path: Path):
    (tmp_path / "Cargo.toml").write_text("[package]\nname = \"x\"\n")
    assert _detect_test_command(tmp_path) == ["cargo", "test"]


def test_detect_test_command_go(tmp_path: Path):
    (tmp_path / "go.mod").write_text("module x\n")
    assert _detect_test_command(tmp_path) == ["go", "test", "./..."]


def test_detect_test_command_makefile(tmp_path: Path):
    (tmp_path / "Makefile").write_text("test:\n\tpytest -q\n")
    assert _detect_test_command(tmp_path) == ["make", "test"]


def test_detect_test_command_python(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest]\n")
    assert _detect_test_command(tmp_path) == ["pytest", "-q"]


def test_detect_test_command_none(tmp_path: Path):
    assert _detect_test_command(tmp_path) == []


@pytest.mark.asyncio
async def test_tester_agent_run_passes():
    settings = _settings()
    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(
        return_value='{"passed": true, "summary": "all good", "failures": []}'
    )

    agent = TesterAgent(client=client)
    with patch.object(agent, "_run_tests", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {
            "command": "pytest -q",
            "returncode": 0,
            "stdout": "passed",
            "stderr": "",
        }
        result = await agent.run("goal", [])

    assert result.passed is True
    assert result.summary == "all good"


@pytest.mark.asyncio
async def test_tester_agent_run_retries_on_bad_json():
    settings = _settings()
    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(
        side_effect=[
            "not json",
            '{"passed": true, "summary": "ok", "failures": []}',
        ]
    )

    agent = TesterAgent(client=client)
    with patch.object(agent, "_run_tests", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {
            "command": "pytest -q",
            "returncode": 1,
            "stdout": "fail",
            "stderr": "",
        }
        result = await agent.run("goal", [])

    assert result.passed is True
    assert client.complete.call_count == 2


@pytest.mark.asyncio
async def test_tester_agent_run_fallback_when_retry_fails():
    settings = _settings()
    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(return_value="also not json")

    agent = TesterAgent(client=client)
    with patch.object(agent, "_run_tests", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {
            "command": "pytest -q",
            "returncode": 1,
            "stdout": "fail",
            "stderr": "",
        }
        result = await agent.run("goal", [])

    assert result.passed is False
    assert "also not json" in result.summary
