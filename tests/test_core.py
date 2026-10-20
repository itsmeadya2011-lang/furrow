"""Smoke/integration tests for the furrow package as a whole."""

from __future__ import annotations

import importlib
from pathlib import Path

import furrow
from furrow import LLMClient, Settings


def test_top_level_exports() -> None:
    assert "LLMClient" in furrow.__all__
    assert "Settings" in furrow.__all__
    assert LLMClient is not None
    assert Settings is not None


def test_agents_package_exports() -> None:
    agents = importlib.import_module("furrow.agents")
    assert hasattr(agents, "PlannerAgent")
    assert hasattr(agents, "WorkerAgent")
    assert hasattr(agents, "TesterAgent")


def test_all_modules_import_cleanly() -> None:
    modules = [
        "furrow.config",
        "furrow.llm",
        "furrow.logging",
        "furrow.core.orchestrator",
        "furrow.agents.planner",
        "furrow.agents.worker",
        "furrow.agents.tester",
        "furrow.agents.prompts",
        "furrow.cli.main",
        "furrow.web.server",
    ]
    for mod in modules:
        importlib.import_module(mod)


def test_console_script_entrypoint() -> None:
    pyproject = Path("/workspace/app/pyproject.toml")
    content = pyproject.read_text()
    assert 'furrow = "furrow.cli.main:main"' in content

    cli_main = importlib.import_module("furrow.cli.main")
    main = cli_main.main
    assert callable(main)
    # Click group should have start and web commands
    commands = getattr(main, "commands", {})
    assert "start" in commands
    assert "web" in commands


def test_prompt_constants_non_empty() -> None:
    prompts = importlib.import_module("furrow.agents.prompts")
    for name in (
        "PLANNER_PROMPT",
        "WORKER_PROMPT",
        "WORKER_TOOL_INSTRUCTIONS",
        "TESTER_PROMPT",
        "TESTER_SUMMARY_PROMPT",
    ):
        val = getattr(prompts, name)
        assert isinstance(val, str)
        assert val.strip() != ""


def test_worker_tool_schemas() -> None:
    worker = importlib.import_module("furrow.agents.worker")
    tools = worker.TOOLS
    assert isinstance(tools, list)
    assert len(tools) == 3
    tool_names = {t.get("name") for t in tools}
    assert tool_names == {"read_file", "write_file", "list_files"}
    for t in tools:
        schema = t.get("input_schema")
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema


def test_worker_and_tester_constants() -> None:
    worker = importlib.import_module("furrow.agents.worker")
    tester = importlib.import_module("furrow.agents.tester")
    assert isinstance(worker.MAX_TOOL_ROUNDS, int)
    assert worker.MAX_TOOL_ROUNDS > 0
    assert isinstance(tester.DEFAULT_TEST_TIMEOUT, int)
    assert tester.DEFAULT_TEST_TIMEOUT > 0


def test_provider_covers_all_three() -> None:
    config = importlib.import_module("furrow.config")
    Provider = config.Provider
    values = {p.value for p in Provider}
    assert values == {"anthropic", "openai", "ollama"}


def test_pyproject_build_backend() -> None:
    pyproject = Path("/workspace/app/pyproject.toml")
    content = pyproject.read_text()
    assert 'build-backend = "setuptools.build_meta"' in content
