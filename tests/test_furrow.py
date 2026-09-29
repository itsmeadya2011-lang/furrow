from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import (
    LLMClient,
    ToolCall,
    ToolResponse,
    _to_anthropic_messages,
    _to_anthropic_tools,
)


class MockLLMClient:
    """Mimics ``furrow.llm.LLMClient`` for testing without network access.

    Every async method returns a configurable value and records its call
    arguments so tests can make assertions about what the agent did.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings(
            provider=Provider.ANTHROPIC,
            model="test-model",
            planner_model="test-planner",
            worker_model="test-worker",
            tester_model="test-tester",
        )
        self.complete_return: str = ""
        self.complete_with_tools_responses: list[ToolResponse] = []
        self.complete_with_tools_return: ToolResponse = ToolResponse()
        self.read_file_return: str = ""
        self.list_files_return: list[str] = []
        self.run_command_return: str = ""
        self.complete_calls: list[dict[str, Any]] = []
        self.complete_with_tools_calls: list[dict[str, Any]] = []
        self.execute_tool_calls: list[dict[str, Any]] = []
        self.read_file_calls: list[dict[str, Any]] = []
        self.write_file_calls: list[dict[str, Any]] = []
        self.list_files_calls: list[dict[str, Any]] = []
        self.run_command_calls: list[dict[str, Any]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls.append({"prompt": prompt, "system": system, "model": model})
        return self.complete_return

    async def complete_with_tools(
        self,
        prompt: str = "",
        system: str = "",
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> ToolResponse:
        self.complete_with_tools_calls.append(
            {
                "prompt": prompt,
                "system": system,
                "model": model,
                "tools": tools,
                "messages": messages,
            }
        )
        if self.complete_with_tools_responses:
            return self.complete_with_tools_responses.pop(0)
        return self.complete_with_tools_return

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.execute_tool_calls.append({"name": name, "arguments": arguments})
        if name == "read_file":
            return await self.read_file(arguments["path"])
        elif name == "write_file":
            await self.write_file(arguments["path"], arguments["content"])
            return ""
        elif name == "list_files":
            return "\n".join(self.list_files(arguments["directory"]))
        elif name == "run_command":
            return await self.run_command(arguments["command"])
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def read_file(self, path: str | Path) -> str:
        self.read_file_calls.append({"path": str(path)})
        return self.read_file_return

    async def write_file(self, path: str | Path, content: str) -> None:
        self.write_file_calls.append({"path": str(path), "content": content})

    def list_files(self, directory: str | Path) -> list[str]:
        self.list_files_calls.append({"directory": str(directory)})
        return self.list_files_return

    async def run_command(self, command: str, timeout: float = 120.0) -> str:
        self.run_command_calls.append({"command": command, "timeout": timeout})
        return self.run_command_return


def _test_settings(**overrides: Any) -> Settings:
    base = {
        "provider": Provider.ANTHROPIC,
        "model": "test-model",
        "planner_model": "test-planner",
        "worker_model": "test-worker",
        "tester_model": "test-tester",
    }
    base.update(overrides)
    return Settings(**base)


# 1. Orchestrator._is_done() logic


def test_is_done_all_completed_and_tests_passed():
    orch = Orchestrator(goal="g", client=MockLLMClient())
    orch._current_tasks = [TaskModel(id="1", description="d", status="completed")]
    orch._tests_passed = True
    assert orch._is_done() is True


def test_is_done_some_pending_and_tests_passed():
    orch = Orchestrator(goal="g", client=MockLLMClient())
    orch._current_tasks = [TaskModel(id="1", description="d", status="pending")]
    orch._tests_passed = True
    assert orch._is_done() is False


def test_is_done_all_completed_tests_not_passed():
    orch = Orchestrator(goal="g", client=MockLLMClient())
    orch._current_tasks = [
        TaskModel(id="1", description="d", status="completed"),
        TaskModel(id="2", description="d", status="completed"),
    ]
    orch._tests_passed = False
    assert orch._is_done() is False


def test_is_done_some_failed():
    orch = Orchestrator(goal="g", client=MockLLMClient())
    orch._current_tasks = [
        TaskModel(id="1", description="d", status="completed"),
        TaskModel(id="2", description="d", status="failed"),
    ]
    orch._tests_passed = True
    assert orch._is_done() is False


def test_is_done_empty_tasks_and_tests_passed():
    orch = Orchestrator(goal="g", client=MockLLMClient())
    orch._current_tasks = []
    orch._tests_passed = True
    assert orch._is_done() is True


def test_is_done_empty_tasks_and_tests_not_passed():
    orch = Orchestrator(goal="g", client=MockLLMClient())
    orch._current_tasks = []
    orch._tests_passed = False
    assert orch._is_done() is False


def test_is_done_tasks_skipped():
    orch = Orchestrator(goal="g", client=MockLLMClient())
    orch._current_tasks = [
        TaskModel(id="1", description="d", status="completed"),
        TaskModel(id="2", description="d", status="skipped"),
    ]
    orch._tests_passed = True
    assert orch._is_done() is False


# 2. Orchestrator._find_cyclic_task_ids()


def test_find_cycles_no_cycles_linear_chain():
    orch = Orchestrator(goal="g", client=MockLLMClient())
    tasks = [
        TaskModel(id="a", description="a", dependencies=[]),
        TaskModel(id="b", description="b", dependencies=["a"]),
        TaskModel(id="c", description="c", dependencies=["b"]),
    ]
    assert orch._find_cyclic_task_ids(tasks) == set()


def test_find_cycles_simple_two_node_cycle():
    orch = Orchestrator(goal="g", client=MockLLMClient())
    tasks = [
        TaskModel(id="a", description="a", dependencies=["b"]),
        TaskModel(id="b", description="b", dependencies=["a"]),
    ]
    assert orch._find_cyclic_task_ids(tasks) == {"a", "b"}


def test_find_cycles_three_node_cycle():
    orch = Orchestrator(goal="g", client=MockLLMClient())
    tasks = [
        TaskModel(id="a", description="a", dependencies=["c"]),
        TaskModel(id="b", description="b", dependencies=["a"]),
        TaskModel(id="c", description="c", dependencies=["b"]),
    ]
    assert orch._find_cyclic_task_ids(tasks) == {"a", "b", "c"}


def test_find_cycles_mixed_cyclic_and_acyclic():
    orch = Orchestrator(goal="g", client=MockLLMClient())
    tasks = [
        TaskModel(id="a", description="a", dependencies=["c"]),
        TaskModel(id="b", description="b", dependencies=["a"]),
        TaskModel(id="c", description="c", dependencies=["b"]),
        TaskModel(id="d", description="d", dependencies=[]),
        TaskModel(id="e", description="e", dependencies=["d"]),
    ]
    assert orch._find_cyclic_task_ids(tasks) == {"a", "b", "c"}


# 3. Orchestrator max_cycles


async def test_orchestrator_stops_at_max_cycles():
    settings = _test_settings(max_cycles=2)
    client = MockLLMClient(settings=settings)
    orch = Orchestrator(goal="do the thing", client=client, settings=settings)

    plan = Plan(
        tasks=[TaskModel(id="1", description="task one", status="pending")],
        rationale="a plan",
    )
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    with patch("furrow.core.orchestrator.console"), patch(
        "furrow.core.orchestrator.Status"
    ), patch("furrow.core.orchestrator.WorkerAgent") as mock_worker, patch(
        "furrow.core.orchestrator.TesterAgent"
    ) as mock_tester:
        mock_worker.return_value.run = AsyncMock(return_value="completed result")
        mock_tester.return_value.run = AsyncMock(
            return_value=TestResult(
                passed=False, summary="failed", failures=["boom"]
            )
        )
        await orch.run()

    assert orch.cycles == 2
    assert orch.planner.plan.call_count == 2


# 4. Tester._run_all_checks()


async def test_tester_run_all_checks_all_sections_enabled():
    settings = _test_settings(lint_enabled=True, type_check_enabled=True)
    client = MockLLMClient(settings=settings)
    tester = TesterAgent(client=client)
    tester._run_command = AsyncMock(return_value="fake output")
    output = await tester._run_all_checks()
    assert "=== TESTS ===" in output
    assert "=== LINT ===" in output
    assert "=== TYPE CHECK ===" in output


async def test_tester_run_all_checks_lint_disabled():
    settings = _test_settings(lint_enabled=False, type_check_enabled=True)
    client = MockLLMClient(settings=settings)
    tester = TesterAgent(client=client)
    tester._run_command = AsyncMock(return_value="fake output")
    output = await tester._run_all_checks()
    assert "=== TESTS ===" in output
    assert "=== LINT ===" not in output
    assert "=== TYPE CHECK ===" in output


async def test_tester_run_all_checks_type_check_disabled():
    settings = _test_settings(lint_enabled=True, type_check_enabled=False)
    client = MockLLMClient(settings=settings)
    tester = TesterAgent(client=client)
    tester._run_command = AsyncMock(return_value="fake output")
    output = await tester._run_all_checks()
    assert "=== TESTS ===" in output
    assert "=== LINT ===" in output
    assert "=== TYPE CHECK ===" not in output


# 5. LLMClient Ollama URL normalization


def test_ollama_url_normalization_adds_v1():
    settings = Settings(
        provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434"
    )
    client = LLMClient(settings=settings)
    assert str(client.ollama.base_url).rstrip("/").endswith("/v1")


def test_ollama_url_normalization_keeps_existing_v1():
    settings = Settings(
        provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434/v1"
    )
    client = LLMClient(settings=settings)
    assert str(client.ollama.base_url).rstrip("/").endswith("/v1")


# 6. LLMClient._to_anthropic_messages


def test_to_anthropic_messages_system_extracted():
    messages = [{"role": "system", "content": "You are helpful"}]
    system, result = _to_anthropic_messages(messages)
    assert system == "You are helpful"
    assert result == []


def test_to_anthropic_messages_user_text():
    messages = [{"role": "user", "content": "Hello there"}]
    system, result = _to_anthropic_messages(messages)
    assert system == ""
    assert result == [{"role": "user", "content": "Hello there"}]


def test_to_anthropic_messages_assistant_with_tool_calls():
    messages = [
        {
            "role": "assistant",
            "content": "Let me check the file",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/tmp/x"}',
                    },
                }
            ],
        }
    ]
    system, result = _to_anthropic_messages(messages)
    assert system == ""
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    blocks = result[0]["content"]
    assert {"type": "text", "text": "Let me check the file"} in blocks
    assert {
        "type": "tool_use",
        "id": "call_1",
        "name": "read_file",
        "input": {"path": "/tmp/x"},
    } in blocks


def test_to_anthropic_messages_tool_result():
    messages = [
        {
            "role": "tool",
            "content": "file contents here",
            "tool_call_id": "call_1",
        }
    ]
    system, result = _to_anthropic_messages(messages)
    assert system == ""
    assert result == [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "output": "file contents here",
                }
            ],
        }
    ]


# 7. LLMClient._to_anthropic_tools


def test_to_anthropic_tools_conversion():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        },
    ]
    result = _to_anthropic_tools(tools)
    assert result == [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    ]


# 8. LLMClient.execute_tool dispatch


async def test_execute_tool_dispatch_read_file():
    client = LLMClient(settings=_test_settings())
    client.read_file = AsyncMock(return_value="file contents")
    result = await client.execute_tool("read_file", {"path": "/tmp/x"})
    assert result == "file contents"
    client.read_file.assert_awaited_once_with("/tmp/x")


async def test_execute_tool_dispatch_write_file():
    client = LLMClient(settings=_test_settings())
    client.write_file = AsyncMock()
    result = await client.execute_tool(
        "write_file", {"path": "/tmp/y", "content": "data"}
    )
    assert result == ""
    client.write_file.assert_awaited_once_with("/tmp/y", "data")


async def test_execute_tool_dispatch_list_files():
    client = LLMClient(settings=_test_settings())
    client.list_files = MagicMock(return_value=["a.py", "b.py"])
    result = await client.execute_tool("list_files", {"directory": "/tmp"})
    assert result == "a.py\nb.py"
    client.list_files.assert_called_once_with("/tmp")


async def test_execute_tool_dispatch_run_command():
    client = LLMClient(settings=_test_settings())
    client.run_command = AsyncMock(return_value="command output")
    result = await client.execute_tool("run_command", {"command": "ls"})
    assert result == "command output"
    client.run_command.assert_awaited_once_with("ls")


async def test_execute_tool_dispatch_unknown_raises():
    client = LLMClient(settings=_test_settings())
    with pytest.raises(ValueError):
        await client.execute_tool("unknown_tool", {})


# 9. WorkerAgent with mock


async def test_worker_returns_text_when_no_tool_calls():
    client = MockLLMClient(settings=_test_settings())
    client.complete_with_tools_return = ToolResponse(text="Done", tool_calls=[])
    task = TaskModel(id="1", description="do something")
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "Done"
    assert len(client.complete_with_tools_calls) == 1
    assert client.execute_tool_calls == []


async def test_worker_loops_on_tool_calls_then_returns_text():
    client = MockLLMClient(settings=_test_settings())
    client.complete_with_tools_responses = [
        ToolResponse(
            text="Let me read it",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="read_file",
                    arguments={"path": "/tmp/x"},
                )
            ],
        ),
        ToolResponse(text="Done", tool_calls=[]),
    ]
    task = TaskModel(id="1", description="do something")
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "Done"
    assert len(client.complete_with_tools_calls) == 2
    assert len(client.execute_tool_calls) == 1
    assert client.execute_tool_calls[0]["name"] == "read_file"
    assert client.execute_tool_calls[0]["arguments"] == {"path": "/tmp/x"}


# 10. Settings defaults


def test_settings_defaults():
    settings = Settings()
    assert settings.lint_command == ["ruff", "check", "."]
    assert settings.type_check_command == ["mypy", "."]
    assert settings.lint_enabled is True
    assert settings.type_check_enabled is True
    assert settings.test_timeout == 120


def test_settings_max_cycles_default():
    settings = Settings()
    assert settings.max_cycles == 0


# 11. ToolCall / ToolResponse models


def test_toolcall_defaults():
    tc = ToolCall(id="1", name="read_file")
    assert tc.id == "1"
    assert tc.name == "read_file"
    assert tc.arguments == {}


def test_toolcall_with_arguments():
    tc = ToolCall(id="1", name="read_file", arguments={"path": "/tmp/x"})
    assert tc.arguments == {"path": "/tmp/x"}


def test_toolresponse_defaults():
    tr = ToolResponse()
    assert tr.text is None
    assert tr.tool_calls == []


def test_toolresponse_with_values():
    tr = ToolResponse(
        text="some text",
        tool_calls=[ToolCall(id="1", name="read_file")],
    )
    assert tr.text == "some text"
    assert len(tr.tool_calls) == 1
    assert tr.tool_calls[0].name == "read_file"


# 12. Existing tests are preserved (regression guard)


def test_existing_test_core_tests_still_present():
    import importlib

    module = importlib.import_module("tests.test_core")
    assert hasattr(module, "test_plan_parse")
    assert hasattr(module, "test_test_result")
