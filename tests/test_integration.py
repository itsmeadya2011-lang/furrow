from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from furrow.agents.tester import TestResult
from furrow.config import Settings, TaskModel
from furrow.core.events import EventType
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import StateStore
from furrow.llm import LLMResponse, ToolCall, Usage


def run(coro):
    """Run a coroutine to completion on a fresh event loop."""
    return asyncio.run(coro)


class FakeLLMClient:
    """Fake LLM client that implements the full LLMClient interface for testing."""

    def __init__(self):
        self.planner_response = ""
        self.planner_responses: list[str] = []
        self.worker_responses: list[LLMResponse] = []
        self.tester_response = ""
        self.call_types: list[str] = []
        self.calls = 0
        self.prompts = []
        self.files: dict[str, str] = {}

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls += 1
        self.prompts.append(prompt)

        if "planner" in system.lower() or "plan" in system.lower():
            self.call_types.append("planner")
            if self.planner_responses:
                return self.planner_responses.pop(0)
            return self.planner_response

        if "tester" in system.lower() or "test" in system.lower():
            self.call_types.append("tester")
            return self.tester_response

        self.call_types.append("complete")
        return ""

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        model: str | None = None,
    ) -> LLMResponse:
        self.calls += 1
        self.prompts.append(str(messages))

        if "worker" in system.lower() or (
            tools and any(t.get("name") == "write_file" for t in tools)
        ):
            self.call_types.append("worker")
            if self.worker_responses:
                return self.worker_responses.pop(0)
            return LLMResponse(text="Task completed.")

        self.call_types.append("chat")
        return LLMResponse(text="")

    async def read_file(self, path: str | Path) -> str:
        path_str = str(path)
        if path_str in self.files:
            return self.files[path_str]
        raise FileNotFoundError(f"File not found: {path_str}")

    async def write_file(self, path: str | Path, content: str) -> None:
        self.files[str(path)] = content

    def list_files(self, directory: str | Path) -> list[str]:
        base = Path(directory)
        if not base.exists():
            return []
        return [str(f.relative_to(base)) for f in base.rglob("*") if f.is_file()]

    @property
    def total_usage(self) -> Usage:
        return Usage(input_tokens=100, output_tokens=50, calls=self.calls, cost_usd=0.001)

    def get_file_content(self, filename: str) -> str | None:
        """Get file content by filename (handles full paths)."""
        for path, content in self.files.items():
            if path.endswith(filename) or path == filename:
                return content
        return None


class MockTester:
    """Mock tester that returns predefined test results."""

    def __init__(self, results: list[TestResult]):
        self.results = results
        self.call_count = 0

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.call_count += 1
        if self.results:
            return self.results.pop(0)
        return TestResult(passed=True, summary="All tests passed", failures=[], skipped=False)


def make_settings(
    tmp_path: Path, max_cycles: int = 5, max_parallel_tasks: int = 2, max_task_retries: int = 1
) -> Settings:
    return Settings(
        _env_file=None,
        workspace=tmp_path,
        max_cycles=max_cycles,
        max_parallel_tasks=max_parallel_tasks,
        max_task_retries=max_task_retries,
        max_worker_iterations=5,
        request_timeout=10.0,
        command_timeout=30.0,
        test_timeout=30.0,
    )


def test_full_pipeline_creates_file(tmp_path: Path):
    """Test the full Orchestrator pipeline: Plan -> Execute -> Test -> Repeat."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    plan_json = json.dumps(
        {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Create hello.txt with content 'hello world'",
                    "files": ["hello.txt"],
                    "dependencies": [],
                }
            ],
            "rationale": "Create a simple text file",
        }
    )

    empty_plan_json = json.dumps({"tasks": [], "rationale": "Goal complete - file created"})

    worker_response_1 = LLMResponse(
        text="I'll create the file hello.txt with the content.",
        tool_calls=[
            ToolCall(
                id="tc1",
                name="write_file",
                arguments={"path": "hello.txt", "content": "hello world"},
            )
        ],
    )

    worker_response_2 = LLMResponse(text="File created successfully.", tool_calls=[])

    tester_result = TestResult(passed=True, summary="All tests passed", failures=[], skipped=False)

    client = FakeLLMClient()
    client.planner_responses = [plan_json, empty_plan_json]
    client.worker_responses = [worker_response_1, worker_response_2]
    client.tester_response = ""

    settings = make_settings(workspace, max_cycles=5)
    store = StateStore(workspace / "state.json")
    orch = Orchestrator(
        goal="create a file hello.txt with content 'hello world'",
        settings=settings,
        state_store=store,
        client=client,
    )
    orch.tester = MockTester([tester_result])

    result = run(orch.run())

    assert result.cycles == 2
    assert result.passed is True
    assert result.stopped_reason == "goal complete"
    assert result.tasks_completed == 1

    # Check the file was created via the fake client
    assert client.get_file_content("hello.txt") == "hello world"

    st = store.load()
    assert st is not None
    assert st.original_goal == "create a file hello.txt with content 'hello world'"
    # State is saved after cycle 1 (cycle 2 has empty plan and breaks before save)
    assert st.cycles == 1
    assert len(st.tasks) == 1
    assert st.tasks[0].id == "t1"
    assert st.tasks[0].status == "completed"

    for entry in st.history:
        assert entry.tests_passed is True


def test_pipeline_with_failing_then_passing_test(tmp_path: Path):
    """Test the full loop with a failing test that gets fixed in the next cycle."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    plan_json_cycle1 = json.dumps(
        {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Create hello.txt with wrong content",
                    "files": ["hello.txt"],
                    "dependencies": [],
                }
            ],
            "rationale": "Create file with wrong content first",
        }
    )

    plan_json_cycle2 = json.dumps(
        {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Fix hello.txt to have correct content",
                    "files": ["hello.txt"],
                    "dependencies": [],
                }
            ],
            "rationale": "Fix the file content",
        }
    )

    empty_plan_json = json.dumps({"tasks": [], "rationale": "Goal complete - file fixed"})

    worker_responses_cycle1 = [
        LLMResponse(
            text="Creating file with wrong content.",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    name="write_file",
                    arguments={"path": "hello.txt", "content": "wrong content"},
                )
            ],
        ),
        LLMResponse(text="Done.", tool_calls=[]),
    ]

    worker_responses_cycle2 = [
        LLMResponse(
            text="Fixing the file content.",
            tool_calls=[
                ToolCall(
                    id="tc2",
                    name="write_file",
                    arguments={"path": "hello.txt", "content": "hello world"},
                )
            ],
        ),
        LLMResponse(text="Fixed.", tool_calls=[]),
    ]

    tester_result_fail = TestResult(
        passed=False,
        summary="Test failed: expected 'hello world' but got 'wrong content'",
        failures=["test_content - AssertionError: expected 'hello world'"],
        skipped=False,
    )

    tester_result_pass = TestResult(
        passed=True, summary="All tests passed", failures=[], skipped=False
    )

    planner_responses = [plan_json_cycle1, plan_json_cycle2, empty_plan_json]
    worker_responses_all = worker_responses_cycle1 + worker_responses_cycle2

    class MultiCycleClient(FakeLLMClient):
        def __init__(self):
            super().__init__()
            self.planner_responses = planner_responses
            self.worker_responses = worker_responses_all

    client = MultiCycleClient()

    settings = make_settings(workspace, max_cycles=5)
    store = StateStore(workspace / "state.json")
    orch = Orchestrator(
        goal="create a file hello.txt with content 'hello world'",
        settings=settings,
        state_store=store,
        client=client,
    )
    orch.tester = MockTester([tester_result_fail, tester_result_pass])

    result = run(orch.run())

    assert result.cycles == 3
    assert result.passed is True
    assert result.stopped_reason == "goal complete"
    assert result.tasks_completed == 1

    # Check the file was fixed
    assert client.get_file_content("hello.txt") == "hello world"

    st = store.load()
    assert st is not None
    # State saved after cycle 1 and cycle 2 (cycle 3 has empty plan)
    assert st.cycles == 2
    assert len(st.history) == 2
    assert st.history[0].tests_passed is False
    assert st.history[1].tests_passed is True


def test_resume_from_persisted_state(tmp_path: Path):
    """Test resuming the Orchestrator from persisted state."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    plan_json = json.dumps(
        {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Create hello.txt with content 'hello world'",
                    "files": ["hello.txt"],
                    "dependencies": [],
                }
            ],
            "rationale": "Create a simple text file",
        }
    )

    empty_plan_json = json.dumps({"tasks": [], "rationale": "Goal complete"})

    worker_response_1 = LLMResponse(
        text="I'll create the file.",
        tool_calls=[
            ToolCall(
                id="tc1",
                name="write_file",
                arguments={"path": "hello.txt", "content": "hello world"},
            )
        ],
    )

    worker_response_2 = LLMResponse(text="File created.", tool_calls=[])

    tester_result = TestResult(passed=True, summary="All tests passed", failures=[], skipped=False)

    client = FakeLLMClient()
    client.planner_responses = [plan_json, empty_plan_json]
    client.worker_responses = [worker_response_1, worker_response_2]
    client.tester_response = ""

    settings = make_settings(workspace, max_cycles=5)
    store = StateStore(workspace / "state.json")
    orch = Orchestrator(
        goal="create a file hello.txt with content 'hello world'",
        settings=settings,
        state_store=store,
        client=client,
    )
    orch.tester = MockTester([tester_result])

    result1 = run(orch.run())

    assert result1.cycles == 2
    assert result1.passed is True
    assert store.exists()

    st = store.load()
    assert st.cycles == 1

    client2 = FakeLLMClient()
    client2.planner_responses = [plan_json, empty_plan_json]
    client2.worker_responses = [worker_response_1, worker_response_2]
    client2.tester_response = ""

    orch2 = Orchestrator(
        goal="different goal", settings=settings, state_store=store, client=client2, resume=True
    )
    orch2.tester = MockTester([tester_result])

    assert orch2.cycles == 1
    assert orch2.original_goal == "create a file hello.txt with content 'hello world'"
    assert orch2.goal == "create a file hello.txt with content 'hello world'"
    assert len(orch2.tasks) == 1
    assert orch2.tasks[0].id == "t1"
    assert orch2.tasks[0].status == "completed"
    assert len(orch2.history) == 1

    # Resume runs: cycle 2 (skips completed task) -> cycle 3 (empty plan)
    result2 = run(orch2.run())

    assert result2.cycles == 3
    assert result2.passed is True


def test_events_emitted_during_pipeline(tmp_path: Path):
    """Test that events are emitted during the full pipeline."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    plan_json = json.dumps(
        {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Create hello.txt",
                    "files": ["hello.txt"],
                    "dependencies": [],
                }
            ],
            "rationale": "Create file",
        }
    )

    empty_plan_json = json.dumps({"tasks": [], "rationale": "Goal complete"})

    worker_response_1 = LLMResponse(
        text="Creating file.",
        tool_calls=[
            ToolCall(
                id="tc1",
                name="write_file",
                arguments={"path": "hello.txt", "content": "hello world"},
            )
        ],
    )
    worker_response_2 = LLMResponse(text="Done.", tool_calls=[])

    tester_result = TestResult(passed=True, summary="All tests passed", failures=[], skipped=False)

    client = FakeLLMClient()
    client.planner_responses = [plan_json, empty_plan_json]
    client.worker_responses = [worker_response_1, worker_response_2]
    client.tester_response = ""

    settings = make_settings(workspace, max_cycles=5)
    store = StateStore(workspace / "state.json")
    orch = Orchestrator(
        goal="create hello.txt", settings=settings, state_store=store, client=client
    )
    orch.tester = MockTester([tester_result])

    q = orch.event_bus.subscribe()
    run(orch.run())

    events = []
    while not q.empty():
        events.append(q.get_nowait())

    event_types = {e.type for e in events}

    assert EventType.CYCLE_STARTED in event_types
    assert EventType.PLAN_CREATED in event_types
    assert EventType.TASK_STARTED in event_types
    assert EventType.TASK_FINISHED in event_types
    assert EventType.TEST_STARTED in event_types
    assert EventType.TEST_FINISHED in event_types
    assert EventType.CYCLE_FINISHED in event_types
    assert EventType.STATE_SAVED in event_types

    plan_created = next(e for e in events if e.type == EventType.PLAN_CREATED)
    assert plan_created.data["tasks"] == 1

    task_started = next(e for e in events if e.type == EventType.TASK_STARTED)
    assert task_started.data["task"] == "t1"

    task_finished = next(e for e in events if e.type == EventType.TASK_FINISHED)
    assert task_finished.data["success"] is True

    test_finished = next(e for e in events if e.type == EventType.TEST_FINISHED)
    assert test_finished.data["passed"] is True


def test_state_persistence_contains_expected_fields(tmp_path: Path):
    """Test that persisted state contains all expected fields."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    plan_json = json.dumps(
        {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Create hello.txt",
                    "files": ["hello.txt"],
                    "dependencies": [],
                }
            ],
            "rationale": "Create file",
        }
    )

    empty_plan_json = json.dumps({"tasks": [], "rationale": "Goal complete"})

    worker_response_1 = LLMResponse(
        text="Creating file.",
        tool_calls=[
            ToolCall(
                id="tc1",
                name="write_file",
                arguments={"path": "hello.txt", "content": "hello world"},
            )
        ],
    )
    worker_response_2 = LLMResponse(text="Done.", tool_calls=[])

    tester_result = TestResult(passed=True, summary="All tests passed", failures=[], skipped=False)

    client = FakeLLMClient()
    client.planner_responses = [plan_json, empty_plan_json]
    client.worker_responses = [worker_response_1, worker_response_2]
    client.tester_response = ""

    settings = make_settings(workspace, max_cycles=5)
    store = StateStore(workspace / "state.json")
    orch = Orchestrator(
        goal="create hello.txt", settings=settings, state_store=store, client=client
    )
    orch.tester = MockTester([tester_result])

    run(orch.run())

    st = store.load()
    assert st is not None

    assert st.goal == "create hello.txt"
    assert st.original_goal == "create hello.txt"
    assert st.cycles == 1
    assert len(st.tasks) == 1
    assert st.tasks[0].id == "t1"
    assert st.tasks[0].description == "Create hello.txt"
    assert st.tasks[0].files == ["hello.txt"]
    assert st.tasks[0].status == "completed"
    assert st.tasks[0].files_changed == ["hello.txt"]
    assert len(st.history) == 1
    assert st.history[0].cycle == 1
    assert st.history[0].tests_passed is True
    assert st.usage.calls > 0
    assert st.no_progress == 0
    assert st.tests_run == 1
    assert st.last_passed is True
    assert st.last_skipped is False


def test_multiple_tasks_dependency_order(tmp_path: Path):
    """Test that multiple tasks with dependencies execute in order."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    plan_json = json.dumps(
        {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Create first.txt",
                    "files": ["first.txt"],
                    "dependencies": [],
                },
                {
                    "id": "t2",
                    "description": "Create second.txt after first",
                    "files": ["second.txt"],
                    "dependencies": ["t1"],
                },
            ],
            "rationale": "Create two files in order",
        }
    )

    empty_plan_json = json.dumps({"tasks": [], "rationale": "Goal complete"})

    worker_responses = [
        LLMResponse(
            text="Creating first.txt",
            tool_calls=[
                ToolCall(
                    id="tc1", name="write_file", arguments={"path": "first.txt", "content": "first"}
                )
            ],
        ),
        LLMResponse(text="Done first.", tool_calls=[]),
        LLMResponse(
            text="Creating second.txt",
            tool_calls=[
                ToolCall(
                    id="tc2",
                    name="write_file",
                    arguments={"path": "second.txt", "content": "second"},
                )
            ],
        ),
        LLMResponse(text="Done second.", tool_calls=[]),
    ]

    tester_result = TestResult(passed=True, summary="All tests passed", failures=[], skipped=False)

    client = FakeLLMClient()
    client.planner_responses = [plan_json, empty_plan_json]
    client.worker_responses = worker_responses
    client.tester_response = ""

    settings = make_settings(workspace, max_cycles=5, max_parallel_tasks=2)
    store = StateStore(workspace / "state.json")
    orch = Orchestrator(
        goal="create two files in order", settings=settings, state_store=store, client=client
    )
    orch.tester = MockTester([tester_result])

    result = run(orch.run())

    assert result.cycles == 2
    assert result.tasks_completed == 2

    # Check files were created
    assert client.get_file_content("first.txt") == "first"
    assert client.get_file_content("second.txt") == "second"

    st = store.load()
    assert st.tasks[0].status == "completed"
    assert st.tasks[1].status == "completed"


def test_tester_failure_triggers_new_plan(tmp_path: Path):
    """Test that a failing test triggers a new plan with fix tasks."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    plan_json_cycle1 = json.dumps(
        {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Create hello.txt with wrong content",
                    "files": ["hello.txt"],
                    "dependencies": [],
                }
            ],
            "rationale": "Create file with wrong content first",
        }
    )

    plan_json_cycle2 = json.dumps(
        {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Fix hello.txt to have correct content",
                    "files": ["hello.txt"],
                    "dependencies": [],
                }
            ],
            "rationale": "Fix the file content based on test failure",
        }
    )

    empty_plan_json = json.dumps({"tasks": [], "rationale": "Goal complete - file fixed"})

    worker_responses_cycle1 = [
        LLMResponse(
            text="Creating file with wrong content.",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    name="write_file",
                    arguments={"path": "hello.txt", "content": "wrong content"},
                )
            ],
        ),
        LLMResponse(text="Done.", tool_calls=[]),
    ]

    worker_responses_cycle2 = [
        LLMResponse(
            text="Fixing the file content.",
            tool_calls=[
                ToolCall(
                    id="tc2",
                    name="write_file",
                    arguments={"path": "hello.txt", "content": "hello world"},
                )
            ],
        ),
        LLMResponse(text="Fixed.", tool_calls=[]),
    ]

    tester_result_fail = TestResult(
        passed=False,
        summary="Test failed: expected 'hello world' but got 'wrong content'",
        failures=["test_content - AssertionError: expected 'hello world'"],
        skipped=False,
    )

    tester_result_pass = TestResult(
        passed=True, summary="All tests passed", failures=[], skipped=False
    )

    planner_responses = [plan_json_cycle1, plan_json_cycle2, empty_plan_json]
    worker_responses_all = worker_responses_cycle1 + worker_responses_cycle2

    class MultiCycleClient(FakeLLMClient):
        def __init__(self):
            super().__init__()
            self.planner_responses = planner_responses
            self.worker_responses = worker_responses_all

    client = MultiCycleClient()

    settings = make_settings(workspace, max_cycles=5)
    store = StateStore(workspace / "state.json")
    orch = Orchestrator(
        goal="create a file hello.txt with content 'hello world'",
        settings=settings,
        state_store=store,
        client=client,
    )
    orch.tester = MockTester([tester_result_fail, tester_result_pass])

    result = run(orch.run())

    assert result.cycles == 3
    assert result.passed is True
    assert result.stopped_reason == "goal complete"
    assert result.tasks_completed == 1

    # Check the file was fixed after test failure
    assert client.get_file_content("hello.txt") == "hello world"

    st = store.load()
    assert st is not None
    assert st.cycles == 2
    assert len(st.history) == 2
    assert st.history[0].tests_passed is False
    assert st.history[1].tests_passed is True


def test_no_test_runner_skips_tests(tmp_path: Path):
    """Test that when no test runner is detected, tests are skipped."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    plan_json = json.dumps(
        {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Create hello.txt",
                    "files": ["hello.txt"],
                    "dependencies": [],
                }
            ],
            "rationale": "Create file",
        }
    )

    empty_plan_json = json.dumps({"tasks": [], "rationale": "Goal complete"})

    worker_response_1 = LLMResponse(
        text="Creating file.",
        tool_calls=[
            ToolCall(
                id="tc1",
                name="write_file",
                arguments={"path": "hello.txt", "content": "hello world"},
            )
        ],
    )
    worker_response_2 = LLMResponse(text="Done.", tool_calls=[])

    class SkipTestClient(FakeLLMClient):
        def __init__(self):
            super().__init__()
            self.worker_responses = [worker_response_1, worker_response_2]
            self.worker_call = 0

        async def chat(
            self,
            messages: list[dict],
            tools: list[dict] | None = None,
            system: str = "",
            model: str | None = None,
        ) -> LLMResponse:
            if "worker" in system.lower() or (
                tools and any(t.get("name") == "write_file" for t in tools)
            ):
                self.call_types.append("worker")
                self.worker_call += 1
                if self.worker_responses:
                    return self.worker_responses.pop(0)
                return LLMResponse(text="Done.", tool_calls=[])
            self.call_types.append("chat")
            return LLMResponse(text="")

    client = SkipTestClient()
    client.planner_responses = [plan_json, empty_plan_json]
    client.tester_response = ""

    settings = make_settings(workspace, max_cycles=5)
    store = StateStore(workspace / "state.json")
    orch = Orchestrator(
        goal="create hello.txt", settings=settings, state_store=store, client=client
    )
    # Don't mock tester - let it run and detect no test runner (skipped)
    # This tests the actual skip behavior

    result = run(orch.run())

    assert result.skipped is True
    assert result.tests_run == 0
    assert result.passed is True
    assert result.stopped_reason == "goal complete"

    st = store.load()
    assert st.last_skipped is True
    assert st.tests_run == 0


def test_max_cycles_limit(tmp_path: Path):
    """Test that the orchestrator stops after max_cycles."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    plan_json = json.dumps(
        {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Create file",
                    "files": ["hello.txt"],
                    "dependencies": [],
                }
            ],
            "rationale": "Create file",
        }
    )

    worker_response_1 = LLMResponse(
        text="Creating.",
        tool_calls=[
            ToolCall(
                id="tc1",
                name="write_file",
                arguments={"path": "hello.txt", "content": "hello world"},
            )
        ],
    )
    worker_response_2 = LLMResponse(text="Done.", tool_calls=[])

    tester_result_fail = TestResult(
        passed=False,
        summary="Tests still failing",
        failures=["test - AssertionError"],
        skipped=False,
    )

    client = FakeLLMClient()
    client.planner_response = plan_json
    client.worker_responses = [worker_response_1, worker_response_2]
    client.tester_response = ""

    settings = make_settings(workspace, max_cycles=3)
    store = StateStore(workspace / "state.json")
    orch = Orchestrator(
        goal="create file but tests fail", settings=settings, state_store=store, client=client
    )
    orch.tester = MockTester([tester_result_fail, tester_result_fail, tester_result_fail])

    result = run(orch.run())

    assert result.cycles == 3
    # The no-progress guard triggers because same plan signature repeats
    assert result.stopped_reason in ("max_cycles reached", "no progress")
    assert result.passed is False


def test_plan_with_no_tasks_completes_goal(tmp_path: Path):
    """Test that an empty plan completes the goal."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    plan_json = json.dumps({"tasks": [], "rationale": "Goal already complete"})

    tester_result = TestResult(passed=True, summary="All tests passed", failures=[], skipped=False)

    class EmptyPlanClient(FakeLLMClient):
        def __init__(self):
            super().__init__()
            self.planner_response = plan_json

    client = EmptyPlanClient()
    client.planner_response = plan_json
    client.tester_response = ""

    settings = make_settings(workspace, max_cycles=5)
    store = StateStore(workspace / "state.json")
    orch = Orchestrator(
        goal="goal already done", settings=settings, state_store=store, client=client
    )
    orch.tester = MockTester([tester_result])

    result = run(orch.run())

    assert result.cycles == 1
    assert result.stopped_reason == "goal complete"
    assert result.tasks_completed == 0
    assert len(orch.tasks) == 0

    # State is not saved when plan is empty (breaks before save)
    # This is expected behavior
    st = store.load()
    # May be None if no state was saved
    if st is not None:
        assert st.cycles == 1


def test_resume_after_max_cycles_reached(tmp_path: Path):
    """Test resuming after max_cycles was reached previously."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    plan_json = json.dumps(
        {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Create file",
                    "files": ["hello.txt"],
                    "dependencies": [],
                }
            ],
            "rationale": "Create file",
        }
    )

    empty_plan_json = json.dumps({"tasks": [], "rationale": "Goal complete"})

    worker_response_1 = LLMResponse(
        text="Creating.",
        tool_calls=[
            ToolCall(
                id="tc1",
                name="write_file",
                arguments={"path": "hello.txt", "content": "hello world"},
            )
        ],
    )
    worker_response_2 = LLMResponse(text="Done.", tool_calls=[])

    tester_result_fail = TestResult(
        passed=False, summary="Tests failing", failures=["test - AssertionError"], skipped=False
    )

    client = FakeLLMClient()
    client.planner_response = plan_json
    client.worker_responses = [worker_response_1, worker_response_2]
    client.tester_response = ""

    settings = make_settings(workspace, max_cycles=2)
    store = StateStore(workspace / "state.json")
    orch = Orchestrator(
        goal="create file but tests fail", settings=settings, state_store=store, client=client
    )
    orch.tester = MockTester([tester_result_fail, tester_result_fail])

    result1 = run(orch.run())
    assert result1.cycles == 2
    assert result1.stopped_reason == "max_cycles reached"

    # Resume with a higher max_cycles to allow continuation
    settings2 = make_settings(workspace, max_cycles=5)
    client2 = FakeLLMClient()
    client2.planner_responses = [plan_json, empty_plan_json]
    client2.worker_responses = [worker_response_1, worker_response_2]
    client2.tester_response = ""

    orch2 = Orchestrator(
        goal="new goal", settings=settings2, state_store=store, client=client2, resume=True
    )
    orch2.tester = MockTester(
        [TestResult(passed=True, summary="Tests passed", failures=[], skipped=False)]
    )

    assert orch2.cycles == 2
    assert orch2.original_goal == "create file but tests fail"

    # Resume runs: cycle 3 (same plan, tests now pass) -> cycle 4 (empty plan)
    result2 = run(orch2.run())
    assert result2.cycles == 4
    assert result2.stopped_reason == "goal complete"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
