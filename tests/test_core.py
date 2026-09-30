import json
from unittest.mock import Mock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent, _has_py_tests
from furrow.config import (
    AgentState,
    CycleState,
    Plan,
    Settings,
    TaskModel,
    TestResult,
)
from furrow.core.orchestrator import Orchestrator


def _make_client(workspace, max_cycles=0):
    client = Mock()
    client.settings.workspace = str(workspace)
    client.settings.max_cycles = max_cycles
    return client


# ---- existing tests ----


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ---- config model tests ----


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_cycle_state():
    cycle = CycleState(cycle_number=1, tasks=[TaskModel(id="1", description="x")])
    assert cycle.test_result is None


def test_agent_state_defaults():
    agent_state = AgentState(goal="build a house")
    assert agent_state.cycles == 0
    assert agent_state.current_cycle is None
    assert agent_state.history == []


def test_settings_test_timeout():
    s = Settings()
    assert s.test_timeout == 120


# ---- planner JSON parsing tests ----


def test_extract_json_bare():
    response = '{"tasks": [{"id": "1", "description": "do thing"}], "rationale": "ok"}'
    data = PlannerAgent._extract_json(response)
    assert data["rationale"] == "ok"
    assert isinstance(data["tasks"], list)


def test_extract_json_markdown():
    response = '```json\n{\n  "tasks": [],\n  "rationale": "ok"\n}\n```'
    data = PlannerAgent._extract_json(response)
    assert data["rationale"] == "ok"


def test_extract_json_in_text():
    response = 'Here is my plan for you:\n{\n  "tasks": [],\n  "rationale": "ok"\n}\nLet me know what you think.'
    data = PlannerAgent._extract_json(response)
    assert data["rationale"] == "ok"


def test_extract_json_invalid():
    with pytest.raises(ValueError):
        PlannerAgent._extract_json("not json at all, just words")


def test_validate_plan_data_valid():
    PlannerAgent._validate_plan_data(
        {"tasks": [TaskModel(id="1", description="x").model_dump()], "rationale": "ok"},
        "response",
    )


def test_validate_plan_data_missing_tasks():
    with pytest.raises(ValueError):
        PlannerAgent._validate_plan_data({"rationale": "ok"}, "response")


def test_validate_plan_data_missing_rationale():
    with pytest.raises(ValueError):
        PlannerAgent._validate_plan_data({"tasks": []}, "response")


def test_validate_plan_data_wrong_types():
    with pytest.raises(TypeError):
        PlannerAgent._validate_plan_data({"tasks": "not a list", "rationale": "ok"}, "response")
    with pytest.raises(TypeError):
        PlannerAgent._validate_plan_data({"tasks": [], "rationale": 123}, "response")


# ---- tester JSON parsing tests ----


def test_tester_extract_json_bare():
    response = '{"passed": true, "summary": "all good", "failures": []}'
    data = TesterAgent._extract_json(response)
    assert data["passed"] is True


def test_tester_extract_json_markdown():
    response = '```json\n{\n  "passed": false,\n  "summary": "broken",\n  "failures": ["x"]\n}\n```'
    data = TesterAgent._extract_json(response)
    assert data["passed"] is False


def test_parse_test_result_valid():
    result = TesterAgent._parse_test_result(
        '{"passed": true, "summary": "all pass", "failures": []}'
    )
    assert result.passed is True
    assert result.summary == "all pass"


def test_parse_test_result_fallback():
    result = TesterAgent._parse_test_result("Tests passed! everything is good")
    assert result.passed is True
    assert result.summary == "Tests passed! everything is good"

    failed = TesterAgent._parse_test_result("all tests failed badly")
    assert failed.passed is False


# ---- tester test runner detection tests ----


def test_detect_from_pyproject_with_pytest(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\naddopts = "-q"\n')
    assert TesterAgent._detect_from_pyproject(tmp_path) == ["pytest", "-q"]


def test_detect_from_pyproject_with_scripts(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project.scripts]\ntest = "python -m pytest"\n')
    assert TesterAgent._detect_from_pyproject(tmp_path) == ["python", "-m", "pytest"]


def test_detect_from_files_python(tmp_path):
    (tmp_path / "test_example.py").write_text("def test_x():\n    assert True\n")
    assert TesterAgent._detect_from_files(tmp_path) == ["pytest", "-q"]


def test_detect_from_files_no_tests(tmp_path):
    (tmp_path / "app.py").write_text("print('hi')\n")
    assert TesterAgent._detect_from_files(tmp_path) is None


def test_detect_no_runner_found(tmp_path):
    agent = TesterAgent(client=_make_client(tmp_path))
    with patch("furrow.agents.tester.shutil.which", return_value=None):
        assert agent._detect_test_command() is None


def test_has_py_tests_finds_tests(tmp_path):
    (tmp_path / "test_foo.py").write_text("def test_foo():\n    assert True\n")
    assert _has_py_tests(tmp_path) is True


# ---- orchestrator tests ----


def test_orchestrator_init(tmp_path):
    client = _make_client(tmp_path)
    orch = Orchestrator(goal="build a house", client=client)
    assert orch.goal == "build a house"
    assert orch.client is client
    assert isinstance(orch.planner, PlannerAgent)
    assert orch.cycles == 0
    assert orch.tasks == []
    assert orch._history == []
    assert orch._last_test_result is None
    assert orch._override_max_cycles is None
    assert orch._status_callback is None


def test_orchestrator_is_done_empty(tmp_path):
    orch = Orchestrator(goal="g", client=_make_client(tmp_path))
    assert orch._is_done() is True


def test_orchestrator_is_done_all_completed(tmp_path):
    orch = Orchestrator(goal="g", client=_make_client(tmp_path))
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_orchestrator_is_done_failed(tmp_path):
    orch = Orchestrator(goal="g", client=_make_client(tmp_path))
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_orchestrator_is_done_partial(tmp_path):
    orch = Orchestrator(goal="g", client=_make_client(tmp_path))
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orch._is_done() is False


def test_orchestrator_get_tasks(tmp_path):
    orch = Orchestrator(goal="g", client=_make_client(tmp_path))
    tasks = [TaskModel(id="1", description="a")]
    orch.tasks = tasks
    assert orch._get_tasks() is orch.tasks


def test_orchestrator_max_cycles_property(tmp_path):
    client = _make_client(tmp_path, max_cycles=5)
    orch = Orchestrator(goal="g", client=client)
    assert orch.max_cycles == 5


def test_orchestrator_state_file(tmp_path):
    client = _make_client(tmp_path)
    orch = Orchestrator(goal="g", client=client)
    assert orch.state_file == tmp_path / ".furrow" / "state.json"


def test_orchestrator_save_state(tmp_path):
    client = _make_client(tmp_path)
    orch = Orchestrator(goal="g", client=client)
    orch.cycles = 1
    cycle = CycleState(
        cycle_number=1,
        tasks=[TaskModel(id="1", description="a", status="completed")],
        test_result=TestResult(passed=True, summary="ok", failures=[]),
    )
    orch._history = [cycle]
    orch.save_state()

    state_path = tmp_path / ".furrow" / "state.json"
    assert state_path.exists()
    data = json.loads(state_path.read_text())
    assert data["goal"] == "g"
    assert data["cycles"] == 1
    assert data["current_cycle"]["cycle_number"] == 1


def test_orchestrator_load_state(tmp_path):
    client = _make_client(tmp_path)
    orch = Orchestrator(goal="g", client=client)

    state = AgentState(
        goal="loaded goal",
        cycles=3,
        current_cycle=CycleState(
            cycle_number=3,
            tasks=[TaskModel(id="1", description="a", status="completed")],
            test_result=TestResult(passed=True, summary="s", failures=[]),
        ),
        history=[],
    )
    state_path = tmp_path / ".furrow" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(state.model_dump_json())

    orch.load_state()
    assert orch.goal == "loaded goal"
    assert orch.cycles == 3
    assert len(orch.tasks) == 1
    assert orch.tasks[0].status == "completed"
    assert orch._last_test_result is not None
    assert orch._last_test_result.passed is True


def test_orchestrator_save_load_roundtrip(tmp_path):
    client = _make_client(tmp_path)
    orch = Orchestrator(goal="orig goal", client=client)
    orch.cycles = 2
    orch._history = [
        CycleState(
            cycle_number=1,
            tasks=[TaskModel(id="1", description="a", status="completed")],
            test_result=TestResult(passed=True, summary="s1", failures=[]),
        ),
        CycleState(cycle_number=2, tasks=[], test_result=None),
    ]
    orch.save_state()

    client2 = _make_client(tmp_path)
    orch2 = Orchestrator(goal="different goal", client=client2)
    orch2.load_state()

    assert orch2.goal == "orig goal"
    assert orch2.cycles == 2
    assert len(orch2._history) == 2
    assert orch2._history[0].cycle_number == 1
    assert orch2._history[1].cycle_number == 2
