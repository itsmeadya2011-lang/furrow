from __future__ import annotations

import json
from pathlib import Path

import pytest

from furrow.agents import tester as tester_mod
from furrow.agents.planner import PlannerAgent, extract_json
from furrow.agents.tester import PYTEST_CANDIDATES

# ----------------------------------------------------------------------
# extract_json
# ----------------------------------------------------------------------


def test_extract_json_from_json_fence():
    text = '```json\n{"tasks": [], "rationale": "nothing to do"}\n```'
    assert json.loads(extract_json(text)) == {"tasks": [], "rationale": "nothing to do"}


def test_extract_json_from_plain_fence():
    text = '```\n{"passed": true, "summary": "ok"}\n```'
    assert json.loads(extract_json(text)) == {"passed": True, "summary": "ok"}


def test_extract_json_from_bare_json():
    text = '{"plan": {"steps": [1, 2, 3]}}'
    assert json.loads(extract_json(text)) == {"plan": {"steps": [1, 2, 3]}}


def test_extract_json_from_bare_json_with_whitespace():
    text = '\n\n   {"a": 1}   \n\n'
    assert json.loads(extract_json(text)) == {"a": 1}


def test_extract_json_with_surrounding_prose():
    text = (
        "Sure! Here is the plan you asked for:\n"
        '{"tasks": [{"id": "1"}], "rationale": "single slice"}\n'
        "Let me know if you want it split further."
    )
    assert json.loads(extract_json(text)) == {
        "tasks": [{"id": "1"}],
        "rationale": "single slice",
    }


def test_extract_json_prose_before_and_after_fence():
    text = (
        "Thinking about it...\n"
        "```json\n"
        '{"nested": {"deep": {"value": 1}}}\n'
        "```\n"
        "That is the plan."
    )
    assert json.loads(extract_json(text)) == {"nested": {"deep": {"value": 1}}}


def test_extract_json_handles_braces_inside_strings():
    payload = {
        "tasks": [{"id": "1", "description": 'use a dict like {"a": 1} in the code'}],
        "rationale": "brace { inside } a string",
    }
    text = "```json\n" + json.dumps(payload) + "\n```"
    assert json.loads(extract_json(text)) == payload


def test_extract_json_bare_with_braces_inside_strings():
    payload = {"note": "closing } then opening { again", "count": 2}
    text = f"Result: {json.dumps(payload)} -- done"
    assert json.loads(extract_json(text)) == payload


def test_extract_json_prefers_fenced_block_over_braces_in_prose():
    text = 'Draft so far: {"placeholder": true}\n```json\n{"real": 1}\n```'
    assert json.loads(extract_json(text)) == {"real": 1}


def test_extract_json_raises_without_braces():
    with pytest.raises(ValueError):
        extract_json("I am sorry, I cannot produce a plan.")


def test_extract_json_raises_on_prose_only_with_empty_fence():
    with pytest.raises(ValueError):
        extract_json("```json\nno object here\n```")


# ----------------------------------------------------------------------
# PlannerAgent.plan
# ----------------------------------------------------------------------

PLAN_PAYLOAD = {
    "tasks": [
        {
            "id": "1",
            "description": "Implement user authentication with JWT",
            "files": ["src/auth.py", "tests/test_auth.py"],
            "dependencies": [],
        },
        {
            "id": "2",
            "description": "Wire up the login route",
            "files": ["src/routes.py"],
            "dependencies": ["1"],
        },
    ],
    "rationale": "Two dependent slices",
}


def make_planner(response, settings, fake_llm_cls):
    client = fake_llm_cls(responses=[response], settings=settings)
    return PlannerAgent(client=client), client


async def test_planner_plan_parses_fenced_json(settings_factory, fake_llm_cls):
    settings = settings_factory()
    response = "Here you go:\n```json\n" + json.dumps(PLAN_PAYLOAD) + "\n```"
    planner, _client = make_planner(response, settings, fake_llm_cls)

    plan = await planner.plan("add authentication")

    assert plan.rationale == "Two dependent slices"
    assert [task.id for task in plan.tasks] == ["1", "2"]
    assert plan.tasks[0].description == "Implement user authentication with JWT"
    assert plan.tasks[0].files == ["src/auth.py", "tests/test_auth.py"]
    assert plan.tasks[0].dependencies == []
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.tasks[0].status == "pending"
    assert plan.tasks[0].result is None


async def test_planner_plan_uses_planner_model_and_goal(settings_factory, fake_llm_cls):
    settings = settings_factory()
    planner, client = make_planner(json.dumps(PLAN_PAYLOAD), settings, fake_llm_cls)

    await planner.plan("add authentication")

    assert len(client.calls) == 1
    assert client.calls[0]["model"] == settings.planner_model
    assert "add authentication" in client.calls[0]["prompt"]


async def test_planner_plan_raises_on_unparseable_response(settings_factory, fake_llm_cls):
    planner, _ = make_planner("I refuse to answer.", settings_factory(), fake_llm_cls)

    with pytest.raises(ValueError) as excinfo:
        await planner.plan("do something impossible")

    assert "Failed to parse plan from LLM" in str(excinfo.value)
    assert "I refuse to answer." in str(excinfo.value)


async def test_planner_plan_raises_on_invalid_json_body(settings_factory, fake_llm_cls):
    planner, _ = make_planner('```json\n{"tasks": [oops], "rationale": "x"}\n```', settings_factory(), fake_llm_cls)

    with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
        await planner.plan("bad json")


async def test_planner_plan_raises_when_shape_is_invalid(settings_factory, fake_llm_cls):
    planner, _ = make_planner('```json\n{"tasks": []}\n```', settings_factory(), fake_llm_cls)

    with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
        await planner.plan("missing rationale")


# ----------------------------------------------------------------------
# TesterAgent._commands
# ----------------------------------------------------------------------


def make_tester(settings, fake_llm_cls):
    return tester_mod.TesterAgent(client=fake_llm_cls(settings=settings))


@pytest.mark.parametrize(
    ("markers", "expected"),
    [
        (["pyproject.toml"], [["python", "-m", "pytest", "-q"]]),
        (["setup.py"], [["python", "-m", "pytest", "-q"]]),
        (["pytest.ini"], [["python", "-m", "pytest", "-q"]]),
        (["Cargo.toml"], [["cargo", "test", "-q"]]),
        (["go.mod"], [["go", "test", "./..."]]),
        (["package.json"], [["npm", "test", "--", "--silent"]]),
        (["package.json", "pnpm-lock.yaml"], [["pnpm", "test", "--", "--silent"]]),
        (["package.json", "yarn.lock"], [["yarn", "test", "--", "--silent"]]),
        (["package.json", "bun.lock"], [["bun", "test", "--", "--silent"]]),
    ],
)
def test_tester_commands_for_project_markers(
    tmp_path: Path, settings_factory, fake_llm_cls, markers, expected
):
    workspace = tmp_path / f"ws_{'_'.join(markers)}"
    workspace.mkdir()
    for marker in markers:
        (workspace / marker).write_text("x", encoding="utf-8")

    tester = make_tester(settings_factory(workspace=workspace), fake_llm_cls)

    assert tester._commands() == expected


def test_tester_commands_falls_back_to_pytest_candidates(tmp_path: Path, settings_factory, fake_llm_cls):
    workspace = tmp_path / "empty_ws"
    workspace.mkdir()
    (workspace / "README.md").write_text("x", encoding="utf-8")

    tester = make_tester(settings_factory(workspace=workspace), fake_llm_cls)

    commands = tester._commands()
    assert commands == [list(cmd) for cmd in PYTEST_CANDIDATES]
    assert commands == [["python", "-m", "pytest", "-q"], ["pytest", "-q"]]


def test_tester_commands_fallback_is_not_shared_module_state(tmp_path: Path, settings_factory, fake_llm_cls):
    workspace = tmp_path / "empty_ws2"
    workspace.mkdir()
    tester = make_tester(settings_factory(workspace=workspace), fake_llm_cls)

    first = tester._commands()
    first[0].append("--mutated")

    assert tester._commands() == [list(cmd) for cmd in PYTEST_CANDIDATES]


def test_tester_commands_prefers_rust_over_python(tmp_path: Path, settings_factory, fake_llm_cls):
    workspace = tmp_path / "polyglot"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text("x", encoding="utf-8")
    (workspace / "Cargo.toml").write_text("x", encoding="utf-8")

    tester = make_tester(settings_factory(workspace=workspace), fake_llm_cls)

    assert tester._commands() == [["cargo", "test", "-q"]]


# ----------------------------------------------------------------------
# TesterAgent.run
# ----------------------------------------------------------------------


def patch_test_output(monkeypatch, output: str) -> None:
    async def fake_run_tests(self) -> str:
        return output

    monkeypatch.setattr(tester_mod.TesterAgent, "_run_tests", fake_run_tests)


async def test_tester_run_parses_json_response(settings_factory, fake_llm_cls, monkeypatch):
    patch_test_output(monkeypatch, "5 passed")
    response = '```json\n{"passed": false, "summary": "2 failed", "failures": ["test_a", "test_b"]}\n```'
    client = fake_llm_cls(responses=[response], settings=settings_factory())
    tester = tester_mod.TesterAgent(client=client)

    result = await tester.run("goal", [])

    assert result.passed is False
    assert result.summary == "2 failed"
    assert result.failures == ["test_a", "test_b"]
    assert client.calls[0]["model"] == client.settings.tester_model
    assert "5 passed" in client.calls[0]["prompt"]


async def test_tester_run_fallback_non_json_passing_output(settings_factory, fake_llm_cls, monkeypatch):
    patch_test_output(monkeypatch, "12 passed in 0.42s")
    client = fake_llm_cls(
        responses=["All tests passed, nothing to report."], settings=settings_factory()
    )
    tester = tester_mod.TesterAgent(client=client)

    result = await tester.run("goal", [])

    assert result.passed is True
    assert result.summary == "All tests passed, nothing to report."
    assert result.failures == []


async def test_tester_run_fallback_non_json_failing_output(settings_factory, fake_llm_cls, monkeypatch):
    patch_test_output(monkeypatch, "3 failed, 1 error")
    client = fake_llm_cls(
        responses=["FAILED tests/test_a.py::test_b - AssertionError: boom"],
        settings=settings_factory(),
    )
    tester = tester_mod.TesterAgent(client=client)

    result = await tester.run("goal", [])

    assert result.passed is False
    assert "AssertionError" in result.summary
    assert result.failures == []


async def test_tester_run_failed_when_test_runner_raises(settings_factory, fake_llm_cls, monkeypatch):
    async def boom(self) -> str:
        raise RuntimeError("no runner available")

    monkeypatch.setattr(tester_mod.TesterAgent, "_run_tests", boom)
    client = fake_llm_cls(responses=[], settings=settings_factory())
    tester = tester_mod.TesterAgent(client=client)

    result = await tester.run("goal", [])

    assert result.passed is False
    assert result.summary == "no runner available"
    assert result.failures == ["no runner available"]
    assert client.calls == []
