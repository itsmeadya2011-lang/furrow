"""Planner, worker, and tester agents -- including the parse, verdict, and timeout fixes."""

from __future__ import annotations

import asyncio
import shlex
import sys
import time
from pathlib import Path

import pytest

# No imported name starting with "Test" may live at module scope: pytest tries to
# collect it (``TesterAgent``, ``TestResult``) and emits a PytestCollectionWarning.
# Both are therefore reached through their module.
from furrow import config
from furrow.agents import tester as tester_module
from furrow.agents.planner import PlannerAgent, extract_json
from furrow.agents.tester import (
    EMPTY_OUTPUT_SUMMARY,
    NO_RUNNER_SUMMARY,
    TIMEOUT_RETURNCODE,
    CommandResult,
    extract_failure_lines,
)
from furrow.agents.worker import MAX_EXCERPT_CHARS, TRUNCATION_MARKER, WorkerAgent
from tests.conftest import PLANNER_MARKER, TESTER_MARKER, WORKER_MARKER, FakeLLM, make_settings

VALID_PLAN_JSON = """{"tasks": [{"id": "1", "description": "build the thing",
    "files": ["src/app.py"], "dependencies": []}],
    "rationale": "small first slice"}"""

# A tester reply that lies: the suite is red, but the model insists it is green.
FALSE_GREEN = (
    '{"passed": true, "summary": "Everything is green", "failures": ["tests/test_x.py::test_y"]}'
)


def py_command(code: str) -> str:
    """A ``test_command`` that runs ``code`` with this interpreter."""
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


# ------------------------------------------------------------------ extract_json


def test_extract_json_parses_raw_json() -> None:
    assert extract_json('{"tasks": [], "rationale": "x"}') == {"tasks": [], "rationale": "x"}


def test_extract_json_parses_a_fenced_json_block() -> None:
    text = '```json\n{"tasks": [], "rationale": "fenced"}\n```'
    assert extract_json(text)["rationale"] == "fenced"


def test_extract_json_parses_a_bare_fence_without_a_language() -> None:
    text = '```\n{"tasks": [], "rationale": "bare"}\n```'
    assert extract_json(text)["rationale"] == "bare"


def test_extract_json_parses_an_unterminated_fence() -> None:
    text = '```json\n{"tasks": [], "rationale": "truncated"}'
    assert extract_json(text)["rationale"] == "truncated"


def test_extract_json_parses_an_object_embedded_in_prose() -> None:
    text = (
        "Sure! Here is the plan you asked for:\n"
        '{"tasks": [{"id": "1", "description": "go"}], "rationale": "embedded"}\n'
        "Let me know if you want changes."
    )
    parsed = extract_json(text)
    assert parsed["rationale"] == "embedded"
    assert parsed["tasks"][0]["id"] == "1"


def test_extract_json_ignores_braces_inside_string_literals() -> None:
    text = 'noise {"rationale": "use { and } freely", "tasks": []} trailing'
    assert extract_json(text)["rationale"] == "use { and } freely"


def test_extract_json_handles_escaped_quotes() -> None:
    text = r'{"rationale": "say \"hi\" then stop", "tasks": []}'
    assert extract_json(text)["rationale"] == 'say "hi" then stop'


def test_extract_json_handles_nested_objects() -> None:
    text = '{"tasks": [{"id": "1", "dependencies": ["0"]}], "rationale": "nested"}'
    assert extract_json(text)["tasks"][0]["dependencies"] == ["0"]


@pytest.mark.parametrize("text", ["", "   \n  "])
def test_extract_json_rejects_empty_input(text: str) -> None:
    with pytest.raises(ValueError, match="Empty response"):
        extract_json(text)


@pytest.mark.parametrize("text", ["I would rather not.", "no json here at all", "[1, 2, 3]"])
def test_extract_json_rejects_garbage(text: str) -> None:
    with pytest.raises(ValueError):
        extract_json(text)


# --------------------------------------------------------------------- planner


def test_plan_returns_a_valid_plan() -> None:
    fake = FakeLLM([VALID_PLAN_JSON], settings=make_settings())
    plan = asyncio.run(PlannerAgent(client=fake).plan("ship the thing"))
    assert isinstance(plan, config.Plan)
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].files == ["src/app.py"]
    assert plan.rationale == "small first slice"


def test_plan_uses_the_planner_model() -> None:
    settings = make_settings(planner_model="planner-model")
    fake = FakeLLM([VALID_PLAN_JSON], settings=settings)
    asyncio.run(PlannerAgent(client=fake).plan("goal"))
    assert fake.models == ["planner-model"]


def test_plan_prompt_carries_the_goal() -> None:
    fake = FakeLLM([VALID_PLAN_JSON], settings=make_settings())
    asyncio.run(PlannerAgent(client=fake).plan("make it fast"))
    assert "Goal: make it fast" in fake.prompts[0]


def test_plan_retries_with_a_corrective_prompt() -> None:
    fake = FakeLLM(["Sure, I could describe the plan in prose.", VALID_PLAN_JSON])
    plan = asyncio.run(PlannerAgent(client=fake).plan("goal"))
    assert plan.tasks[0].id == "1"
    assert len(fake.calls) == 2
    assert "not valid JSON" in fake.prompts[1]
    assert "Goal: goal" in fake.prompts[1]  # the retry keeps the original objective


def test_plan_retries_an_empty_response() -> None:
    fake = FakeLLM(["", "   ", VALID_PLAN_JSON], settings=make_settings(max_retries=2))
    plan = asyncio.run(PlannerAgent(client=fake).plan("goal"))
    assert plan.tasks
    assert len(fake.calls) == 3


def test_plan_raises_with_a_cause_after_exhausting_retries() -> None:
    fake = FakeLLM(["nonsense"] * 5, settings=make_settings(max_retries=2))
    with pytest.raises(ValueError) as excinfo:
        asyncio.run(PlannerAgent(client=fake).plan("goal"))
    # Regression: the failure is re-raised ``from`` the parse error.
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "Failed to parse plan" in str(excinfo.value)
    assert len(fake.calls) == 3  # 1 attempt + 2 retries


def test_plan_fails_without_any_api_key_present() -> None:
    settings = make_settings(max_retries=1)
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None
    fake = FakeLLM(["garbage", "still garbage"], settings=settings)
    with pytest.raises(ValueError) as excinfo:
        asyncio.run(PlannerAgent(client=fake).plan("goal"))
    assert excinfo.value.__cause__ is not None


def test_plan_retry_count_is_capped() -> None:
    fake = FakeLLM(["garbage"] * 10, settings=make_settings(max_retries=99))
    with pytest.raises(ValueError):
        asyncio.run(PlannerAgent(client=fake).plan("goal"))
    assert len(fake.calls) == 4  # hard cap of 3 extra attempts


def test_plan_propagates_the_complete_flag() -> None:
    fake = FakeLLM(['{"tasks": [], "rationale": "done", "complete": true}'])
    assert asyncio.run(PlannerAgent(client=fake).plan("goal")).complete is True


def test_plan_defaults_complete_to_false_when_absent() -> None:
    fake = FakeLLM([VALID_PLAN_JSON])
    assert asyncio.run(PlannerAgent(client=fake).plan("goal")).complete is False


# ---------------------------------------------------------------------- worker


def test_worker_returns_the_canned_response(tmp_workspace: Path) -> None:
    task = config.TaskModel(id="1", description="build the thing")
    fake = FakeLLM(["changed 3 files"], settings=make_settings(workspace=tmp_workspace))
    assert asyncio.run(WorkerAgent(task=task, client=fake).run()) == "changed 3 files"


def test_worker_uses_the_worker_model(tmp_workspace: Path) -> None:
    settings = make_settings(workspace=tmp_workspace, worker_model="worker-model")
    fake = FakeLLM(["ok"], settings=settings)
    task = config.TaskModel(id="1", description="thing")
    asyncio.run(WorkerAgent(task=task, client=fake).run())
    assert fake.models == ["worker-model"]


def test_worker_prompt_includes_file_excerpts(tmp_workspace: Path) -> None:
    task = config.TaskModel(id="1", description="build", files=["src/app.py"])
    fake = FakeLLM(["ok"], settings=make_settings(workspace=tmp_workspace))
    asyncio.run(WorkerAgent(task=task, client=fake).run())
    prompt = fake.prompts[0]
    assert "Task: build" in prompt
    assert "Files to touch: src/app.py" in prompt
    assert "--- src/app.py ---" in prompt
    assert "return a + b" in prompt  # the actual file contents


def test_worker_prompt_truncates_a_large_file(tmp_workspace: Path) -> None:
    big = tmp_workspace / "big.py"
    big.write_text("x" * (MAX_EXCERPT_CHARS * 2), encoding="utf-8")
    task = config.TaskModel(id="1", description="build", files=["big.py"])
    fake = FakeLLM(["ok"], settings=make_settings(workspace=tmp_workspace))
    asyncio.run(WorkerAgent(task=task, client=fake).run())
    prompt = fake.prompts[0]
    assert TRUNCATION_MARKER.strip() in prompt
    assert "x" * (MAX_EXCERPT_CHARS + 1) not in prompt


def test_worker_skips_an_oversized_file(tmp_workspace: Path) -> None:
    huge = tmp_workspace / "huge.bin"
    huge.write_bytes(b"\0" * (256 * 1024 + 1))
    task = config.TaskModel(id="1", description="build", files=["huge.bin"])
    fake = FakeLLM(["ok"], settings=make_settings(workspace=tmp_workspace))
    asyncio.run(WorkerAgent(task=task, client=fake).run())
    assert "(skipped: larger than" in fake.prompts[0]


def test_worker_skips_a_missing_file_without_raising(tmp_workspace: Path) -> None:
    task = config.TaskModel(id="1", description="build", files=["nope/missing.py", "src/app.py"])
    fake = FakeLLM(["survived"], settings=make_settings(workspace=tmp_workspace))
    result = asyncio.run(WorkerAgent(task=task, client=fake).run())
    assert result == "survived"
    prompt = fake.prompts[0]
    assert "(not found: nope/missing.py)" in prompt
    assert "return a + b" in prompt  # the readable file still made it in


def test_worker_without_files(tmp_workspace: Path) -> None:
    task = config.TaskModel(id="1", description="build")
    fake = FakeLLM(["ok"], settings=make_settings(workspace=tmp_workspace))
    asyncio.run(WorkerAgent(task=task, client=fake).run())
    prompt = fake.prompts[0]
    assert "Files to touch: any" in prompt
    assert "--- " not in prompt


def test_worker_prompt_lists_dependencies(tmp_workspace: Path) -> None:
    task = config.TaskModel(id="2", description="build", dependencies=["1", "0"])
    fake = FakeLLM(["ok"], settings=make_settings(workspace=tmp_workspace))
    asyncio.run(WorkerAgent(task=task, client=fake).run())
    assert "Depends on these completed tasks: 1, 0" in fake.prompts[0]


def test_worker_reads_absolute_paths(tmp_workspace: Path) -> None:
    target = tmp_workspace / "elsewhere.py"
    target.write_text("ABSOLUTE_MARKER = 1\n", encoding="utf-8")
    task = config.TaskModel(id="1", description="build", files=[str(target)])
    fake = FakeLLM(["ok"], settings=make_settings(workspace=Path("/definitely/not/here")))
    asyncio.run(WorkerAgent(task=task, client=fake).run())
    assert "ABSOLUTE_MARKER" in fake.prompts[0]


# ---------------------------------------------------------------------- tester


def make_tester(fake: FakeLLM) -> tester_module.TesterAgent:
    return tester_module.TesterAgent(client=fake)


def test_tester_llm_cannot_report_green_on_a_red_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    # The regression that mattered: ``passed = "passed" in output.lower()`` turned
    # "5 passed, 2 failed" into a green build.
    async def fake_run_tests(self):
        return CommandResult(
            output="5 passed, 2 failed\nFAILED tests/test_x.py::test_y",
            returncode=1,
            command=["pytest", "-q"],
        )

    monkeypatch.setattr(tester_module.TesterAgent, "_run_tests", fake_run_tests)
    fake = FakeLLM([FALSE_GREEN])
    result = asyncio.run(make_tester(fake).run("goal", []))

    assert result.passed is False
    assert result.summary == "Everything is green"
    assert result.failures == ["tests/test_x.py::test_y"]


def test_tester_llm_cannot_report_red_on_a_green_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_tests(self):
        return CommandResult(output="ok", returncode=0, command=["pytest", "-q"])

    monkeypatch.setattr(tester_module.TesterAgent, "_run_tests", fake_run_tests)
    fake = FakeLLM(['{"passed": false, "summary": "something looked wrong", "failures": ["x"]}'])
    result = asyncio.run(make_tester(fake).run("goal", []))

    assert result.passed is True
    assert result.summary == "something looked wrong"


def test_tester_reports_a_nonzero_exit_code(tmp_workspace: Path) -> None:
    settings = make_settings(
        workspace=tmp_workspace,
        test_command=py_command("print('FAILED tests/test_x.py::test_y'); raise SystemExit(3)"),
        test_timeout=30,
    )
    fake = FakeLLM(['{"passed": false, "summary": "three", "failures": []}'], settings=settings)
    result = asyncio.run(make_tester(fake).run("goal", []))

    assert result.passed is False
    assert "Exit code: 3" in fake.prompts[0]  # the code is observable, not swallowed
    assert "NON-ZERO, so the suite FAILED" in fake.prompts[0]
    assert result.failures == ["FAILED tests/test_x.py::test_y"]


def test_tester_populates_failures_from_the_output(tmp_workspace: Path) -> None:
    settings = make_settings(
        workspace=tmp_workspace,
        test_command=py_command("print('FAILED tests/test_x.py::test_y'); raise SystemExit(1)"),
    )
    fake = FakeLLM([], default="", settings=settings)
    result = asyncio.run(make_tester(fake).run("goal", []))

    assert result.passed is False
    assert result.failures == ["FAILED tests/test_x.py::test_y"]


def test_tester_passes_when_the_command_exits_zero(tmp_workspace: Path) -> None:
    settings = make_settings(
        workspace=tmp_workspace, test_command=py_command("print('2 passed')"), test_timeout=30
    )
    fake = FakeLLM(['{"passed": true, "summary": "all green", "failures": []}'], settings=settings)
    result = asyncio.run(make_tester(fake).run("goal", []))

    assert result.passed is True
    assert result.summary == "all green"
    assert result.failures == []


def test_tester_without_a_runner_never_calls_the_llm(
    monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path
) -> None:
    monkeypatch.setenv("PATH", "")  # no runner can be found on an empty PATH
    fake = FakeLLM(
        ['{"passed": false, "summary": "boom", "failures": []}'],
        settings=make_settings(workspace=tmp_workspace, test_command=None),
    )
    result = asyncio.run(make_tester(fake).run("goal", []))

    assert result.passed is True
    assert "No test runner found" in result.summary
    assert result.summary == NO_RUNNER_SUMMARY
    assert fake.calls == []


def test_tester_falls_back_to_the_exit_code_when_the_reply_is_unparseable(
    tmp_workspace: Path,
) -> None:
    settings = make_settings(
        workspace=tmp_workspace,
        test_command=py_command("print('Traceback (most recent call last)'); raise SystemExit(1)"),
    )
    fake = FakeLLM([], default="the suite looks fine to me", settings=settings)
    result = asyncio.run(make_tester(fake).run("goal", []))

    assert result.passed is False
    assert result.failures == ["Traceback (most recent call last)"]


def test_tester_survives_an_llm_failure(tmp_workspace: Path) -> None:
    settings = make_settings(
        workspace=tmp_workspace, test_command=py_command("raise SystemExit(1)"), test_timeout=30
    )
    fake = FakeLLM([RuntimeError("the API is down")], settings=settings)
    result = asyncio.run(make_tester(fake).run("goal", []))

    assert result.passed is False
    assert result.summary == EMPTY_OUTPUT_SUMMARY


def test_tester_normalises_a_single_failure_string(tmp_workspace: Path) -> None:
    settings = make_settings(
        workspace=tmp_workspace, test_command=py_command("raise SystemExit(1)"), test_timeout=30
    )
    fake = FakeLLM(
        ['{"passed": true, "summary": "one failure", "failures": "tests/test_x.py::test_y"}'],
        settings=settings,
    )
    result = asyncio.run(make_tester(fake).run("goal", []))

    assert result.failures == ["tests/test_x.py::test_y"]


def test_tester_falls_back_to_raw_output_when_no_summary_is_given(tmp_workspace: Path) -> None:
    settings = make_settings(
        workspace=tmp_workspace, test_command=py_command("print('3 passed')"), test_timeout=30
    )
    fake = FakeLLM(['{"passed": true, "summary": "   ", "failures": []}'], settings=settings)
    result = asyncio.run(make_tester(fake).run("goal", []))

    assert result.summary == "3 passed"


async def test_tester_kills_a_command_that_exceeds_the_timeout(tmp_workspace: Path) -> None:
    # Regression: a killed-but-unwaited child left a zombie behind and blocked the loop.
    settings = make_settings(
        workspace=tmp_workspace,
        test_command=py_command("import time; time.sleep(30)"),
        test_timeout=1,
    )
    fake = FakeLLM(['{"passed": true, "summary": "sure", "failures": []}'], settings=settings)

    started = time.monotonic()
    result = await make_tester(fake).run("goal", [])
    elapsed = time.monotonic() - started

    assert elapsed < 10
    assert result.passed is False
    assert "timed out after 1s and was killed" in result.summary or "timed out" in fake.prompts[0]
    assert f"Exit code: {TIMEOUT_RETURNCODE}" in fake.prompts[0]


def test_tester_prompt_lists_the_tasks_it_just_ran(tmp_workspace: Path) -> None:
    settings = make_settings(
        workspace=tmp_workspace, test_command=py_command("raise SystemExit(0)"), test_timeout=30
    )
    fake = FakeLLM(
        [], default='{"passed": true, "summary": "ok", "failures": []}', settings=settings
    )
    task = config.TaskModel(id="1", description="build", status=config.TaskStatus.COMPLETED)
    asyncio.run(make_tester(fake).run("ship it", [task]))

    prompt = fake.prompts[0]
    assert "Goal: ship it" in prompt
    assert "- [1] build (status: completed)" in prompt
    assert sys.executable in prompt


def test_extract_failure_lines_ignores_unrelated_output() -> None:
    output = "collecting ...\nassert 1 == 2\nFAILED tests/test_x.py::test_y\nsome ok line"
    assert extract_failure_lines(output) == ["assert 1 == 2", "FAILED tests/test_x.py::test_y"]


def test_extract_failure_lines_of_clean_output_is_empty() -> None:
    assert extract_failure_lines("4 passed in 0.12s") == []


def test_extract_failure_lines_is_capped() -> None:
    output = "\n".join(f"FAILED test_{index}" for index in range(50))
    assert len(extract_failure_lines(output)) == 20


def test_agent_prompts_are_distinguishable() -> None:
    # The routing keys used by FakeLLM depend on these markers staying unique.
    from furrow.agents.prompts import PLANNER_PROMPT, TESTER_PROMPT, WORKER_PROMPT

    prompts = {
        PLANNER_MARKER: PLANNER_PROMPT,
        WORKER_MARKER: WORKER_PROMPT,
        TESTER_MARKER: TESTER_PROMPT,
    }
    assert PLANNER_MARKER in PLANNER_PROMPT
    assert WORKER_MARKER in WORKER_PROMPT
    assert TESTER_MARKER in TESTER_PROMPT
    for marker, prompt in prompts.items():
        others = [other for other in prompts if other != marker]
        assert all(other not in prompt for other in others)
