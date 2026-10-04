"""Tests for the three agents: planner, worker and tester.

The fake LLM client from ``conftest`` supplies model replies and the real (aiofiles-backed) file
helpers, so the worker writes to a genuine temporary workspace and the tester runs against a
scripted subprocess. The headline regressions covered here are: the planner surviving fenced and
prose-wrapped JSON, the worker actually writing files while refusing every path escape, and the
tester's verdict coming from the exit code instead of from ``"passed" in text``.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from furrow.agents import PLANNER_PROMPT, TESTER_PROMPT, WORKER_PROMPT, extract_json
from furrow.agents.planner import PlannerAgent, normalize_plan
from furrow.agents.tester import (
    MAX_FAILURE_CHARS,
    MAX_FAILURES,
    TesterAgent,
    _collect_failures,
)
from furrow.agents.worker import WorkerAgent, apply_search_replace
from furrow.config import Settings, TaskModel, TestResult
from tests.conftest import FakeLLMClient, ProcessOutcome, SubprocessStub, build_settings

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def make_task(
    description: str = "do the thing",
    files: list[str] | None = None,
    task_id: str = "1",
    dependencies: list[str] | None = None,
) -> TaskModel:
    return TaskModel(
        id=task_id, description=description, files=files or [], dependencies=dependencies or []
    )


def bounded(awaitable: Awaitable[Any], timeout: float = 5.0) -> Awaitable[Any]:
    """Hard deadline so a hang fails this test instead of the whole suite."""
    return asyncio.wait_for(awaitable, timeout)


async def run_worker(client: FakeLLMClient, task: TaskModel) -> str:
    return await WorkerAgent(task=task, client=client).run()


async def run_tester(
    client: FakeLLMClient, tasks: list[TaskModel] | None = None
) -> TestResult:
    return await TesterAgent(client=client).run("ship it", tasks or [])


PLAN_JSON = (
    '{"tasks": [{"id": "1", "description": "write auth", "files": ["src/auth.py"],'
    ' "dependencies": []}], "rationale": "start with auth"}'
)


def fenced(plan: str, language: str = "json") -> str:
    return f"Here is the plan:\n```{language}\n{plan}\n```\nThat is all."


# --------------------------------------------------------------------------------------
# extract_json
# --------------------------------------------------------------------------------------


def test_extract_json_reads_bare_json() -> None:
    assert extract_json('{"tasks": ["a"]}') == {"tasks": ["a"]}


def test_extract_json_reads_a_json_fenced_block() -> None:
    assert extract_json('```json\n{"tasks": [1]}\n```') == {"tasks": [1]}


@pytest.mark.parametrize("language", ["json", "JSON", "javascript", ""])
def test_extract_json_ignores_the_fence_language(language: str) -> None:
    assert extract_json(f"```{language}\n[1, 2]\n```") == [1, 2]


def test_extract_json_prefers_the_fenced_body_over_surrounding_prose() -> None:
    text = 'Prose {"not": "the plan"} more prose.\n```json\n{"the": "plan"}\n```\n'
    assert extract_json(text) == {"the": "plan"}


def test_extract_json_finds_json_embedded_in_prose() -> None:
    text = 'Sure! Here you go:\n{"tasks": [{"id": "1"}]}\nHope that helps.'
    assert extract_json(text) == {"tasks": [{"id": "1"}]}


def test_extract_json_tolerates_an_unterminated_fence() -> None:
    assert extract_json('```json\n{"tasks": []}') == {"tasks": []}


def test_extract_json_handles_a_closing_brace_inside_a_string() -> None:
    text = 'Result: {"rationale": "wrap a } brace", "tasks": []} enjoy'
    assert extract_json(text) == {"rationale": "wrap a } brace", "tasks": []}


def test_extract_json_handles_escaped_quotes_inside_strings() -> None:
    text = '{"tasks": [{"description": "print(\\"}\\")"}]}'
    assert extract_json(text) == {"tasks": [{"description": 'print("}")'}]}


def test_extract_json_handles_nested_objects_in_prose() -> None:
    text = 'Plan: {"tasks": [], "meta": {"depth": {"n": 1}}} done'
    assert extract_json(text) == {"tasks": [], "meta": {"depth": {"n": 1}}}


def test_extract_json_accepts_a_top_level_array() -> None:
    assert extract_json('["step one", "step two"]') == ["step one", "step two"]


@pytest.mark.parametrize("text", ["", "   \n ", "I am afraid I cannot help with that."])
def test_extract_json_rejects_text_with_no_json(text: str) -> None:
    with pytest.raises(ValueError, match=r"no JSON value|empty LLM response"):
        extract_json(text)


def test_extract_json_error_mentions_the_offending_response() -> None:
    with pytest.raises(ValueError, match="chicken"):
        extract_json("chicken")


# --------------------------------------------------------------------------------------
# normalize_plan
# --------------------------------------------------------------------------------------


def test_normalize_plan_accepts_the_documented_shape() -> None:
    plan = normalize_plan({"tasks": [{"id": "a", "description": "d"}], "rationale": "r"})
    assert plan.rationale == "r"
    assert [task.id for task in plan.tasks] == ["a"]


def test_normalize_plan_accepts_a_bare_list_of_task_objects() -> None:
    plan = normalize_plan(
        [{"id": "a", "description": "first"}, {"id": "b", "description": "second"}]
    )
    assert [task.description for task in plan.tasks] == ["first", "second"]
    assert "a, b" in plan.rationale


def test_normalize_plan_accepts_a_list_of_plain_strings() -> None:
    plan = normalize_plan(["do this", "do that"])
    assert [task.id for task in plan.tasks] == ["1", "2"]
    assert [task.description for task in plan.tasks] == ["do this", "do that"]


def test_normalize_plan_uses_steps_and_reason_as_aliases() -> None:
    plan = normalize_plan({"steps": ["aliased"], "reason": "because"})
    assert plan.tasks[0].description == "aliased"
    assert plan.rationale == "because"


def test_normalize_plan_synthesises_a_rationale_when_missing() -> None:
    plan = normalize_plan({"tasks": [{"id": "x", "description": "d"}]})
    assert plan.rationale == "Plan of x."


def test_normalize_plan_synthesises_positional_ids_when_missing() -> None:
    plan = normalize_plan({"tasks": [{"description": "one"}, {"description": "two"}]})
    assert [task.id for task in plan.tasks] == ["1", "2"]


def test_normalize_plan_disambiguates_duplicate_ids() -> None:
    plan = normalize_plan([{"id": "1", "description": "a"}, {"id": "1", "description": "b"}])
    assert [task.id for task in plan.tasks] == ["1", "1-2"]


def test_normalize_plan_drops_malformed_task_entries() -> None:
    plan = normalize_plan(
        {"tasks": [None, 42, {}, {"description": ""}, {"description": "the real one"}]}
    )
    assert [task.description for task in plan.tasks] == ["the real one"]


def test_normalize_plan_accepts_alternative_description_keys() -> None:
    plan = normalize_plan([{"title": "from title"}, {"summary": "from summary"}, {"goal": "g"}])
    assert [task.description for task in plan.tasks] == ["from title", "from summary", "g"]


def test_normalize_plan_coerces_comma_separated_files_and_dependencies() -> None:
    plan = normalize_plan(
        [{"id": "1", "description": "a"}, {"id": "2", "description": "b, dependencies: 1"}]
    )
    assert plan.tasks[0].files == []
    plan = normalize_plan(
        {"tasks": [{"id": "1", "description": "a", "files": "x.py, y.py", "dependencies": "2, 1"}]}
    )
    assert plan.tasks[0].files == ["x.py", "y.py"]


def test_normalize_plan_prunes_dangling_duplicate_and_self_dependencies() -> None:
    plan = normalize_plan(
        {
            "tasks": [
                {"id": "1", "description": "a"},
                {"id": "2", "description": "b", "dependencies": ["1", "1", "2", "ghost"]},
            ]
        }
    )
    assert plan.tasks[1].dependencies == ["1"]


def test_normalize_plan_rejects_a_payload_with_no_usable_task() -> None:
    with pytest.raises(ValueError, match="no usable task"):
        normalize_plan({"tasks": [None, 7], "rationale": "r"})


def test_normalize_plan_rejects_a_scalar_payload() -> None:
    with pytest.raises(ValueError, match="must be an object or list"):
        normalize_plan(42)


def test_normalize_plan_rejects_a_non_list_tasks_field() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        normalize_plan({"tasks": 5})


# --------------------------------------------------------------------------------------
# PlannerAgent
# --------------------------------------------------------------------------------------


async def test_planner_returns_a_plan_from_a_bare_json_reply(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    fake_llm = make_fake_llm(PLAN_JSON)
    plan = await PlannerAgent(client=fake_llm).plan("add auth")
    assert plan.rationale == "start with auth"
    assert plan.tasks[0].description == "write auth"
    assert plan.tasks[0].files == ["src/auth.py"]


async def test_planner_sends_the_goal_as_the_user_turn_and_the_prompt_as_system(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    fake_llm = make_fake_llm(PLAN_JSON)
    await PlannerAgent(client=fake_llm).plan("add auth")
    call = fake_llm.last_call()
    assert "add auth" in call.prompt
    assert call.system == PLANNER_PROMPT
    assert call.model == fake_llm.settings.planner_model


async def test_planner_uses_the_planner_model_not_the_default_model(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    client = make_fake_llm(PLAN_JSON, model="main-model", planner_model="cheap-planner")
    await PlannerAgent(client=client).plan("goal")
    assert client.models == ["cheap-planner"]


async def test_planner_accepts_a_fenced_reply(make_fake_llm: Callable[..., FakeLLMClient]) -> None:
    # Regression: a fenced reply used to crash the entire run instead of being parsed.
    client = make_fake_llm(fenced(PLAN_JSON))
    plan = await PlannerAgent(client=client).plan("add auth")
    assert [task.id for task in plan.tasks] == ["1"]
    assert client.call_count == 1


async def test_planner_accepts_json_wrapped_in_prose(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    client = make_fake_llm(f"Thinking about it...\n{PLAN_JSON}\nDone.")
    plan = await PlannerAgent(client=client).plan("add auth")
    assert plan.tasks[0].description == "write auth"


async def test_planner_retries_with_a_corrective_prompt_after_an_unusable_reply(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    client = make_fake_llm(["I would rather not.", fenced(PLAN_JSON, language="JSON")])
    plan = await PlannerAgent(client=client).plan("add auth")

    assert plan.tasks[0].description == "write auth"
    assert client.call_count == 2
    retry_prompt = client.prompts[1]
    assert "add auth" in retry_prompt, "the corrective retry must keep the goal"
    assert "could not be used" in retry_prompt
    assert "I would rather not." in retry_prompt


async def test_planner_raises_after_exhausting_its_retries(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    client = make_fake_llm(["nope", "still nope", "nope again"])
    with pytest.raises(ValueError, match="Failed to parse a usable plan after 3 attempts"):
        await PlannerAgent(client=client).plan("add auth")
    assert client.call_count == 3


async def test_planner_error_message_echoes_the_last_reply(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    client = make_fake_llm("total nonsense")
    with pytest.raises(ValueError, match="total nonsense"):
        await PlannerAgent(client=client).plan("add auth")


async def test_planner_accepts_a_bare_list_reply(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    client = make_fake_llm('["first slice", "second slice"]')
    plan = await PlannerAgent(client=client).plan("add auth")
    assert [task.description for task in plan.tasks] == ["first slice", "second slice"]
    assert [task.id for task in plan.tasks] == ["1", "2"]
    assert plan.rationale


async def test_planner_tolerates_a_reply_with_no_rationale(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    client = make_fake_llm('{"tasks": [{"id": "x", "description": "d"}]}')
    plan = await PlannerAgent(client=client).plan("add auth")
    assert plan.rationale == "Plan of x."


async def test_planner_drops_malformed_entries_instead_of_failing_the_plan(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    client = make_fake_llm(
        '{"tasks": [{"id": "1", "description": "keep me"}, null, {"description": ""}],'
        ' "rationale": "partial plan"}'
    )
    plan = await PlannerAgent(client=client).plan("add auth")
    assert [task.description for task in plan.tasks] == ["keep me"]


async def test_planner_normalises_duplicate_and_missing_ids(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    client = make_fake_llm(
        '{"tasks": [{"id": "dup", "description": "a"}, {"id": "dup", "description": "b"},'
        ' {"description": "c"}]}'
    )
    plan = await PlannerAgent(client=client).plan("add auth")
    assert [task.id for task in plan.tasks] == ["dup", "dup-2", "3"]


async def test_planner_asks_for_bare_json_in_the_prompt(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    fake_llm = make_fake_llm(PLAN_JSON)
    await PlannerAgent(client=fake_llm).plan("add auth")
    assert "bare JSON" in fake_llm.prompts[0]


async def test_planner_builds_its_own_client_when_none_is_given(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Settings] = []

    def fake_init(settings_arg: Settings | None = None, **kwargs: Any) -> FakeLLMClient:
        seen.append(settings_arg or settings)
        return FakeLLMClient(settings, responses=PLAN_JSON)

    monkeypatch.setattr("furrow.agents.planner.LLMClient", fake_init)
    plan = await PlannerAgent(settings=settings).plan("add auth")
    assert plan.tasks[0].id == "1"
    assert seen == [settings]


# --------------------------------------------------------------------------------------
# WorkerAgent: writing files
# --------------------------------------------------------------------------------------


async def test_file_block_creates_a_file_inside_the_workspace(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    reply = (
        "Here you go.\n\n<<<FILE: src/app.py>>>\n"
        "def main():\n    return 1\n<<<END FILE>>>\n\nDone."
    )
    client = make_fake_llm(reply)
    summary = await run_worker(client, make_task(files=["src/app.py"]))

    assert (tmp_workspace / "src" / "app.py").read_text() == "def main():\n    return 1\n"
    assert "wrote src/app.py" in summary
    assert "Here you go." in summary, "prose around the blocks is surfaced to the user"


async def test_file_block_creates_missing_parent_directories(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    client = make_fake_llm("<<<FILE: a/b/c/d.py>>>\nx = 1\n<<<END FILE>>>")
    await run_worker(client, make_task())
    assert (tmp_workspace / "a" / "b" / "c" / "d.py").read_text() == "x = 1\n"


async def test_file_block_overwrites_an_existing_file(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    target = tmp_workspace / "app.py"
    target.write_text("old contents\n")
    client = make_fake_llm("<<<FILE: app.py>>>\nnew contents\n<<<END FILE>>>")

    summary = await run_worker(client, make_task())

    assert target.read_text() == "new contents\n"
    assert "wrote app.py" in summary


async def test_file_block_without_a_closing_marker_still_applies(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    client = make_fake_llm("<<<FILE: truncated.py>>>\nvalue = 42\n")
    summary = await run_worker(client, make_task())

    assert (tmp_workspace / "truncated.py").read_text() == "value = 42\n"
    assert "wrote truncated.py" in summary


async def test_several_blocks_in_one_reply_are_all_applied(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    client = make_fake_llm(
        "<<<FILE: one.py>>>\n1\n<<<END FILE>>>\n"
        "<<<FILE: two.py>>>\n2\n<<<END FILE>>>\n"
    )
    summary = await run_worker(client, make_task())

    assert (tmp_workspace / "one.py").read_text() == "1\n"
    assert (tmp_workspace / "two.py").read_text() == "2\n"
    assert "wrote one.py" in summary
    assert "wrote two.py" in summary


async def test_response_without_blocks_reports_no_changes_and_writes_nothing(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    client = make_fake_llm("I looked at the code and it already does the right thing.")
    summary = await run_worker(client, make_task())

    assert summary.startswith("no file changes")
    assert client.list_files(tmp_workspace) == []


async def test_worker_sends_the_prompt_as_system_and_uses_the_worker_model(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    client = make_fake_llm("nothing to do", worker_model="worker-x")
    await run_worker(client, make_task(description="add a parser"))

    call = client.last_call()
    assert call.system == WORKER_PROMPT
    assert call.model == "worker-x"
    assert "add a parser" in call.prompt
    assert "Task id: 1" in call.prompt


async def test_worker_prompt_includes_the_workspace_layout(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    (tmp_workspace / "existing.py").write_text("x = 1\n")
    (tmp_workspace / ".hidden").mkdir()
    (tmp_workspace / ".hidden" / "secret.py").write_text("nope\n")
    client = make_fake_llm("nothing to do")

    await run_worker(client, make_task())

    prompt = client.prompts[0]
    assert "Workspace layout" in prompt
    assert "existing.py" in prompt
    assert "secret.py" not in prompt, "hidden paths are not advertised to the model"


# --------------------------------------------------------------------------------------
# WorkerAgent: reading context
# --------------------------------------------------------------------------------------


async def test_task_files_are_read_into_the_prompt(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    source = tmp_workspace / "src"
    source.mkdir()
    (source / "app.py").write_text("def existing():\n    return 'old'\n")
    client = make_fake_llm("nothing to do")

    await run_worker(client, make_task(files=["src/app.py"]))

    prompt = client.prompts[0]
    assert "existing file: src/app.py" in prompt
    assert "def existing():" in prompt
    assert "Existing files you may edit: src/app.py" in prompt


async def test_task_files_that_do_not_exist_yet_are_reported_as_new(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    client = make_fake_llm("<<<FILE: src/new.py>>>\nx = 1\n<<<END FILE>>>")
    await run_worker(client, make_task(files=["src/new.py"]))
    assert "treat them as new files" in client.prompts[0]


async def test_a_task_with_no_files_tells_the_model_to_choose_paths(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    client = make_fake_llm("nothing to do")
    await run_worker(client, make_task(files=[]))
    assert "choose the paths yourself" in client.prompts[0]
    assert "(not specified)" in client.prompts[0]


async def test_a_task_file_outside_the_workspace_is_not_leaked_into_the_prompt(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_path: Path, tmp_workspace: Path
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    client = make_fake_llm("nothing to do")

    await run_worker(client, make_task(files=["../secret.txt", "~/secret.txt"]))

    assert "TOP SECRET" not in client.prompts[0]


# --------------------------------------------------------------------------------------
# WorkerAgent: edits
# --------------------------------------------------------------------------------------


def edit_reply(path: str, search: str, replace: str) -> str:
    return (
        f"<<<EDIT: {path}>>>\n<<<SEARCH>>>\n{search}\n<<<REPLACE>>>\n{replace}\n<<<END EDIT>>>"
    )


async def test_edit_block_applies_a_search_and_replace(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    target = tmp_workspace / "app.py"
    target.write_text("import os\n\n\ndef main():\n    return os.getcwd()\n")
    client = make_fake_llm(
        edit_reply("app.py", "    return os.getcwd()", "    return Path.cwd()")
    )

    summary = await run_worker(client, make_task(files=["app.py"]))

    assert "return Path.cwd()" in target.read_text()
    assert "import os" in target.read_text()
    assert "edited app.py" in summary


async def test_edit_block_may_span_several_lines(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    target = tmp_workspace / "app.py"
    target.write_text("a = 1\nb = 2\nc = 3\n")
    client = make_fake_llm(edit_reply("app.py", "a = 1\nb = 2", "a = 10"))

    await run_worker(client, make_task())

    assert target.read_text() == "a = 10\nc = 3\n"


async def test_edit_with_a_missing_search_is_skipped_and_reported(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    target = tmp_workspace / "app.py"
    target.write_text("a = 1\n")
    client = make_fake_llm(edit_reply("app.py", "this line is absent", "replacement"))

    summary = await run_worker(client, make_task())

    assert target.read_text() == "a = 1\n", "an unmatched SEARCH must never be guessed"
    assert "skipped edit to app.py" in summary
    assert "not found" in summary


async def test_edit_with_an_ambiguous_search_is_skipped_and_reported(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    target = tmp_workspace / "app.py"
    target.write_text("x = 1\nx = 1\n")
    client = make_fake_llm(edit_reply("app.py", "x = 1", "x = 2"))

    summary = await run_worker(client, make_task())

    assert target.read_text() == "x = 1\nx = 1\n"
    assert "not unique" in summary


async def test_edit_of_a_missing_file_asks_for_a_file_block(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    client = make_fake_llm(edit_reply("ghost.py", "old", "new"))
    summary = await run_worker(client, make_task())
    assert "use a FILE block" in summary


async def test_edit_block_missing_its_search_or_replace_marker_is_skipped(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    target = tmp_workspace / "app.py"
    target.write_text("a = 1\n")
    client = make_fake_llm("<<<EDIT: app.py>>>\n<<<SEARCH>>>\na = 1\n<<<END EDIT>>>")

    summary = await run_worker(client, make_task())

    assert target.read_text() == "a = 1\n"
    assert "missing SEARCH or REPLACE" in summary


def test_search_replace_reports_an_empty_search() -> None:
    updated, reason = apply_search_replace("a = 1\n", "   ", "b = 2")
    assert updated is None
    assert reason == "SEARCH text is empty"


def test_search_replace_applies_an_exact_match() -> None:
    updated, reason = apply_search_replace("a = 1\nb = 2\n", "a = 1", "a = 9")
    assert reason == ""
    assert updated == "a = 9\nb = 2\n"


# --------------------------------------------------------------------------------------
# WorkerAgent: path safety
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "escape",
    [
        "../outside.py",
        "a/../../outside.py",
        "./../outside.py",
        "nested/../../../outside.py",
    ],
)
async def test_file_block_cannot_escape_via_relative_traversal(
    escape: str,
    make_fake_llm: Callable[..., FakeLLMClient],
    tmp_path: Path,
    tmp_workspace: Path,
) -> None:
    client = make_fake_llm(f"<<<FILE: {escape}>>>\nowned = True\n<<<END FILE>>>")

    summary = await run_worker(client, make_task())

    assert not (tmp_path / "outside.py").exists()
    assert "outside the workspace" in summary
    assert client.list_files(tmp_workspace) == []


async def test_absolute_path_outside_the_workspace_is_refused(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_path: Path, tmp_workspace: Path
) -> None:
    target = tmp_path / "absolute_escape.py"
    client = make_fake_llm(f"<<<FILE: {target}>>>\nowned = True\n<<<END FILE>>>")

    summary = await run_worker(client, make_task())

    assert not target.exists()
    assert "outside the workspace" in summary


async def test_system_path_is_refused_and_never_written(
    make_fake_llm: Callable[..., FakeLLMClient],
) -> None:
    passwd = Path("/etc/passwd")
    before = passwd.read_bytes() if passwd.is_file() else None
    client = make_fake_llm("<<<FILE: /etc/passwd>>>\nowned = True\n<<<END FILE>>>")

    summary = await run_worker(client, make_task())

    assert "outside the workspace" in summary
    if before is not None:
        assert passwd.read_bytes() == before


async def test_home_relative_path_is_refused(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    client = make_fake_llm("<<<FILE: ~/furrow-escape.py>>>\nowned = True\n<<<END FILE>>>")

    summary = await run_worker(client, make_task())

    assert not (fake_home / "furrow-escape.py").exists()
    assert "outside the workspace" in summary


async def test_symlink_pointing_outside_the_workspace_is_refused(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_path: Path, tmp_workspace: Path
) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("original\n")
    (tmp_workspace / "link.py").symlink_to(outside)
    client = make_fake_llm("<<<FILE: link.py>>>\nowned = True\n<<<END FILE>>>")

    summary = await run_worker(client, make_task())

    assert outside.read_text() == "original\n", "a symlink must not be followed out of the tree"
    assert "outside the workspace" in summary


async def test_symlinked_directory_escape_is_refused(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_path: Path, tmp_workspace: Path
) -> None:
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    (tmp_workspace / "escape").symlink_to(outside_dir, target_is_directory=True)
    client = make_fake_llm("<<<FILE: escape/evil.py>>>\nowned = True\n<<<END FILE>>>")

    summary = await run_worker(client, make_task())

    assert not (outside_dir / "evil.py").exists()
    assert "outside the workspace" in summary


async def test_edit_block_cannot_escape_the_workspace(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_path: Path
) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("secret = 1\n")
    client = make_fake_llm(edit_reply("../outside.py", "secret = 1", "secret = 2"))

    summary = await run_worker(client, make_task())

    assert outside.read_text() == "secret = 1\n"
    assert "outside the workspace" in summary


@pytest.mark.parametrize("raw_path", ["", ".", "./", "   "])
async def test_the_workspace_root_itself_is_refused(
    raw_path: str,
    make_fake_llm: Callable[..., FakeLLMClient],
    tmp_workspace: Path,
) -> None:
    client = make_fake_llm(f"<<<FILE: {raw_path}>>>\nowned = True\n<<<END FILE>>>")
    summary = await run_worker(client, make_task())
    assert "outside the workspace" in summary
    assert client.list_files(tmp_workspace) == []


async def test_an_existing_directory_target_is_refused(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    (tmp_workspace / "pkg").mkdir()
    client = make_fake_llm("<<<FILE: pkg>>>\nowned = True\n<<<END FILE>>>")
    summary = await run_worker(client, make_task())
    assert "outside the workspace" in summary


async def test_a_refused_block_does_not_stop_the_safe_blocks(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    client = make_fake_llm(
        "<<<FILE: ../escape.py>>>\nowned = True\n<<<END FILE>>>\n"
        "<<<FILE: safe.py>>>\nsafe = True\n<<<END FILE>>>"
    )
    summary = await run_worker(client, make_task())

    assert (tmp_workspace / "safe.py").read_text() == "safe = True\n"
    assert "wrote safe.py" in summary
    assert "outside the workspace" in summary


async def test_quoting_and_backticks_around_a_path_are_tolerated(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    client = make_fake_llm("<<<FILE: `quoted.py`>>>\nx = 1\n<<<END FILE>>>")
    await run_worker(client, make_task())
    assert (tmp_workspace / "quoted.py").read_text() == "x = 1\n"


async def test_worker_does_not_raise_when_the_write_fails(
    make_fake_llm: Callable[..., FakeLLMClient], tmp_workspace: Path
) -> None:
    client = make_fake_llm("<<<FILE: unwritable/app.py>>>\nx = 1\n<<<END FILE>>>")
    (tmp_workspace / "unwritable").write_text("i am a file, not a directory")
    summary = await run_worker(client, make_task())
    assert "failed to write unwritable/app.py" in summary


# --------------------------------------------------------------------------------------
# TesterAgent: exit code decides
# --------------------------------------------------------------------------------------


@pytest.fixture
def python_project(tmp_workspace: Path) -> Path:
    """A workspace that looks like a Python project, so pytest is a legitimate runner."""
    (tmp_workspace / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    return tmp_workspace


async def test_exit_code_zero_means_passed(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    subprocess_stub.configure(returncode=0, stdout="3 passed in 0.01s")
    client = make_fake_llm('{"passed": true, "summary": "3 tests passed", "failures": []}')

    result = await run_tester(client)

    assert result.passed is True
    assert result.summary == "3 tests passed"
    assert result.failures == []


async def test_the_test_runs_in_the_workspace_with_a_python_project_layout(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    client = make_fake_llm("{}")
    await run_tester(client)

    assert subprocess_stub.command[:2] == ("pytest", "-q")
    assert subprocess_stub.kwargs[0]["cwd"] == str(python_project)


async def test_non_zero_exit_code_means_failed_with_failures_extracted(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    output = (
        "============================= test session starts =============================\n"
        "FAILED tests/test_auth.py::test_login - AssertionError: expected 401, got 200\n"
        "========================= 1 failed, 2 passed in 0.10s ========================\n"
    )
    subprocess_stub.configure(returncode=1, stdout=output)
    client = make_fake_llm("{}")

    result = await run_tester(client)

    assert result.passed is False
    assert [failure for failure in result.failures if "test_login" in failure]
    assert len(result.failures) == 1, "nothing is invented from the exit code alone"
    assert "exited 1" in result.summary


async def test_non_zero_exit_code_with_unrecognisable_output_still_reports_a_failure(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    subprocess_stub.configure(returncode=2, stdout="something went sideways")
    client = make_fake_llm("{}")

    result = await run_tester(client)

    assert result.passed is False
    assert "exited 2" in result.failures[0]
    assert "something went sideways" in result.failures[0]


async def test_stdout_and_stderr_are_both_considered(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    subprocess_stub.configure(returncode=1, stdout="", stderr="ERROR: boom\n")
    client = make_fake_llm("{}")

    result = await run_tester(client)

    assert result.passed is False
    assert any("boom" in failure for failure in result.failures)


async def test_tester_asks_the_llm_to_phrase_the_verdict(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    subprocess_stub.configure(returncode=0, stdout="ok")
    client = make_fake_llm('{"summary": "all green", "failures": []}', tester_model="tester-x")

    await run_tester(client)

    call = client.last_call()
    assert call.system == TESTER_PROMPT
    assert call.model == "tester-x"
    assert "Exit code: 0" in call.prompt
    assert "passed=true" in call.prompt


async def test_tester_prompt_states_the_verdict_as_false_for_a_failing_run(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    subprocess_stub.configure(returncode=1, stdout="FAILED tests/test_a.py::test_b")
    client = make_fake_llm("{}")

    await run_tester(client)

    assert "passed=false" in client.prompts[0]


async def test_tester_prompt_lists_the_tasks_it_was_given(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    subprocess_stub.configure(returncode=0, stdout="ok")
    client = make_fake_llm("{}")
    tasks = [make_task(description="write the parser", task_id="7")]

    await run_tester(client, tasks)

    assert "7: write the parser" in client.prompts[0]


# --------------------------------------------------------------------------------------
# TesterAgent: the LLM can never overturn the exit code
# --------------------------------------------------------------------------------------


async def test_a_lying_llm_cannot_turn_a_failed_run_into_a_pass(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    subprocess_stub.configure(returncode=1, stdout="FAILED tests/test_a.py::test_b - boom")
    client = make_fake_llm('{"passed": true, "summary": "everything passed", "failures": []}')

    result = await run_tester(client)

    assert result.passed is False
    assert result.failures, "the scraped failures must survive a lying summary"


async def test_prose_claiming_success_cannot_turn_a_failed_run_into_a_pass(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    # Regression: the verdict used to come from ``"passed" in text``.
    subprocess_stub.configure(returncode=1, stdout="FAILED tests/test_a.py::test_b")
    client = make_fake_llm("The test suite passed; nothing to report.")

    result = await run_tester(client)

    assert result.passed is False


async def test_unparseable_llm_reply_keeps_the_exit_code_verdict(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    subprocess_stub.configure(returncode=1, stdout="FAILED tests/test_a.py::test_b")
    client = make_fake_llm("```\nnot json at all\n```")

    result = await run_tester(client)

    assert result.passed is False
    assert "exited 1" in result.summary


async def test_a_crashing_llm_keeps_the_exit_code_verdict(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    subprocess_stub.configure(returncode=1, stdout="FAILED tests/test_a.py::test_b")
    client = make_fake_llm(RuntimeError("the LLM exploded"))

    result = await run_tester(client)

    assert result.passed is False
    assert "exited 1" in result.summary


async def test_a_passing_run_cannot_be_turned_into_a_failure_by_the_llm(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    subprocess_stub.configure(returncode=0, stdout="1 passed")
    client = make_fake_llm('{"passed": false, "summary": "looks broken to me"}')

    result = await run_tester(client)

    assert result.passed is True


async def test_a_non_object_llm_reply_keeps_the_decided_summary(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    subprocess_stub.configure(returncode=1, stdout="FAILED tests/test_a.py::test_b")
    client = make_fake_llm('["not", "an", "object"]')

    result = await run_tester(client)

    assert result.passed is False
    assert "exited 1" in result.summary


async def test_llm_failure_list_is_ignored_when_it_is_empty(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    subprocess_stub.configure(returncode=1, stdout="FAILED tests/test_a.py::test_b - boom")
    client = make_fake_llm('{"passed": false, "summary": "s", "failures": []}')

    result = await run_tester(client)

    assert any("boom" in failure for failure in result.failures)


async def test_llm_may_replace_the_summary_and_failures_but_not_the_verdict(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    subprocess_stub.configure(returncode=1, stdout="FAILED tests/test_a.py::test_b")
    client = make_fake_llm(
        '{"passed": true, "summary": "login returns 401 as expected",'
        ' "failures": ["tests/test_auth.py::test_login - AssertionError"]}'
    )

    result = await run_tester(client)

    assert result.passed is False
    assert result.summary == "login returns 401 as expected"
    assert result.failures == ["tests/test_auth.py::test_login - AssertionError"]


# --------------------------------------------------------------------------------------
# TesterAgent: runner detection
# --------------------------------------------------------------------------------------


def no_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend nothing is executable.

    In CI ``sys.executable -m pytest`` is always importable, so detection can never fail
    naturally; this is the only way to exercise the "no runner" branch hermetically.
    """
    monkeypatch.setattr(TesterAgent, "_executable", staticmethod(lambda command: False))


async def test_no_runner_detected_fails_without_consulting_the_llm(
    make_fake_llm: Callable[..., FakeLLMClient],
    tmp_workspace: Path,
    subprocess_stub: SubprocessStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_runner(monkeypatch)
    client = make_fake_llm('{"passed": true, "summary": "looks fine to me"}')

    result = await run_tester(client)

    assert result.passed is False
    assert "No test runner detected" in result.summary
    assert client.call_count == 0, "a hallucinated pass is exactly the bug this guards"
    assert subprocess_stub.commands == []


async def test_node_project_runs_npm_and_not_pytest(
    make_fake_llm: Callable[..., FakeLLMClient],
    tmp_workspace: Path,
    subprocess_stub: SubprocessStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_workspace / "package.json").write_text("{}\n")
    only_npm = staticmethod(lambda command: command[0] == "npm")
    monkeypatch.setattr(TesterAgent, "_executable", only_npm)
    subprocess_stub.configure(returncode=0, stdout="1 passing")
    client = make_fake_llm('{"passed": true, "summary": "js tests pass"}')

    result = await run_tester(client)

    assert subprocess_stub.command[:3] == ("npm", "test", "--")
    assert result.passed is True


async def test_stray_pytest_cannot_hijack_a_javascript_project(
    settings: Settings,
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_workspace / "package.json").write_text("{}\n")
    tester = TesterAgent(settings=settings)
    monkeypatch.setattr(
        TesterAgent, "_executable", staticmethod(lambda command: command[0] == "pytest")
    )
    assert await bounded(tester._detect_runner()) is None


async def test_python_markers_select_the_pytest_runner(
    settings: Settings,
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_workspace / "setup.cfg").write_text("[metadata]\n")
    monkeypatch.setattr(TesterAgent, "_executable", staticmethod(lambda command: True))
    tester = TesterAgent(settings=settings)
    assert await bounded(tester._detect_runner()) == ["pytest", "-q"]


@pytest.mark.parametrize(
    ("marker", "expected_head"),
    [("Cargo.toml", "cargo"), ("go.mod", "go")],
)
async def test_rust_and_go_markers_select_their_own_runners(
    marker: str,
    expected_head: str,
    settings: Settings,
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_workspace / marker).write_text("# manifest\n")
    monkeypatch.setattr(TesterAgent, "_executable", staticmethod(lambda command: True))
    tester = TesterAgent(settings=settings)
    command = await bounded(tester._detect_runner())
    assert command is not None
    assert command[0] == expected_head


async def test_a_workspace_with_no_manifest_falls_back_to_pytest(
    settings: Settings,
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        TesterAgent, "_executable", staticmethod(lambda command: command[0] == "pytest")
    )
    tester = TesterAgent(settings=settings)
    assert await bounded(tester._detect_runner()) == ["pytest", "-q"]


async def test_executable_detection_uses_sys_executable_only_when_pytest_is_importable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert TesterAgent._executable([sys.executable, "-m", "pytest", "-q"]) is True
    assert TesterAgent._executable(["definitely-not-on-path-xyz"]) is False


# --------------------------------------------------------------------------------------
# TesterAgent: timeouts
# --------------------------------------------------------------------------------------


async def test_a_hung_run_is_killed_and_reported_as_a_failure(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
) -> None:
    subprocess_stub.queue(ProcessOutcome.of(returncode=None, hang=True))
    client = make_fake_llm('{"passed": true, "summary": "trust me"}', test_timeout=0.05)

    result = await run_tester(client)

    assert result.passed is False
    assert "timed out" in result.summary
    assert result.failures, "a timeout must be reported as a failure, not ignored"
    assert subprocess_stub.process.killed, "the child must be killed"
    assert subprocess_stub.process.reaped, "the child must be reaped so it cannot linger"


async def test_a_real_hung_child_process_is_killed_and_leaves_no_orphan(
    make_settings: Callable[..., Settings],
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_create = asyncio.create_subprocess_exec
    spawned: list[asyncio.subprocess.Process] = []

    async def spy(*command: str, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await real_create(*command, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    monkeypatch.setattr(
        TesterAgent,
        "_candidates",
        lambda self: [[sys.executable, "-c", "import time; time.sleep(30)"]],
    )
    monkeypatch.setattr(TesterAgent, "_executable", staticmethod(lambda command: True))
    client = FakeLLMClient(make_settings(test_timeout=0.2), responses="{}")

    result = await bounded(run_tester(client))

    assert result.passed is False
    assert "timed out" in result.summary
    assert len(spawned) == 1
    assert spawned[0].returncode == -signal.SIGKILL, "the hung child must not outlive the run"


async def test_an_unparseable_timeout_setting_falls_back_to_the_default(
    python_project: Path,
) -> None:
    valid = build_settings(workspace=python_project)
    broken = valid.model_copy(update={"test_timeout": "not a number"})
    assert TesterAgent(settings=broken)._timeout() == 300.0


async def test_the_configured_timeout_is_used(
    python_project: Path, make_fake_llm: Callable[..., FakeLLMClient]
) -> None:
    client = make_fake_llm("{}", test_timeout=42.0)
    assert TesterAgent(client=client)._timeout() == 42.0


async def test_the_configured_timeout_is_passed_to_the_wait(
    make_fake_llm: Callable[..., FakeLLMClient],
    python_project: Path,
    subprocess_stub: SubprocessStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[float | None] = []
    real_wait_for = asyncio.wait_for

    async def spy(awaitable: Any, timeout: float | None = None, **kwargs: Any) -> Any:
        seen.append(timeout)
        return await real_wait_for(awaitable, timeout=5.0, **kwargs)

    monkeypatch.setattr(asyncio, "wait_for", spy)
    subprocess_stub.configure(returncode=0, stdout="ok")
    client = make_fake_llm("{}", test_timeout=42.0)

    await run_tester(client)

    assert seen == [42.0]


# --------------------------------------------------------------------------------------
# TesterAgent: failure extraction
# --------------------------------------------------------------------------------------


def test_failure_extraction_keeps_pytest_failure_lines() -> None:
    output = "FAILED tests/test_a.py::test_b - AssertionError\n1 failed, 2 passed"
    failures = _collect_failures(output)
    assert any("test_a.py::test_b" in failure for failure in failures)
    assert any("1 failed" in failure for failure in failures)


def test_failure_extraction_keeps_traceback_bodies() -> None:
    output = "Traceback (most recent call last):\n  File \"a.py\", line 1\n    boom\n"
    failures = _collect_failures(output)
    assert any("File \"a.py\"" in failure for failure in failures)


def test_failure_extraction_keeps_the_run_summary_line() -> None:
    # The scraper only runs for a non-zero exit, where the summary line is useful context.
    assert _collect_failures("2 passed in 0.01s\n") == ["2 passed in 0.01s"]


def test_failure_extraction_ignores_blank_and_unremarkable_lines() -> None:
    assert _collect_failures("\n\nrunning 3 tests\n") == []


def test_failure_extraction_is_capped() -> None:
    output = "\n".join(f"FAILED tests/test_{index}.py::test_x" for index in range(100))
    assert len(_collect_failures(output)) == MAX_FAILURES


def test_failure_extraction_collapses_and_truncates_long_lines() -> None:
    output = "ERROR " + ("x" * (MAX_FAILURE_CHARS * 2))
    failures = _collect_failures(output)
    assert len(failures) == 1
    assert len(failures[0]) <= MAX_FAILURE_CHARS + 3
