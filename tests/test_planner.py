"""Contract tests for :mod:`furrow.agents.planner` -- JSON recovery and plan building.

Two contracts are protected here.

**Response recovery.** ``extract_json_object`` must dig a JSON object out of whatever a
model actually emits: bare JSON, a ```` ```json ```` fence, a bare ```` ``` ```` fence,
prose before and after the fence, and -- crucially -- braces and backslash escapes
*inside* string values, which naive brace counting gets wrong. Anything that is not an
object (``[1,2,3]``, prose, an empty response, an unbalanced brace) must raise
``ValueError`` so ``PlannerAgent.plan`` can retry instead of building a broken plan.

**Plan construction.** ``PlannerAgent.plan()`` asks for raw JSON with an explicit
``system=`` prompt and the per-role ``planner_model``, repairs model sloppiness (blank
task ids become positional ids, descriptions are stripped), caps the number of tasks at
``max(5, settings.max_parallel_tasks)`` while noting the truncation in the rationale, and
raises ``ValueError`` only after its bounded retries are exhausted.

No network access: every test drives a scripted :class:`tests.conftest.FakeLLM`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from furrow.agents.planner import (
    MAX_ATTEMPTS,
    SYSTEM_PROMPT,
    PlannerAgent,
    extract_json_object,
    strip_code_fences,
)
from furrow.config import Plan, Settings
from tests.conftest import FakeLLM, make_settings_for


def _planner(
    tmp_workspace: Path,
    responses: list[str],
    **overrides: object,
) -> tuple[PlannerAgent, FakeLLM]:
    """A ``PlannerAgent`` wired to a scripted, offline client."""
    settings = make_settings_for(tmp_workspace, **overrides)
    fake = FakeLLM(responses=list(responses), settings=settings)
    return PlannerAgent(client=fake), fake


class TestStripCodeFences:
    """``strip_code_fences`` removes a wrapping fence, and is a no-op otherwise."""

    def test_removes_a_json_tagged_fence(self) -> None:
        assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_removes_an_untagged_fence(self) -> None:
        assert strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_removes_an_indented_fence_and_surrounding_whitespace(self) -> None:
        assert strip_code_fences('  \n  ```json\n{"a": 1}\n```  \n') == '{"a": 1}'

    def test_leaves_bare_json_untouched_apart_from_whitespace(self) -> None:
        assert strip_code_fences('  \n {"a": 1}\n  ') == '{"a": 1}'

    def test_leaves_a_bare_json_array_untouched(self) -> None:
        assert strip_code_fences("  [1, 2, 3] ") == "[1, 2, 3]"

    def test_only_a_trailing_fence_is_removed_when_prose_precedes_it(self) -> None:
        # Documented asymmetry: only the *last* line is dropped as a fence. Because
        # the first line is prose rather than a fence, the opening ```` ```json ````
        # line survives -- this is a line-based strip, not a fence-pair strip.
        # ``extract_json_object`` still finds the payload regardless.
        assert (
            strip_code_fences('Here you go:\n```json\n{"a": 1}\n```')
            == 'Here you go:\n```json\n{"a": 1}'
        )


class TestExtractJsonObject:
    """``extract_json_object`` finds the outermost object in arbitrary model output."""

    def test_parses_bare_json(self) -> None:
        assert extract_json_object('{"tasks": [], "rationale": "r"}') == {
            "tasks": [],
            "rationale": "r",
        }

    def test_parses_a_json_tagged_fence(self) -> None:
        text = '```json\n{"tasks": [1], "rationale": "r"}\n```'
        assert extract_json_object(text) == {"tasks": [1], "rationale": "r"}

    def test_parses_an_untagged_fence(self) -> None:
        assert extract_json_object('```\n{"tasks": [1]}\n```') == {"tasks": [1]}

    def test_ignores_prose_before_and_after_the_fence(self) -> None:
        text = 'Sure, here is the plan:\n```json\n{"tasks": [1]}\n```\nLet me know.'
        assert extract_json_object(text) == {"tasks": [1]}

    def test_ignores_prose_around_bare_json(self) -> None:
        assert extract_json_object('Answer: {"tasks": []} -- done') == {"tasks": []}

    def test_keeps_braces_inside_string_values(self) -> None:
        text = json.dumps({"rationale": "use { and } carefully", "tasks": []})
        assert extract_json_object(text)["rationale"] == "use { and } carefully"

    def test_keeps_escaped_quotes_and_braces_inside_string_values(self) -> None:
        text = '{"rationale": "a \\"quoted }\\" brace", "tasks": []}'
        assert extract_json_object(text)["rationale"] == 'a "quoted }" brace'

    def test_keeps_nested_objects_and_a_close_brace_inside_a_value(self) -> None:
        assert extract_json_object('{"outer": {"inner": "}"}, "tasks": []}') == {
            "outer": {"inner": "}"},
            "tasks": [],
        }

    def test_skips_a_leading_broken_brace_and_parses_the_next_object(self) -> None:
        assert extract_json_object('{oops then {"tasks": []}') == {"tasks": []}

    @pytest.mark.parametrize("text", ["[1,2,3]", "not json at all", "", "   \n  "])
    def test_rejects_anything_that_is_not_an_object(self, text: str) -> None:
        with pytest.raises(ValueError, match="no JSON object found"):
            extract_json_object(text)

    def test_rejects_an_unbalanced_object(self) -> None:
        with pytest.raises(ValueError, match="no JSON object found"):
            extract_json_object("{not json}")

    def test_rejects_an_empty_fenced_response(self) -> None:
        with pytest.raises(ValueError, match="no JSON object found"):
            extract_json_object("```json\n```")


class TestPlanHappyPath:
    """A well-formed response becomes a validated ``Plan`` on the first attempt."""

    async def test_returns_a_plan_from_a_single_call(self, tmp_workspace: Path) -> None:
        planner, fake = _planner(
            tmp_workspace,
            ['{"tasks": [{"id": "a", "description": "write a"}], "rationale": "because"}'],
        )
        plan = await planner.plan("do the thing")
        assert isinstance(plan, Plan)
        assert [task.id for task in plan.tasks] == ["a"]
        assert plan.tasks[0].description == "write a"
        assert plan.rationale == "because"
        assert fake.calls == 1

    async def test_keeps_declared_files_and_dependencies(self, tmp_workspace: Path) -> None:
        planner, _ = _planner(
            tmp_workspace,
            [
                (
                    '{"tasks": [{"id": "b", "description": "d", "files": ["b.py"],'
                    ' "dependencies": ["a"]}], "rationale": "r"}'
                )
            ],
        )
        plan = await planner.plan("goal")
        assert plan.tasks[0].files == ["b.py"]
        assert plan.tasks[0].dependencies == ["a"]

    async def test_rationale_defaults_to_empty_when_absent(self, tmp_workspace: Path) -> None:
        planner, _ = _planner(tmp_workspace, ['{"tasks": []}'])
        plan = await planner.plan("goal")
        assert plan.rationale == ""

    async def test_prompt_carries_the_goal(self, tmp_workspace: Path) -> None:
        planner, fake = _planner(tmp_workspace, ['{"tasks": [], "rationale": "r"}'])
        await planner.plan("make the parser fast")
        assert "make the parser fast" in fake.prompts[0]


class TestPlanRetry:
    """Malformed output is fed back to the model a bounded number of times."""

    async def test_retries_after_malformed_json_and_succeeds(self, tmp_workspace: Path) -> None:
        planner, fake = _planner(
            tmp_workspace,
            ["I could not do it", '{"tasks": [{"id": "a", "description": "d"}], "rationale": "r"}'],
        )
        plan = await planner.plan("goal")
        assert fake.calls == 2
        assert [task.id for task in plan.tasks] == ["a"]

    async def test_retry_prompt_quotes_the_rejected_output(self, tmp_workspace: Path) -> None:
        planner, fake = _planner(
            tmp_workspace,
            ["totally not json", '{"tasks": [], "rationale": "r"}'],
        )
        await planner.plan("goal")
        assert "totally not json" in fake.prompts[1]
        assert "Malformed output" in fake.prompts[1]
        assert "no JSON object found" in fake.prompts[1]

    async def test_raises_value_error_after_exhausting_the_retries(
        self, tmp_workspace: Path
    ) -> None:
        planner, fake = _planner(tmp_workspace, ["a", "b", "c", "d"])
        with pytest.raises(ValueError, match="Failed to parse plan from LLM after"):
            await planner.plan("goal")
        assert fake.calls == MAX_ATTEMPTS == 3

    async def test_error_message_carries_the_last_response(self, tmp_workspace: Path) -> None:
        planner, _ = _planner(tmp_workspace, ["aaa", "bbb", "ccc"])
        with pytest.raises(ValueError, match="Last response: ccc"):
            await planner.plan("goal")

    async def test_a_structurally_wrong_payload_is_retried_not_raised(
        self, tmp_workspace: Path
    ) -> None:
        planner, fake = _planner(
            tmp_workspace,
            ['{"tasks": "not a list", "rationale": "r"}', '{"tasks": [], "rationale": "r"}'],
        )
        plan = await planner.plan("goal")
        assert fake.calls == 2
        assert plan.tasks == []
        assert "'tasks' must be a list" in fake.prompts[1]

    async def test_a_non_object_task_entry_is_retried_not_raised(self, tmp_workspace: Path) -> None:
        planner, fake = _planner(
            tmp_workspace,
            ['{"tasks": ["just a string"], "rationale": "r"}', '{"tasks": [], "rationale": "r"}'],
        )
        plan = await planner.plan("goal")
        assert fake.calls == 2
        assert plan.tasks == []
        assert "each task must be an object" in fake.prompts[1]


class TestPlanNormalisation:
    """Model sloppiness is repaired before the plan is handed to the orchestrator."""

    async def test_blank_task_ids_become_positional_ids(self, tmp_workspace: Path) -> None:
        planner, _ = _planner(
            tmp_workspace,
            [
                (
                    '{"tasks": [{"id": "  ", "description": "a"}, {"id": "", "description": "b"},'
                    ' {"description": "c"}], "rationale": "r"}'
                )
            ],
        )
        plan = await planner.plan("goal")
        assert [task.id for task in plan.tasks] == ["1", "2", "3"]

    async def test_explicit_ids_are_not_renumbered(self, tmp_workspace: Path) -> None:
        planner, _ = _planner(
            tmp_workspace,
            [
                (
                    '{"tasks": [{"id": "b", "description": "x"}, {"id": "a", "description": "y"}],'
                    ' "rationale": "r"}'
                )
            ],
        )
        plan = await planner.plan("goal")
        assert [task.id for task in plan.tasks] == ["b", "a"]

    async def test_descriptions_are_stripped(self, tmp_workspace: Path) -> None:
        planner, _ = _planner(
            tmp_workspace,
            ['{"tasks": [{"id": "a", "description": "   write a.py   "}], "rationale": " r "}'],
        )
        plan = await planner.plan("goal")
        assert plan.tasks[0].description == "write a.py"
        assert plan.rationale == "r"

    async def test_missing_description_becomes_empty(self, tmp_workspace: Path) -> None:
        planner, _ = _planner(tmp_workspace, ['{"tasks": [{"id": "a"}], "rationale": "r"}'])
        plan = await planner.plan("goal")
        assert plan.tasks[0].description == ""


class TestPlanTaskCap:
    """The plan is bounded so one cycle cannot melt the machine."""

    @staticmethod
    def _response(count: int) -> str:
        """A well-formed plan response carrying ``count`` real task objects."""
        tasks = [{"id": f"t{i}", "description": f"task {i}"} for i in range(count)]
        return json.dumps({"tasks": tasks, "rationale": "long plan"})

    async def test_cap_is_at_least_five_even_when_parallelism_is_small(
        self, tmp_workspace: Path
    ) -> None:
        planner, fake = _planner(tmp_workspace, [self._response(9)], max_parallel_tasks=2)
        plan = await planner.plan("goal")
        assert fake.calls == 1
        assert len(plan.tasks) == 5
        assert [task.id for task in plan.tasks] == ["t0", "t1", "t2", "t3", "t4"]

    async def test_cap_follows_max_parallel_tasks_when_it_is_larger(
        self, tmp_workspace: Path
    ) -> None:
        planner, _ = _planner(tmp_workspace, [self._response(12)], max_parallel_tasks=8)
        plan = await planner.plan("goal")
        assert len(plan.tasks) == 8
        assert [task.id for task in plan.tasks] == [f"t{i}" for i in range(8)]

    async def test_truncation_is_noted_in_the_rationale(self, tmp_workspace: Path) -> None:
        planner, fake = _planner(tmp_workspace, [self._response(9)], max_parallel_tasks=3)
        plan = await planner.plan("goal")
        # One call, no retry: the payload was valid JSON, so the cap -- not an
        # exhausted retry budget -- is the only reason tasks went missing.
        assert fake.calls == 1
        assert len(plan.tasks) == 5
        assert [task.id for task in plan.tasks] == ["t0", "t1", "t2", "t3", "t4"]
        assert plan.rationale == "long plan\n[truncated to 5 tasks]"

    async def test_a_plan_at_the_cap_is_not_truncated(self, tmp_workspace: Path) -> None:
        planner, _ = _planner(tmp_workspace, [self._response(5)], max_parallel_tasks=1)
        plan = await planner.plan("goal")
        assert len(plan.tasks) == 5
        assert plan.rationale == "long plan"
        assert "truncated" not in plan.rationale


class TestPlanClientContract:
    """The planner must actually pass ``system=`` and ask for the planner's model."""

    async def test_system_prompt_is_sent_on_every_attempt(self, tmp_workspace: Path) -> None:
        planner, fake = _planner(
            tmp_workspace,
            ["nope", "still nope", '{"tasks": [], "rationale": "r"}'],
        )
        await planner.plan("goal")
        assert fake.systems == [SYSTEM_PROMPT, SYSTEM_PROMPT, SYSTEM_PROMPT]

    async def test_planner_model_is_requested(self, tmp_workspace: Path) -> None:
        planner, fake = _planner(
            tmp_workspace,
            ['{"tasks": [], "rationale": "r"}'],
            planner_model="planner-model",
            model="default-model",
        )
        await planner.plan("goal")
        assert fake.models == ["planner-model"]
        assert planner.client.settings.model_for("planner") == "planner-model"

    async def test_an_explicit_client_is_kept(self, tmp_workspace: Path) -> None:
        planner, fake = _planner(tmp_workspace, ['{"tasks": [], "rationale": "r"}'])
        assert planner.client is fake

    async def test_settings_can_be_supplied_without_a_client(self, tmp_workspace: Path) -> None:
        settings = make_settings_for(tmp_workspace)
        planner = PlannerAgent(settings=settings)
        assert isinstance(planner.client.settings, Settings)
