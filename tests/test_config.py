"""Contract tests for :mod:`furrow.config` -- settings, task and verdict models.

The contracts protected here are the ones the rest of Furrow silently relies on:

* ``TaskModel.status`` keeps its legacy *string* behaviour even though the field is a
  ``TaskStatus`` enum, so ``task.status == "pending"`` stays true for old callers.
* ``max_parallel_tasks`` is bounded to ``1..32`` (a value of 0 would deadlock the
  orchestrator's semaphore, 99 would melt the host), while ``max_cycles`` accepts 0 as
  "unlimited" and rejects negatives.
* ``Settings.model_for`` resolves per-role models and falls back to ``model``.
* ``TestResult.verified`` defaults to ``True`` -- a verdict is trustworthy unless the
  tester says otherwise -- and ``TestResult`` opts out of pytest collection.
* Mutable defaults (``files``/``dependencies``) are per-instance, not shared.

Every test builds its own ``Settings``; the module-level ``furrow.config.settings``
singleton is constructed at import time and must not be used here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import Provider, Settings, TaskModel, TaskStatus, TestResult


class TestTaskStatusBackwardCompat:
    """``status`` is an enum field that must still compare equal to its raw string."""

    def test_default_status_equals_the_legacy_pending_string(self) -> None:
        model = TaskModel(id="1", description="x")
        assert model.status == "pending"

    def test_default_status_is_the_pending_enum_member(self) -> None:
        model = TaskModel(id="1", description="x")
        assert model.status is TaskStatus.PENDING

    def test_completed_and_failed_statuses_compare_to_their_strings(self) -> None:
        model = TaskModel(id="1", description="x", status=TaskStatus.COMPLETED)
        assert model.status == "completed"
        model.status = TaskStatus.FAILED
        assert model.status == "failed"

    def test_status_enum_values_are_stable_strings(self) -> None:
        assert [member.value for member in TaskStatus] == ["pending", "completed", "failed"]


class TestTaskModel:
    """Field defaults and the human-readable one-liner."""

    def test_short_formats_id_and_description(self) -> None:
        assert TaskModel(id="7", description="write the parser").short() == ("[7] write the parser")

    def test_files_and_dependencies_default_to_empty_lists(self) -> None:
        model = TaskModel(id="1", description="x")
        assert model.files == []
        assert model.dependencies == []

    def test_default_lists_are_not_shared_between_instances(self) -> None:
        first = TaskModel(id="1", description="x")
        second = TaskModel(id="2", description="y")
        first.files.append("a.py")
        first.dependencies.append("0")
        assert second.files == []
        assert second.dependencies == []

    def test_explicit_files_and_dependencies_are_kept(self) -> None:
        model = TaskModel(id="1", description="x", files=["a.py"], dependencies=["0"])
        assert model.files == ["a.py"]
        assert model.dependencies == ["0"]

    def test_result_defaults_to_none(self) -> None:
        assert TaskModel(id="1", description="x").result is None


class TestSettingsBounds:
    """Numeric guard rails that keep the orchestrator's control flow sane."""

    def test_max_parallel_tasks_zero_is_rejected(
        self, make_settings: Callable[..., Settings], tmp_workspace: Path
    ) -> None:
        with pytest.raises(ValidationError):
            make_settings(tmp_workspace, max_parallel_tasks=0)

    def test_max_parallel_tasks_above_the_ceiling_is_rejected(
        self, make_settings: Callable[..., Settings], tmp_workspace: Path
    ) -> None:
        with pytest.raises(ValidationError):
            make_settings(tmp_workspace, max_parallel_tasks=99)

    def test_max_parallel_tasks_accepts_the_documented_range(
        self, make_settings: Callable[..., Settings], tmp_workspace: Path
    ) -> None:
        assert make_settings(tmp_workspace, max_parallel_tasks=1).max_parallel_tasks == 1
        assert make_settings(tmp_workspace, max_parallel_tasks=32).max_parallel_tasks == 32

    def test_max_cycles_zero_is_accepted_and_means_unlimited(
        self, make_settings: Callable[..., Settings], tmp_workspace: Path
    ) -> None:
        assert make_settings(tmp_workspace, max_cycles=0).max_cycles == 0

    def test_max_cycles_negative_is_rejected(
        self, make_settings: Callable[..., Settings], tmp_workspace: Path
    ) -> None:
        with pytest.raises(ValidationError):
            make_settings(tmp_workspace, max_cycles=-1)

    def test_max_parallel_tasks_default_is_five(
        self, make_settings: Callable[..., Settings], tmp_workspace: Path
    ) -> None:
        assert make_settings(tmp_workspace).max_parallel_tasks == 5


class TestSettingsModelRouting:
    """``model_for`` decides which model pays for each role."""

    def test_planner_role_resolves_to_planner_model(
        self, make_settings: Callable[..., Settings], tmp_workspace: Path
    ) -> None:
        settings = make_settings(tmp_workspace, planner_model="planner-model")
        assert settings.model_for("planner") == "planner-model"

    def test_worker_role_resolves_to_worker_model(
        self, make_settings: Callable[..., Settings], tmp_workspace: Path
    ) -> None:
        settings = make_settings(tmp_workspace, worker_model="worker-model")
        assert settings.model_for("worker") == "worker-model"

    def test_tester_role_resolves_to_tester_model(
        self, make_settings: Callable[..., Settings], tmp_workspace: Path
    ) -> None:
        settings = make_settings(tmp_workspace, tester_model="tester-model")
        assert settings.model_for("tester") == "tester-model"

    def test_unknown_role_falls_back_to_the_default_model(
        self, make_settings: Callable[..., Settings], tmp_workspace: Path
    ) -> None:
        settings = make_settings(tmp_workspace, model="default-model")
        assert settings.model_for("anything-else") == "default-model"

    def test_default_provider_is_anthropic(
        self, make_settings: Callable[..., Settings], tmp_workspace: Path
    ) -> None:
        assert make_settings(tmp_workspace).provider is Provider.ANTHROPIC


class TestSettingsLogLevel:
    """The validator exists so a lowercase ``debug`` still configures logging."""

    def test_log_level_is_uppercased(
        self, make_settings: Callable[..., Settings], tmp_workspace: Path
    ) -> None:
        assert make_settings(tmp_workspace, log_level="debug").log_level == "DEBUG"

    def test_log_level_default_is_info(
        self, make_settings: Callable[..., Settings], tmp_workspace: Path
    ) -> None:
        assert make_settings(tmp_workspace).log_level == "INFO"


class TestSettingsWorkspace:
    """The workspace is the root every agent is confined to."""

    def test_workspace_is_kept_as_a_path(
        self, make_settings: Callable[..., Settings], tmp_workspace: Path
    ) -> None:
        settings = make_settings(tmp_workspace)
        assert settings.workspace == tmp_workspace
        assert isinstance(settings.workspace, Path)


class TestTestResult:
    """The verdict model carries an explicit trust flag."""

    def test_verified_defaults_to_true(self) -> None:
        assert TestResult(passed=True, summary="1 passed").verified is True

    def test_failures_default_to_an_empty_list(self) -> None:
        assert TestResult(passed=False, summary="1 failed").failures == []

    def test_verified_can_be_set_to_false(self) -> None:
        result = TestResult(passed=False, summary="no runner", verified=False)
        assert result.verified is False

    def test_opted_out_of_pytest_collection(self) -> None:
        assert TestResult.__test__ is False
