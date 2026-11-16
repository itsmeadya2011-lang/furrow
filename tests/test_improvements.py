"""Tests for the new furrow improvements."""

from __future__ import annotations

import pytest

from furrow.agents.worker import (
    _is_dangerous_segment,
    _text_is_denied,
    is_command_denied,
    resolve_in_workspace,
)
from furrow.config import Settings


class TestCommandSecurity:
    """Tests for the improved command security validation."""

    def test_allowed_programs(self, tmp_path):
        """Test that allowed programs bypass denial."""
        settings = Settings(
            _env_file=None,
            workspace=tmp_path,
            allowed_programs=["mycustomtool"],
        )
        # This should be allowed because mycustomtool is in allowed_programs
        assert not _is_dangerous_segment("mycustomtool --arg", tmp_path, settings)

    def test_allowed_commands(self, tmp_path):
        """Test that allowed command patterns bypass denial."""
        settings = Settings(
            _env_file=None,
            workspace=tmp_path,
            allowed_commands=["special-command"],
        )
        # This should be allowed because special-command is in allowed_commands
        assert not _is_dangerous_segment("special-command --flag", tmp_path, settings)

    def test_deny_force_push_configurable(self, tmp_path):
        """Test that force push denial can be disabled."""
        settings = Settings(
            _env_file=None,
            workspace=tmp_path,
            deny_force_push=False,
        )
        # Force push should be allowed when deny_force_push=False
        assert not _is_dangerous_segment("git push --force", tmp_path, settings)

    def test_deny_workspace_escape_configurable(self, tmp_path):
        """Test that workspace escape denial can be disabled."""
        settings = Settings(
            _env_file=None,
            workspace=tmp_path,
            deny_workspace_escape=False,
        )
        # Redirection outside workspace should be allowed when deny_workspace_escape=False
        assert not _is_dangerous_segment("echo test > /tmp/outside", tmp_path, settings)

    def test_is_command_denied_with_settings(self, tmp_path):
        """Test is_command_denied with custom settings."""
        settings = Settings(
            _env_file=None,
            workspace=tmp_path,
            allowed_commands=["allowed-cmd"],
        )
        # Should be allowed
        assert not is_command_denied("allowed-cmd", tmp_path, settings)
        # Should be denied
        assert is_command_denied("rm -rf /", tmp_path, settings)


class TestGitTools:
    """Tests for git integration tools."""

    @pytest.mark.asyncio
    async def test_git_status(self, tmp_path, monkeypatch):
        """Test git_status tool."""
        from furrow.agents.worker import WorkerAgent
        from furrow.config import TaskModel

        # Initialize a git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)

        task = TaskModel(id="1", description="test")
        worker = WorkerAgent(task=task, workspace=tmp_path)

        # Mock the _run_command to return a fake git status
        async def mock_run_command(cmd):
            if cmd == "git status":
                return "On branch main\nNo commits yet\nUntracked files:\n  (use \"git add <file>...\" to include in what will be committed)\n    test.py"
            return ""

        monkeypatch.setattr(worker, "_run_command", mock_run_command)

        result = await worker._git_status()
        assert "On branch main" in result
        assert "test.py" in result

    @pytest.mark.asyncio
    async def test_git_diff(self, tmp_path, monkeypatch):
        """Test git_diff tool."""
        from furrow.agents.worker import WorkerAgent
        from furrow.config import TaskModel

        task = TaskModel(id="1", description="test")
        worker = WorkerAgent(task=task, workspace=tmp_path)

        async def mock_run_command(cmd):
            if cmd == "git diff":
                return "diff --git a/test.py b/test.py\n+new line"
            if cmd == "git diff --cached":
                return "diff --git a/test.py b/test.py\n+staged line"
            return ""

        monkeypatch.setattr(worker, "_run_command", mock_run_command)

        # Test unstaged diff
        result = await worker._git_diff()
        assert "new line" in result

        # Test staged diff
        result = await worker._git_diff(staged=True)
        assert "staged line" in result

    @pytest.mark.asyncio
    async def test_git_commit(self, tmp_path, monkeypatch):
        """Test git_commit tool."""
        from furrow.agents.worker import WorkerAgent
        from furrow.config import TaskModel

        task = TaskModel(id="1", description="test")
        worker = WorkerAgent(task=task, workspace=tmp_path)

        async def mock_run_command(cmd):
            if cmd == "git add -A":
                return ""
            if cmd == "git status --porcelain":
                return "M test.py"
            if cmd.startswith("git commit -m"):
                return "[main abc1234] Test commit\n 1 file changed, 1 insertion(+)"
            return ""

        monkeypatch.setattr(worker, "_run_command", mock_run_command)

        result = await worker._git_commit(message="Test commit", add_all=True)
        assert "Test commit" in result

    @pytest.mark.asyncio
    async def test_git_commit_empty_message(self, tmp_path):
        """Test git_commit with empty message returns error."""
        from furrow.agents.worker import WorkerAgent
        from furrow.config import TaskModel

        task = TaskModel(id="1", description="test")
        worker = WorkerAgent(task=task, workspace=tmp_path)

        result = await worker._git_commit(message="")
        assert "error: commit message cannot be empty" in result

    @pytest.mark.asyncio
    async def test_git_log(self, tmp_path, monkeypatch):
        """Test git_log tool."""
        from furrow.agents.worker import WorkerAgent
        from furrow.config import TaskModel

        task = TaskModel(id="1", description="test")
        worker = WorkerAgent(task=task, workspace=tmp_path)

        async def mock_run_command(cmd):
            if cmd == "git log --oneline -n 10":
                return "abc1234 First commit\ndef5678 Second commit"
            return ""

        monkeypatch.setattr(worker, "_run_command", mock_run_command)

        result = await worker._git_log(limit=10, oneline=True)
        assert "abc1234" in result
        assert "First commit" in result


class TestCostTracking:
    """Tests for cost tracking."""

    def test_usage_add(self):
        """Test Usage.add method."""
        from furrow.config import Usage

        u1 = Usage(input_tokens=100, output_tokens=50, calls=1, cost_usd=0.001)
        u2 = Usage(input_tokens=200, output_tokens=100, calls=2, cost_usd=0.002)

        u1.add(u2)

        assert u1.input_tokens == 300
        assert u1.output_tokens == 150
        assert u1.calls == 3
        assert u1.cost_usd == 0.003
        assert u1.total_tokens == 450

    def test_model_pricing_override(self, tmp_path):
        """Test custom model pricing override."""
        from furrow.config import Settings
        from furrow.costs import ModelPricing

        custom_pricing = {
            "my-model": ModelPricing(input_per_million=1.0, output_per_million=2.0)
        }
        settings = Settings(
            _env_file=None,
            workspace=tmp_path,
            model_pricing=custom_pricing,
        )

        pricing = settings.pricing_overrides()
        assert "my-model" in pricing
        assert pricing["my-model"].input_per_million == 1.0
        assert pricing["my-model"].output_per_million == 2.0


class TestPlanManager:
    """Tests for the plan management (merged into PlannerAgent)."""

    @pytest.mark.asyncio
    async def test_merge_plan_preserves_completed(self, tmp_path):
        """Test that completed tasks with same description/files are preserved."""
        from furrow.agents.planner import PlannerAgent
        from furrow.config import Plan, Settings, TaskModel
        from furrow.llm import LLMClient

        settings = Settings(_env_file=None, workspace=tmp_path, max_task_retries=1)
        client = LLMClient(settings=settings)
        planner = PlannerAgent(client=client, settings=settings)

        # Existing completed task
        existing = TaskModel(
            id="1", description="Implement auth", files=["auth.py"], status="completed"
        )
        tasks = [existing]

        # New plan with same task
        new_plan = Plan(
            tasks=[TaskModel(id="1", description="Implement auth", files=["auth.py"])],
            rationale="test",
        )

        planner.merge_plan(tasks, new_plan)

        # Task should remain completed and be in skip_ids
        assert tasks[0].status == "completed"
        assert "1" in planner.skip_ids

    @pytest.mark.asyncio
    async def test_merge_plan_resets_failed_with_retries(self, tmp_path):
        """Test that failed tasks with retries left are reset to pending."""
        from furrow.agents.planner import PlannerAgent
        from furrow.config import Plan, Settings, TaskModel
        from furrow.llm import LLMClient

        settings = Settings(_env_file=None, workspace=tmp_path, max_task_retries=2)
        client = LLMClient(settings=settings)
        planner = PlannerAgent(client=client, settings=settings)

        # Existing failed task with 1 attempt (retries left)
        existing = TaskModel(
            id="1", description="Old task", files=["old.py"], status="failed", attempts=1
        )
        tasks = [existing]

        # New plan with updated task
        new_plan = Plan(
            tasks=[TaskModel(id="1", description="New task", files=["new.py"])],
            rationale="test",
        )

        planner.merge_plan(tasks, new_plan)

        # Task should be pending (retry available)
        assert tasks[0].status == "pending"
        assert tasks[0].description == "New task"
        assert tasks[0].files == ["new.py"]

    @pytest.mark.asyncio
    async def test_merge_plan_keeps_failed_exhausted(self, tmp_path):
        """Test that failed tasks with exhausted retries stay failed."""
        from furrow.agents.planner import PlannerAgent
        from furrow.config import Plan, Settings, TaskModel
        from furrow.llm import LLMClient

        settings = Settings(_env_file=None, workspace=tmp_path, max_task_retries=1)
        client = LLMClient(settings=settings)
        planner = PlannerAgent(client=client, settings=settings)

        # Existing failed task with 2 attempts (retries exhausted)
        existing = TaskModel(
            id="1", description="Old task", files=["old.py"], status="failed", attempts=2
        )
        tasks = [existing]

        # New plan with same task
        new_plan = Plan(
            tasks=[TaskModel(id="1", description="Old task", files=["old.py"])],
            rationale="test",
        )

        planner.merge_plan(tasks, new_plan)

        # Task should remain failed
        assert tasks[0].status == "failed"

    @pytest.mark.asyncio
    async def test_plan_waves_dependency_order(self, tmp_path):
        """Test that plan_waves respects dependencies."""
        from furrow.agents.planner import PlannerAgent
        from furrow.config import Settings, TaskModel
        from furrow.llm import LLMClient

        settings = Settings(_env_file=None, workspace=tmp_path)
        client = LLMClient(settings=settings)
        planner = PlannerAgent(client=client, settings=settings)

        tasks = [
            TaskModel(id="1", description="Task 1"),
            TaskModel(id="2", description="Task 2", dependencies=["1"]),
            TaskModel(id="3", description="Task 3", dependencies=["1"]),
            TaskModel(id="4", description="Task 4", dependencies=["2", "3"]),
        ]

        waves = planner.plan_waves(tasks)

        # Wave 1: task 1 (no deps)
        # Wave 2: tasks 2, 3 (depend on 1)
        # Wave 3: task 4 (depends on 2, 3)
        assert len(waves) == 3
        assert [t.id for t in waves[0]] == ["1"]
        assert set(t.id for t in waves[1]) == {"2", "3"}
        assert [t.id for t in waves[2]] == ["4"]


class TestResolveInWorkspace:
    """Tests for resolve_in_workspace function."""

    def test_resolve_relative_path(self, tmp_path):
        """Test resolving a relative path within workspace."""
        result = resolve_in_workspace(tmp_path, "subdir/file.txt")
        assert result is not None
        assert result == tmp_path / "subdir" / "file.txt"

    def test_resolve_absolute_path_in_workspace(self, tmp_path):
        """Test resolving an absolute path within workspace."""
        result = resolve_in_workspace(tmp_path, str(tmp_path / "file.txt"))
        assert result is not None
        assert result == tmp_path / "file.txt"

    def test_resolve_escapes_workspace(self, tmp_path):
        """Test that paths escaping workspace return None."""
        result = resolve_in_workspace(tmp_path, "../outside.txt")
        assert result is None

    def test_resolve_empty_path(self, tmp_path):
        """Test that empty path returns None."""
        result = resolve_in_workspace(tmp_path, "")
        assert result is None

    def test_resolve_absolute_outside(self, tmp_path):
        """Test absolute path outside workspace returns None."""
        result = resolve_in_workspace(tmp_path, "/etc/passwd")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])