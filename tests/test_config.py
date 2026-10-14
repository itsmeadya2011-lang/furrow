from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


class TestSettingsDefaults:
    def test_provider_defaults_to_anthropic(self):
        settings = Settings()
        assert settings.provider == Provider.ANTHROPIC

    def test_max_parallel_tasks_defaults_to_5(self):
        settings = Settings()
        assert settings.max_parallel_tasks == 5

    def test_max_cycles_defaults_to_0(self):
        settings = Settings()
        assert settings.max_cycles == 0

    def test_model_defaults(self):
        settings = Settings()
        assert settings.model == "claude-sonnet-4-20250514"
        assert settings.planner_model == "claude-3-5-haiku-20241022"
        assert settings.worker_model == "claude-3-5-sonnet-20241022"
        assert settings.tester_model == "claude-3-5-sonnet-20241022"

    def test_workspace_defaults_to_cwd(self):
        settings = Settings()
        assert settings.workspace == Settings().workspace


class TestTaskModelDefaults:
    def test_status_defaults_to_pending(self):
        task = TaskModel(id="1", description="Test task")
        assert task.status == "pending"

    def test_files_defaults_to_empty_list(self):
        task = TaskModel(id="1", description="Test task")
        assert task.files == []

    def test_dependencies_defaults_to_empty_list(self):
        task = TaskModel(id="1", description="Test task")
        assert task.dependencies == []

    def test_result_defaults_to_none(self):
        task = TaskModel(id="1", description="Test task")
        assert task.result is None

    def test_can_override_defaults(self):
        task = TaskModel(
            id="1",
            description="Test task",
            status="completed",
            files=["src/main.py"],
            dependencies=["2"],
            result="Done",
        )
        assert task.status == "completed"
        assert task.files == ["src/main.py"]
        assert task.dependencies == ["2"]
        assert task.result == "Done"


class TestPlanModel:
    def test_plan_with_tasks(self):
        tasks = [
            TaskModel(id="1", description="Task 1"),
            TaskModel(id="2", description="Task 2"),
        ]
        plan = Plan(tasks=tasks, rationale="Test plan")
        assert len(plan.tasks) == 2
        assert plan.rationale == "Test plan"

    def test_plan_empty_tasks(self):
        plan = Plan(tasks=[], rationale="No tasks")
        assert plan.tasks == []


class TestTestResultModel:
    def test_test_result_with_failures(self):
        result = TestResult(
            passed=False, summary="Tests failed", failures=["test_foo failed", "test_bar failed"]
        )
        assert result.passed is False
        assert result.summary == "Tests failed"
        assert result.failures == ["test_foo failed", "test_bar failed"]

    def test_test_result_passed_empty_failures(self):
        result = TestResult(passed=True, summary="All passed", failures=[])
        assert result.passed is True
        assert result.failures == []

    def test_test_result_failures_defaults_to_empty_list(self):
        result = TestResult(passed=True, summary="All passed")
        assert result.failures == []
