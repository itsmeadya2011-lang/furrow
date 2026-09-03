import pytest
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


# ---------------------------------------------------------------------------
# Existing tests (preserved verbatim)
# ---------------------------------------------------------------------------


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------


class TestProvider:
    @pytest.mark.parametrize(
        "member, expected",
        [
            (Provider.ANTHROPIC, "anthropic"),
            (Provider.OPENAI, "openai"),
            (Provider.OLLAMA, "ollama"),
        ],
    )
    def test_member_values(self, member, expected):
        assert member.value == expected

    def test_all_values(self):
        assert {m.value for m in Provider} == {"anthropic", "openai", "ollama"}

    def test_is_str_subclass(self):
        assert issubclass(Provider, str)


# ---------------------------------------------------------------------------
# TaskModel
# ---------------------------------------------------------------------------


class TestTaskModel:
    def test_defaults(self):
        t = TaskModel(id="t1", description="do the thing")
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None

    def test_custom_values(self):
        t = TaskModel(
            id="t2",
            description="other thing",
            files=["a.py", "b.py"],
            dependencies=["t1"],
            status="done",
            result="success",
        )
        assert t.id == "t2"
        assert t.files == ["a.py", "b.py"]
        assert t.dependencies == ["t1"]
        assert t.status == "done"
        assert t.result == "success"

    def test_model_dump_round_trip(self):
        original = TaskModel(
            id="t3",
            description="round trip",
            files=["x.py"],
            dependencies=[],
            status="running",
            result=None,
        )
        data = original.model_dump()
        restored = TaskModel.model_validate(data)
        assert restored.model_dump() == data


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class TestPlan:
    def test_empty_tasks(self):
        p = Plan(tasks=[], rationale="no tasks")
        assert p.tasks == []

    def test_multiple_tasks(self):
        p = Plan(
            tasks=[
                TaskModel(id="1", description="first"),
                TaskModel(id="2", description="second"),
                TaskModel(id="3", description="third"),
            ],
            rationale="three tasks",
        )
        assert len(p.tasks) == 3
        assert [t.id for t in p.tasks] == ["1", "2", "3"]
        assert p.rationale == "three tasks"

    def test_model_dump(self):
        p = Plan(
            tasks=[TaskModel(id="1", description="x")],
            rationale="r",
        )
        data = p.model_dump()
        assert data["rationale"] == "r"
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["id"] == "1"


# ---------------------------------------------------------------------------
# TestResult
# ---------------------------------------------------------------------------


class TestTestResult:
    def test_default_failures(self):
        t = TestResult(passed=False, summary="fail")
        assert t.failures == []

    def test_passed_false_by_default_when_not_set(self):
        t = TestResult(passed=False, summary="oops")
        assert t.passed is False

    def test_failures_with_content(self):
        t = TestResult(
            passed=False,
            summary="unit tests failed",
            failures=["AssertionError in test_foo", "Timeout in test_bar"],
        )
        assert not t.passed
        assert len(t.failures) == 2
        assert "AssertionError in test_foo" in t.failures

    def test_passed_true_with_empty_failures(self):
        t = TestResult(passed=True, summary="all good", failures=[])
        assert t.passed is True
        assert t.failures == []

    @pytest.mark.parametrize(
        "passed, failures",
        [
            (True, []),
            (False, []),
            (False, ["one failure"]),
            (False, ["a", "b", "c"]),
        ],
    )
    def test_parametrized_passed_and_failures(self, passed, failures):
        t = TestResult(passed=passed, summary="s", failures=failures)
        assert t.passed is passed
        assert t.failures == failures


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        # Ensure no FURROW_ env vars leak into these tests.
        for key in list(monkeypatch._environment.keys()):
            if key.startswith("FURROW_"):
                monkeypatch.delenv(key, raising=False)

    def test_default_provider(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.provider.value == "anthropic"

    def test_default_model(self):
        s = Settings()
        assert s.model == "claude-sonnet-4-20250514"

    def test_default_planner_model(self):
        s = Settings()
        assert s.planner_model == "claude-3-5-haiku-20241022"

    def test_default_worker_model(self):
        s = Settings()
        assert s.worker_model == "claude-3-5-sonnet-20241022"

    def test_default_tester_model(self):
        s = Settings()
        assert s.tester_model == "claude-3-5-sonnet-20241022"

    def test_default_max_parallel_tasks(self):
        s = Settings()
        assert s.max_parallel_tasks == 5

    def test_default_max_cycles(self):
        s = Settings()
        assert s.max_cycles == 0

    def test_default_ollama_base_url(self):
        s = Settings()
        assert s.ollama_base_url == "http://localhost:11434"

    def test_default_log_level(self):
        s = Settings()
        assert s.log_level == "INFO"

    def test_api_keys_default_none(self):
        s = Settings()
        assert s.anthropic_api_key is None
        assert s.openai_api_key is None

    def test_workspace_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = Settings()
        assert s.workspace == tmp_path

    @pytest.mark.parametrize(
        "provider_str, expected",
        [
            ("anthropic", Provider.ANTHROPIC),
            ("openai", Provider.OPENAI),
            ("ollama", Provider.OLLAMA),
        ],
    )
    def test_provider_from_string(self, provider_str, expected):
        s = Settings(provider=provider_str)
        assert s.provider == expected

    @pytest.mark.parametrize(
        "env_key, env_value, expected",
        [
            ("FURROW_MODEL", "gpt-4o", "gpt-4o"),
            ("FURROW_LOG_LEVEL", "DEBUG", "DEBUG"),
            ("FURROW_MAX_PARALLEL_TASKS", "10", 10),
            ("FURROW_MAX_CYCLES", "3", 3),
        ],
    )
    def test_env_override(self, env_key, env_value, expected, monkeypatch):
        monkeypatch.setenv(env_key, str(env_value))
        s = Settings()
        assert getattr(s, env_key.split("_", 1)[1].lower()) == expected
