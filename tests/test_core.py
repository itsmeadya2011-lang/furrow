import pytest
from furrow.config import Plan, TaskModel, TestResult, Settings


# --- Config model tests ---

def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_failures():
    t = TestResult(passed=False, summary="fail", failures=["err1", "err2"])
    assert t.failures == ["err1", "err2"]


def test_settings_default():
    s = Settings()
    assert s.provider.value == "anthropic"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_provider_enum():
    assert Settings(provider="openai").provider.value == "openai"
    assert Settings(provider="ollama").provider.value == "ollama"


# --- Orchestrator tests ---

class FakeTask:
    def __init__(self, id: str, status: str = "pending", result: str | None = None):
        self.id = id
        self.status = status
        self.result = result


class FakePlan:
    def __init__(self, tasks):
        self.tasks = tasks

    def model_dump(self):
        return {"tasks": [t.__dict__ for t in self.tasks], "rationale": "test"}


class TestOrchestratorIsDone:
    """Verify _is_done logic with persisted tasks."""

    @pytest.mark.asyncio
    async def test_empty_tasks_is_done(self):
        from furrow.core.orchestrator import Orchestrator
        o = Orchestrator(goal="test")
        assert o._is_done() is True

    @pytest.mark.asyncio
    async def test_all_completed_is_done(self):
        from furrow.core.orchestrator import Orchestrator
        o = Orchestrator(goal="test")
        o.tasks = [FakeTask("1", "completed"), FakeTask("2", "completed")]
        assert o._is_done() is True

    @pytest.mark.asyncio
    async def test_any_failed_not_done(self):
        from furrow.core.orchestrator import Orchestrator
        o = Orchestrator(goal="test")
        o.tasks = [FakeTask("1", "completed"), FakeTask("2", "failed")]
        assert o._is_done() is False

    @pytest.mark.asyncio
    async def test_pending_not_done(self):
        from furrow.core.orchestrator import Orchestrator
        o = Orchestrator(goal="test")
        o.tasks = [FakeTask("1", "completed"), FakeTask("2", "pending")]
        assert o._is_done() is False

    @pytest.mark.asyncio
    async def test_mixed_status_not_done(self):
        from furrow.core.orchestrator import Orchestrator
        o = Orchestrator(goal="test")
        o.tasks = [FakeTask("1", "completed"), FakeTask("2", "failed"), FakeTask("3", "pending")]
        assert o._is_done() is False

    @pytest.mark.asyncio
    async def test_partial_completion_not_done(self):
        from furrow.core.orchestrator import Orchestrator
        o = Orchestrator(goal="test")
        o.tasks = [FakeTask("1", "completed"), FakeTask("2", "pending")]
        assert o._is_done() is False


class TestOrchestratorMaxCycles:
    """Verify max_cycles enforcement."""

    @pytest.mark.asyncio
    async def test_max_cycles_zero_unlimited(self):
        from furrow.core.orchestrator import Orchestrator
        o = Orchestrator(goal="test")
        assert o._max_cycles == float("inf")

    @pytest.mark.asyncio
    async def test_max_cycles_set_via_settings(self):
        from furrow.config import Settings
        from furrow.core.orchestrator import Orchestrator
        s = Settings(max_cycles=3)
        o = Orchestrator(goal="test", settings=s)
        assert o._max_cycles == 3

    @pytest.mark.asyncio
    async def test_max_cycles_enforced(self):
        from furrow.config import Settings
        from furrow.core.orchestrator import Orchestrator
        s = Settings(max_cycles=2)
        o = Orchestrator(goal="test", settings=s)
        assert o._max_cycles == 2
        assert o.cycles == 0


# --- Tester tests ---

class TestTesterResultParsing:
    """Verify tester JSON parsing and fallback behavior."""

    @pytest.mark.asyncio
    async def test_valid_json_response(self):
        from furrow.agents.tester import TesterAgent
        from furrow.llm import LLMClient

        class MockClient:
            settings = Settings()

            async def complete(self, *args, **kwargs):
                return '{"passed": true, "summary": "All tests passed", "failures": []}'

        tester = TesterAgent(client=MockClient())
        result = await tester.run("test goal", [])
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_false_json_response(self):
        from furrow.agents.tester import TesterAgent
        from furrow.config import Settings

        class MockClient:
            settings = Settings()

            async def complete(self, *args, **kwargs):
                return '{"passed": false, "summary": "2 tests failed", "failures": ["test1", "test2"]}'

        tester = TesterAgent(client=MockClient())
        result = await tester.run("test goal", [])
        assert result.passed is False
        assert result.summary == "2 tests failed"
        assert result.failures == ["test1", "test2"]

    @pytest.mark.asyncio
    async def test_invalid_json_fallback(self):
        from furrow.agents.tester import TesterAgent
        from furrow.config import Settings

        class MockClient:
            settings = Settings()

            async def complete(self, *args, **kwargs):
                return "This is not JSON at all"

        tester = TesterAgent(client=MockClient())
        result = await tester.run("test goal", [])
        assert result.passed is False
        assert "not valid JSON" in result.summary
        assert len(result.failures) > 0

    @pytest.mark.asyncio
    async def test_empty_response_fallback(self):
        from furrow.agents.tester import TesterAgent
        from furrow.config import Settings

        class MockClient:
            settings = Settings()

            async def complete(self, *args, **kwargs):
                return ""

        tester = TesterAgent(client=MockClient())
        result = await tester.run("test goal", [])
        assert result.passed is False
        assert "not valid JSON" in result.summary


# --- LLM client tests ---

class TestLLMClient:
    def test_unsupported_provider_raises(self):
        import asyncio
        from furrow.config import Provider
        from furrow.llm import LLMClient

        s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="fake")
        client = LLMClient(settings=s)
        # Monkeypatch provider to an invalid value to test the branch
        client.settings.provider = "unknown"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unsupported provider"):
            asyncio.run(client.complete("test"))

    def test_anthropic_missing_key_raises(self):
        from furrow.config import Provider, Settings
        from furrow.llm import LLMClient

        s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
        client = LLMClient(settings=s)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            import asyncio
            asyncio.run(client.complete("test"))

    def test_openai_missing_key_raises(self):
        from furrow.config import Provider, Settings
        from furrow.llm import LLMClient

        s = Settings(provider=Provider.OPENAI, openai_api_key=None)
        client = LLMClient(settings=s)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            import asyncio
            asyncio.run(client.complete("test"))
