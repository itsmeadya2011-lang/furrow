import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_task_model_with_files():
    t = TaskModel(id="2", description="impl", files=["src/a.py", "tests/test_a.py"])
    assert len(t.files) == 2


def test_plan_empty_tasks():
    p = Plan(tasks=[], rationale="nothing to do")
    assert p.tasks == []


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="2 failures", failures=["err1", "err2"])
    assert not t.passed
    assert len(t.failures) == 2


def test_provider_enum():
    from furrow.config import Provider
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_settings_defaults():
    from furrow.config import Settings
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_validate_provider_anthropic():
    from furrow.config import Settings, Provider
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    s.validate_provider()  # should not raise


def test_settings_validate_provider_anthropic_missing():
    from furrow.config import Settings, Provider
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        s.validate_provider()


def test_settings_validate_provider_openai_missing():
    from furrow.config import Settings, Provider
    s = Settings(provider=Provider.OPENAI, openai_api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        s.validate_provider()


def test_settings_validate_provider_ollama():
    from furrow.config import Settings, Provider
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    s.validate_provider()  # should not raise


def test_settings_extra_ignore():
    from furrow.config import Settings
    s = Settings(some_unknown_field="value")
    assert not hasattr(s, "some_unknown_field")