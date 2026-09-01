import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_version_string():
    import furrow

    assert furrow.__version__ == "0.1.0"


def test_import_does_not_require_api_keys():
    import furrow

    assert furrow is not None
    assert hasattr(furrow, "__version__")


def test_lazy_exports():
    from furrow import LLMClient, Settings

    assert LLMClient is not None
    assert Settings is not None


def test_settings_default_max_tokens():
    from furrow import Settings

    s = Settings()
    assert s.max_tokens == 4096


def test_provider_ollama_exists():
    from furrow.config import Provider

    assert Provider.OLLAMA.value == "ollama"
