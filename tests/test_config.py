import pytest
from pydantic import ValidationError

from furrow.config import Provider, Settings


class TestSettingsValidation:
    def test_max_parallel_tasks_must_be_positive(self):
        with pytest.raises(ValueError, match="at least 1"):
            Settings(max_parallel_tasks=0)

    def test_max_parallel_tasks_negative_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            Settings(max_parallel_tasks=-1)

    def test_max_parallel_tasks_valid(self):
        s = Settings(max_parallel_tasks=3)
        assert s.max_parallel_tasks == 3

    def test_max_cycles_must_be_non_negative(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            Settings(max_cycles=-1)

    def test_max_cycles_zero_is_valid(self):
        s = Settings(max_cycles=0)
        assert s.max_cycles == 0

    def test_max_cycles_positive_valid(self):
        s = Settings(max_cycles=10)
        assert s.max_cycles == 10

    def test_log_level_validation_uppercase(self):
        s = Settings(log_level="debug")
        assert s.log_level == "DEBUG"

    def test_log_level_invalid_raises(self):
        with pytest.raises(ValidationError):
            Settings(log_level="VERBOSE")

    def test_provider_default(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC

    def test_provider_openai(self):
        s = Settings(provider=Provider.OPENAI)
        assert s.provider == Provider.OPENAI

    def test_provider_ollama(self):
        s = Settings(provider=Provider.OLLAMA)
        assert s.provider == Provider.OLLAMA

    def test_ollama_base_url_default(self):
        s = Settings()
        assert s.ollama_base_url == "http://localhost:11434"
