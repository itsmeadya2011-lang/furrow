from furrow.config import Settings


def test_config_new_fields():
    fields = Settings.model_fields
    assert fields["max_tokens"].default == 4096
    assert fields["max_retries"].default == 3
    assert fields["request_timeout"].default == 120.0
    # Pre-existing fields are still present.
    assert fields["max_parallel_tasks"].default == 5
    assert fields["max_cycles"].default == 0


def test_config_defaults_instantiate():
    # Should instantiate with defaults regardless of env overrides.
    s = Settings()
    assert hasattr(s, "max_tokens") and s.max_tokens >= 1
    assert hasattr(s, "max_retries") and s.max_retries >= 1
    assert s.request_timeout > 0
