import pytest
from pydantic import ValidationError

from furrow.config import Settings


def test_settings_reject_invalid_cycle_and_parallel_bounds():
    with pytest.raises(ValidationError):
        Settings(max_cycles=-1)

    with pytest.raises(ValidationError):
        Settings(max_parallel_tasks=0)

    with pytest.raises(ValidationError):
        Settings(test_timeout=0)


def test_settings_allow_unlimited_cycles_and_positive_parallelism():
    settings = Settings(max_cycles=0, max_parallel_tasks=1, test_timeout=0.1)

    assert settings.max_cycles == 0
    assert settings.max_parallel_tasks == 1
    assert settings.test_timeout == 0.1
