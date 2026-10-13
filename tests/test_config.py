from __future__ import annotations

import pytest
from pydantic import ValidationError

from furrow.config import Provider, Settings


def test_valid_settings_accepted():
    s = Settings(max_parallel_tasks=3, worker_rounds=5, max_cycles=2, test_timeout=60)
    assert s.max_parallel_tasks == 3
    assert s.worker_rounds == 5
    assert s.max_cycles == 2
    assert s.test_timeout == 60


def test_defaults():
    s = Settings()
    assert s.max_parallel_tasks == 5
    assert s.worker_rounds == 8
    assert s.max_cycles == 0
    assert s.test_timeout == 120
    assert s.provider == Provider.ANTHROPIC


def test_max_parallel_tasks_below_one_raises():
    with pytest.raises(ValidationError):
        Settings(max_parallel_tasks=0)


def test_max_parallel_tasks_negative_raises():
    with pytest.raises(ValidationError):
        Settings(max_parallel_tasks=-3)


def test_worker_rounds_below_one_raises():
    with pytest.raises(ValidationError):
        Settings(worker_rounds=0)


def test_worker_rounds_negative_raises():
    with pytest.raises(ValidationError):
        Settings(worker_rounds=-1)


def test_max_cycles_negative_raises():
    with pytest.raises(ValidationError):
        Settings(max_cycles=-1)


def test_max_cycles_zero_allowed():
    s = Settings(max_cycles=0)
    assert s.max_cycles == 0


def test_env_prefix_furrow(monkeypatch):
    monkeypatch.setenv("FURROW_MAX_CYCLES", "7")
    monkeypatch.setenv("FURROW_WORKER_ROUNDS", "4")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "9")
    s = Settings()
    assert s.max_cycles == 7
    assert s.worker_rounds == 4
    assert s.max_parallel_tasks == 9


def test_env_prefix_provider(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    s = Settings()
    assert s.provider == Provider.OPENAI


def test_env_prefix_model(monkeypatch):
    monkeypatch.setenv("FURROW_MODEL", "gpt-4o")
    s = Settings()
    assert s.model == "gpt-4o"


def test_env_overridden_by_explicit_value(monkeypatch):
    monkeypatch.setenv("FURROW_MAX_CYCLES", "7")
    s = Settings(max_cycles=1)
    assert s.max_cycles == 1


def test_test_command_optional():
    s = Settings(test_command="pytest -q")
    assert s.test_command == "pytest -q"


def test_workspace_is_path(tmp_path):
    s = Settings(workspace=tmp_path)
    assert s.workspace == tmp_path
