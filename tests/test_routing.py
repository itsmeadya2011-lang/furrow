"""Tests for multi-provider model routing.

Furrow can send different agent roles to different providers and models.
Each role entry is optional and falls back to the global defaults when unset,
so a router with no entries behaves exactly like the old flat configuration.
"""

from __future__ import annotations

import pytest

from furrow.config import Provider, RouteEntry, Settings


def test_route_entry_defaults_to_unset():
    entry = RouteEntry()
    assert entry.provider is None
    assert entry.model is None


def test_router_is_empty_by_default():
    s = Settings(_env_file=None)
    assert s.routing.is_empty() is True


def test_route_for_falls_back_to_global_defaults():
    s = Settings(_env_file=None, provider=Provider.OPENAI, model="gpt-4o")
    provider, model = s.route_for("planner")
    assert provider == Provider.OPENAI
    assert model == "claude-3-5-haiku-20241022"

    provider, model = s.route_for("worker")
    assert provider == Provider.OPENAI
    assert model == "claude-3-5-sonnet-20241022"


def test_route_for_unknown_role_uses_global_defaults():
    s = Settings(_env_file=None)
    provider, model = s.route_for("nonexistent")
    assert provider == Provider.ANTHROPIC
    assert model == s.model


def test_route_for_uses_role_provider_when_set():
    s = Settings(_env_file=None, provider=Provider.ANTHROPIC)
    s.routing.worker = RouteEntry(provider=Provider.OPENAI)
    provider, model = s.route_for("worker")
    assert provider == Provider.OPENAI
    assert model == s.worker_model


def test_route_for_uses_role_model_when_set():
    s = Settings(_env_file=None)
    s.routing.planner = RouteEntry(model="my-planner")
    provider, model = s.route_for("planner")
    assert provider == Provider.ANTHROPIC
    assert model == "my-planner"


def test_route_for_uses_role_provider_and_model_together():
    s = Settings(_env_file=None)
    s.routing.worker = RouteEntry(provider=Provider.OLLAMA, model="qwen2.5-coder:7b")
    provider, model = s.route_for("worker")
    assert provider == Provider.OLLAMA
    assert model == "qwen2.5-coder:7b"


def test_router_not_empty_when_any_role_has_an_override():
    s = Settings(_env_file=None)
    s.routing.tester = RouteEntry(provider=Provider.OPENAI)
    assert s.routing.is_empty() is False


def test_router_round_trips_through_settings_dump_and_validate():
    s = Settings(
        _env_file=None,
        routing={
            "planner": {"provider": "openai", "model": "gpt-4o-mini"},
            "worker": {"provider": "ollama", "model": "llama3"},
        },
    )
    assert s.routing.planner.provider == Provider.OPENAI
    assert s.routing.planner.model == "gpt-4o-mini"
    assert s.routing.worker.provider == Provider.OLLAMA
    assert s.routing.worker.model == "llama3"
    assert s.routing.tester.provider is None

    dumped = s.model_dump()
    rebuilt = Settings.model_validate(dumped)
    assert rebuilt.routing.planner.provider == Provider.OPENAI
    assert rebuilt.routing.worker.model == "llama3"


def test_router_reads_from_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "FURROW_ROUTING",
        '{"planner":{"provider":"openai","model":"gpt-4o-mini"},'
        '"worker":{"provider":"ollama","model":"qwen"}}',
    )
    s = Settings(_env_file=None)
    assert s.routing.planner.provider == Provider.OPENAI
    assert s.routing.planner.model == "gpt-4o-mini"
    assert s.routing.worker.provider == Provider.OLLAMA
    assert s.routing.worker.model == "qwen"


def test_router_overrides_take_precedence_over_environment(monkeypatch: pytest.MonkeyPatch):
    """An explicit routing kwarg overrides the FURROW_ROUTING env var."""
    monkeypatch.setenv(
        "FURROW_ROUTING",
        '{"planner":{"provider":"openai","model":"gpt-4o-mini"}}',
    )
    s = Settings(_env_file=None, routing={"worker": {"provider": "ollama"}})
    assert s.routing.planner.provider == Provider.OPENAI
    assert s.routing.planner.model == "gpt-4o-mini"
    assert s.routing.worker.provider == Provider.OLLAMA
    assert s.routing.worker.model is None


def test_router_is_a_copy_of_the_setting():
    """Mutating the router of one settings object must not affect another."""
    s1 = Settings(_env_file=None)
    s2 = Settings(_env_file=None)
    s1.routing.worker = RouteEntry(provider=Provider.OPENAI)
    assert s2.routing.worker.provider is None