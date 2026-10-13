from __future__ import annotations

import pytest

from furrow.config import Settings
from tests.fakes import FakeLLMClient, make_settings


@pytest.fixture
def make_llm():
    def _make(
        responses=None,
        workspace=None,
        settings: Settings | None = None,
        **settings_kwargs,
    ):
        if settings is None:
            settings = make_settings(workspace=workspace, **settings_kwargs)
        return FakeLLMClient(settings=settings, responses=responses)

    return _make


@pytest.fixture
def dummy_test_cmd(tmp_path):
    script = tmp_path / "dummy_test.sh"
    script.write_text("#!/bin/sh\necho 'dummy test output'\nexit 0\n")
    script.chmod(0o755)
    return str(script)


@pytest.fixture
def recorded_bus():
    from furrow.core.events import EventBus

    bus = EventBus()
    events: list[dict] = []
    bus.subscribe(events.append)
    return bus, events
