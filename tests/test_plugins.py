from __future__ import annotations

from typing import Any

import pytest

from furrow.plugins import (
    HOOK_NAMES,
    Plugin,
    PluginRegistry,
    registry,
    run_hook,
    run_hooks,
)


class RecordingPlugin(Plugin):
    """A plugin that records every hook call and its kwargs."""

    name = "recording"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _record(self, hook_name: str, **kwargs: Any) -> None:
        self.calls.append({"hook": hook_name, "kwargs": kwargs})

    def setup(self, settings: Any) -> None:
        self._record("setup", settings=settings)

    def before_plan(self, **kwargs: Any) -> None:
        self._record("before_plan", **kwargs)

    def after_plan(self, **kwargs: Any) -> None:
        self._record("after_plan", **kwargs)

    def before_execute(self, **kwargs: Any) -> None:
        self._record("before_execute", **kwargs)

    def after_execute(self, **kwargs: Any) -> None:
        self._record("after_execute", **kwargs)

    def before_test(self, **kwargs: Any) -> None:
        self._record("before_test", **kwargs)

    def after_test(self, **kwargs: Any) -> None:
        self._record("after_test", **kwargs)

    def before_cycle(self, **kwargs: Any) -> None:
        self._record("before_cycle", **kwargs)

    def after_cycle(self, **kwargs: Any) -> None:
        self._record("after_cycle", **kwargs)


class FailingPlugin(Plugin):
    """A plugin whose hook raises, to verify failures are caught."""

    name = "failing"

    def before_plan(self, **kwargs: Any) -> None:
        raise RuntimeError("plugin boom")


class SilentPlugin(Plugin):
    """A plugin that runs after the failing one, to verify it still gets called."""

    name = "silent"

    def before_plan(self, **kwargs: Any) -> None:
        self.called = True


class NoNamePlugin(Plugin):
    """A plugin without a usable name."""


def test_plugin_defaults_are_no_ops():
    p = Plugin()
    assert p.name == ""
    p.setup(None)  # type: ignore[arg-type]
    p.before_plan()
    p.after_plan()
    p.before_execute()
    p.after_execute()
    p.before_test()
    p.after_test()
    p.before_cycle()
    p.after_cycle()


def test_plugin_registry_register_and_get():
    reg = PluginRegistry()
    p = RecordingPlugin()
    reg.register(p)
    assert reg.get("recording") is p
    assert reg.all() == [p]


def test_plugin_registry_ignores_missing_name():
    reg = PluginRegistry()
    p = NoNamePlugin()
    reg.register(p)
    assert reg.get("") is None
    assert reg.all() == []


def test_plugin_registry_replaces_by_name():
    reg = PluginRegistry()
    p1 = RecordingPlugin()
    p2 = RecordingPlugin()
    reg.register(p1)
    reg.register(p2)
    assert reg.get("recording") is p2
    assert reg.all() == [p2]


def test_plugin_registry_setup_is_called():
    reg = PluginRegistry()
    p = RecordingPlugin()
    settings = object()
    reg.register(p, settings=settings)
    assert p.calls == [{"hook": "setup", "kwargs": {"settings": settings}}]


def test_run_hook_calls_all_plugins_in_order():
    reg = registry
    reg.reset()
    p1 = RecordingPlugin()
    p2 = RecordingPlugin()
    reg.register(p1)
    reg.register(p2)

    run_hook("before_plan", goal="build something")
    assert p1.calls[-1]["hook"] == "before_plan"
    assert p1.calls[-1]["kwargs"]["goal"] == "build something"
    assert p2.calls[-1]["hook"] == "before_plan"
    assert p2.calls[-1]["kwargs"]["goal"] == "build something"


def test_run_hook_unknown_name_is_ignored(caplog):
    run_hook("not_a_real_hook")
    assert "Unknown plugin hook requested" in caplog.text


def test_run_hook_catches_failing_plugin_and_still_runs_others():
    reg = registry
    reg.reset()
    failing = FailingPlugin()
    silent = SilentPlugin()
    silent.called = False
    reg.register(failing)
    reg.register(silent)

    run_hook("before_plan")
    # The failing plugin's exception is caught; the silent plugin still runs.
    assert silent.called is True


def test_run_hooks_is_alias():
    assert run_hooks is run_hook


def test_registry_all_returns_copy():
    reg = PluginRegistry()
    p = RecordingPlugin()
    reg.register(p)
    plugins = reg.all()
    plugins.append(Plugin())  # type: ignore[arg-type]
    assert reg.all() == [p]


def test_registry_reset_clears_all():
    reg = PluginRegistry()
    p = RecordingPlugin()
    reg.register(p)
    reg.reset()
    assert reg.get("recording") is None
    assert reg.all() == []


def test_global_registry_singleton():
    assert registry is not None
    assert isinstance(registry, PluginRegistry)


def test_hook_names_cover_all_lifecycle_points():
    assert "before_plan" in HOOK_NAMES
    assert "after_plan" in HOOK_NAMES
    assert "before_execute" in HOOK_NAMES
    assert "after_execute" in HOOK_NAMES
    assert "before_test" in HOOK_NAMES
    assert "after_test" in HOOK_NAMES
    assert "before_cycle" in HOOK_NAMES
    assert "after_cycle" in HOOK_NAMES


def test_recording_plugin_records_all_hooks():
    p = RecordingPlugin()
    p.before_plan(goal="x")
    p.after_plan(goal="x", plan="plan")
    p.before_execute(plan="plan", tasks=[])
    p.after_execute(plan="plan", tasks=[])
    p.before_test(goal="x", tasks=[])
    p.after_test(goal="x", tasks=[], test_result=None)
    p.before_cycle(cycle=1, goal="x")
    p.after_cycle(cycle=1, goal="x", result=None)

    hooks = {c["hook"] for c in p.calls}
    assert hooks == HOOK_NAMES
