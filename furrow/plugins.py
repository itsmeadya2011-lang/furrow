"""Plugin system for Furrow.

Plugins can observe and extend each lifecycle point of a run (plan, execute,
test and cycle boundaries).  The module is deliberately defensive: a failing
plugin must never break a run, so every hook call is isolated and logged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)

ENTRY_POINT_GROUP = "furrow.plugins"

HOOK_NAMES: frozenset[str] = frozenset(
    {
        "before_plan",
        "after_plan",
        "before_execute",
        "after_execute",
        "before_test",
        "after_test",
        "before_cycle",
        "after_cycle",
    }
)


class Plugin:
    """Base class for Furrow plugins.

    Subclasses set :attr:`name` and may override :meth:`setup` and any of the
    lifecycle hooks.  Hook implementations must be defensive; the registry also
    catches and logs exceptions so one bad plugin cannot break the others.
    """

    name: str = ""

    def setup(self, settings: Settings) -> None:
        """Called once when the plugin is registered."""
        return None

    def before_plan(self, **kwargs: Any) -> None:
        """Called before the planner produces a plan (e.g. ``goal``)."""
        return None

    def after_plan(self, **kwargs: Any) -> None:
        """Called after a plan is produced (e.g. ``goal``, ``plan``)."""
        return None

    def before_execute(self, **kwargs: Any) -> None:
        """Called before planned tasks are executed (e.g. ``plan``, ``tasks``)."""
        return None

    def after_execute(self, **kwargs: Any) -> None:
        """Called after planned tasks are executed (e.g. ``tasks``, ``result``)."""
        return None

    def before_test(self, **kwargs: Any) -> None:
        """Called before the tester runs (e.g. ``goal``, ``tasks``)."""
        return None

    def after_test(self, **kwargs: Any) -> None:
        """Called after the tester runs (e.g. ``test_result``, ``tasks``)."""
        return None

    def before_cycle(self, **kwargs: Any) -> None:
        """Called before a cycle starts (e.g. ``cycle``, ``goal``)."""
        return None

    def after_cycle(self, **kwargs: Any) -> None:
        """Called after a cycle finishes (e.g. ``cycle``, ``goal``, ``result``)."""
        return None


class PluginRegistry:
    """Ordered registry of plugins keyed by plugin name."""

    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}

    def register(self, plugin: Plugin, settings: Settings | None = None) -> None:
        """Register *plugin*, optionally running its setup with *settings*.

        A plugin without a usable name is ignored.  Registering a name that is
        already present replaces the previous plugin while keeping its position
        in the registration order.
        """
        try:
            raw_name = getattr(plugin, "name", "")
        except Exception as exc:  # pragma: no cover - defensive against odd plugins
            logger.error("Could not read plugin name: %s", exc)
            return

        if not isinstance(raw_name, str) or not raw_name.strip():
            logger.warning("Ignoring plugin with missing or invalid name: %r", plugin)
            return

        name = raw_name.strip()
        if name in self._plugins:
            logger.warning("Replacing already registered plugin %s", name)

        self._plugins[name] = plugin

        if settings is not None:
            try:
                plugin.setup(settings)
            except Exception as exc:
                logger.error("Plugin %s failed during setup: %s", name, exc, exc_info=True)

    def get(self, name: str) -> Plugin | None:
        """Return the plugin registered under *name*, if any."""
        return self._plugins.get(name)

    def all(self) -> list[Plugin]:
        """Return all registered plugins in registration order."""
        return list(self._plugins.values())

    def discover_entry_points(self) -> None:
        """Load and register plugins advertised under ``furrow.plugins``.

        Discovery is best-effort: missing metadata support, broken entry points
        or failing plugin factories are logged and otherwise ignored.
        """
        try:
            from importlib.metadata import entry_points
        except Exception as exc:  # pragma: no cover - importlib.metadata is stdlib
            logger.warning("Plugin discovery unavailable: %s", exc)
            return

        try:
            all_entry_points = entry_points()
        except Exception as exc:
            logger.warning("Could not read entry points: %s", exc)
            return

        try:
            if hasattr(all_entry_points, "select"):
                selected = all_entry_points.select(group=ENTRY_POINT_GROUP)
            else:  # pragma: no cover - legacy Python fallback
                selected = getattr(all_entry_points, ENTRY_POINT_GROUP, [])
        except Exception as exc:
            logger.warning("Could not select %s entry points: %s", ENTRY_POINT_GROUP, exc)
            return

        for entry_point in selected:
            ep_name = getattr(entry_point, "name", "<unknown>")
            try:
                loaded = entry_point.load()
            except Exception as exc:
                logger.warning("Could not load plugin entry point %s: %s", ep_name, exc)
                continue

            plugin = self._coerce_plugin(loaded)
            if plugin is None:
                logger.warning(
                    "Entry point %s did not provide a Furrow plugin", ep_name
                )
                continue

            self.register(plugin)

    @staticmethod
    def _coerce_plugin(loaded: Any) -> Plugin | None:
        """Turn an entry-point object into a plugin instance, if possible."""
        if isinstance(loaded, Plugin):
            return loaded

        if isinstance(loaded, type) and issubclass(loaded, Plugin):
            try:
                return loaded()
            except Exception as exc:
                logger.error("Could not instantiate plugin class %s: %s", loaded, exc)
                return None

        return None

    def reset(self) -> None:
        """Remove all registered plugins (primarily for tests)."""
        self._plugins.clear()


registry = PluginRegistry()


def run_hook(hook_name: str, **kwargs: Any) -> None:
    """Call *hook_name* on every registered plugin in registration order.

    Exceptions raised by a plugin are caught and logged so a single bad plugin
    cannot break the remaining plugins or the caller.
    """
    if hook_name not in HOOK_NAMES:
        logger.warning("Unknown plugin hook requested: %s", hook_name)
        return

    for plugin in registry.all():
        try:
            name = getattr(plugin, "name", "<unknown>")
        except Exception:  # pragma: no cover - defensive against odd plugins
            name = "<unknown>"

        try:
            hook = getattr(plugin, hook_name, None)
        except Exception as exc:  # pragma: no cover - defensive against odd plugins
            logger.error("Could not resolve hook %s on plugin %s: %s", hook_name, name, exc)
            continue

        if not callable(hook):
            continue

        try:
            hook(**kwargs)
        except Exception as exc:
            logger.error(
                "Plugin %s failed in hook %s: %s", name, hook_name, exc, exc_info=True
            )


run_hooks = run_hook

__all__ = [
    "ENTRY_POINT_GROUP",
    "HOOK_NAMES",
    "Plugin",
    "PluginRegistry",
    "registry",
    "run_hook",
    "run_hooks",
]