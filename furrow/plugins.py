from __future__ import annotations

import contextlib
import importlib
import inspect
from collections.abc import Awaitable, Iterable
from importlib import metadata
from pathlib import Path
from typing import Any

from furrow.config import Settings, TaskModel, TestResult, WorkerResult
from furrow.config import settings as global_settings

ENTRY_POINT_GROUP = "furrow.plugins"


class Plugin:
    """Base class for Furrow plugins.

    Subclasses can override hook methods to extend Furrow behavior.
    """

    name: str = ""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        if not self.name:
            self.name = self.__class__.__name__

    # Lifecycle
    def setup(self, plugin_manager: PluginManager) -> None:
        """Called once after the plugin is instantiated and registered."""

    # Orchestrator hooks
    async def pre_cycle(self, orchestrator: Any, cycle_number: int) -> None:
        """Called before each cycle."""

    async def post_cycle(self, orchestrator: Any, cycle_number: int, result: Any) -> None:
        """Called after each cycle."""

    async def pre_task_execution(self, task: TaskModel) -> None:
        """Called before a task runs."""

    async def post_task_execution(self, task: TaskModel, result: WorkerResult) -> None:
        """Called after a task runs."""

    # Worker extension points
    def get_worker_tools(self) -> list[dict[str, Any]]:
        """Return additional worker tool schemas in Furrow's neutral format."""
        return []

    def execute_worker_tool(
        self, name: str, arguments: dict[str, Any], worker: Any
    ) -> Awaitable[str | None] | str | None:
        """Execute a custom worker tool. Return a string result or None to defer."""
        return None

    # Tester extension points
    def detect_test_command(
        self, tester: Any, workspace: Path
    ) -> Awaitable[list[str] | None] | list[str] | None:
        """Return a test command to use, or None to defer to other plugins/defaults."""
        return None

    def run_tests(
        self, tester: Any, goal: str, tasks: list[TaskModel]
    ) -> Awaitable[TestResult | None] | TestResult | None:
        """Run tests and return a TestResult, or None to defer."""
        return None


class PluginManager:
    """Loads and manages Furrow plugins and hook dispatch."""

    def __init__(
        self,
        settings: Settings | None = None,
        plugins: Iterable[Plugin] | None = None,
    ) -> None:
        self.settings = settings if settings is not None else global_settings
        self._plugins: list[Plugin] = list(plugins or [])

    @property
    def plugins(self) -> list[Plugin]:
        return list(self._plugins)

    def register(self, plugin: Plugin) -> Plugin:
        self._plugins.append(plugin)
        with contextlib.suppress(Exception):
            # Plugin setup must not break host initialization.
            plugin.setup(self)
        return plugin

    @staticmethod
    def _instantiate(obj: Any, settings: Settings) -> Plugin | None:
        try:
            if inspect.isclass(obj) and issubclass(obj, Plugin):
                try:
                    return obj(settings=settings)
                except TypeError:
                    return obj()
            if isinstance(obj, Plugin):
                return obj
            if callable(obj):
                try:
                    inst = obj(settings=settings)
                except TypeError:
                    inst = obj()
                if isinstance(inst, Plugin):
                    return inst
        except Exception:
            return None
        return None

    @staticmethod
    def _load_entry_points(group: str = ENTRY_POINT_GROUP) -> list[Any]:
        try:
            eps = metadata.entry_points()
            if hasattr(eps, "select"):
                selected = list(eps.select(group=group))
            else:  # pragma: no cover - legacy API
                selected = list(eps.get(group, []))
            loaded = []
            for ep in selected:
                try:
                    loaded.append(ep.load())
                except Exception:
                    continue
            return loaded
        except Exception:
            return []

    @staticmethod
    def _import_dotted(path: str) -> Any | None:
        module_path, sep, attr = path.rpartition(".")
        if not sep or not module_path:
            return None
        try:
            module = importlib.import_module(module_path)
            return getattr(module, attr, None)
        except Exception:
            return None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> PluginManager:
        resolved = settings if settings is not None else global_settings
        manager = cls(settings=resolved)
        # Entry points
        for obj in cls._load_entry_points():
            inst = cls._instantiate(obj, resolved)
            if inst is not None:
                manager.register(inst)
        # Dotted import paths from settings.plugins
        for path in list(getattr(resolved, "plugins", None) or []):
            if not isinstance(path, str):
                continue
            obj = cls._import_dotted(path)
            if obj is None:
                continue
            inst = cls._instantiate(obj, resolved)
            if inst is not None:
                manager.register(inst)
        return manager

    async def _call(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        for plugin in self._plugins:
            method = getattr(plugin, method_name, None)
            if method is None:
                continue
            try:
                result = method(*args, **kwargs)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                # Hooks must not crash the host.
                continue

    async def pre_cycle(self, orchestrator: Any, cycle_number: int) -> None:
        await self._call("pre_cycle", orchestrator, cycle_number)

    async def post_cycle(self, orchestrator: Any, cycle_number: int, result: Any) -> None:
        await self._call("post_cycle", orchestrator, cycle_number, result)

    async def pre_task_execution(self, task: TaskModel) -> None:
        await self._call("pre_task_execution", task)

    async def post_task_execution(self, task: TaskModel, result: WorkerResult) -> None:
        await self._call("post_task_execution", task, result)

    def get_worker_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        seen: set[str] = set()
        for plugin in self._plugins:
            try:
                extra = plugin.get_worker_tools() or []
            except Exception:
                continue
            for tool in extra:
                if not isinstance(tool, dict):
                    continue
                name = str(tool.get("name", ""))
                if not name or name in seen:
                    continue
                seen.add(name)
                tools.append(tool)
        return tools

    async def execute_worker_tool(
        self, name: str, arguments: dict[str, Any], worker: Any
    ) -> str | None:
        for plugin in self._plugins:
            try:
                result = plugin.execute_worker_tool(name, arguments, worker)
                if inspect.isawaitable(result):
                    result = await result
            except Exception:
                continue
            if result is not None:
                return str(result)
        return None

    async def detect_test_command(self, tester: Any, workspace: Path) -> list[str] | None:
        for plugin in self._plugins:
            try:
                result = plugin.detect_test_command(tester, workspace)
                if inspect.isawaitable(result):
                    result = await result
            except Exception:
                continue
            if result:
                return list(result)
        return None

    async def run_tests(self, tester: Any, goal: str, tasks: list[TaskModel]) -> TestResult | None:
        for plugin in self._plugins:
            try:
                result = plugin.run_tests(tester, goal, tasks)
                if inspect.isawaitable(result):
                    result = await result
            except Exception:
                continue
            if isinstance(result, TestResult):
                return result
        return None
