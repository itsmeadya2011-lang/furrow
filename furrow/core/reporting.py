from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text

from furrow.core.events import Event, EventType

if TYPE_CHECKING:
    from furrow.core.events import EventBus

logger = logging.getLogger(__name__)

_MAX_DISPLAY = 200


def _truncate(value: str, limit: int = _MAX_DISPLAY) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _as_int(value: object, default: int = 0) -> int:
    """Best-effort int coercion so a malformed payload cannot kill a render."""
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    """Best-effort float coercion so a malformed payload cannot kill a render."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class RichReporter:
    """Subscribes to an EventBus and renders human-readable progress to a
    rich Console."""

    def __init__(self, bus: EventBus, console: Console | None = None) -> None:
        self._console = console or Console()
        self._unsubscribe = bus.subscribe(self.handle)

    def handle(self, event: Event) -> None:
        """Render a single event to the console."""
        data = event.data
        try:
            if event.type is EventType.RUN_STARTED:
                self._render_run_started(data)
            elif event.type is EventType.CYCLE_STARTED:
                self._render_cycle_started(data)
            elif event.type is EventType.PLAN_CREATED:
                self._render_plan_created(data)
            elif event.type is EventType.TASK_STARTED:
                self._render_task_started(data)
            elif event.type is EventType.TASK_COMPLETED:
                self._render_task_completed(data)
            elif event.type is EventType.TASK_FAILED:
                self._render_task_failed(data)
            elif event.type is EventType.TASK_SKIPPED:
                self._render_task_skipped(data)
            elif event.type is EventType.TESTS_COMPLETED:
                self._render_tests_completed(data)
            elif event.type is EventType.CYCLE_COMPLETED:
                self._render_cycle_completed(data)
            elif event.type is EventType.LOG:
                self._render_log(data)
            elif event.type is EventType.RUN_COMPLETED:
                self._render_run_completed(data)
            elif event.type is EventType.RUN_FAILED:
                self._render_run_failed(data)
        except Exception:
            logger.exception("RichReporter failed to render %s", event.type)

    def close(self) -> None:
        """Unsubscribe from the event bus."""
        self._unsubscribe()

    # ── Individual renderers ──────────────────────────────────────────────

    def _render_run_started(self, data: dict[str, object]) -> None:
        goal = str(data.get("goal", ""))
        max_cycles = data.get("max_cycles", 0)
        body = Text.assemble("Goal: ", (goal, "bold white"), "\n")
        body.append(f"Max cycles: {max_cycles}")
        self._console.print(
            Panel.fit(
                body,
                title="[bold cyan]Furrow Run Started[/bold cyan]",
                border_style="cyan",
            )
        )

    def _render_cycle_started(self, data: dict[str, object]) -> None:
        cycle = _as_int(data.get("cycle", 0))
        self._console.print(Text(f"═══ Cycle {cycle} ═══", style="bold cyan"))

    def _render_plan_created(self, data: dict[str, object]) -> None:
        plan = data.get("plan", {})
        rationale = data.get("rationale")
        renderable: RenderableType = Pretty(plan)
        if rationale:
            renderable = Group(
                Pretty(plan),
                Text.assemble(("\nRationale: ", "italic"), str(rationale)),
            )
        self._console.print(
            Panel(
                renderable,
                title="[bold blue]Plan Created[/bold blue]",
                border_style="blue",
                box=box.ROUNDED,
            )
        )

    def _render_task_started(self, data: dict[str, object]) -> None:
        task_id = str(data.get("task_id", ""))
        description = str(data.get("description", ""))
        self._console.print(Text(f"task {task_id}: {description}", style="dim"))

    def _render_task_completed(self, data: dict[str, object]) -> None:
        task_id = str(data.get("task_id", ""))
        result = _truncate(str(data.get("result", "")))
        duration = _as_float(data.get("duration", 0.0))
        self._console.print(
            Text(
                f"task {task_id} completed ({duration:.1f}s)",
                style="green",
            )
        )
        if result:
            self._console.print(Text(result, style="dim"))

    def _render_task_failed(self, data: dict[str, object]) -> None:
        task_id = str(data.get("task_id", ""))
        error = _truncate(str(data.get("error", "")))
        duration = _as_float(data.get("duration", 0.0))
        self._console.print(
            Text(
                f"task {task_id} failed: {error} ({duration:.1f}s)",
                style="red",
            )
        )

    def _render_task_skipped(self, data: dict[str, object]) -> None:
        task_id = str(data.get("task_id", ""))
        reason = _truncate(str(data.get("reason", "")))
        self._console.print(Text(f"task {task_id} skipped: {reason}", style="yellow"))

    def _render_tests_completed(self, data: dict[str, object]) -> None:
        passed = bool(data.get("passed", False))
        summary = str(data.get("summary", ""))
        failures = data.get("failures", [])
        style = "green" if passed else "red"
        self._console.print(Text(summary, style=style))
        if failures:
            for failure in failures:
                self._console.print(Text(f"  • {_truncate(str(failure))}"))

    def _render_cycle_completed(self, data: dict[str, object]) -> None:
        passed = bool(data.get("passed", False))
        status = "passed" if passed else "failed"
        self._console.print(Text(f"Cycle completed: {status}", style="dim"))

    def _render_log(self, data: dict[str, object]) -> None:
        message = str(data.get("message", ""))
        level = str(data.get("level", "info")).lower()
        if level == "warning":
            style = "yellow"
        elif level == "error":
            style = "red"
        else:
            style = "dim"
        self._console.print(Text(message, style=style))

    def _render_run_completed(self, data: dict[str, object]) -> None:
        reason = str(data.get("reason", ""))
        self._console.print(
            Panel(
                Text(reason, style="green"),
                title="[bold green]Run Completed[/bold green]",
                border_style="green",
            )
        )

    def _render_run_failed(self, data: dict[str, object]) -> None:
        error = _truncate(str(data.get("error", "")))
        self._console.print(
            Panel(
                Text(error, style="red"),
                title="[bold red]Run Failed[/bold red]",
                border_style="red",
            )
        )
