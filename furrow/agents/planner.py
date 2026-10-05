"""Planning agent: turns a goal into a validated, normalised plan."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import ValidationError

from furrow.agents.errors import AgentError
from furrow.agents.jsonutil import extract_json
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

__all__ = ["PlannerAgent"]

log = structlog.get_logger(__name__)

#: One repair attempt is allowed after the first malformed reply.
_MAX_ATTEMPTS = 2
#: Upper bound on ``files`` entries retained for a single task.
_MAX_FILES_PER_TASK = 25
#: Default when ``Settings.max_prompt_files`` is absent.
_DEFAULT_MAX_PROMPT_FILES = 200

_REPAIR_SUFFIX = (
    "\n\nYour previous reply could not be parsed ({error}). "
    "Reply again with ONLY the bare JSON document described above: no markdown "
    "fences, no prose, no comments, no trailing commas, no extra keys."
)


class PlannerAgent:
    """Ask the LLM for a task plan and make it structurally sound."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        """Store the client, constructing a default one when none is supplied.

        Args:
            client: Shared :class:`~furrow.llm.LLMClient`; created if omitted.
            settings: Explicit settings; defaults to the client's own settings.
        """
        if client is None:
            self.client = LLMClient(settings=settings) if settings is not None else LLMClient()
        else:
            self.client = client
        self.settings = settings if settings is not None else getattr(self.client, "settings", None)

    async def plan(self, goal: str) -> Plan:
        """Produce a plan for ``goal``.

        The first reply is used as-is; if it cannot be decoded or validated,
        one repair round-trip is attempted before giving up.

        Args:
            goal: The user's objective.

        Returns:
            A :class:`~furrow.config.Plan` with unique task ids, pending
            statuses, resolvable dependencies and clamped file lists.

        Raises:
            AgentError: If the model never produced a usable plan.
        """
        prompt = self._build_prompt(goal)
        last_error = "unknown error"

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            response = await self.client.complete(
                prompt,
                system=PLANNER_SYSTEM,
                model=self._setting("planner_model", None),
            )
            try:
                data = extract_json(response)
            except AgentError as exc:
                last_error = str(exc)
                log.warning(
                    "planner_response_unparseable", attempt=attempt, error=last_error[:300]
                )
                prompt += _REPAIR_SUFFIX.format(error=last_error[:300])
                continue

            try:
                return _normalise_plan(data)
            except (AgentError, ValidationError, TypeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "planner_response_invalid", attempt=attempt, error=last_error[:300]
                )
                prompt += _REPAIR_SUFFIX.format(error=last_error[:300])

        raise AgentError(
            f"Planner could not produce a valid plan after {_MAX_ATTEMPTS} attempts. "
            f"Last error: {last_error}"
        )

    def _build_prompt(self, goal: str) -> str:
        """Assemble the planner prompt: instructions, goal, file listing."""
        workspace = self._workspace()
        listing = self._file_listing(workspace)
        max_files = self._setting("max_prompt_files", _DEFAULT_MAX_PROMPT_FILES)
        shown = listing[:max_files] if max_files and max_files > 0 else listing
        if shown:
            body = "\n".join(shown)
            if len(shown) < len(listing):
                body += f"\n... [{len(listing) - len(shown)} more files omitted]"
        else:
            body = "(the repository appears to be empty or could not be listed)"

        return (
            f"{PLANNER_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Repository files (relative to {workspace}):\n{body}\n"
        )

    def _file_listing(self, workspace: Path) -> list[str]:
        """List workspace files, tolerating a missing or unreadable directory."""
        try:
            return list(self.client.list_files(workspace))
        except OSError:
            return []

    def _workspace(self) -> Path:
        """Return the configured workspace as a :class:`~pathlib.Path`."""
        return Path(self._setting("workspace", Path.cwd()))

    def _setting(self, name: str, default: Any) -> Any:
        """Read a setting from the explicit settings or the client's settings."""
        holder = self.settings if self.settings is not None else getattr(
            self.client, "settings", None
        )
        value = getattr(holder, name, default)
        return default if value is None else value


def _normalise_plan(data: Any) -> Plan:
    """Validate raw decoded plan data into a sound :class:`Plan`.

    Repairs duplicate or missing task ids, drops hallucinated ``status`` and
    ``result`` fields, removes dependencies on unknown ids, and clamps the
    per-task ``files`` list.

    Args:
        data: The value returned by :func:`~furrow.agents.jsonutil.extract_json`.

    Returns:
        The validated plan.

    Raises:
        AgentError: If ``data`` is not a JSON object with a task list.
    """
    if not isinstance(data, dict):
        raise AgentError(f"Expected a JSON object for the plan, got {type(data).__name__}.")

    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise AgentError("Plan JSON is missing a non-empty 'tasks' array.")

    data = dict(data)
    tasks = [_normalise_task(raw, index) for index, raw in enumerate(raw_tasks)]
    data["tasks"] = _dedupe_ids(tasks)
    data.setdefault("rationale", "")

    plan = Plan.model_validate(data)
    _drop_unknown_dependencies(plan)
    for task in plan.tasks:
        task.status = "pending"
        task.result = None
    return plan


def _normalise_task(raw: Any, index: int) -> dict[str, Any]:
    """Return a sanitised task dict for ``raw`` at position ``index``."""
    if not isinstance(raw, dict):
        raise AgentError(f"Task at position {index} is {type(raw).__name__}, expected an object.")

    task = {key: value for key, value in raw.items() if key not in {"status", "result"}}
    identifier = task.get("id")
    if not isinstance(identifier, str) or not identifier.strip():
        task["id"] = str(index + 1)
    task["description"] = str(task.get("description") or "").strip()
    task["files"] = [item for item in _as_list(task.get("files")) if isinstance(item, str)]
    task["files"] = task["files"][:_MAX_FILES_PER_TASK]
    task["dependencies"] = [
        item for item in _as_list(task.get("dependencies")) if isinstance(item, str)
    ]
    return task


def _dedupe_ids(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Renumber tasks to ``"1"``, ``"2"``, ... when the model duplicated ids.

    Dependencies are remapped through the old-to-new mapping so a plan that
    repeats an id still keeps its ordering information; references that cannot
    be mapped are removed later by :func:`_drop_unknown_dependencies`.
    """
    seen: set[str] = set()
    for task in tasks:
        identifier = str(task["id"])
        if identifier in seen:
            return _renumber(tasks)
        seen.add(identifier)
    return tasks


def _renumber(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign sequential ids in place and remap dependency references."""
    mapping: dict[str, str] = {}
    for index, task in enumerate(tasks):
        old = str(task["id"])
        new = str(index + 1)
        task["id"] = new
        mapping.setdefault(old, new)
    for task in tasks:
        task["dependencies"] = [mapping.get(str(item), str(item)) for item in task["dependencies"]]
    return tasks


def _as_list(value: Any) -> list[Any]:
    """Coerce a model-supplied field into a list, treating scalars as one item."""
    if isinstance(value, list):
        return value
    if value is None or isinstance(value, (dict, tuple, set)):
        return []
    return [value]


def _drop_unknown_dependencies(plan: Plan) -> None:
    """Remove dependency ids that do not match any task in the plan."""
    known = {task.id for task in plan.tasks}
    for task in plan.tasks:
        kept: list[str] = []
        for dependency in task.dependencies:
            if dependency in known and dependency != task.id and dependency not in kept:
                kept.append(dependency)
        task.dependencies = kept