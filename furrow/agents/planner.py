"""Turn a goal into a small, sanitized :class:`~furrow.config.Plan`.

The planner is the one place where a bad model reply can wreck a whole run: a plan
with duplicate ids, unparseable text, or a task that references a task that does not
exist would surface as a confusing failure three layers down.  So this module is
deliberately paranoid -- it retries once with a strict repair message, and then
normalizes whatever comes back field by field, dropping anything that does not make
sense instead of propagating it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SCHEMA
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient
from furrow.util import JSONExtractionError, extract_json_object, truncate

if TYPE_CHECKING:
    from furrow.config import Settings

__all__ = ["PlannerAgent", "PlannerError"]

_MAX_GOAL_CHARS = 4000
_MAX_FOCUS_CHARS = 2000
_MAX_CONTEXT_CHARS = 4000
_MAX_LAYOUT_FILES = 40
_MAX_LAYOUT_CHARS = 2000
_MAX_ECHO_CHARS = 800
_DEFAULT_MAX_TASKS = 5


class PlannerError(RuntimeError):
    """Raised when the planner model will not produce a usable plan."""


class PlannerAgent:
    """Plans a goal into at most ``planner_max_tasks`` independent tasks."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client if client is not None else LLMClient(settings=settings)
        self.settings: Settings = settings if settings is not None else self.client.settings

    async def plan(
        self,
        goal: str,
        *,
        focus: str | None = None,
        context: str | None = None,
    ) -> Plan:
        """Build a plan for ``goal``.

        Args:
            goal: What the user wants achieved.
            focus: What the previous cycle found failing or missing; when given it is
                highlighted so the model repairs instead of re-planning from scratch.
            context: Optional caller-supplied summary of the repository (the CLI and
                web server build one); the file listing is added on top of it.

        Returns:
            A normalized :class:`Plan`.  An empty ``tasks`` list means the model
            judged the goal already satisfied.

        Raises:
            PlannerError: Both the first attempt and the single repair retry failed.
        """
        prompt = self._build_prompt(goal, focus, context)
        raw = await self._complete(prompt)
        try:
            return self._parse(raw)
        except (JSONExtractionError, ValueError) as first_error:
            repaired = await self._complete(self._repair_prompt(prompt, raw, first_error))
            try:
                return self._parse(repaired)
            except (JSONExtractionError, ValueError) as exc:
                raise PlannerError(
                    f"planner model did not return a usable plan ({exc}); "
                    f"reply={truncate(repaired, limit=_MAX_ECHO_CHARS)!r}"
                ) from exc

    async def _complete(self, prompt: str) -> str:
        """Send one planner request, letting provider errors propagate to the caller."""
        return await self.client.complete(prompt, system=PLANNER_PROMPT, model=self._model())

    def _model(self) -> str | None:
        return getattr(self.client.settings, "planner_model", None)

    def _max_tasks(self) -> int:
        raw = getattr(self.settings, "planner_max_tasks", _DEFAULT_MAX_TASKS)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return _DEFAULT_MAX_TASKS

    def _build_prompt(self, goal: str, focus: str | None, context: str | None) -> str:
        parts = [f"Goal:\n{truncate(goal, limit=_MAX_GOAL_CHARS)}"]
        if focus:
            focus = truncate(focus.strip(), limit=_MAX_FOCUS_CHARS)
            parts.append(f"Focus for this cycle (what is still broken or missing):\n{focus}")
        if context:
            context = truncate(context.strip(), limit=_MAX_CONTEXT_CHARS)
            parts.append(f"Repository context:\n{context}")
        layout = self._repo_layout()
        if layout:
            parts.append(f"Repository layout (truncated):\n{layout}")
        return "\n\n".join(parts)

    def _repo_layout(self) -> str:
        """Return a capped, newline-joined listing so it cannot swamp the prompt."""
        try:
            files = self.client.list_files(max_files=_MAX_LAYOUT_FILES)
        except Exception as exc:  # a broken listing must not stop planning
            return f"(file listing unavailable: {exc})"
        if not files:
            return ""
        return truncate("\n".join(str(name) for name in files), limit=_MAX_LAYOUT_CHARS)

    def _repair_prompt(self, prompt: str, raw: str, error: Exception) -> str:
        return (
            f"{prompt}\n\n"
            "Your previous reply could not be used as a plan.\n"
            f"Problem: {error}\n"
            f"Previous reply (truncated):\n{truncate(raw, limit=_MAX_ECHO_CHARS)}\n\n"
            "Reply again with ONE raw JSON object and nothing else -- no markdown fences, "
            "no prose before or after it -- in exactly this shape:\n"
            f"{PLANNER_SCHEMA}\n"
            'Use "tasks": [] with a rationale if the goal is already satisfied.'
        )

    def _parse(self, raw: str) -> Plan:
        """Parse and normalize one model reply, raising ``ValueError`` when unusable."""
        data = extract_json_object(raw)
        rationale = data.get("rationale")
        rationale = rationale.strip() if isinstance(rationale, str) else ""
        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, list):
            raise ValueError("plan JSON has no 'tasks' array")
        tasks = self._normalize(raw_tasks)
        if raw_tasks and not tasks:
            # Every task was unusable: almost certainly a format problem worth one retry.
            raise ValueError("no task in the plan had a usable description")
        return Plan(tasks=tasks, rationale=rationale)

    def _normalize(self, raw_tasks: list[Any]) -> list[TaskModel]:
        """Coerce a list of loosely-typed task dicts into validated, unique tasks.

        Descriptions are fingerprinted case-insensitively and ignoring internal
        whitespace, so "Add a  README" and "add a readme" collapse into one task instead
        of costing two worker runs.
        """
        limit = self._max_tasks()
        kept: list[TaskModel] = []
        seen_descriptions: set[str] = set()
        used_ids: set[str] = set()

        for index, entry in enumerate(raw_tasks):
            if len(kept) >= limit:
                break
            if not isinstance(entry, dict):
                continue
            description = entry.get("description")
            if not isinstance(description, str) or not description.strip():
                continue
            description = description.strip()
            fingerprint = " ".join(description.lower().split())
            if fingerprint in seen_descriptions:
                continue
            task_id = self._unique_id(entry.get("id"), index, used_ids)
            try:
                task = TaskModel(
                    id=task_id,
                    description=description,
                    files=_as_str_list(entry.get("files")),
                    dependencies=_as_str_list(entry.get("dependencies")),
                )
            except ValueError:
                # A single malformed task must not cost the whole plan.
                continue
            seen_descriptions.add(fingerprint)
            used_ids.add(task_id)
            kept.append(task)

        known = {task.id for task in kept}
        for task in kept:
            task.dependencies = [
                dep
                for dep in dict.fromkeys(task.dependencies)
                if dep != task.id and dep in known
            ]
        return kept

    @staticmethod
    def _unique_id(raw_id: Any, index: int, used_ids: set[str]) -> str:
        candidate = raw_id.strip() if isinstance(raw_id, str) else ""
        if candidate and candidate not in used_ids:
            return candidate
        # Duplicate or missing ids would make the dependency graph ambiguous.
        fallback = str(index + 1)
        suffix = 2
        while fallback in used_ids:
            fallback = f"{index + 1}-{suffix}"
            suffix += 1
        return fallback


def _as_str_list(value: Any) -> list[str]:
    """Coerce a model-supplied field into a list of non-empty strings."""
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            out.append(str(item))
    return out
