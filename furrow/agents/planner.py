from __future__ import annotations

import inspect
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from collections.abc import Iterable

    from furrow.config import Settings

logger = logging.getLogger(__name__)

# The prompt asks for 1-5 tasks; this is only a guard against runaway output.
MAX_TASKS = 10
# Bounded project listing so the planner sees the real tree but the prompt stays small.
MAX_CONTEXT_FILES = 200
MAX_CONTEXT_CHARS = 8_000
# Snippet sizes used when reporting/repairing a malformed model response.
ERROR_SNIPPET_CHARS = 500
REPAIR_SNIPPET_CHARS = 1_500

_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\r?\n?(.*?)```", re.DOTALL)


def _outermost_span(text: str, opener: str, closer: str) -> str | None:
    """Return the outermost balanced ``opener...closer`` span, or ``None``.

    Brackets inside JSON string literals are ignored, so a description
    containing ``"}"`` cannot truncate the payload.
    """
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json(text: str) -> str:
    """Pull the JSON payload out of a raw LLM response.

    Strategy, in order:

    1. Look for a ```json / ``` fence anywhere in the response and take its body.
       This handles the very common case of prose or commentary *before* the
       fence ("Here is the plan: ```json {...} ```").
    2. Otherwise, if the response begins with ``[``, scan for the outermost
       balanced ``[...]`` region so a top-level array of tasks is not
       truncated to its first element.
    3. Otherwise scan for the outermost balanced ``{...}`` region, ignoring
       braces inside string literals, which handles a bare object with
       trailing prose. If that object sits directly inside ``[...]`` (a bare
       array of tasks after prose, e.g. "Here is the plan: [{...}]"),
       return the whole array so the list-shaped plan survives.
    4. Otherwise return the text unchanged, so ``json.loads`` raises the error
       that gets reported back to the model during the repair retry.
    """
    match = _FENCE_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            return candidate
    if text.strip().startswith("["):
        candidate = _outermost_span(text, "[", "]")
        if candidate:
            return candidate
    candidate = _outermost_span(text, "{", "}")
    if candidate:
        prefix = text[: text.find(candidate)].rstrip()
        if prefix.endswith("["):
            array = _outermost_span(text[len(prefix) - 1 :], "[", "]")
            if array is not None:
                return array
        return candidate
    return text.strip()


def _snippet(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"


def _coerce_str_list(value: Any) -> list[str]:
    """Coerce a model-supplied value into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        parts = [part for part in parts if part]
        return parts or ([value.strip()] if value.strip() else [])
    if isinstance(value, (list, tuple, set)):
        return [text for text in (str(item).strip() for item in value) if text]
    text = str(value).strip()
    return [text] if text else []


def _normalise_task(raw: Any, index: int) -> dict[str, Any] | None:
    """Turn one model-supplied task into a ``TaskModel``-safe dict of known fields.

    Unknown keys are dropped, execution state (``status``/``result``) is discarded
    so a hallucinated status cannot leak into the run, and missing/odd values are
    replaced with sane defaults instead of failing validation. Non-object entries
    (e.g. ``"N/A"``) are logged and dropped.
    """
    if not isinstance(raw, dict):
        logger.warning(
            "Skipping planner task #%d: expected a JSON object, got %s",
            index + 1,
            type(raw).__name__,
        )
        return None

    identifier = raw.get("id")
    identifier = str(identifier).strip() if identifier is not None else ""
    identifier = identifier or str(index + 1)

    description = (
        raw.get("description") or raw.get("task") or raw.get("summary") or raw.get("title")
    )
    description = str(description).strip() if description is not None else ""
    if not description:
        description = f"Task {identifier}: infer intent from the goal"
        logger.warning("Planner task %s had no description; using a placeholder.", identifier)

    return {
        "id": identifier,
        "description": description,
        "files": _coerce_str_list(raw.get("files")),
        "dependencies": _coerce_str_list(raw.get("dependencies")),
    }


def _normalise_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce a decoded plan payload into valid ``Plan`` kwargs."""
    raw_tasks = payload.get("tasks")
    if (
        raw_tasks is None
        and "tasks" not in payload
        and payload
        and all(isinstance(value, dict) for value in payload.values())
    ):
        # Accept a bare {"1": {...}, "2": {...}} mapping shape at the top level.
        raw_tasks = [payload[key] for key in sorted(payload, key=str)]
    if isinstance(raw_tasks, dict):
        # Accept a {"1": {...}, "2": {...}} mapping shape.
        raw_tasks = [raw_tasks[key] for key in sorted(raw_tasks, key=str)]
    if raw_tasks is None:
        raw_tasks = []
    elif not isinstance(raw_tasks, list):
        raw_tasks = [raw_tasks]

    if len(raw_tasks) > MAX_TASKS:
        logger.warning(
            "Planner returned %d tasks; clamping to the first %d.", len(raw_tasks), MAX_TASKS
        )

    tasks: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_tasks[:MAX_TASKS]):
        try:
            task = _normalise_task(raw, index)
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("Discarding unparsable planner task #%d: %s", index + 1, exc)
            continue
        if task is not None:
            tasks.append(task)

    rationale = payload.get("rationale") or payload.get("explanation") or payload.get("reason")
    rationale = str(rationale).strip() if rationale is not None else ""
    if not rationale:
        logger.warning("Planner response had no rationale; defaulting to an empty string.")

    return {"tasks": tasks, "rationale": rationale or "(planner provided no rationale)"}


def parse_plan(text: str) -> Plan:
    """Extract, decode and normalise a plan from a raw LLM response.

    Raises ``ValueError`` when the response cannot be turned into a valid plan.
    """
    payload = json.loads(extract_json(text))
    if isinstance(payload, list):
        # Tolerate a bare array of tasks.
        payload = {"tasks": payload}
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object at the top level, got {type(payload).__name__}")
    return Plan(**_normalise_plan(payload))


def _flatten_context(value: Any) -> str:
    """Flatten failure/extra context (a string, or a list of strings) into text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        return "\n".join(f"- {item}" for item in value if str(item).strip())
    return str(value).strip()


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        # NOTE: relies on the sibling contract that LLMClient.__init__ accepts
        # settings=None and falls back to the global settings. If that regresses,
        # pass an explicit Settings() here.
        self.client = client or LLMClient(settings=settings)

    async def _project_context(self) -> str:
        """Bounded listing of the project tree, so plans reference real files."""
        try:
            root = Path(self.client.settings.workspace)
            listing = self.client.list_files(root)
            # list_files may be sync or async depending on the client version.
            if inspect.isawaitable(listing):
                listing = await listing
        except Exception as exc:  # filesystem/permission errors must not kill the plan
            logger.warning("No project context available for the planner: %s", exc)
            return ""

        try:
            all_paths = [str(path) for path in (listing or [])]
        except TypeError as exc:
            logger.warning("Ignoring unusable project listing from client: %s", exc)
            return ""

        if not all_paths:
            return ""
        paths = all_paths[:MAX_CONTEXT_FILES]
        body = "\n".join(f"- {path}" for path in paths)
        if len(body) > MAX_CONTEXT_CHARS:
            body = body[:MAX_CONTEXT_CHARS] + "\n... [listing truncated]"
        if len(all_paths) > MAX_CONTEXT_FILES:
            body += f"\n... [{len(all_paths) - MAX_CONTEXT_FILES} more files omitted]"
        return f"Project root: {root}\nExisting files:\n{body}"

    async def plan(
        self,
        goal: str,
        failure_context: str | Iterable[str] | None = None,
        **extra_context: Any,
    ) -> Plan:
        """Produce a plan for ``goal``.

        ``failure_context`` is optional prior-cycle context (test failures, prior
        outcomes) as a string or list of strings; ``plan(goal)`` still works.
        Any other keyword argument is accepted and appended as labelled context so
        the orchestrator can pass extra failure information without a signature
        change.
        """
        sections = [f"Goal: {goal}"]
        project_context = await self._project_context()
        if project_context:
            sections.append(project_context)

        blocks: list[str] = []
        flattened = _flatten_context(failure_context)
        if flattened:
            blocks.append(f"Previous cycle context (test failures / prior outcomes):\n{flattened}")
        for key, value in extra_context.items():
            flattened = _flatten_context(value)
            if flattened:
                blocks.append(f"{key.replace('_', ' ')}:\n{flattened}")
        if blocks:
            sections.append("\n\n".join(blocks))

        prompt = "\n\n".join([PLANNER_PROMPT, *sections]) + "\n"
        model = self.client.settings.planner_model

        response = await self.client.complete(prompt, model=model)
        try:
            return parse_plan(response)
        except (ValueError, TypeError) as exc:
            # `exc` is unbound once the except block exits, so keep the text.
            initial_error = str(exc) or type(exc).__name__
            logger.warning(
                "Planner response was not valid plan JSON (%s); requesting a repair.",
                initial_error,
            )

        repair_prompt = (
            f"{prompt}\n"
            "Your previous reply could not be parsed:\n"
            f"---\n{_snippet(response, REPAIR_SNIPPET_CHARS)}\n---\n"
            "Reply with ONLY the raw JSON object matching the schema above: no markdown "
            'fences, no prose, no commentary. Start with "{" and end with "}".'
        )
        repaired = await self.client.complete(repair_prompt, model=model)
        try:
            plan_obj = parse_plan(repaired)
        except (ValueError, TypeError) as second_error:
            logger.error("Planner repair attempt also failed: %s", second_error)
            raise ValueError(
                "Planner failed to return valid plan JSON after one repair attempt. "
                f"Initial error: {initial_error}. Repair error: {second_error}. "
                "Bad response (first "
                f"{ERROR_SNIPPET_CHARS} chars): "
                f"{_snippet(repaired or response, ERROR_SNIPPET_CHARS)!r}"
            ) from second_error

        logger.info("Planner repair attempt succeeded.")
        return plan_obj
