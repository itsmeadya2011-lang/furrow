"""Planning agent plus the JSON extraction helpers shared with the tester agent."""

from __future__ import annotations

import json
import re
import warnings
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.config import settings as default_settings
from furrow.llm import LLMClient, LLMResponseError

if TYPE_CHECKING:
    from furrow.config import Settings

#: Longest fragment of an LLM response echoed back inside an exception message.
#: Keeps a runaway (or secret-laden) completion out of the logs.
MAX_ECHO_CHARS = 500

# A fenced block, with or without an info string: ```json ... ``` or ``` ... ```.
_FENCE_RE = re.compile(r"```[^\n`]*\r?\n(?P<body>.*?)```", re.DOTALL)

#: Re-exported so ``from furrow.agents.planner import LLMResponseError`` keeps
#: working. This used to be a *second*, unrelated ``ValueError`` subclass defined
#: here, so ``except furrow.llm.LLMResponseError`` did not catch a planner parse
#: failure. There is now exactly one definition, in :mod:`furrow.llm`, and it
#: subclasses both ``LLMError``/``RuntimeError`` and ``ValueError``.
__all__ = [
    "MAX_ECHO_CHARS",
    "PlannerAgent",
    "extract_json_object",
    "strip_code_fences",
    "truncate_for_log",
]


def truncate_for_log(text: str, limit: int = MAX_ECHO_CHARS) -> str:
    """Return at most the first ``limit`` characters of ``text``, marked when cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text)} chars total]"


def strip_code_fences(text: str) -> str:
    """Return the body of the first markdown code fence in ``text``, else ``text``."""
    match = _FENCE_RE.search(text)
    if match is None:
        return text
    return match.group("body").strip()


def _outermost_object_span(text: str) -> tuple[int, int] | None:
    """Locate the first balanced ``{...}`` span, ignoring braces inside strings."""
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
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
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                return start, index + 1
    return None


def extract_json_object(response: str, *, context: str = "LLM response") -> dict[str, Any]:
    """Parse a JSON object out of a possibly decorated LLM response.

    Handles the shapes models actually emit: a bare object, an object wrapped in a
    markdown code fence, and an object surrounded by prose. Anything else raises
    :class:`LLMResponseError` - including valid JSON that is not an object (an
    array, a string, a number), which used to escape as a bare ``TypeError`` from
    ``Plan(**data)``.

    Raises:
        LLMResponseError: The response contains no JSON object, or contains JSON
            of some other type.
    """
    candidates: list[str] = []
    for candidate in (strip_code_fences(response).strip(), response.strip()):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        raise LLMResponseError(f"{context} was empty")

    # Valid JSON that is not an object, remembered so the error can say so precisely.
    non_object: tuple[str, str] | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except ValueError:  # json.JSONDecodeError is a subclass of ValueError
            continue
        if isinstance(data, dict):
            return data
        non_object = (candidate, type(data).__name__)

    # Fallback: pull the outermost {...} out of a fenced/prose-wrapped response.
    for candidate in candidates:
        span = _outermost_object_span(candidate)
        if span is None:
            continue
        snippet = candidate[span[0] : span[1]].strip()
        try:
            data = json.loads(snippet)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data

    if non_object is not None:
        candidate, kind = non_object
        raise LLMResponseError(
            f"{context} is valid JSON of type {kind}, expected a JSON object: "
            f"{truncate_for_log(candidate)}"
        )
    raise LLMResponseError(
        f"{context} does not contain a JSON object: {truncate_for_log(response)}"
    )


class PlannerAgent:
    """Turns a goal into a :class:`~furrow.config.Plan` using an LLM.

    Args:
        client: Injected LLM client. When omitted a default one is built.
        settings: Configuration used to build the default client. Ignored - with
            a warning - when it conflicts with ``client.settings``; the explicitly
            passed ``client`` always wins, as it did before.
    """

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        if client is None:
            # LLMClient() (no argument) resolves to the global settings singleton;
            # passing settings=None explicitly used to store None and break later.
            self.client = LLMClient() if settings is None else LLMClient(settings=settings)
        else:
            client_settings = getattr(client, "settings", None)
            if (
                settings is not None
                and client_settings is not None
                and settings is not client_settings
            ):
                warnings.warn(
                    "PlannerAgent got both client= and settings=; the client's own settings "
                    "take precedence and the settings argument is ignored.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.client = client
        resolved = getattr(self.client, "settings", None)
        if resolved is None:
            resolved = settings if settings is not None else default_settings
        self.settings: Settings = resolved

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.settings.planner_model)
        # extract_json_object already raises a truncated, typed LLMResponseError.
        data = extract_json_object(response, context="Planner response")
        try:
            return Plan(**data)
        except (ValueError, TypeError) as e:
            raise LLMResponseError(
                f"Failed to build plan from LLM JSON ({e}); "
                f"top-level keys were: {sorted(data)[:20]}"
            ) from e