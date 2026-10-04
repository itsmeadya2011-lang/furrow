"""Tolerant parsing helpers shared by the Furrow agents.

Language models are asked for JSON but do not reliably deliver it: they wrap the
payload in a markdown fence, prefix it with "Sure, here is the plan:", or invent
status names such as ``"done"``. Every agent needs the same three steps — unwrap
the fence, isolate the object, validate it against a model — and none of them
should re-implement them. The fourth helper, :func:`coerce_status`, absorbs
free-form status strings into the :class:`~furrow.config.TaskStatus` enum so a
scheduler never trips over an unknown status value.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypeVar

from pydantic import ValidationError

from furrow.config import TaskStatus

if TYPE_CHECKING:
    from pydantic import BaseModel

__all__ = [
    "coerce_status",
    "extract_json",
    "parse_json_response",
    "strip_code_fence",
]

ModelT = TypeVar("ModelT", bound="BaseModel")

_FENCE = "```"

_STATUS_KEY = "status"

_STATUS_ALIASES: dict[str, TaskStatus] = {
    "pending": TaskStatus.PENDING,
    "todo": TaskStatus.PENDING,
    "new": TaskStatus.PENDING,
    "planned": TaskStatus.PENDING,
    "queued": TaskStatus.PENDING,
    "not_started": TaskStatus.PENDING,
    "running": TaskStatus.RUNNING,
    "in_progress": TaskStatus.RUNNING,
    "in-progress": TaskStatus.RUNNING,
    "active": TaskStatus.RUNNING,
    "started": TaskStatus.RUNNING,
    "completed": TaskStatus.COMPLETED,
    "complete": TaskStatus.COMPLETED,
    "done": TaskStatus.COMPLETED,
    "success": TaskStatus.COMPLETED,
    "succeeded": TaskStatus.COMPLETED,
    "passed": TaskStatus.COMPLETED,
    "ok": TaskStatus.COMPLETED,
    "failed": TaskStatus.FAILED,
    "failure": TaskStatus.FAILED,
    "error": TaskStatus.FAILED,
    "errored": TaskStatus.FAILED,
    "broken": TaskStatus.FAILED,
    "failed_with_errors": TaskStatus.FAILED,
    "skipped": TaskStatus.SKIPPED,
    "skip": TaskStatus.SKIPPED,
    "cancelled": TaskStatus.SKIPPED,
    "canceled": TaskStatus.SKIPPED,
    "blocked": TaskStatus.SKIPPED,
}


def strip_code_fence(text: str) -> str:
    """Remove a surrounding markdown code fence, if one is present.

    Args:
        text: Raw text that may or may not be fenced.

    Returns:
        The trimmed payload without the opening ```` ```json ```` line or the
        closing ```` ``` ```` line. Unfenced text is returned unchanged apart
        from surrounding whitespace.
    """
    stripped = text.strip()
    if not stripped.startswith(_FENCE):
        return stripped
    lines = stripped.splitlines()
    lines = lines[1:]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip().startswith(_FENCE):
        lines.pop()
    return "\n".join(lines).strip()


def extract_json(text: str) -> str:
    """Return the JSON object substring of a possibly chatty response.

    The fence is removed first, then the payload is taken as everything from
    the first ``{`` to the last ``}``, which survives both leading prose and
    trailing commentary such as "Let me know if you want changes!".

    Args:
        text: Raw text returned by the model.

    Returns:
        The candidate JSON object substring.

    Raises:
        ValueError: If no opening or closing brace is present, so there is
            nothing that could be a JSON object.
    """
    candidate = strip_code_fence(text)
    start = candidate.find("{")
    if start < 0:
        raise ValueError("response contains no '{' and therefore no JSON object")
    end = candidate.rfind("}")
    if end < start:
        raise ValueError("response contains '{' but no closing '}'")
    return candidate[start : end + 1]


def parse_json_response(text: str, model_cls: type[ModelT], context: str) -> ModelT:
    """Parse and validate an LLM response into ``model_cls``.

    Args:
        text: Raw text returned by the model, fenced or not.
        model_cls: The pydantic model to validate the decoded payload against.
        context: Short description of what was being parsed, included in the
            error message so a caller can tell which stage failed.

    Returns:
        A validated instance of ``model_cls``. Any ``status`` field in the
        payload is passed through :func:`coerce_status` first, because
        pydantic would otherwise reject the synonyms models actually emit.

        Raises:
            ValueError: If the text holds no JSON object, holds malformed JSON, is
            not a JSON object, or fails validation. The message carries
            ``context`` and the raw text so the failure can be logged or fed
            back to the model for a repair attempt.
    """
    try:
        data = json.loads(extract_json(text))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"could not decode JSON for {context}: {exc}; raw response: {text!r}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"expected a JSON object for {context} but decoded "
            f"{type(data).__name__}; raw response: {text!r}"
        )
    try:
        return model_cls.model_validate(_normalise_statuses(data))
    except ValidationError as exc:
        raise ValueError(
            f"invalid payload for {context}: {exc}; raw response: {text!r}"
        ) from exc


def _normalise_statuses(payload: object) -> object:
    """Rewrite every ``status`` string in a decoded payload via coercion.

    The recursion covers nested lists and objects so a plan's task statuses are
    reached without the caller knowing the payload shape. Values that are not
    strings are left alone, and only the key ``status`` is touched.

    Args:
        payload: A decoded JSON value.

    Returns:
        An equivalent value whose ``status`` entries are canonical
        :class:`~furrow.config.TaskStatus` values.
    """
    if isinstance(payload, dict):
        return {key: _normalise_field(key, item) for key, item in payload.items()}
    if isinstance(payload, list):
        return [_normalise_statuses(item) for item in payload]
    return payload


def _normalise_field(key: str, value: object) -> object:
    """Coerce one decoded ``key``/``value`` pair.

    Args:
        key: The JSON key the value was stored under.
        value: The decoded value.

    Returns:
        The canonical status for a ``status`` string, otherwise the value with
        its own contents normalised.
    """
    if key == _STATUS_KEY and isinstance(value, str):
        return coerce_status(value).value
    return _normalise_statuses(value)


def coerce_status(value: object) -> TaskStatus:
    """Map a free-form status onto :class:`~furrow.config.TaskStatus`.

    Models routinely answer ``"done"``, ``"complete"`` or ``"in progress"``
    where the enum expects ``"completed"``, ``"completed"`` or ``"running"``.
    Refusing such a value would abort a run over a synonym, so anything
    unrecognised degrades to ``TaskStatus.PENDING``.

    Args:
        value: Any status value: a real enum member, a string, ``None`` or an
            arbitrary object.

    Returns:
        The matching :class:`~furrow.config.TaskStatus`, defaulting to
        :attr:`~furrow.config.TaskStatus.PENDING` when nothing matches.
    """
    if isinstance(value, TaskStatus):
        return value
    if isinstance(value, str):
        normalised = value.strip().lower().replace(" ", "_").replace("-", "_")
        if normalised in _STATUS_ALIASES:
            return _STATUS_ALIASES[normalised]
    return TaskStatus.PENDING
