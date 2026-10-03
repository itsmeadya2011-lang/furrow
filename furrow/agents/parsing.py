"""Tolerant parsing of LLM output into Furrow models.

Real model output is noisy: JSON arrives wrapped in ```` ```json ```` fences,
prefixed with "Here is the plan:", suffixed with commentary, or encoded twice.
Every helper here degrades gracefully instead of raising -- recoverable noise
is stripped, unknown keys are dropped, missing fields are defaulted, and only a
genuinely unusable payload raises :class:`ValueError`.

Standard library plus the models in :mod:`furrow.config`; no other dependencies.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from furrow.config import Plan, TaskModel, TaskStatus, TestResult

__all__ = [
    "MAX_EXCERPT_CHARS",
    "coerce_plan",
    "coerce_test_result",
    "excerpt",
    "extract_json_object",
    "looks_like_failure",
    "looks_like_pass",
    "parse_json_response",
    "strip_code_fences",
]

MAX_EXCERPT_CHARS = 500
MAX_DECODE_PASSES = 3

_FENCE_RE = re.compile(r"```[ \t]*\w*[ \t]*\r?\n?(.*?)(?:```|\Z)", re.DOTALL)
_OPENERS = "{["

# Explicit, unambiguous failure wording. Checked after the negated occurrences of
# the same wording have been removed, so that a response like "3 passed, 1 error"
# is never scored as a pass.
_FAILURE_SIGNALS = ("fail", "error", "traceback", "assertionerror", "not passed")
_PASS_SIGNALS = ("passed", "passes", "passing", "all tests pass", "success", "green")

# WHY the negation is applied by *removing* negated occurrences instead of
# short-circuiting the whole text: a negation only cancels the occurrence it
# actually negates. "0 errors, but 12 failures" negates the error count, not the
# failure count, so returning False for the entire message (the old behaviour)
# scored a mixed, genuinely broken run as clean. Every pattern below matches a
# negation plus the one signal it cancels, so only that occurrence is deleted;
# whatever signal survives is a real one. The leading ``\b`` and the letter-only
# fillers keep a negation word from matching inside a longer word ("nonzero
# errors", "not passed"). Both a leading negation ("no errors", "0 failed") and a
# trailing zero count ("errors: 0") are recognised, because reporters emit both.
_NEGATION_WORDS = r"(?:no|zero|without|never|0)"
_FILLERS = r"(?:[a-z]+\s+){0,2}?"
_LABELS = r"(?:\s+[a-z]+){0,2}?"
_NEGATED_FAILURE_RE = re.compile(
    rf"\b(?:{_NEGATION_WORDS}\s+{_FILLERS}(?:fail|error)\w*\b"
    rf"|(?:fail|error)\w*{_LABELS}[:=]\s*(?:0|zero)\b)"
)
_NEGATED_PASS_RE = re.compile(
    rf"\b{_NEGATION_WORDS}\s+{_FILLERS}(?:pass\w*|success\w*|green)\b"
)

_TASK_KEYS = frozenset(
    {"id", "description", "files", "dependencies", "status", "result", "error", "attempts"}
)
_TRUE_WORDS = frozenset({"true", "yes", "y", "ok", "okay", "1", "pass", "passed", "success"})
_FALSE_WORDS = frozenset({"false", "no", "n", "0", "fail", "failed", "failure", "error"})


def excerpt(text: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    """Return a single-line, length-capped view of ``text`` safe for messages."""
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "...[truncated]"


def strip_code_fences(text: str) -> str:
    """Return the body of the first markdown code fence in ``text``.

    Handles ```` ```json ```` and bare ```` ``` ```` wrappers as well as an
    unterminated fence (everything after the opening marker is the body). Text
    with no fence is returned stripped, unchanged.
    """
    match = _FENCE_RE.search(text)
    if match is None:
        return text.strip()
    return match.group(1).strip()


def _scan_balanced(text: str, start: int) -> str | None:
    """Return the balanced JSON object/array starting at ``text[start]``.

    Braces and brackets inside JSON string literals are ignored, and escapes are
    honoured, so a description containing ``}`` cannot truncate the value.
    Returns ``None`` when the candidate never balances.
    """
    opener = text[start]
    closer = "}" if opener == "{" else "]"
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


def extract_json_object(text: str) -> Any:
    """Return the first balanced JSON object or array found anywhere in ``text``.

    Prose before or after the payload is ignored. Raises :class:`ValueError`,
    carrying a truncated excerpt of the offending text, when nothing parses.
    """
    candidate_source = strip_code_fences(text)
    try:
        return json.loads(candidate_source)
    except json.JSONDecodeError:
        pass

    for index, char in enumerate(candidate_source):
        if char not in _OPENERS:
            continue
        balanced = _scan_balanced(candidate_source, index)
        if balanced is None:
            continue
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"no JSON object or array found in response: {excerpt(candidate_source)}")


def parse_json_response(text: str) -> Any:
    """Parse LLM output into Python data, tolerating fences, prose and double encoding."""
    current: str = text
    for _ in range(MAX_DECODE_PASSES):
        try:
            decoded = json.loads(current)
        except json.JSONDecodeError:
            break
        if not isinstance(decoded, str):
            return decoded
        current = decoded
    return extract_json_object(current)


def _coerce_text(value: Any) -> str:
    """Render a scalar (or sequence of scalars) as a human-readable string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Sequence):
        return ", ".join(_coerce_text(item) for item in value if item is not None)
    return str(value)


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = _coerce_text(value)
    return text or None


def _coerce_id(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = _coerce_text(value).strip()
    return text or fallback


def _coerce_bool(value: Any) -> bool | None:
    """Coerce a truthy/falsey LLM value; ``None`` means "the model said nothing"."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False
        return None
    return None


def _coerce_str_list(value: Any) -> list[str]:
    """Coerce a string, scalar, or sequence into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [text for text in (_coerce_text(item).strip() for item in value) if text]
    rendered = _coerce_text(value).strip()
    return [rendered] if rendered else []


def _coerce_status(value: Any) -> TaskStatus:
    if isinstance(value, TaskStatus):
        return value
    if isinstance(value, str):
        try:
            return TaskStatus(value.strip().lower())
        except ValueError:
            return TaskStatus.PENDING
    return TaskStatus.PENDING


def _coerce_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        return int(value.strip())
    return 0


def _coerce_task(item: Any, position: int) -> TaskModel | None:
    """Build a :class:`TaskModel` from one plan entry; ``None`` drops empty entries."""
    if item is None:
        return None
    fallback_id = str(position)
    if not isinstance(item, Mapping):
        description = _coerce_text(item).strip()
        if not description:
            return None
        return TaskModel(id=fallback_id, description=description)

    known = {key: item[key] for key in _TASK_KEYS if key in item}
    return TaskModel(
        id=_coerce_id(known.get("id"), fallback_id),
        description=_coerce_text(known.get("description")),
        files=_coerce_str_list(known.get("files")),
        dependencies=_coerce_str_list(known.get("dependencies")),
        status=_coerce_status(known.get("status")),
        result=_coerce_optional_text(known.get("result")),
        error=_coerce_optional_text(known.get("error")),
        attempts=_coerce_count(known.get("attempts")),
    )


def _coerce_tasks(value: Any) -> list[TaskModel]:
    """Coerce the ``tasks`` field of a plan payload into task models."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = [value]
    if isinstance(value, str):
        value = [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            task
            for task in (
                _coerce_task(item, position) for position, item in enumerate(value, start=1)
            )
            if task is not None
        ]
    raise ValueError(f"'tasks' must be a list of task objects, got {type(value).__name__}")


def _as_json_data(data: Any) -> Any:
    """Re-parse a payload that arrived as a raw JSON string."""
    if not isinstance(data, str):
        return data
    try:
        return parse_json_response(data)
    except ValueError:
        raise ValueError(f"payload is a string that does not contain JSON: {excerpt(data)}") from None


def coerce_plan(data: Any) -> Plan:
    """Coerce loose LLM output into a :class:`Plan`.

    Accepts a plan mapping, a bare list of tasks, a single task mapping, or any
    of those still wrapped in a JSON string. Unknown keys are dropped, ``id`` is
    coerced to ``str``, and a bare string in ``files``/``dependencies`` becomes a
    single-element list. Raises :class:`ValueError` only when the payload is
    neither a mapping nor a list.
    """
    data = _as_json_data(data)

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return Plan(tasks=_coerce_tasks(data), rationale="")

    if isinstance(data, Mapping):
        raw_tasks = data.get("tasks")
        if raw_tasks is None and "description" in data:
            return Plan(tasks=_coerce_tasks([data]), rationale="")
        return Plan(
            tasks=_coerce_tasks(raw_tasks),
            rationale=_coerce_text(data.get("rationale")),
        )

    raise ValueError(
        f"plan payload must be an object or a list of tasks, got {type(data).__name__}: "
        f"{excerpt(repr(data))}"
    )


def looks_like_failure(text: str) -> bool:
    """True when ``text`` contains a failure signal that is not itself negated.

    Negated occurrences ("no errors", "0 failed", "without failures") are removed
    first, so a message that negates one count and reports another failure is
    still a failure.
    """
    lowered = _NEGATED_FAILURE_RE.sub(" ", text.lower())
    return any(signal in lowered for signal in _FAILURE_SIGNALS)


def looks_like_pass(text: str) -> bool:
    """True when ``text`` contains a success signal that is not itself negated.

    Uses the same per-occurrence negation rule as :func:`looks_like_failure`, so
    "no tests passed" is not mistaken for a pass.
    """
    lowered = _NEGATED_PASS_RE.sub(" ", text.lower())
    return any(signal in lowered for signal in _PASS_SIGNALS)


def coerce_test_result(data: Any) -> TestResult:
    """Coerce loose LLM output into a :class:`TestResult`.

    A plain string is treated as the summary, and the verdict is then derived:
    an explicit failure signal in the text wins over a positive signal, so
    "3 passed, 1 error" is never scored as a pass, and a negation cancels only
    the count it negates, so "0 errors, but 12 failures" fails too. Raises
    :class:`ValueError` only when the payload carries no usable verdict.
    """
    if isinstance(data, str):
        try:
            data = _as_json_data(data)
        except ValueError:
            return _test_result_from_mapping({"summary": data})
    if isinstance(data, str):
        return _test_result_from_mapping({"summary": data})
    if isinstance(data, Mapping):
        return _test_result_from_mapping(data)
    raise ValueError(
        f"test result payload must be an object, got {type(data).__name__}: {excerpt(repr(data))}"
    )


def _test_result_from_mapping(data: Mapping[str, Any]) -> TestResult:
    summary = _coerce_text(data.get("summary") or data.get("detail") or data.get("reason"))
    raw_failures = data.get("failures")
    if raw_failures is None and "errors" in data:
        raw_failures = data["errors"]
    failures = _coerce_str_list(raw_failures)
    passed = _coerce_bool(data.get("passed") if "passed" in data else data.get("ok"))
    if passed is None:
        # Silence is not success: an empty summary with no stated verdict fails.
        passed = bool(summary.strip()) and not failures and not looks_like_failure(summary)
    return TestResult(passed=passed, summary=summary, failures=failures)
