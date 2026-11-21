from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from furrow.config import Plan, TaskModel, TestResult


def strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` / ``` ... ``` fences. If the text contains exactly one
    fenced block, return that block's inner content (fence info string dropped).
    Otherwise strip fence lines from the text and return the remainder."""
    if not isinstance(text, str):
        text = str(text)

    fence_pattern = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
    matches = fence_pattern.findall(text)
    if len(matches) == 1:
        return matches[0]
    result = re.sub(r"```[^\n]*\n", "", text)
    result = re.sub(r"\n```[ \t]*$", "", result)
    result = re.sub(r"^```[^\n]*\n", "", result)
    result = re.sub(r"```[ \t]*$", "", result)
    return result


def _iter_json_candidates(text: str) -> Iterator[str]:
    """Yield every balanced ``{...}`` / ``[...]`` substring in *text*, in order.

    A candidate starts at a ``{`` or ``[`` and ends at the matching ``}`` /
    ``]`` with all nested braces and brackets balanced; braces inside JSON
    string literals are ignored. Scanning resumes one character after each
    yielded candidate, so a valid object nested inside an invalid outer object
    is still found.
    """
    if not isinstance(text, str):
        text = str(text)

    n = len(text)
    i = 0
    while i < n:
        if text[i] not in "{[":
            i += 1
            continue
        start = i
        depth_brace = 0
        depth_bracket = 0
        in_string = False
        j = i
        found: str | None = None
        while j < n:
            c = text[j]
            if in_string:
                if c == "\\":
                    j += 2
                    continue
                if c == '"':
                    in_string = False
                j += 1
                continue
            if c == '"':
                in_string = True
                j += 1
                continue
            if c == "{":
                depth_brace += 1
                j += 1
                continue
            if c == "}":
                depth_brace -= 1
                j += 1
                if (
                    depth_brace == 0
                    and depth_bracket == 0
                    and text[start] == "{"
                    and text[j - 1] == "}"
                ):
                    found = text[start:j]
                    break
                continue
            if c == "[":
                depth_bracket += 1
                j += 1
                continue
            if c == "]":
                depth_bracket -= 1
                j += 1
                if (
                    depth_brace == 0
                    and depth_bracket == 0
                    and text[start] == "["
                    and text[j - 1] == "]"
                ):
                    found = text[start:j]
                    break
                continue
            j += 1
        if found is not None:
            yield found
            i = start + 1
        else:
            i += 1


def extract_json(text: str) -> str:
    """Return the first balanced JSON object or array substring found in *text*.

    Braces inside JSON string literals are ignored. The first balanced ``{...}``
    / ``[...]`` group is returned even when it is not valid JSON; use
    :func:`parse_json` when the first *parseable* value is needed instead.
    """
    if not isinstance(text, str):
        text = str(text)

    for candidate in _iter_json_candidates(text):
        return candidate
    raise ValueError("No JSON object or array found in output")


def parse_json(text: str) -> Any:
    """Return the first candidate in *text* that parses as JSON.

    Every balanced ``{...}`` / ``[...]`` substring is tried in order, so a stray
    bracket or a leading array in the surrounding prose no longer hides a valid
    object. Raises :class:`ValueError` (never ``json.JSONDecodeError``) when no
    candidate parses or when there is no candidate at all.
    """
    if not isinstance(text, str):
        text = str(text)

    first_error: str | None = None
    for candidate in _iter_json_candidates(text):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, RecursionError) as exc:
            if first_error is None:
                first_error = str(exc)
    if first_error is not None:
        raise ValueError(first_error)
    raise ValueError("No JSON object or array found in output")


def _first_json_mapping(text: str) -> dict[str, Any]:
    """Return the first candidate in *text* that parses to a JSON object.

    Non-object candidates (arrays, scalars) are skipped so a leading array in
    the prose cannot hide the plan object. Raises :class:`ValueError` when
    nothing parses, or when the parsed values are not objects.
    """
    if not isinstance(text, str):
        text = str(text)

    parsed_any = False
    for candidate in _iter_json_candidates(text):
        try:
            data = json.loads(candidate)
        except (ValueError, RecursionError):
            continue
        parsed_any = True
        if isinstance(data, dict):
            return data
    if parsed_any:
        raise ValueError("Plan JSON must be a mapping/object")
    raise ValueError("No JSON object or array found in output")


def parse_plan(text: str) -> Plan:
    """Parse a planner response into Plan.
    - parse the JSON
    - if it is not a mapping raise ValueError
    - accept missing 'rationale' (default '') and missing 'tasks' (default [])
    - validate tasks via TaskModel
    - on any failure raise ValueError whose message includes the underlying reason
      and a short excerpt (first 300 chars) of the original text"""
    try:
        data = _first_json_mapping(text)
        rationale = data.get("rationale", "")
        if rationale is None:
            rationale = ""
        tasks_data = data.get("tasks", [])
        if tasks_data is None:
            tasks_data = []
        tasks = []
        if isinstance(tasks_data, list):
            seen_ids: set[str] = set()
            for t in tasks_data:
                if isinstance(t, dict):
                    task_id = t.get("id")
                    if not isinstance(task_id, str):
                        raise ValueError("Each task must have a string id")
                    if task_id in seen_ids:
                        raise ValueError(f"duplicate task id: {task_id!r}")
                    seen_ids.add(task_id)
                    tasks.append(TaskModel(**t))
                else:
                    raise ValueError("Each task must be an object")
        else:
            raise ValueError("tasks must be a list")
        return Plan(tasks=tasks, rationale=rationale)
    except (ValueError, TypeError) as e:
        excerpt = text[:300] if isinstance(text, str) else str(text)[:300]
        raise ValueError(f"{e}\nExcerpt: {excerpt}") from e


def parse_test_result(text: str) -> TestResult:
    """Parse a tester response into TestResult. NEVER raise."""
    try:
        try:
            data = parse_json(text)
            if isinstance(data, dict):
                return TestResult(**data)
        except (ValueError, TypeError):
            pass

        s = strip_code_fences(text) if isinstance(text, str) else str(text)
        lower = s.lower()

        passed = True
        m = re.search(r'"passed"\s*:\s*(true|false)', lower)
        if m:
            passed = m.group(1) == "true"
        else:
            m_true = re.search(r"\bpassed\b\s*[:=]\s*(true|yes|success)\b", lower)
            m_false = re.search(r"\b(failed|failure)\b\s*[:=]\s*(true|yes)\b", lower)
            if m_false:
                passed = False
            elif m_true:
                passed = True
            else:
                m_success = re.search(
                    r"\b(no failures?|0 failed|all tests passed|tests passed|success)\b",
                    lower,
                )
                if m_success:
                    passed = True
                else:
                    m_fail = re.search(
                        r"\b(fail|failure|failed|error|traceback|assertionerror|exception)\b",
                        lower,
                    )
                    passed = not m_fail

        summary = s.strip()[:500]

        failure_words = re.compile(
            r"\b(fail|failure|failed|error|traceback|assertionerror|exception)\b",
            re.IGNORECASE,
        )
        failures = []
        seen = set()
        for line in s.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if failure_words.search(line_stripped):
                if line_stripped not in seen:
                    seen.add(line_stripped)
                    failures.append(line_stripped)
                if len(failures) >= 10:
                    break

        return TestResult(passed=passed, summary=summary, failures=failures)
    except (ValueError, TypeError):
        try:
            s = strip_code_fences(text) if isinstance(text, str) else str(text)
        except (ValueError, TypeError):
            s = str(text) if text is not None else ""
        return TestResult(passed=True, summary=s[:500], failures=[])
