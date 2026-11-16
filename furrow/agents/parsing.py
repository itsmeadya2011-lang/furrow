from __future__ import annotations

import json
import re
from typing import Any

from furrow.config import Plan, TaskModel, TestResult

# Compiled once at import time so parse_test_result does not recompile the
# same pattern on every call.
_FAILURE_WORDS_RE = re.compile(
    r"\b(fail|failure|failed|error|traceback|assertionerror|exception)\b",
    re.IGNORECASE,
)

# Matches a fenced code block: ```<optional-info>\n<body>```
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` / ``` ... ``` fences. If the text contains exactly one
    fenced block, return that block's inner content (fence info string dropped).
    Otherwise strip fence lines from the text and return the remainder."""
    if not isinstance(text, str):
        text = str(text)

    matches = _FENCE_RE.findall(text)
    if len(matches) == 1:
        return matches[0]
    # No single fenced block: strip fence markers wherever they appear.
    result = re.sub(r"```[^\n]*\n?", "", text)
    result = re.sub(r"```", "", result)
    return result


def extract_json(text: str) -> str:
    """Return the first balanced JSON object or array substring found in `text`.

    Scan character by character. Track depth of {} and [] together; ignore braces
    inside JSON string literals and respect backslash escapes inside strings.
    Start at the first '{' or '[' that yields a balanced, non-empty value.
    Raise ValueError('No JSON object or array found in output') when none exists."""
    if not isinstance(text, str):
        text = str(text)

    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch in "{[":
            start = i
            depth_brace = 0
            depth_bracket = 0
            in_string = False
            j = i
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
                else:
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
                            return text[start:j]
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
                            return text[start:j]
                        continue
                    j += 1
                    continue
        i += 1

    raise ValueError("No JSON object or array found in output")


def parse_json(text: str) -> Any:
    """json.loads(extract_json(text)). Raise ValueError with the JSONDecodeError text
    on malformed JSON (never let json.JSONDecodeError escape)."""
    try:
        return json.loads(extract_json(text))
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc


def parse_plan(text: str) -> Plan:
    """Parse a planner response into Plan.
    - parse the JSON
    - if it is not a mapping raise ValueError
    - accept missing 'rationale' (default '') and missing 'tasks' (default [])
    - validate tasks via TaskModel
    - on any failure raise ValueError whose message includes the underlying reason
      and a short excerpt (first 300 chars) of the original text"""
    try:
        data = parse_json(text)
        if not isinstance(data, dict):
            raise ValueError("Plan JSON must be a mapping/object")
        rationale = data.get("rationale", "")
        if rationale is None:
            rationale = ""
        tasks_data = data.get("tasks", [])
        if tasks_data is None:
            tasks_data = []
        tasks = []
        if isinstance(tasks_data, list):
            for t in tasks_data:
                if isinstance(t, dict):
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
                # No success signal: the output could not be interpreted, so
                # treat it as a failure rather than silently reporting a pass.
                passed = bool(m_success)

        summary = s.strip()[:500]

        failures = []
        seen = set()
        for line in s.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if _FAILURE_WORDS_RE.search(line_stripped):
                if line_stripped not in seen:
                    seen.add(line_stripped)
                    failures.append(line_stripped)
                if len(failures) >= 10:
                    break

        if not passed and not failures:
            failures = ["unparseable test output"]

        return TestResult(passed=passed, summary=summary, failures=failures)
    except (ValueError, TypeError):
        try:
            s = strip_code_fences(text) if isinstance(text, str) else str(text)
        except (ValueError, TypeError):
            s = str(text) if text is not None else ""
        # An unparseable tester response carries no signal: report it as a
        # failure rather than silently treating it as a pass.
        return TestResult(
            passed=False,
            summary=(s[:500] if s else "unparseable tester response"),
            failures=["unparseable tester response"],
        )
