from __future__ import annotations

import json
import re
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

        # Zero-count filtering patterns (shared with the failure-line filter
        # below so that "5 passed, 0 failed" and "FAILED (errors=0)" are not
        # read as failures).
        zero_count_re = re.compile(
            r"\b(?:0|no|none|zero)\s+(?:fail(?:ed|ure|ures)?|error(?:s)?)\b"
            r"|\b(?:fail(?:ed|ure|ures)?|error(?:s)?)\b\s*[:=]?\s*(?:0|none|zero)\b"
            r"(?=\s*[,;)]|\s*$)",
            re.IGNORECASE,
        )
        all_zero_summary_re = re.compile(
            r"\bfail(?:ed|ure|ures)?\b\s*\(\s*"
            r"(?:\w+\s*[:=]\s*(?:0|none|zero)\s*[,;]?\s*)*\)",
            re.IGNORECASE,
        )

        def _strip_zero_phrases(text: str) -> str:
            """Remove zero-count phrases so surviving keywords are real failures."""
            remainder = all_zero_summary_re.sub(" ", text)
            remainder = zero_count_re.sub(" ", remainder)
            return remainder

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
                    # Filter out zero-count / all-zero summary blocks before
                    # deciding: "FAILED (failures=0, errors=0)" is a green run.
                    filtered = _strip_zero_phrases(lower)
                    m_fail = re.search(
                        r"\b(fail|failure|failed|error|traceback|assertionerror|exception)\b",
                        filtered,
                    )
                    passed = not m_fail

        summary = s.strip()[:500]

        failure_words = re.compile(
            r"\b(fail|failure|failed|error|traceback|assertionerror|exception)\b",
            re.IGNORECASE,
        )

        def _looks_like_failure(line: str) -> bool:
            text = line.strip()
            if not text:
                return False
            if not any(k in text.lower() for k in (
                "fail", "error", "traceback", "assertion"
            )):
                return False
            remainder = _strip_zero_phrases(text)
            return any(k in remainder.lower() for k in (
                "fail", "error", "traceback", "assertion"
            ))

        failures = []
        seen = set()
        for line in s.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if failure_words.search(line_stripped) and _looks_like_failure(line_stripped):
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
