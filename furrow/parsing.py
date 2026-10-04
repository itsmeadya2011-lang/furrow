"""Robust extraction of JSON payloads from LLM text.

Models routinely wrap JSON in markdown fences or surround it with prose, so a
bare ``json.loads(response)`` is not enough. The helpers here recover the payload
in several escalating steps and always fail with
:class:`~furrow.errors.StructuredOutputError` rather than a bare ``ValueError``.

This module intentionally depends only on ``pydantic`` and ``furrow.errors`` so
it can be imported from anywhere without creating import cycles.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from furrow.errors import StructuredOutputError

T = TypeVar("T", bound=BaseModel)

MAX_RAW_ECHO = 500
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)
_OPENERS = "{["
_CLOSERS = "]}"


def strip_code_fences(text: str) -> str:
    """Return the contents of the first markdown code fence, or text unchanged."""
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def _find_balanced(text: str) -> str | None:
    """Return the first balanced ``{...}``/``[...]`` span in ``text``, or None.

    String literals and backslash escapes are skipped so that braces inside JSON
    strings do not throw off the depth counter.
    """
    start = -1
    depth = 0
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
        elif char in _OPENERS:
            if depth == 0:
                start = index
            depth += 1
        elif char in _CLOSERS and depth > 0:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json(text: str) -> str:
    """Return the JSON payload contained in ``text``.

    Attempts the whole stripped text, then the first fenced block, then the first
    balanced brace/bracket span, returning the first candidate that ``json.loads``
    accepts. Raises :class:`StructuredOutputError` when none of them do.
    """
    candidates: list[str] = [text.strip(), strip_code_fences(text)]
    balanced = _find_balanced(text)
    if balanced is not None:
        candidates.append(balanced)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            json.loads(candidate)
        except ValueError:
            continue
        return candidate

    raise StructuredOutputError("Could not extract JSON from model response", raw=text)


def parse_json(text: str) -> Any:
    """Extract and ``json.loads`` a JSON payload, raising on failure."""
    return json.loads(extract_json(text))


def parse_model(text: str, model: type[T]) -> T:
    """Extract JSON from ``text`` and validate it into the pydantic model ``model``.

    Raises :class:`StructuredOutputError` if no JSON can be extracted, or if the
    payload does not satisfy ``model``.
    """
    payload = parse_json(text)
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        summary = str(exc)[:MAX_RAW_ECHO]
        raise StructuredOutputError(
            f"Model response did not match {model.__name__}: {summary}",
            raw=text,
        ) from exc
