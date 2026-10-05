"""Tolerant extraction of JSON payloads from language-model output.

Models rarely emit exactly the bytes we asked for: they wrap payloads in markdown
fences, add a sentence of commentary before or after, or get cut off mid-object by the
token budget. ``json.loads`` on that raw text raises, and the exception used to escape
the orchestrator and kill an otherwise healthy run. The helpers here recover the payload
so a caller can either use it or, failing that, retry deliberately.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel

__all__ = [
    "JSONExtractionError",
    "extract_json",
    "parse_model_json",
    "strip_code_fences",
    "truncate",
]

T = TypeVar("T", bound=BaseModel)

# Bracket pairs and quote characters the scanner must respect. No regex can balance
# nested brackets while ignoring brackets inside string literals, so extraction walks the
# text one character at a time instead of pattern matching.
_BRACKET_PAIRS: dict[str, str] = {"{": "}", "[": "]"}
_QUOTES = "\"'"

# A fence only counts when it opens the text (modulo whitespace). Backticks in the middle
# of prose are content rather than structure, and mangling them would corrupt JSON that
# legitimately contains a markdown snippet inside a string value. The closing pattern is
# anchored to the end of the text so only a trailing fence is ever removed.
_FENCE_OPEN_RE = re.compile(r"\A\s*```[ \t]*[A-Za-z0-9_+#.-]*[ \t]*\r?\n?")
_FENCE_CLOSE_RE = re.compile(r"\r?\n?[ \t]*```[ \t]*\s*\Z")


class JSONExtractionError(ValueError):
    """Raised when no balanced JSON value can be recovered from model output.

    Subclasses ``ValueError`` so existing handlers that already catch loose parsing
    failures keep working. It stays a distinct type because callers branch on it: a
    retry can plausibly fix unparseable output, whereas a pydantic ``ValidationError``
    means the model returned well-formed JSON of the wrong shape and a retry would just
    reproduce the same mistake.
    """


def strip_code_fences(text: str) -> str:
    """Remove a single wrapping markdown code fence, if present.

    Handles ```json ... ```, ```JSON ... ```, ``` ... ``` and an unterminated
    fence. Returns the text unchanged when there is no fence.

    Deliberately dumb: the fence body is returned as-is even when it is not JSON, because
    deciding what the payload is belongs to :func:`extract_json`. Only a fence at the very
    start of the text is unwrapped, so a backtick sequence embedded mid-string survives
    untouched.
    """
    opening = _FENCE_OPEN_RE.match(text)
    if opening is None:
        return text
    return _FENCE_CLOSE_RE.sub("", text[opening.end() :], count=1)


def _skip_string(text: str, start: int) -> int:
    """Return the index one past the closing quote of the string literal at `start`.

    Single-quoted literals are accepted alongside double-quoted ones because models
    sometimes emit JavaScript-style quoting; the caller's parser is what ultimately
    decides whether the payload is valid JSON.
    """
    quote = text[start]
    index = start + 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == quote:
            return index + 1
        index += 1
    raise JSONExtractionError(f"unterminated string literal starting at index {start}")


def _scan_value(text: str, start: int) -> int:
    """Return the index one past the closing bracket of the container at `start`.

    Tracks nesting with an explicit stack rather than recursion so that pathological
    input cannot exhaust the Python stack, and skips whole string literals so brackets
    and escaped quotes inside them are not mistaken for structure. Raises
    :class:`JSONExtractionError` when the container never closes, which is what a
    response truncated by the token budget looks like.
    """
    if start >= len(text) or text[start] not in _BRACKET_PAIRS:
        raise JSONExtractionError(f"no JSON object or array begins at index {start}")

    stack = [_BRACKET_PAIRS[text[start]]]
    index = start + 1
    while True:
        if index >= len(text):
            raise JSONExtractionError(
                f"JSON value starting at index {start} is never closed before the end of "
                "the text; the response was probably truncated mid-object"
            )
        char = text[index]
        if char in _QUOTES:
            index = _skip_string(text, index)
            continue
        if char in _BRACKET_PAIRS:
            stack.append(_BRACKET_PAIRS[char])
            index += 1
            continue
        if char == stack[-1]:
            stack.pop()
            index += 1
            if not stack:
                return index
            continue
        index += 1


def extract_json(text: str) -> str:
    """Return the most plausible JSON value embedded in `text`.

    Locates the first ``{`` or ``[`` and returns everything up to its matching closer,
    tolerating markdown fences, leading commentary and trailing commentary.

    When several values are present the first one wins, so a model that shows an example
    payload before the real answer yields the example. The tempting alternative, picking
    whichever candidate happens to parse, is worse than it looks: a sub-span of a deeply
    nested value can parse when the value itself cannot, so "first valid candidate"
    sometimes returns a fragment rather than the payload. The cost of taking the first is
    that prose containing balanced braces before the payload (``set {env} first``) can
    shadow the real value, and that unparseable output is returned rather than rejected
    here. :func:`parse_model_json` turns both of those into a retryable
    :class:`JSONExtractionError`, which is the right place for the decision anyway.

    Raises:
        JSONExtractionError: when no balanced value can be found, including the
            truncated-response case where the brackets never balance.
    """
    body = strip_code_fences(text)
    first_failure: str | None = None
    for index, char in enumerate(body):
        if char not in _BRACKET_PAIRS:
            continue
        try:
            end = _scan_value(body, index)
        except JSONExtractionError as exc:
            if first_failure is None:
                first_failure = str(exc)
            continue
        return body[index:end]
    detail = f"; first candidate: {first_failure}" if first_failure else ""
    raise JSONExtractionError(
        f"no balanced JSON value in model output ({truncate(body, 200)!r}){detail}"
    )


def parse_model_json(text: str, model: type[T]) -> T:
    """Parse LLM output into a pydantic model, tolerating fences and prose.

    Raises:
        JSONExtractionError: when no balanced, syntactically valid JSON value is found.
        pydantic.ValidationError: when the payload does not match `model`; that
            distinction matters because callers retry only the former.
    """
    payload = extract_json(text)
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, RecursionError) as exc:
        # A malformed payload is worth one retry; a wrong-shaped one is not, which is why
        # the two failure modes below produce different exception types.
        raise JSONExtractionError(
            f"found a balanced value for {model.__name__} but it is not valid JSON: "
            f"{truncate(payload, 200)!r} ({exc})"
        ) from exc
    # ValidationError is deliberately allowed to propagate: the orchestrator treats it as
    # a permanent shape error and skips the retry that a JSONExtractionError would earn.
    return model.model_validate(data)


def truncate(text: str, limit: int = 4000) -> str:
    """Shorten `text` to `limit` characters, marking the elision.

    Used to keep huge test-suite output from blowing up an LLM prompt. The marker states
    how much was dropped, because a silently shortened log is worse than no log when
    diagnosing a failure.

    Raises:
        ValueError: when `limit` is zero or negative, where "the first `limit`
            characters" has no sensible meaning and returning an empty string would look
            like a successful, genuinely empty result.
    """
    if limit <= 0:
        raise ValueError(f"truncate() requires a positive limit, got {limit}")
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"{text[:limit]}\n... [truncated {dropped} characters] ..."
