"""Helpers for turning LLM output text into Python values and pydantic models.

LLM responses are unreliable containers for JSON: they arrive wrapped in markdown code
fences, prefixed with "Here is the plan:", or followed by a chatty sign-off. The planner
and tester both used to call :func:`json.loads` directly on that raw text, which failed
on every fenced response. All parsing goes through this module instead.

Pure and dependency-light: no ``furrow`` imports, no I/O, no third-party imports beyond
pydantic (for ``BaseModel``/``TypeVar``).
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

__all__ = ["extract_json", "parse_model_json", "strip_code_fences", "truncate"]

TModel = TypeVar("TModel", bound=BaseModel)

# Opening ```/```json fence, anchored at the start of the text. The tag is optional and
# \r?\n keeps CRLF responses working.
_OPENING_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*[ \t]*\r?\n?")
# Closing fence, anchored at the end so that ``` occurring mid-prose survives.
_CLOSING_FENCE = re.compile(r"\r?\n?```[ \t]*\r?\n?\s*$")
# Every index at which a JSON object or array could begin.
_JSON_START = re.compile(r"[\{\[]")


def strip_code_fences(text: str) -> str:
    """Remove a surrounding ```/```json markdown fence, if present.

    Handles ```` ```json\\n{...}\\n``` ````, a bare ```` ``` ```` fence, an unterminated
    fence (the opening fence is dropped and the remainder returned), and unfenced text
    (returned unchanged). Both LF and CRLF line endings are accepted.
    """
    return _CLOSING_FENCE.sub("", _OPENING_FENCE.sub("", text))


def extract_json(text: str) -> Any:
    """Parse the first complete JSON value in text.

    Tolerates leading prose and trailing chatter. Raises :class:`ValueError` carrying a
    truncated snippet of the offending input (never the whole multi-thousand-token
    response) when no complete value can be decoded.
    """
    stripped = strip_code_fences(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Fall back to scanning for the first index that starts a decodable value, so that
    # "Here is the plan: {...} Hope this helps" still yields the object.
    decoder = json.JSONDecoder()
    for match in _JSON_START.finditer(stripped):
        try:
            value, _ = decoder.raw_decode(stripped, match.start())
        except json.JSONDecodeError:
            continue
        return value

    raise ValueError(f"no JSON value found in model output: {truncate(text, 500)}")


def parse_model_json(text: str, model: type[TModel]) -> TModel:
    """Validate the first JSON value in ``text`` against ``model``.

    Combines :func:`extract_json` with ``model.model_validate``. Both the JSON failure
    and any :class:`pydantic.ValidationError` (itself a :class:`ValueError`) are re-raised
    as a single :class:`ValueError` naming the expected model, chained via ``from exc``
    so the original decode/validation detail is preserved in the traceback.
    """
    try:
        return model.model_validate(extract_json(text))
    except ValidationError as exc:
        raise ValueError(f"model output did not match {model.__name__}: {_brief(exc)}") from exc
    except ValueError as exc:  # includes json.JSONDecodeError
        raise ValueError(f"could not parse {model.__name__} from model output: {exc}") from exc


def truncate(text: str, limit: int = 2000) -> str:
    """Shorten text to ``limit`` chars, appending a ``... (N chars truncated)`` marker."""
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"{text[:limit]}... ({dropped} chars truncated)"


def _brief(exc: ValidationError) -> str:
    """Render a pydantic ValidationError as a short single-line summary."""
    return truncate("; ".join(_error_lines(exc)), 500)


def _error_lines(exc: ValidationError) -> list[str]:
    """Flatten a pydantic ValidationError into one message per field error."""
    return [
        f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()
    ]
