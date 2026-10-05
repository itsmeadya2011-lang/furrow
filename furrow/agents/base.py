"""Shared plumbing for the Furrow agents.

The agents used to duplicate their constructor and their ``complete()`` call, which
made them impossible to construct without a live ``LLMClient`` and left model selection
copy-pasted three ways.  Everything common now lives here:

* :func:`extract_json` unwraps the markdown fences and prose that LLMs wrap JSON in.
* :class:`SupportsComplete` is the minimal LLM surface an agent needs, so tests can
  inject a fake client.
* :class:`BaseAgent` resolves settings and the model to use, and sends the agent
  persona as the *system* message instead of burying it in the user message.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from furrow.config import Settings
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

_FENCE = "```"


def _strip_code_fence(text: str) -> str:
    """Remove a leading ``` / ```json fence and its matching trailing ```."""
    if not text.startswith(_FENCE):
        return text
    first_newline = text.find("\n")
    if first_newline == -1:
        return ""
    body = text[first_newline + 1 :]
    stripped = body.rstrip()
    if stripped.endswith(_FENCE):
        stripped = stripped[: -len(_FENCE)]
    return stripped.strip()


_CLOSING = {"{": "}", "[": "]"}


def _find_json_span(text: str) -> str | None:
    """Return the first balanced ``{...}`` or ``[...]`` span in ``text``.

    The scan anchors on the leftmost ``{`` or ``[`` and tracks nesting with a stack
    of expected closing brackets, so a top-level array of objects comes back whole
    instead of being truncated to its first element.  JSON string state and escapes
    are honoured throughout: a ``{``, ``[`` or ``"`` inside a string value never
    affects nesting.  ``None`` is returned when no span closes.
    """
    start = -1
    for index, char in enumerate(text):
        if char in _CLOSING:
            start = index
            break
    if start == -1:
        return None
    stack: list[str] = []
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
        elif char in _CLOSING:
            stack.append(_CLOSING[char])
        elif char in ("}", "]"):
            if not stack or stack.pop() != char:
                return None
            if not stack:
                return text[start : index + 1]
    return None


def extract_json(text: str) -> str:
    """Strip markdown fences and prose wrappers from an LLM JSON response.

    Returns the JSON payload on its own, whether that payload is a top-level object
    (``{...}``) or a top-level array (``[...]``) of objects.  Empty/``None`` input and
    input without a balanced span are returned unchanged, so the caller sees the raw
    response and can report it as unparseable.
    """
    if not text:
        return text
    candidate = _strip_code_fence(text.strip())
    return _find_json_span(candidate) or candidate


@runtime_checkable
class SupportsComplete(Protocol):
    """The LLM surface the agents depend on, so a fake client can be injected."""

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str: ...


class BaseAgent:
    """Base class for the planner, worker and tester agents.

    Subclasses set :attr:`model_attr` to the name of the settings field that holds
    their model, pass their persona as ``system`` and only the variable part of the
    request (goal, task, test output) as the user prompt.
    """

    model_attr: str = "model"

    def __init__(
        self,
        client: SupportsComplete | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings: Settings = settings if settings is not None else default_settings
        self.client: SupportsComplete = (
            client if client is not None else LLMClient(settings=self.settings)
        )

    @property
    def _effective_settings(self) -> Settings:
        """Settings to resolve models from: the client's own when it has any.

        The orchestrator injects a client whose ``settings`` may be a different
        instance from the global singleton, so the injected client's settings win.
        """
        return getattr(self.client, "settings", None) or self.settings

    def _model(self, model_attr: str | None = None) -> str:
        """Return the model name configured in ``model_attr`` (default: the class one)."""
        attr = model_attr or self.model_attr
        return str(getattr(self._effective_settings, attr))

    async def ask(self, prompt: str, model_attr: str | None = None, system: str = "") -> str:
        """Send ``prompt`` to the LLM with the agent persona in the system message."""
        return await self.client.complete(prompt, system=system, model=self._model(model_attr))