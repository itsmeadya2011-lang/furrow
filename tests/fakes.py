from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from furrow.config import Settings
from furrow.llm import LLMClient, extract_json


def make_settings(workspace=None, **overrides: Any) -> Settings:
    kwargs: dict[str, Any] = {}
    if workspace is not None:
        kwargs["workspace"] = workspace
    kwargs.update(overrides)
    return Settings(**kwargs)


class FakeLLMClient(LLMClient):
    """LLMClient with a scripted response queue and full call/tool-call recording.

    ``responses`` is a queue consumed in call order regardless of which method is
    invoked (the agents always call the methods in a deterministic sequence).

    Each queued item may be:
    - a dict: returned directly by ``complete_json`` (bypasses extract_json);
      for ``complete``/``complete_messages`` it is serialized to a JSON string.
    - a str: returned verbatim by ``complete``/``complete_messages``; fed through
      ``extract_json`` by ``complete_json``.
    - a BaseException instance: raised when the queue slot is consumed.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        responses: Iterable[Any] | None = None,
    ) -> None:
        if settings is None:
            settings = make_settings()
        super().__init__(settings=settings)
        self._queue: list[Any] = list(responses or [])
        self.calls: list[tuple[str, Any, Any, Any]] = []
        self.tool_calls: list[tuple[str, ...]] = []
        self._pos = 0

    def _next(self) -> Any:
        if self._pos >= len(self._queue):
            raise IndexError("FakeLLMClient ran out of scripted responses")
        item = self._queue[self._pos]
        self._pos += 1
        return item

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append(("complete", prompt, system, model))
        item = self._next()
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return item
        return json.dumps(item)

    async def complete_messages(
        self, messages: list[dict], system: str = "", model: str | None = None
    ) -> str:
        self.calls.append(("complete_messages", messages, system, model))
        item = self._next()
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return item
        return json.dumps(item)

    async def complete_json(self, prompt: str, system: str = "", model: str | None = None) -> dict:
        self.calls.append(("complete_json", prompt, system, model))
        item = self._next()
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, dict):
            return item
        text = item if isinstance(item, str) else json.dumps(item)
        return extract_json(text)

    def list_files(self, directory: str | Any = ".") -> list[str]:
        self.tool_calls.append(("list_files", directory))
        return super().list_files(directory)

    async def read_file(self, path: str | Any) -> str:
        self.tool_calls.append(("read_file", str(path)))
        return await super().read_file(path)

    async def write_file(self, path: str | Any, content: str) -> None:
        self.tool_calls.append(("write_file", str(path), content))
        await super().write_file(path, content)
