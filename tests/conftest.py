from __future__ import annotations

from typing import Any


class MockLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, str | None]] = []
        self.writes: list[tuple[str, str]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append((prompt, system, model))
        if not self.responses:
            return ""
        return self.responses.pop(0)

    async def read_file(self, path: str) -> str:
        return ""

    async def write_file(self, path: str, content: str) -> None:
        self.writes.append((path, content))

    def list_files(self, directory: str) -> list[str]:
        return []
