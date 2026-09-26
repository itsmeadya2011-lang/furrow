from __future__ import annotations

from pathlib import Path
from typing import Any

from furrow.config import Settings


class FakeClient:
    """Minimal LLMClient substitute for unit tests."""

    def __init__(
        self, responses: list[str] | None = None, settings: Settings | None = None
    ) -> None:
        self.responses = list(responses or [])
        self._index = 0
        self.settings = settings or Settings()
        self.written: dict[str, str] = {}

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        if self._index < len(self.responses):
            value = self.responses[self._index]
        elif self.responses:
            value = self.responses[-1]
        else:
            value = ""
        self._index += 1
        return value

    async def read_file(self, path: str | Path) -> str:
        p = Path(path)
        return p.read_text() if p.exists() else ""

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self.written[str(p)] = content

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]

    @property
    def anthropic(self) -> Any:
        raise RuntimeError("anthropic not configured")

    @property
    def openai(self) -> Any:
        raise RuntimeError("openai not configured")


class FakeTester:
    """Stand-in for TesterAgent in orchestrator tests."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls = 0

    async def run(self, goal: str, tasks: list[Any]) -> Any:
        self.calls += 1
        return self.result
