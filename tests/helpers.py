"""Offline test doubles and payload builders shared by the furrow test suite.

Everything in here never touches the network: :class:`FakeLLMClient` is a
scripted stand-in for :class:`furrow.llm.LLMClient` that only performs the
file IO the real client performs.
"""

from __future__ import annotations

import inspect
import json
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from furrow.agents.prompts import PLANNER_PROMPT, TESTER_PROMPT, WORKER_PROMPT
from furrow.config import Settings
from furrow.llm import LLMClient

SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv"}

ResponseFn = Callable[[str, str, str | None], "str | Awaitable[str]"]


class FakeLLMClient(LLMClient):
    """A scripted, offline :class:`LLMClient`.

    Responses can be queued per agent role (``plans`` / ``workers`` / ``tests``)
    or supplied through a single ``response_fn`` callable. ``write_file``
    really writes to disk so worker behaviour can be observed end to end.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        responses: Iterable[str] | None = None,
        response_fn: ResponseFn | None = None,
        plans: Iterable[str] | None = None,
        workers: Iterable[str] | None = None,
        tests: Iterable[str] | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self.responses: deque[str] = deque(responses or [])
        self.response_fn = response_fn
        self.plans: deque[str] = deque(plans or [])
        self.workers: deque[str] = deque(workers or [])
        self.tests: deque[str] = deque(tests or [])
        self.calls: list[dict[str, Any]] = []
        self.writes: list[tuple[Path, str]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        if self.response_fn is not None:
            result = self.response_fn(prompt, system, model)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        if prompt.startswith(PLANNER_PROMPT) and self.plans:
            return self.plans.popleft()
        if prompt.startswith(WORKER_PROMPT) and self.workers:
            return self.workers.popleft()
        if prompt.startswith(TESTER_PROMPT) and self.tests:
            return self.tests.popleft()
        if self.responses:
            return self.responses.popleft()
        raise AssertionError(f"FakeLLMClient: no scripted response for prompt {prompt[:120]!r}")

    async def read_file(self, path: str | Path) -> str:
        return Path(path).read_text(encoding="utf-8")

    async def write_file(self, path: str | Path, content: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.writes.append((target, content))

    def list_files(self, directory: str | Path) -> list[str]:
        base = Path(directory)
        if not base.exists():
            return []
        found: list[str] = []
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(base)
            if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
                continue
            found.append(str(rel))
        return found


def make_plan(
    *descriptions: str,
    rationale: str = "break the goal into small steps",
    ids: list[str] | None = None,
) -> str:
    """Build a planner response containing one task per description."""
    tasks = []
    for index, description in enumerate(descriptions, start=1):
        task_id = str(index) if ids is None else ids[index - 1]
        tasks.append({"id": task_id, "description": description, "files": [], "dependencies": []})
    return json.dumps({"tasks": tasks, "rationale": rationale})


def make_worker(
    summary: str = "implemented the task",
    edits: list[dict[str, Any]] | None = None,
) -> str:
    """Build a worker response with the given edits."""
    return json.dumps({"summary": summary, "edits": edits or []})


def make_test_result(
    passed: bool,
    summary: str = "all good",
    failures: list[str] | None = None,
) -> str:
    """Build a tester response with an explicit verdict."""
    return json.dumps({"passed": passed, "summary": summary, "failures": failures or []})
