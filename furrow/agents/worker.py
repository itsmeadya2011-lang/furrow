from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

# Directives the worker may emit in each turn.
_READ_RE = re.compile(r"^READ:\s*(\S.*)$", re.MULTILINE)
_WRITE_START_RE = re.compile(r"^WRITE:\s*(\S.*)$", re.MULTILINE)
_CONTENT_START = "<<<CONTENT"
_CONTENT_END = ">>>CONTENT"
_FINISH_RE = re.compile(r"^FINISH\s*$", re.MULTILINE)


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: "Settings | None" = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self._edits: list[str] = []
        self._reads: list[str] = []

    async def run(self) -> str:
        files = ", ".join(self.task.files) if self.task.files else "any"
        context: dict[str, str] = {}
        turn = 0
        max_turns = int(getattr(self.settings, "max_turns", 5))

        # Seed context with the task's declared files so the worker can edit them in place.
        for path in self.task.files:
            try:
                content = await self.client.read_file(self.settings.workspace / path)
                context[path] = content
                self._reads.append(path)
            except Exception:
                context[path] = ""

        while turn < max_turns:
            turn += 1
            prompt = WORKER_PROMPT.format(
                task_description=self.task.description,
                files=files,
            )
            # Inject current file context so the model can reference what it has read.
            if context:
                context_block = "\n".join(
                    f"--- {p} ---\n{c}" for p, c in context.items()
                )
                prompt += f"\n\nCurrent file context:\n{context_block}\n"

            response = await self.client.complete(
                prompt, model=self.client.settings.worker_model
            )
            directive = self._parse_directive(response)
            if directive is None:
                # No recognizable directive; treat as finished.
                break
            kind, payload = directive
            if kind == "FINISH":
                break
            if kind == "READ":
                path = payload.strip()
                self._reads.append(path)
                try:
                    context[path] = await self.client.read_file(
                        self.settings.workspace / path
                    )
                except Exception as e:
                    context[path] = f"<unable to read {path}: {e}>"
                continue
            if kind == "WRITE":
                path, content = payload
                self._edits.append(path)
                await self.client.write_file(
                    self.settings.workspace / path, content
                )
                context[path] = content
                continue

        return self._summary()

    def _parse_directive(self, response: str) -> tuple[str, object] | None:
        """Parse the first actionable directive from the model's response.

        Returns ("READ", path), ("WRITE", (path, content)), ("FINISH", None), or None.
        """
        lines = response.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            m = _FINISH_RE.match(line)
            if m:
                return ("FINISH", None)
            m = _READ_RE.match(line)
            if m:
                return ("READ", m.group(1))
            m = _WRITE_START_RE.match(line)
            if m:
                path = m.group(1).strip()
                # Expect the next line to be <<<CONTENT; skip it, then collect
                # content lines until >>>CONTENT.
                i += 1
                if i < len(lines) and lines[i] == _CONTENT_START:
                    i += 1
                content_lines: list[str] = []
                while i < len(lines) and lines[i] != _CONTENT_END:
                    content_lines.append(lines[i])
                    i += 1
                content = "\n".join(content_lines)
                return ("WRITE", (path, content))
            i += 1
        return None

    def _summary(self) -> str:
        parts: list[str] = []
        if self._edits:
            parts.append(f"wrote {len(self._edits)} file(s): " + ", ".join(self._edits))
        if self._reads:
            parts.append(f"read {len(self._reads)} file(s): " + ", ".join(self._reads))
        if not parts:
            return "no changes made"
        return "; ".join(parts)
