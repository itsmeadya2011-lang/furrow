from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)


def _parse_file_operations(response: str) -> list[tuple[str, str]]:
    """Parse the LLM response for file-write directives.

    Recognised formats:

    1. JSON array (with or without trailing text):
       [{"file": "path/to/file.py", "content": "..."}, ...]

    2. Markdown code blocks with filename headers:
       ```
       ```path/to/file.py
       ...content...
       ```
       ```
    """
    # Try JSON: extract the first JSON array in the response
    stripped = response.strip()
    json_start = stripped.find("[")
    if json_start >= 0:
        # Find the matching closing bracket
        depth = 0
        json_end = -1
        for i, ch in enumerate(stripped[json_start:], json_start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break
        if json_end > json_start:
            json_str = stripped[json_start:json_end]
            try:
                items = json.loads(json_str)
                ops = []
                for item in items:
                    if isinstance(item, dict) and "file" in item:
                        ops.append((item["file"], item.get("content", "")))
                if ops:
                    return ops
            except json.JSONDecodeError:
                pass

    # Fall back to markdown code blocks with filename headers
    # Format: ```filename
    #         content
    #         ```
    pattern = r"```([^\n\s]+)\s*\n(.*?)```"
    blocks = re.findall(pattern, response, re.DOTALL)

    ops = []
    for filename, content in blocks:
        # Skip language-tagged blocks (like ```python, ```json)
        if filename in ("python", "json", "javascript", "typescript", "rust", "go", "yaml", "yml", "toml", "bash", "sh"):
            continue
        ops.append((filename.strip(), content.strip()))

    return ops


class WorkerAgent:
    """Executes a single assigned task.

    The worker sends the task description to the LLM and then *actually writes*
    the resulting files to disk by parsing structured output from the model.
    """

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # Provide the worker with content of files it should touch
        project_context = await self._build_context()

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"{project_context}\n"
            f"Output format:\n"
            f"  1. First write a JSON array of file operations:\n"
            f"     [{{\"file\": \"path/to/file.py\", \"content\": \"file content\"}}, ...]\n"
            f"  2. Then provide a brief summary of changes.\n"
        )

        response = await self.client.complete(
            prompt,
            model=self.client.settings.worker_model,
        )

        # Parse and write files
        ops = _parse_file_operations(response)
        written: list[str] = []
        for file_path, content in ops:
            try:
                await self.client.write_file(file_path, content)
                written.append(file_path)
                log.info("Wrote file", file=file_path, task_id=self.task.id)
            except Exception as e:
                log.error("Failed to write file", file=file_path, error=str(e), task_id=self.task.id)
                console_print(f"[red]Failed to write {file_path}: {e}[/red]")

        if written:
            return f"Wrote {len(written)} file(s): {', '.join(written)}.\nSummary: {response[:500]}"
        return response

    async def _build_context(self) -> str:
        """Read existing file contents so the LLM has project context."""
        if not self.task.files:
            return ""

        sections = ["## Existing code context:\n"]
        for rel_path in self.task.files:
            abs_path = self.client.settings.workspace / rel_path
            if abs_path.exists():
                try:
                    content = await self.client.read_file(abs_path)
                    sections.append(f"--- {rel_path} ---\n```\n{content}\n```\n")
                except Exception as e:
                    sections.append(f"--- {rel_path} --- (read error: {e})\n")
            else:
                sections.append(f"--- {rel_path} --- (new file)\n")
        return "\n".join(sections)


def console_print(msg: str) -> None:
    """Print to console (imported lazily to avoid circular imports)."""
    from rich.console import Console

    Console().print(msg)
