from __future__ import annotations

import asyncio
import os
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

MAX_FILE_CHARS = 12000
MAX_TOTAL_CHARS = 80000

FILE_BLOCK_PATTERN = re.compile(r"### FILE:\s*(.+?)\n```(?:\w+)?\n(.*?)\n```", re.DOTALL)
DIFF_BLOCK_PATTERN = re.compile(r"### DIFF:\s*(.+?)\n```diff\n(.*?)\n```", re.DOTALL)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def _read_file_contents(self, path: str) -> tuple[str, bool]:
        try:
            contents = await self.client.read_file(path)
            truncated = False
            if len(contents) > MAX_FILE_CHARS:
                contents = contents[:MAX_FILE_CHARS] + "\n... [TRUNCATED]"
                truncated = True
            return contents, truncated
        except Exception as e:
            return f"<Error reading file: {e}>", False

    async def _build_prompt(self) -> tuple[str, bool]:
        sections = [WORKER_PROMPT]
        total_chars = len(WORKER_PROMPT)
        truncated = False

        sections.append(f"Task: {self.task.description}\n")

        if self.task.files:
            sections.append("Files to touch:")
            for f in self.task.files:
                sections.append(f"  - {f}")
            sections.append("")
            sections.append("File contents (read before editing):")

            for path in self.task.files:
                contents, file_truncated = await self._read_file_contents(path)
                file_section = f"--- FILE: {path} ---\n{contents}\n--- END FILE: {path} ---\n\n"
                if total_chars + len(file_section) > MAX_TOTAL_CHARS:
                    truncated = True
                    sections.append(f"--- FILE: {path} ---\n<Remaining files omitted due to context limit>\n--- END FILE: {path} ---\n\n")
                    break
                sections.append(file_section)
                total_chars += len(file_section)
                if file_truncated:
                    truncated = True
        else:
            sections.append("Files to touch: any (search the workspace as needed)")

        return "\n".join(sections), truncated

    async def _run_verification(self, changed_files: list[str]) -> str:
        verification_output = ""
        candidates = []
        for path in changed_files:
            if path.endswith(".py"):
                candidates.append(["python", "-m", "py_compile", path])
            elif path.endswith((".ts", ".tsx", ".js", ".jsx")):
                candidates.append(["npx", "tsc", "--noEmit", path])

        if not candidates:
            return ""

        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    out = stdout.decode() + stderr.decode()
                    verification_output += f"$ {' '.join(cmd)}\n{out}"
                    if proc.returncode != 0:
                        verification_output += f"[Verification failed with exit code {proc.returncode}]\n"
                except asyncio.TimeoutError:
                    proc.kill()
                    verification_output += f"$ {' '.join(cmd)}\n[Timed out]\n"
            except FileNotFoundError:
                continue
            except Exception as e:
                verification_output += f"$ {' '.join(cmd)}\n[Error: {e}]\n"

        return verification_output

    def _write_files(self, response: str) -> list[str]:
        changed_files: list[str] = []
        for match in FILE_BLOCK_PATTERN.finditer(response):
            path = match.group(1).strip()
            content = match.group(2)
            try:
                p = os.path.expanduser(path)
                p = os.path.expandvars(p)
                os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    f.write(content)
                changed_files.append(path)
            except Exception as e:
                pass
        return changed_files

    async def run(self) -> str:
        prompt, truncated = await self._build_prompt()
        if truncated:
            prompt += "\n\nNote: Some file contents were truncated due to context limits. Focus on the most relevant sections."

        prompt += """
After implementing, provide your response as:
1. A concise summary of changes.
2. If you changed files, provide the full updated content for each file using this exact format:
   ### FILE: path/to/file.py
   ```python
   full updated file content
   ```
3. If a file is too large, provide a unified diff instead:
   ### DIFF: path/to/file.py
   ```diff
   diff --git a/path/to/file.py ...
   ```
4. List quick verification commands you ran or recommend running (e.g., `python -m py_compile path/to/file.py`).
"""

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        changed_files = self._write_files(response)

        verification_output = ""
        if changed_files:
            verification_output = await self._run_verification(changed_files)

        if verification_output:
            response += f"\n\nVerification output:\n{verification_output}"

        return response
