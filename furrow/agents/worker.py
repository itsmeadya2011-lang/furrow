from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        """Execute the task by reading files, generating changes, and applying them."""
        # Read context files if specified
        context = await self._gather_context()

        # Build the prompt with context
        prompt = self._build_prompt(context)

        # Get implementation plan from LLM
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse and apply file changes
        changes = self._parse_file_changes(response)
        applied = await self._apply_changes(changes)

        return f"Task '{self.task.description}': {applied}"

    async def _gather_context(self) -> str:
        """Read context files to provide to the LLM."""
        if not self.task.files:
            return "No specific files specified. Use workspace root for context."

        context_parts = []
        for file_path in self.task.files[:10]:  # Limit to 10 files
            full_path = settings.workspace / file_path
            if full_path.exists() and full_path.is_file():
                try:
                    content = await self.client.read_file(full_path)
                    context_parts.append(f"--- {file_path} ---\n{content}\n")
                except Exception as e:
                    context_parts.append(f"--- {file_path} ---\n[Could not read: {e}]\n")
            else:
                context_parts.append(f"--- {file_path} ---\n[File does not exist yet]\n")

        return "\n".join(context_parts) if context_parts else "No context files available."

    def _build_prompt(self, context: str) -> str:
        """Build the full prompt for the LLM."""
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        return f"""{WORKER_PROMPT}

## Task
{self.task.description}

## Files to touch
{files_str}

## Current file contents
{context}

## Instructions
Analyze the task and current code. Then provide file changes using this exact format:

FILE: <path>
<full file content or code to write>

FILE: <path>
<full file content or code to write>

Make minimal, targeted changes. Only include files you actually modified."""

    def _parse_file_changes(self, response: str) -> dict[str, str]:
        """Parse FILE: sections from LLM response into {path: content} dict."""
        changes: dict[str, str] = {}

        # Match FILE: path followed by content until next FILE: or end
        pattern = r"FILE:\s*(.+?)\n(.*?)(?=FILE:|\Z)"
        matches = re.findall(pattern, response, re.DOTALL)

        for path, content in matches:
            path = path.strip()
            content = content.strip()
            if path and content:
                changes[path] = content

        return changes

    async def _apply_changes(self, changes: dict[str, str]) -> str:
        """Apply parsed file changes to the filesystem."""
        if not changes:
            return "No file changes detected in response."

        applied = []
        for file_path, content in changes.items():
            try:
                full_path = settings.workspace / file_path
                await self.client.write_file(full_path, content)
                applied.append(file_path)
            except Exception as e:
                return f"Failed to write {file_path}: {e}"

        return f"Applied changes to: {', '.join(applied)}"
