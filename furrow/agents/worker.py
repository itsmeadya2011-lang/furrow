from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    pass

logger = structlog.get_logger("furrow.worker")


class FileOperation(BaseModel):
    path: str = Field(..., description="File path relative to workspace root")
    operation: str = Field(..., description="create, modify, or delete")
    content: str | None = Field(None, description="New file content (for create/modify)")


class WorkerResult(BaseModel):
    task_id: str
    status: str = "completed"
    files_modified: list[str] = Field(default_factory=list)
    summary: str = ""
    error: str | None = None


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = workspace or Path.cwd()

    async def run(self) -> WorkerResult:
        # Read existing files referenced by the task
        file_contents: dict[str, str] = {}
        for file_path in self.task.files:
            full_path = self.workspace / file_path
            if full_path.exists():
                try:
                    file_contents[file_path] = await self.client.read_file(full_path)
                except Exception:
                    file_contents[file_path] = ""

        # Build context for the LLM
        context_parts = [f"Task: {self.task.description}"]
        if file_contents:
            context_parts.append("\nCurrent file contents:")
            for path, content in file_contents.items():
                context_parts.append(f"\n--- {path} ---\n{content}")
        else:
            context_parts.append("\nNo existing files to read. Create new files as needed.")

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            + "\n".join(context_parts)
            + f"\n\nWorkspace root: {self.workspace}\n"
            + "Return JSON only with file operations to perform:\n"
            + '{\n  "operations": [\n    {"path": "src/foo.py", "operation": "modify", "content": "..."}\n  ],\n  "summary": "Brief summary of changes"\n}\n'
        )

        try:
            response = await self.client.complete(
                prompt, model=self.client.settings.worker_model
            )
            data = json.loads(response)
            operations = data.get("operations", [])
            summary = data.get("summary", "No summary provided")

            files_modified: list[str] = []
            for op in operations:
                op_path = op.get("path", "")
                op_type = op.get("operation", "modify")
                op_content = op.get("content")

                if not op_path:
                    continue

                full_path = self.workspace / op_path

                if op_type == "delete":
                    if full_path.exists():
                        full_path.unlink()
                    files_modified.append(op_path)
                elif op_type in ("create", "modify"):
                    if op_content is None:
                        continue
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    await self.client.write_file(full_path, op_content)
                    files_modified.append(op_path)

            return WorkerResult(
                task_id=self.task.id,
                status="completed",
                files_modified=files_modified,
                summary=summary,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            return WorkerResult(
                task_id=self.task.id,
                status="failed",
                summary=f"Failed to parse LLM response: {e}",
                error=str(e),
            )
        except Exception as e:  # noqa: BLE001
            return WorkerResult(
                task_id=self.task.id,
                status="failed",
                summary=f"Worker error: {e}",
                error=str(e),
            )