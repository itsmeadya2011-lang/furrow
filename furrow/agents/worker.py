from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from furrow.agents.prompts import WORKER_PROMPT, WORKSPACE_CONTEXT_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        max_retries: int = 3,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_retries = max_retries

    async def run(self) -> str:
        try:
            logger.info(
                "worker_started",
                task_id=self.task.id,
                description=self.task.description,
            )

            workspace_files = self._get_workspace_context()
            file_contents = await self._get_relevant_file_contents()
            prompt = self._build_prompt(workspace_files, file_contents)

            result = await self._get_llm_response_with_retry(prompt)
            if result is None:
                return f"Failed to parse LLM response after {self.max_retries} attempts."

            issues = await self._apply_changes(result)
            summary = result.get("summary") or "Task completed."
            if issues:
                result.setdefault("issues", []).extend(issues)
                summary += " Issues: " + "; ".join(result["issues"])

            logger.info(
                "worker_completed",
                task_id=self.task.id,
                changed_files=[f["path"] for f in result.get("changed_files", [])],
            )
            return summary
        except Exception as e:  # noqa: BLE001
            logger.error("worker_error", task_id=self.task.id, error=str(e))
            return f"Worker failed on task {self.task.id}: {e}"

    def _get_workspace_context(self) -> list[str]:
        workspace = self.client.settings.workspace
        files = self.client.list_files(workspace)
        logger.debug("workspace_context", task_id=self.task.id, file_count=len(files))
        return files

    async def _get_relevant_file_contents(self) -> str:
        task_files = self.task.files
        if not task_files:
            return "(no specific files requested)"

        contents: list[str] = []
        for file_path in task_files:
            try:
                full_path = Path(file_path)
                if not full_path.is_absolute():
                    full_path = self.client.settings.workspace / file_path
                content = await self.client.read_file(full_path)
                contents.append(f"--- {file_path} ---\n{content}")
            except OSError as e:
                logger.debug("file_read_skipped", file_path=file_path, error=str(e))
                contents.append(f"--- {file_path} ---\n(file could not be read: {e})")
        return "\n\n".join(contents)

    def _build_prompt(self, workspace_files: list[str], file_contents: str) -> str:
        file_list = "\n".join(f"  - {f}" for f in workspace_files) or "(workspace is empty)"
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        workspace = str(self.client.settings.workspace)

        context = WORKSPACE_CONTEXT_PROMPT.format(
            workspace=workspace,
            file_list=file_list,
            file_contents=file_contents,
        )
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"{context}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_str}\n"
        )
        return prompt

    async def _get_llm_response_with_retry(self, prompt: str) -> dict[str, Any] | None:
        last_error: str | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.client.complete(
                    prompt, model=self.client.settings.worker_model
                )
                result = self._parse_response(response)
                logger.debug("llm_response_parsed", task_id=self.task.id, attempt=attempt)
                return result
            except (json.JSONDecodeError, ValueError) as e:
                last_error = str(e)
                logger.warning(
                    "parse_retry",
                    task_id=self.task.id,
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error=last_error,
                )
        logger.error(
            "parse_failed",
            task_id=self.task.id,
            max_retries=self.max_retries,
            error=last_error,
        )
        return None

    def _parse_response(self, response: str) -> dict[str, Any]:
        text = response.strip()

        if text.startswith("```json"):
            text = text.removeprefix("```json").removesuffix("```").strip()
        elif text.startswith("```"):
            text = text.removeprefix("```").removesuffix("```").strip()

        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]

        data: dict[str, Any] = json.loads(text)

        if not isinstance(data, dict):
            raise TypeError(f"Expected a JSON object, got {type(data).__name__}")

        changed_files = data.get("changed_files", [])
        if not isinstance(changed_files, list):
            raise TypeError("changed_files must be a list")

        normalized: list[dict[str, str]] = []
        for item in changed_files:
            if isinstance(item, str):
                normalized.append({"path": item, "content": ""})
            elif isinstance(item, dict) and "path" in item:
                normalized.append(
                    {"path": item["path"], "content": item.get("content", "")}
                )
            else:
                raise ValueError(f"Invalid changed_files entry: {item!r}")
        data["changed_files"] = normalized

        data["summary"] = data.get("summary") or "Task completed."
        data["issues"] = data.get("issues") or []
        return data

    async def _apply_changes(self, result: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        for file_info in result.get("changed_files", []):
            path = file_info.get("path", "")
            content = file_info.get("content", "")
            if not path:
                issues.append("Skipped file entry without a path")
                continue
            try:
                full_path = Path(path)
                if not full_path.is_absolute():
                    full_path = self.client.settings.workspace / path
                await self.client.write_file(full_path, content)
                logger.info("file_written", task_id=self.task.id, path=path)
            except OSError as e:
                msg = f"Failed to write {path}: {e}"
                logger.error("file_write_failed", task_id=self.task.id, path=path, error=str(e))
                issues.append(msg)
        return issues
