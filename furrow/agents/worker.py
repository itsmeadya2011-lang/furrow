from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

MAX_TOOL_ITERATIONS = 10
MAX_TOOL_OUTPUT_CHARS = 20000

TOOL_PROTOCOL = """Tool protocol:
- Respond with exactly one JSON object per turn and nothing else (no prose, no
  markdown fences) when you want to call a tool.
- After each call you receive a "Tool result" block with the output, or an error
  message. Read it before deciding the next step.
- Relative paths are resolved against the workspace root.
- Keep iterating until the task is really written to disk. Multiple reads of the
  same file are wasted rounds; read what you need, then write.
- When the implementation is complete, stop calling tools and reply with plain
  text, or with {"done": true, "summary": "..."}.
"""


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        if client is None:
            client = LLMClient(settings=settings or default_settings)
        self.client = client

    async def run(self) -> str:
        model, workspace = self._resolve_model()
        system = f"{WORKER_PROMPT}\n\n{TOOL_PROTOCOL}"
        files = ", ".join(self.task.files) if self.task.files else "any"
        messages: list[dict[str, str]] = [
            {"role": "User", "content": f"Task: {self.task.description}\nFiles to touch: {files}\n"}
        ]
        summary = ""

        for _ in range(MAX_TOOL_ITERATIONS):
            prompt = self._render(messages)
            response = await self.client.complete(prompt, system=system, model=model)
            payload = self._parse(response)

            if payload is None:
                return response.strip()

            if isinstance(payload, dict) and payload.get("done"):
                summary = str(payload.get("summary") or "").strip()
                return summary or response.strip()

            calls = self._tool_calls(payload)
            if not calls:
                summary = str(payload.get("summary", "")) if isinstance(payload, dict) else ""
                return summary.strip() or response.strip()

            for call in calls:
                messages.append({"role": "Worker", "content": json.dumps(call)})
                result = await self._run_tool(call, workspace)
                messages.append({"role": "Tool result", "content": result})

        exhausted = f"Worker stopped after {MAX_TOOL_ITERATIONS} tool rounds without finishing."
        return summary or exhausted

    def _resolve_model(self) -> tuple[str, Path]:
        if self.client is None:
            raise RuntimeError("WorkerAgent requires an LLMClient, but none was configured.")
        client_settings = getattr(self.client, "settings", None)
        if client_settings is None:
            raise RuntimeError(
                "WorkerAgent requires an LLMClient built with settings, but settings is None."
            )
        workspace = Path(client_settings.workspace)
        return client_settings.worker_model, workspace

    async def _run_tool(self, call: dict[str, Any], workspace: Path) -> str:
        name = call.get("tool") or call.get("name")
        try:
            if name == "read_file":
                path = self._argument_path(call, "path", workspace)
                if path is None:
                    return 'Error: read_file requires a "path" argument.'
                content = await self.client.read_file(path)
                return self._truncate(f"read_file {path}\n{content}")

            if name == "write_file":
                path = self._argument_path(call, "path", workspace)
                if path is None:
                    return 'Error: write_file requires a "path" argument.'
                body = call.get("content")
                if not isinstance(body, str):
                    return 'Error: write_file requires a string "content" argument.'
                await self.client.write_file(path, body)
                return f"Wrote {len(body)} characters to {path}"

            if name == "list_files":
                path = self._argument_path(call, "directory", workspace)
                if path is None:
                    return 'Error: list_files requires a "directory" argument.'
                files = self.client.list_files(path)
                return self._truncate(f"list_files {path}\n" + "\n".join(files))

            known = "read_file, write_file, list_files"
            return f"Error: unknown tool {name!r}. Available tools: {known}."
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    @staticmethod
    def _argument_path(call: dict[str, Any], key: str, workspace: Path) -> Path | None:
        raw = call.get(key)
        if not isinstance(raw, str) or not raw.strip():
            return None
        path = Path(raw.strip()).expanduser()
        return path if path.is_absolute() else workspace / path

    @staticmethod
    def _tool_calls(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            raw = payload.get("tool_calls")
            if raw is None:
                raw = [payload] if payload.get("tool") else None
        elif isinstance(payload, list):
            raw = payload
        else:
            raw = None
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        calls = [c for c in raw if isinstance(c, dict) and (c.get("tool") or c.get("name"))]
        return calls

    @staticmethod
    def _parse(response: str) -> Any | None:
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        if not text.startswith(("{", "[")):
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _render(messages: list[dict[str, str]]) -> str:
        return "\n\n".join(f"{message['role']}:\n{message['content']}" for message in messages)

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_TOOL_OUTPUT_CHARS:
            return text
        note = f"\n... [truncated at {MAX_TOOL_OUTPUT_CHARS} characters]"
        return text[:MAX_TOOL_OUTPUT_CHARS] + note
