from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

_MAX_ITERATIONS = 12


def _extract_tool_call(response: str) -> dict[str, Any] | None:
    text = response.replace("\r\n", "\n").replace("\r", "\n")
    
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    
    obj = _find_first_top_level_json_object(text)
    if obj is not None:
        try:
            data = json.loads(obj)
            if isinstance(data, dict) and "tool" in data:
                return data
        except json.JSONDecodeError:
            pass
    return None


def _find_first_top_level_json_object(text: str) -> str | None:
    in_string = False
    escape_next = False
    string_char = None
    depth = 0
    start = None
    
    for i, ch in enumerate(text):
        if not in_string:
            if ch == '"' or ch == "'":
                in_string = True
                string_char = ch
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        return text[start:i+1]
        else:
            if escape_next:
                escape_next = False
            elif ch == "\\":
                escape_next = True
            elif ch == string_char:
                in_string = False
                string_char = None
    return None


def _is_safe_path(workspace: Path, target: Path) -> bool:
    resolved = (workspace / target).resolve()
    return resolved.is_relative_to(workspace.resolve())


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Any | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    @property
    def _workspace(self) -> Path:
        return self.client.settings.workspace

    def _build_prompt(self) -> str:
        file_list = ", ".join(self.task.files) if self.task.files else "any"
        tool_instructions = """

Tool protocol:
Respond with exactly one JSON object specifying the operation to perform:

{
  "tool": "read",
  "path": "relative/path/to/file"
}

{
  "tool": "write",
  "path": "relative/path/to/file",
  "content": "file contents"
}

{
  "tool": "run",
  "command": "pytest -q",
  "cwd": null,
  "timeout": 30
}

{
  "tool": "finish",
  "summary": "Description of completed work"
}

All paths are relative to the workspace. After each tool result, continue with the next operation. When done, use tool "finish".
"""
        return f"{WORKER_PROMPT}{tool_instructions}\n\nTask: {self.task.description}\nFiles to touch: {file_list}\n"

    async def _execute_tool(self, tool_call: dict[str, Any]) -> str:
        tool = tool_call.get("tool")
        workspace = self._workspace

        if tool == "read":
            path = tool_call.get("path")
            if not path or not isinstance(path, str):
                return json.dumps({"error": "Missing required field: path"})
            path = Path(path)
            if not _is_safe_path(workspace, path):
                return json.dumps({"error": f"Path traversal blocked: {path}"})
            target = workspace / path
            if not target.exists():
                return json.dumps({"error": f"File not found: {path}"})
            try:
                content = await self.client.read_file(target)
                return json.dumps({"ok": True, "content": content})
            except Exception as e:
                return json.dumps({"error": str(e)})

        elif tool == "write":
            path = tool_call.get("path")
            if not path or not isinstance(path, str):
                return json.dumps({"error": "Missing required field: path"})
            path = Path(path)
            if not _is_safe_path(workspace, path):
                return json.dumps({"error": f"Path traversal blocked: {path}"})
            content = tool_call.get("content")
            if content is None:
                return json.dumps({"error": "Missing required field: content"})
            target = workspace / path
            try:
                await self.client.write_file(target, content)
                return json.dumps({"ok": True, "written": str(target.relative_to(workspace))})
            except Exception as e:
                return json.dumps({"error": str(e)})

        elif tool == "run":
            command = tool_call.get("command")
            if not command or not isinstance(command, str):
                return json.dumps({"error": "Missing required field: command"})
            cwd_val = tool_call.get("cwd")
            timeout = tool_call.get("timeout", 30.0)
            if cwd_val is not None:
                cwd = (workspace / Path(cwd_val)).resolve()
            else:
                cwd = workspace.resolve()
            if not cwd.is_relative_to(workspace.resolve()):
                return json.dumps({"error": f"Invalid cwd: {cwd_val}"})
            try:
                result = await self.client.run_command(command, cwd=cwd, timeout=timeout)
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"error": str(e)})

        elif tool == "finish":
            return json.dumps({"finished": True, "summary": tool_call.get("summary", "")})

        return json.dumps({"error": f"Unknown tool: {tool}"})

    async def run(self) -> str:
        prompt = self._build_prompt()
        performed_work = False
        for _ in range(_MAX_ITERATIONS):
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            tool_call = _extract_tool_call(response)
            if tool_call is None:
                raise RuntimeError("No parseable tool call found in response")
            tool = tool_call.get("tool")
            if tool == "finish":
                if not performed_work:
                    raise RuntimeError("Finish called before any non-finish tool operation")
                result = await self._execute_tool(tool_call)
                return result
            result = await self._execute_tool(tool_call)
            performed_work = True
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and "error" in parsed:
                    raise RuntimeError(parsed["error"])
            except json.JSONDecodeError:
                pass
            prompt = f"Previous response:\n{response}\n\nTool result:\n{result}\n\nContinue with the next operation:"
        raise RuntimeError(f"Reached maximum iterations ({_MAX_ITERATIONS}) without finish.")
