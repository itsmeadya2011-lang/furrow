from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path
from typing import Any


JSON_ONLY_INSTRUCTION = (
    "Respond with ONLY a JSON object, no markdown, no explanation. One of: "
    '{"action":"read_file",...}, {"action":"write_file",...}, '
    '{"action":"run_command",...}, or {"action":"done","summary":"..."}.'
)


def _format_conversation(conversation: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in conversation
    )


def _resolve_path(path: str, workspace: Path) -> Path:
    requested_path = Path(path).expanduser()
    resolved_path = (
        requested_path if requested_path.is_absolute() else workspace / requested_path
    ).resolve()
    try:
        resolved_path.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Path must be within the workspace: {path}") from exc
    return resolved_path


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit]


async def _read_file(path: str, client: Any, workspace: Path) -> str:
    resolved_path = _resolve_path(path, workspace)
    contents = await client.read_file(resolved_path)
    return _truncate(str(contents), 4000)


async def _write_file(path: str, content: str, client: Any, workspace: Path) -> str:
    resolved_path = _resolve_path(path, workspace)
    await client.write_file(resolved_path, content)
    return f"Wrote {len(content.encode('utf-8'))} bytes to {resolved_path}"


async def _run_command(command: str, workspace: Path) -> str:
    try:
        args = shlex.split(command)
    except ValueError as exc:
        return f"Command failed: {exc}"

    if not args:
        return "Command failed: empty command."

    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
            shell=False,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return "Command timed out after 120 seconds."
    except Exception as exc:
        return f"Command failed: {exc}"

    output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
    return _truncate(output, 6000)


async def _execute_action(action: dict[str, Any], client: Any, workspace: Path) -> str:
    action_name = action.get("action")
    if action_name == "read_file":
        path = action.get("path")
        if not isinstance(path, str):
            return "Error: read_file requires a string path."
        try:
            return await _read_file(path, client, workspace)
        except Exception as exc:
            return f"Error: {exc}"

    if action_name == "write_file":
        path = action.get("path")
        content = action.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            return "Error: write_file requires string path and content."
        try:
            return await _write_file(path, content, client, workspace)
        except Exception as exc:
            return f"Error: {exc}"

    if action_name == "run_command":
        command = action.get("command")
        if not isinstance(command, str):
            return "Error: run_command requires a string command."
        return await _run_command(command, workspace)

    return f"Error: unknown action {action_name!r}."


async def run_tool_loop(
    client: Any,
    system_prompt: str,
    user_message: str,
    max_rounds: int = 10,
    workspace: str | Path = ".",
    model: str | None = None,
) -> str:
    workspace_path = Path(workspace).expanduser().resolve()
    conversation: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    last_result = ""

    for _ in range(max_rounds):
        prompt = f"{_format_conversation(conversation)}\n\n{JSON_ONLY_INSTRUCTION}"
        response = await client.complete(prompt=prompt, model=model)
        response_text = response or ""
        conversation.append({"role": "assistant", "content": response_text})

        try:
            action = json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            last_result = response_text
            conversation.append(
                {
                    "role": "user",
                    "content": "Invalid JSON response. Retry with valid JSON only.",
                }
            )
            continue

        if not isinstance(action, dict):
            result = "Error: response must be a JSON object."
        elif action.get("action") == "done":
            # Return the full action object as JSON so callers (e.g. the tester)
            # can parse passed/summary/failures out of it. Falling back to just
            # the summary string would make the result unparseable as JSON.
            return json.dumps(action)
        else:
            result = await _execute_action(action, client, workspace_path)

        last_result = result
        conversation.append({"role": "user", "content": f"Tool result: {result}"})

    return f"Max rounds ({max_rounds}) reached without completion. Last result: {last_result}"
