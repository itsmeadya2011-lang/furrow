from __future__ import annotations

import json
from pathlib import Path

from furrow.agents.prompts import WORKER_PROMPT, WORKER_SYSTEM_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


def _parse_action(text: str) -> dict | None:
    try:
        cleaned = text
        if "```json" in cleaned:
            start = cleaned.find("```json") + len("```json")
            end = cleaned.find("```", start)
            if end != -1:
                cleaned = cleaned[start:end]
        elif "```" in cleaned:
            start = cleaned.find("```") + len("```")
            end = cleaned.find("```", start)
            if end != -1:
                cleaned = cleaned[start:end]
        s = cleaned.find("{")
        e = cleaned.rfind("}")
        if s == -1 or e == -1 or e < s:
            return None
        obj = json.loads(cleaned[s : e + 1])
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    return None


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        if client is not None:
            self.client = client
            self.settings = getattr(client, "settings", settings or Settings())
        else:
            self.settings = settings or Settings()
            self.client = LLMClient(settings=self.settings)
        self.task = task

    async def run(self) -> str:
        messages: list[dict] = []
        task = self.task
        initial = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {task.description}\n"
            f"Files to touch: {', '.join(task.files) if task.files else 'any'}\n"
            f"Dependencies: {', '.join(task.dependencies) if task.dependencies else 'none'}\n"
        )
        messages.append({"role": "user", "content": initial})

        max_rounds = getattr(self.settings, "worker_rounds", 8)
        for round_num in range(1, max_rounds + 1):
            try:
                resp = await self.client.complete_messages(
                    messages,
                    system=WORKER_SYSTEM_PROMPT,
                    model=self.settings.worker_model,
                )
            except (RuntimeError, OSError, ValueError) as e:
                return f"Worker error in round {round_num}: {e}"

            action = _parse_action(resp)
            if action is None:
                messages.append({"role": "assistant", "content": resp})
                messages.append(
                    {
                        "role": "user",
                        "content": "Tool result:\nInvalid action format. Respond with ONLY a valid JSON action object.",
                    }
                )
                continue

            action_type = action.get("action")
            if action_type == "finish":
                summary = str(action.get("summary", "Done"))
                return summary

            if action_type == "list_files":
                directory = str(action.get("directory", "."))
                try:
                    files = self.client.list_files(directory)
                    out = json.dumps(files)
                    if len(out) > 20000:
                        out = json.dumps(files[:200])
                    messages.append({"role": "assistant", "content": resp})
                    messages.append({"role": "user", "content": f"Tool result:\n{out}"})
                except (RuntimeError, OSError, ValueError) as e:
                    messages.append({"role": "assistant", "content": resp})
                    messages.append({"role": "user", "content": f"Tool result:\nError: {e}"})
                continue

            if action_type == "read_file":
                path = str(action.get("path", ""))
                try:
                    content = await self.client.read_file(path)
                    if len(content) > 8000:
                        content = content[:8000] + "\n... (truncated)"
                    messages.append({"role": "assistant", "content": resp})
                    messages.append({"role": "user", "content": f"Tool result:\n{content}"})
                except (RuntimeError, OSError, ValueError, FileNotFoundError) as e:
                    messages.append({"role": "assistant", "content": resp})
                    messages.append({"role": "user", "content": f"Tool result:\nError: {e}"})
                continue

            if action_type == "write_file":
                path = str(action.get("path", ""))
                content = str(action.get("content", ""))
                try:
                    ws = Path(self.settings.workspace).resolve()
                    p = Path(path)
                    target = (ws / p).resolve() if not p.is_absolute() else p.resolve()
                    if ws not in target.parents and target != ws:
                        raise ValueError("Path escapes workspace root")
                    await self.client.write_file(path, content)
                    messages.append({"role": "assistant", "content": resp})
                    messages.append({"role": "user", "content": "Tool result:\nok"})
                except (RuntimeError, OSError, ValueError, FileNotFoundError, PermissionError) as e:
                    messages.append({"role": "assistant", "content": resp})
                    messages.append({"role": "user", "content": f"Tool result:\nError: {e}"})
                continue

            messages.append({"role": "assistant", "content": resp})
            messages.append(
                {
                    "role": "user",
                    "content": "Tool result:\nUnsupported action. Use list_files/read_file/write_file/finish.",
                }
            )

        return f"Worker hit the round limit ({max_rounds} rounds) without calling finish. Partial work may be incomplete."
