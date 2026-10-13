from __future__ import annotations

import json
from typing import Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        if client is not None:
            self.client = client
            self.settings = getattr(client, "settings", settings or Settings())
        else:
            self.settings = settings or Settings()
            self.client = LLMClient(settings=self.settings)

    async def plan(self, goal: str, history: list[dict] | None = None) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        if history:
            prompt += "\nPrevious cycle history:\n" + json.dumps(history, indent=2) + "\n"
            prompt += (
                "\nIf the goal is already fully complete based on history, return "
                '{"tasks": [], "rationale": "Goal already complete"}.\n'
                "Otherwise return only the remaining work."
            )

        try:
            data = await self.client.complete_json(
                prompt, model=self.client.settings.planner_model
            )
        except ValueError:
            try:
                data = await self.client.complete_json(
                    prompt + "\nYour previous response was not valid JSON. Return only the JSON object.",
                    model=self.client.settings.planner_model,
                )
            except ValueError as e:
                raise ValueError(f"Failed to parse plan from LLM: {e}") from e

        try:
            if not isinstance(data, dict):
                raise TypeError("Plan data must be a dict")
            tasks_raw = data.get("tasks", [])
            if not isinstance(tasks_raw, list):
                tasks_raw = []

            tasks: list[TaskModel] = []
            for t in tasks_raw:
                if not isinstance(t, dict):
                    continue
                try:
                    tid = str(t.get("id", "")).strip()
                    desc = str(t.get("description", "")).strip()
                    if not tid or not desc:
                        continue
                    files = t.get("files", [])
                    if not isinstance(files, list):
                        files = []
                    files = [str(f) for f in files]
                    deps = t.get("dependencies", [])
                    if not isinstance(deps, list):
                        deps = []
                    deps = [str(d) for d in deps]
                    tasks.append(
                        TaskModel(
                            id=tid,
                            description=desc,
                            files=files,
                            dependencies=deps,
                            status="pending",
                        )
                    )
                except (TypeError, ValueError, AttributeError):
                    continue
            data_clean: dict[str, Any] = {"tasks": tasks}
            if "rationale" in data:
                data_clean["rationale"] = str(data["rationale"])
            return Plan(**data_clean)
        except (TypeError, ValueError, AttributeError) as e:
            raise ValueError(f"Failed to validate plan from LLM: {e}\nResponse: {data}") from e
