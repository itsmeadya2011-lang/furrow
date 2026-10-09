from __future__ import annotations

import json
from typing import Any

from furrow.config import Settings


class FakeLLMClient:
    """Canned LLM that dispatches on prompt content."""

    def __init__(self, plans=None, tests=None, settings=None):
        self.settings = settings or Settings()
        self.plans = list(plans or [])
        self.tests = list(tests or [])
        self.plan_calls = 0
        self.test_calls = 0
        self.prompts: list[str] = []

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        if "planning agent" in prompt:
            idx = min(self.plan_calls, len(self.plans) - 1) if self.plans else 0
            self.plan_calls += 1
            return self.plans[idx] if self.plans else json.dumps(
                {"tasks": [], "rationale": "nothing"}
            )
        if "tester agent" in prompt:
            idx = min(self.test_calls, len(self.tests) - 1) if self.tests else 0
            self.test_calls += 1
            return self.tests[idx] if self.tests else json.dumps(
                {"passed": True, "summary": "ok", "failures": []}
            )
        return "worker completed the task"


class EventCollector:
    """Collects orchestrator events for assertions."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __call__(self, event: dict) -> Any:
        self.events.append(event)
        return None

    def names(self) -> list[str]:
        return [e["event"] for e in self.events]


def make_plan_json(tasks=None, rationale="why"):
    """Build a canned planner JSON string."""
    if tasks is None:
        tasks = [
            {
                "id": "1",
                "description": "do thing",
                "files": ["a.py"],
                "dependencies": [],
            }
        ]
    return json.dumps({"tasks": tasks, "rationale": rationale})


def make_test_json(passed=True, summary="ok", failures=None):
    """Build a canned tester JSON string."""
    return json.dumps(
        {"passed": passed, "summary": summary, "failures": failures or []}
    )