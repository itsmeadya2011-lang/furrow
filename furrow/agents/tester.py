from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.tools import ANTHROPIC_TOOLS, OPENAI_TOOLS, run_command

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        # First, run the actual tests
        test_output = await self._run_tests()

        # If no test runner found, try to determine pass/fail from task results
        if "No test runner found" in test_output:
            # Check if any tasks failed
            failed_tasks = [t for t in tasks if t.status == "failed"]
            if failed_tasks:
                return TestResult(
                    passed=False,
                    summary=f"No test runner found. {len(failed_tasks)} task(s) failed.",
                    failures=[f"Task {t.id}: {t.result}" for t in failed_tasks],
                )
            return TestResult(
                passed=True,
                summary="No test runner found, but all tasks completed.",
                failures=[],
            )

        # Use LLM to analyze test output
        prompt = (
            f"{TESTER_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Test output:\n{test_output}\n"
        )

        # Choose tool schema format based on provider
        if self.client.settings.provider.value == "anthropic":
            tools = ANTHROPIC_TOOLS
        else:
            tools = OPENAI_TOOLS

        async def tool_executor(tool_name: str, tool_input: dict) -> str:
            if tool_name == "run_command":
                return await run_command(tool_input["command"])
            elif tool_name == "read_file":
                from furrow.tools import read_file
                return await read_file(tool_input["path"])
            return f"Unknown tool: {tool_name}"

        response = await self.client.complete_with_tools(
            prompt=prompt,
            system=TESTER_PROMPT,
            model=self.client.settings.tester_model,
            tools=tools,
            tool_executor=tool_executor,
            max_tool_rounds=5,
        )

        # Parse the JSON response
        try:
            # Try to extract JSON from the response (may be wrapped in markdown)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: simple heuristic parsing
        passed = "passed" in response.lower() and "fail" not in response.lower()
        failures = re.findall(r'(?:FAIL|ERROR|Error|error)[:\s]+([^\n]+)', test_output)
        return TestResult(passed=passed, summary=response, failures=failures)

    async def _run_tests(self) -> str:
        """Run available test commands and return combined output."""
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        outputs = []
        for cmd in candidates:
            try:
                output = await run_command(" ".join(cmd))
                if output and "not found" not in output.lower():
                    outputs.append(f"=== {' '.join(cmd)} ===\n{output}")
            except Exception:
                continue
        if not outputs:
            return "No test runner found."
        return "\n\n".join(outputs)
