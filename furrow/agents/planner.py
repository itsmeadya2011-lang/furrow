from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog
import tenacity

from furrow.agents.prompts import PLANNER_PROMPT, WORKSPACE_CONTEXT_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)

MAX_PLAN_ATTEMPTS = 3
MAX_WORKSPACE_FILES = 50


def _log_retry(retry_state: tenacity.RetryCallState) -> None:
    if retry_state.outcome is None or not retry_state.outcome.failed:
        return
    ex = retry_state.outcome.exception()
    logger.warning(
        "plan_retrying_after_parse_failure",
        error=str(ex),
        attempt=retry_state.attempt_number,
    )


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = self.client.settings

    def _workspace_context(self) -> str:
        workspace = self.settings.workspace
        files = self.client.list_files(workspace)
        if not files:
            return ""
        visible = files[:MAX_WORKSPACE_FILES]
        suffix = (
            f"\n...and {len(files) - len(visible)} more"
            if len(files) > MAX_WORKSPACE_FILES
            else ""
        )
        file_list = "\n".join(f"  - {f}" for f in visible) + suffix
        logger.debug("workspace_context", file_count=len(files))
        return WORKSPACE_CONTEXT_PROMPT.format(
            workspace=str(workspace),
            file_list=file_list,
            file_contents="(planning phase - file contents not loaded yet)",
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON object from text, handling markdown code fences and extra text."""
        # Strip leading/trailing whitespace
        text = text.strip()
        # If the text is wrapped in markdown code fences, remove them
        if text.startswith("```") and text.endswith("```"):
            # Extract the content between the first and last triple backticks
            lines = text.splitlines()
            # Remove the first line if it starts with ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove the last line if it ends with ```
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        # Find the first '{' and the last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return text  # No JSON object found, return as is
        return text[start:end+1]

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(MAX_PLAN_ATTEMPTS),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        retry=tenacity.retry_if_exception_type((json.JSONDecodeError, ValueError)),
        reraise=True,
        before_sleep=_log_retry,
    )
    async def _generate_plan(self, prompt: str) -> Plan:
        response = await self.client.complete(
            prompt, model=self.settings.planner_model
        )
        logger.debug("plan_response_received", response_length=len(response))
        json_str = self._extract_json(response)
        logger.debug("json_extracted", json_str=json_str[:2000])
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM returned invalid JSON (parse error): {e.msg}\n"
                f"Raw LLM response (truncated):\n{response[:2000]}\n"
                f"Extracted JSON (truncated):\n{json_str[:2000]}"
            ) from e
        if not isinstance(data, dict) or "tasks" not in data:
            raise ValueError(
                "LLM response is valid JSON but missing required 'tasks' field.\n"
                f"Raw LLM response (truncated):\n{response[:2000]}\n"
                f"Extracted JSON (truncated):\n{json_str[:2000]}"
            )
        return Plan(**data)

    async def plan(self, goal: str) -> Plan:
        logger.info("planning_started", goal=goal)
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n{self._workspace_context()}"
        try:
            plan = await self._generate_plan(prompt)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("planning_failed", goal=goal, error=str(e))
            raise ValueError(
                f"Failed to parse a valid plan from the planner model after "
                f"{MAX_PLAN_ATTEMPTS} attempts.\nLast error:\n{e}"
            ) from e
        logger.info("planning_succeeded", goal=goal, task_count=len(plan.tasks))
        return plan
