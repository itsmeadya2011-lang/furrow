from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.jsonutil import parse_json_model
from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        # Resolve settings once so the class does not reach through
        # self.client.settings for every field on every call.
        self.settings = self.client.settings

    async def plan(self, goal: str, feedback: str = "", context: str = "") -> Plan:
        """Produce a Plan for *goal*, optionally informed by prior-cycle *feedback*
        (failure text) and a bounded workspace *context* summary.

        The LLM is asked for JSON; if its response is malformed (the most common
        failure mode is a markdown fence or surrounding prose) the request is
        retried up to ``settings.llm_max_retries`` times, each time including the
        parse error and the offending response so the model can correct itself.
        A ``ValueError`` is raised only after the retries are exhausted.
        """
        prompt = _build_prompt(goal, feedback, context)
        last_error: ValueError | None = None
        response = ""
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                response = await self.client.complete(prompt, model=self.settings.planner_model)
                return parse_json_model(response, Plan)
            except ValueError as e:
                last_error = e
                if attempt >= self.settings.llm_max_retries:
                    break
                # Feed the error and the bad response back so the model can fix it.
                prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response failed to parse: {e}\n"
                    f"Previous response:\n{response}\n\n"
                    f"Please return only valid JSON matching the required shape."
                )
        assert last_error is not None  # for type checkers; loop always sets it
        raise ValueError(
            f"Failed to parse plan after {self.settings.llm_max_retries + 1} attempts: {last_error}"
        )


def _build_prompt(goal: str, feedback: str, context: str) -> str:
    sections = [PLANNER_PROMPT, f"Goal: {goal}"]
    if feedback:
        sections.append(
            "Feedback from the previous cycle (real failures the plan must address):\n" + feedback
        )
    if context:
        sections.append(f"Workspace context:\n{context}")
    return "\n\n".join(sections)
