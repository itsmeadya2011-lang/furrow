from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


def _response_text(response: object) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, (bytes, bytearray)):
        return bytes(response).decode("utf-8", errors="replace")
    return str(response)


def _accepts_model_argument(client: Any) -> bool:
    try:
        parameters = inspect.signature(client.complete).parameters.values()
    except (TypeError, ValueError):
        return True

    for parameter in parameters:
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "model" and parameter.kind != inspect.Parameter.POSITIONAL_ONLY:
            return True
    return False


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings
        if self.settings is None:
            client_settings = getattr(client, "settings", None)
            self.settings = client_settings if client_settings is not None else default_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)

    async def plan(self, goal: str, context: str | None = None) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\n<goal>\n{goal}\n</goal>\n"
        if context is not None:
            prompt += f"<context>\n{context}\n</context>\n"

        planner_model = getattr(self.settings, "planner_model", None) or default_settings.planner_model
        last_error: Exception | None = None
        responses: list[str] = []

        for _ in range(2):
            try:
                if _accepts_model_argument(self.client):
                    response = await self.client.complete(prompt, model=planner_model)
                else:
                    response = await self.client.complete(prompt)
            except (IndexError, StopIteration) as exc:
                if responses:
                    last_error = exc
                    break
                raise

            raw_response = _response_text(response)
            responses.append(raw_response)
            try:
                data = json.loads(raw_response)
                if not isinstance(data, dict):
                    raise ValueError(
                        f"plan response must be a JSON object, got {type(data).__name__}"
                    )
                return Plan.model_validate(data)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = exc

        attempt_details = "\n".join(
            f"Attempt {index} response: {response!r}"
            for index, response in enumerate(responses, start=1)
        )
        raise ValueError(
            "Failed to parse and validate plan from LLM after 2 attempts.\n"
            f"{attempt_details}\n"
            f"Response: {responses[-1]}\n"
            f"Last error: {last_error}"
        ) from last_error

