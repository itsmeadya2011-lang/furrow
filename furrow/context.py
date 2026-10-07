from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.llm import LLMClient
from furrow.config import Settings

if TYPE_CHECKING:
    from furrow.config import Settings


async def compact_context(
    client: LLMClient,
    goal: str,
    task_history: list[dict],
    settings: Settings | None = None,
    max_tokens: int = 4000,
) -> str:
    """Compact the context by summarizing task history."""
    if not task_history:
        return goal

    history_text = "\n".join(
        f"Cycle {i+1}: {h.get('summary', 'No summary')}" for i, h in enumerate(task_history)
    )

    prompt = f"""Summarize the following task history for a coding agent working on this goal:

Goal: {goal}

Task History:
{history_text}

Provide a concise summary that captures:
1. What has been accomplished so far
2. What remains to be done
3. Any important context or decisions made

Keep the summary under {max_tokens // 4} words."""

    return await client.complete(prompt, model=settings.model if settings else None)


async def should_compact(task_history: list[dict], threshold: int = 10) -> bool:
    """Determine if context should be compacted."""
    return len(task_history) >= threshold