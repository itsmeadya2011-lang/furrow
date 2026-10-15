#!/usr/bin/env python3
"""Test TesterAgent.run method when API keys are not set."""

import asyncio
import os
from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, Settings


async def test_tester_agent_no_api_keys():
    # Ensure no API keys are set
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("FURROW_ANTHROPIC_API_KEY", None)
    os.environ.pop("FURROW_OPENAI_API_KEY", None)

    settings = Settings(
        provider="anthropic",
        anthropic_api_key=None,
        openai_api_key=None,
    )

    agent = TesterAgent(settings=settings)

    goal = "Test goal"
    tasks = [TaskModel(id="1", description="Test task")]

    try:
        result = await agent.run(goal, tasks)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Exception raised: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(test_tester_agent_no_api_keys())