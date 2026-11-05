from __future__ import annotations

from furrow.agents.prompts import (
    PLANNER_PROMPT,
    PLANNER_SYSTEM,
    TESTER_PROMPT,
    TESTER_SYSTEM,
    WORKER_SYSTEM,
    WORKER_TOOLS,
)


def test_prompts_are_non_empty_strings() -> None:
    for value in (
        PLANNER_PROMPT,
        PLANNER_SYSTEM,
        TESTER_PROMPT,
        TESTER_SYSTEM,
        WORKER_SYSTEM,
    ):
        assert isinstance(value, str)
        assert value.strip()


def test_worker_tools_schema_shape() -> None:
    assert isinstance(WORKER_TOOLS, list)
    names = [tool["name"] for tool in WORKER_TOOLS]
    assert names == ["read_file", "write_file", "list_files", "run_command"]
    for tool in WORKER_TOOLS:
        assert "description" in tool and tool["description"]
        params = tool["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
