from __future__ import annotations

import inspect
import re

import pytest

from furrow.agents.prompts import (
    PLANNER_PROMPT,
    PLANNER_SYSTEM,
    TESTER_PROMPT,
    TESTER_SYSTEM,
    WORKER_SYSTEM,
    WORKER_TOOLS,
)
from furrow.agents.worker import WorkerAgent

EXPECTED_TOOL_NAMES = {"read_file", "write_file", "list_files", "run_command"}
PROMPT_CONSTANTS = {
    "PLANNER_PROMPT": PLANNER_PROMPT,
    "TESTER_PROMPT": TESTER_PROMPT,
    "PLANNER_SYSTEM": PLANNER_SYSTEM,
    "WORKER_SYSTEM": WORKER_SYSTEM,
    "TESTER_SYSTEM": TESTER_SYSTEM,
}


def test_worker_tools_is_non_empty_list_of_dicts():
    assert isinstance(WORKER_TOOLS, list)
    assert WORKER_TOOLS
    for tool in WORKER_TOOLS:
        assert isinstance(tool, dict)
        assert {"name", "description", "parameters"} <= tool.keys()


def test_worker_tool_names_are_exactly_the_expected_set():
    names = [tool["name"] for tool in WORKER_TOOLS]
    assert set(names) == EXPECTED_TOOL_NAMES
    assert len(names) == len(set(names))


def test_every_tool_description_is_a_non_empty_string():
    for tool in WORKER_TOOLS:
        assert isinstance(tool["description"], str)
        assert tool["description"].strip()


@pytest.mark.parametrize("tool", WORKER_TOOLS, ids=lambda t: t["name"])
def test_tool_parameters_are_a_valid_object_schema(tool):
    params = tool["parameters"]
    assert isinstance(params, dict)
    assert params.get("type") == "object"
    properties = params.get("properties")
    assert isinstance(properties, dict)
    for prop_name, prop_schema in properties.items():
        assert isinstance(prop_schema, dict), prop_name
        assert "type" in prop_schema, prop_name
    required = params.get("required", [])
    assert isinstance(required, list)
    assert all(isinstance(field, str) for field in required)
    assert set(required) <= set(properties)


@pytest.mark.parametrize(
    ("tool_name", "expected_properties"),
    [
        ("run_command", {"command"}),
        ("read_file", {"path"}),
        ("write_file", {"path", "content"}),
        ("list_files", {"path"}),
    ],
)
def test_tool_exposes_expected_properties(tool_name, expected_properties):
    tool = next(t for t in WORKER_TOOLS if t["name"] == tool_name)
    assert expected_properties <= set(tool["parameters"]["properties"])


@pytest.mark.parametrize("name", sorted(PROMPT_CONSTANTS))
def test_prompts_are_non_empty_and_have_no_placeholders(name):
    prompt = PROMPT_CONSTANTS[name]
    assert isinstance(prompt, str)
    assert prompt.strip()
    assert "TODO" not in prompt
    assert "FIXME" not in prompt


def test_tool_names_match_the_worker_dispatcher():
    source = inspect.getsource(WorkerAgent._execute_tool)
    dispatched = set(re.findall(r'name\s*==\s*"([^"]+)"', source))
    assert dispatched == EXPECTED_TOOL_NAMES