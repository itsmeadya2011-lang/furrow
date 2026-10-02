"""Packaging + entry-point guarantees for the ``furrow`` CLI.

``pyproject.toml`` ships ``[project.scripts] furrow = "furrow.cli.main:main"`` and
``[tool.setuptools.packages.find] include = ["furrow*"]``. Both halves have to
hold for an installed ``furrow`` command to work at all: a missing
``furrow/cli/__init__.py`` makes ``find_packages`` skip the directory, so the
package is silently dropped from the wheel and the console script dies on import
while the source checkout keeps working. These tests pin that down.

They also pin the model-override contract shared by the two front ends: a
supplied model must reach the agents, and an omitted one must not disturb the
deliberate per-role defaults or the process-wide ``settings`` singleton.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from typing import Any

import click
import pytest
from setuptools import find_packages

from furrow.cli.main import ROLE_MODEL_FIELDS, build_run_settings
from furrow.config import Settings
from furrow.config import settings as global_settings
from furrow.web import server

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


# --------------------------------------------------------------------------- #
# 1. the package is discovered
# --------------------------------------------------------------------------- #


def test_furrow_cli_is_an_importable_package() -> None:
    module = importlib.import_module("furrow.cli")
    module_main = importlib.import_module("furrow.cli.main")

    assert module.__name__ == "furrow.cli"
    # Not just importable: importable as a *module*. A non-module object bound
    # to the same name would make `import ... as` hand callers the wrong thing.
    assert inspect.ismodule(module_main)
    assert module_main.__name__ == "furrow.cli.main"


def test_find_packages_ships_the_cli_subpackage() -> None:
    # A directory with no __init__.py is not a package, so it never reaches the
    # wheel -- which is exactly how the console script breaks while the checkout
    # still runs.
    assert (REPO_ROOT / "furrow" / "cli" / "__init__.py").is_file()

    assert "furrow.cli" in find_packages(where=str(REPO_ROOT), include=["furrow*"])


def test_find_packages_matches_the_include_pattern_declared_in_pyproject() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^include\s*=\s*\[(?P<body>[^\]]*)\]', text, re.MULTILINE)

    assert match is not None, "pyproject.toml declares no packages.find include list"
    patterns = re.findall(r'"([^"]+)"', match.group("body"))

    assert patterns == ["furrow*"]
    assert "furrow.cli" in find_packages(where=str(REPO_ROOT), include=patterns)


def test_the_console_script_target_is_importable_and_callable() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    # tomllib does not exist on 3.10 (the floor in pyproject.toml), so read the
    # one line we care about with a regex rather than a TOML parser.
    match = re.search(r'^\s*furrow\s*=\s*"(?P<target>[^"]+)"', text, re.MULTILINE)

    assert match is not None, "pyproject.toml declares no [project.scripts] furrow entry"

    module_name, _, attribute = match.group("target").partition(":")
    assert module_name == "furrow.cli.main"

    # Deliberately not ``import furrow.cli.main as cli_main``: that ``as`` form
    # binds the *attribute* ``main`` on the ``furrow.cli`` package, so a
    # re-export of the click Group in ``furrow/cli/__init__.py`` would silently
    # shadow the submodule and every assertion below would pass against the
    # wrong object (``Group.main`` is itself callable). ``import_module`` always
    # returns the module itself.
    module = importlib.import_module(module_name)

    assert inspect.ismodule(module)
    assert module.__name__ == "furrow.cli.main"

    entry_point = getattr(module, attribute)

    assert isinstance(entry_point, click.Group)
    assert callable(entry_point)


# --------------------------------------------------------------------------- #
# 2. a supplied model reaches the agents
# --------------------------------------------------------------------------- #

ROLE_FIELDS = ("planner_model", "worker_model", "tester_model")


def test_role_fields_cover_every_agent_role() -> None:
    assert tuple(ROLE_MODEL_FIELDS) == ROLE_FIELDS
    assert tuple(server.ROLE_MODEL_FIELDS) == ROLE_FIELDS


def test_build_run_settings_propagates_a_supplied_model_to_every_role() -> None:
    run_settings = build_run_settings(model="x")

    assert run_settings.model == "x"
    for field in ROLE_FIELDS:
        assert getattr(run_settings, field) == "x", field


def test_settings_for_model_propagates_a_supplied_model_to_every_role() -> None:
    copy = server.settings_for_model("x")

    assert copy is not None
    assert copy.model == "x"
    for field in ROLE_FIELDS:
        assert getattr(copy, field) == "x", field


# --------------------------------------------------------------------------- #
# 3. an omitted model changes nothing
# --------------------------------------------------------------------------- #


def test_build_run_settings_without_a_model_keeps_the_role_defaults() -> None:
    run_settings = build_run_settings()

    for field in ROLE_FIELDS:
        assert getattr(run_settings, field) == getattr(global_settings, field)
    # Not merely equal to each other: the deliberate per-role defaults survive.
    assert run_settings.planner_model == Settings.model_fields["planner_model"].default
    assert run_settings.worker_model == Settings.model_fields["worker_model"].default
    assert run_settings.tester_model == Settings.model_fields["tester_model"].default


def test_build_run_settings_leaves_other_omitted_options_alone() -> None:
    run_settings = build_run_settings(provider="openai")

    assert run_settings.provider == "openai"
    assert run_settings.max_cycles == global_settings.max_cycles
    assert run_settings.max_parallel_tasks == global_settings.max_parallel_tasks
    for field in ROLE_FIELDS:
        assert getattr(run_settings, field) == getattr(global_settings, field)


def test_settings_for_model_without_a_model_forges_no_copy() -> None:
    # Unchanged contract: nothing to override -> None, so the caller keeps using
    # the env-derived singleton and its per-role defaults.
    assert server.settings_for_model(None) is None
    assert server.settings_for_model("") is None


# --------------------------------------------------------------------------- #
# 4. the singleton is never mutated
# --------------------------------------------------------------------------- #


def test_the_settings_singleton_is_untouched_by_both_front_ends() -> None:
    before: dict[str, Any] = global_settings.model_dump()

    build_run_settings(model="x", provider="openai", max_cycles=3, max_parallel_tasks=2)
    server.settings_for_model("x")

    assert global_settings.model_dump() == before


def test_an_override_does_not_leak_into_the_next_caller() -> None:
    server.settings_for_model("x")
    before: dict[str, Any] = global_settings.model_dump()

    later = build_run_settings()

    for field in ROLE_FIELDS:
        assert getattr(later, field) == getattr(global_settings, field)
    assert global_settings.model_dump() == before


# --------------------------------------------------------------------------- #
# 5. the two front ends agree
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", ["x", "gpt-4o-mini", "some/model:v2"])
def test_both_front_ends_produce_the_same_role_models(model: str) -> None:
    from_cli = build_run_settings(model=model)
    from_web = server.settings_for_model(model)

    assert from_web is not None
    for field in ROLE_FIELDS:
        assert getattr(from_cli, field) == getattr(from_web, field) == model
    assert from_cli.model == from_web.model == model
