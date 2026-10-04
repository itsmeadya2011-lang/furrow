"""Package-contract tests for the ``furrow`` distribution.

This file used to hold two placeholder assertions that exercised :class:`Plan` and
:class:`TestResult` -- both of which now have real homes in ``test_models.py``, so the
assertions were duplicate value wearing a test's clothes. It is repurposed instead of
deleted because the obvious replacement name, ``test_orchestrator.py``, belongs to a
different wave that owns the orchestrator's own tests. What lives *here* is the set of
facts that only hold for the package as a whole and that a unit test of any one module
cannot see:

* the version in ``furrow/__init__.py`` matches the distribution metadata, so ``pip`` and
  ``import furrow`` never disagree;
* every name in ``furrow.__all__`` really exists (a stale ``__all__`` breaks
  ``from furrow import *`` and star-importing downstream code);
* the console-script entry point named in ``pyproject.toml`` resolves to a real command;
* the deprecated ``furrow.config`` re-exports still point at the same objects the
  ``furrow.models`` originals do, which is what keeps every existing import working;
* ``import furrow`` stays *light*: it must not drag in ``anthropic``/``openai``, which is
  why ``LLMClient`` is deliberately absent from the root namespace. Measured in a
  subprocess, because this session's own imports would make an in-process check vacuous.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import furrow

#: Repository root, i.e. the directory holding ``pyproject.toml``.
REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _project_table() -> str:
    """Body of the ``[project]`` table in pyproject.toml.

    Hand-rolled because Python 3.10 has no ``tomllib`` and the suite must not gain a
    dependency just to read two strings.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    assert "[project]" in text, "pyproject.toml has no [project] table"
    return text.split("[project]", 1)[1].split("\n[", 1)[0]


def test_version_matches_pyproject():
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', _project_table())

    assert match is not None, "no `version = \"...\"` in the [project] table"
    assert furrow.__version__ == match.group(1)


def test_everything_in_all_is_importable():
    assert "FileChange" in furrow.__all__
    for name in furrow.__all__:
        assert hasattr(furrow, name), f"furrow.__all__ advertises missing name {name!r}"
        assert getattr(furrow, name) is not None


def test_all_is_sorted():
    # A stable __all__ keeps `from furrow import *` diffable and the module greppable.
    assert furrow.__all__ == sorted(furrow.__all__)


def test_cli_entry_point_resolves():
    from furrow.cli import main

    assert callable(main)


def test_the_console_script_target_is_importable():
    # `[project.scripts] furrow = "furrow.cli.main:main"` -- if that path ever rots, the
    # installed `furrow` command breaks while every in-repo test still passes.
    declared = re.search(
        r'(?m)^furrow\s*=\s*"(?P<target>[\w.]+:[a-zA-Z_]\w*)"',
        PYPROJECT.read_text(encoding="utf-8"),
    )

    assert declared is not None, "pyproject.toml no longer declares the furrow console script"

    module_name, _, attribute = declared.group("target").partition(":")
    module = __import__(module_name, fromlist=[attribute])
    assert callable(getattr(module, attribute))
    # and the same object is reachable from the subpackage, which is how CLI tests will
    # reach it (`from furrow.cli import main`).
    import furrow.cli

    assert furrow.cli.main is getattr(module, attribute)


def test_config_still_re_exports_the_domain_models():
    """`furrow.config` keeps a backwards-compatibility shim for the models.

    `agents/planner.py`, `agents/worker.py`, `agents/tester.py` and
    `core/orchestrator.py` all still do `from furrow.config import TaskModel, Plan,
    TestResult`. The re-export must stay the *same object* as the `furrow.models`
    original, or `isinstance` checks and `TaskStatus` comparisons would start failing
    across the two import paths.
    """
    from furrow import config, models
    from furrow.config import Plan, TaskModel, TestResult

    assert TaskModel is models.TaskModel
    assert Plan is models.Plan
    assert TestResult is models.TestResult
    assert config.FileChange is models.FileChange
    assert config.TaskStatus is models.TaskStatus
    assert config.WorkerResult is models.WorkerResult


def test_config_advertises_the_shimmed_names_in_all():
    from furrow import config

    for name in ("FileChange", "Plan", "TaskModel", "TaskStatus", "TestResult", "WorkerResult"):
        assert name in config.__all__


def test_the_root_namespace_does_not_expose_the_llm_client():
    """`LLMClient` is deliberately absent from `furrow`'s root namespace.

    Re-exporting it would make `import furrow` import `furrow.llm`, which imports the
    anthropic and openai SDKs: ~1.5 s of import time on every `furrow --version`, and a
    hard dependency on two optional extras just to print help. Callers import it directly
    instead (`from furrow.llm import LLMClient`), which is asserted next.
    """
    assert not hasattr(furrow, "LLMClient")
    assert "LLMClient" not in furrow.__all__

    from furrow.llm import LLMClient

    assert LLMClient.__module__ == "furrow.llm"


def test_importing_furrow_loads_no_provider_sdk():
    """A fresh interpreter must be able to import `furrow` with no SDK in ``sys.modules``.

    Measured in a subprocess on purpose: by the time this test runs, conftest and the other
    test modules have already imported `furrow.llm`, so an in-process check of
    ``sys.modules`` would only prove that the suite is slow.
    """
    probe = (
        "import sys, furrow; "
        "print(sorted(m for m in ('anthropic', 'openai', 'httpx', 'tenacity') "
        "if m in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                part for part in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if part
            ),
        },
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]", completed.stdout
