"""Tests for the package entry point: :mod:`furrow`'s lazy public surface.

``furrow/__init__.py`` resolves ``LLMClient`` and ``Settings`` on first access
through a PEP 562 module-level ``__getattr__``.  ``furrow.config`` ends with
``settings = Settings()``, executed at import time, so eager re-exports turned a
malformed ``.env`` / ``FURROW_*`` value into a raw pydantic ``ValidationError``
raised out of ``import furrow`` itself - before the friendly error handling in
``furrow.cli.main`` could run.  The tests below pin both halves of that contract:
the names still resolve for every existing caller, and a bare ``import furrow``
stays free of the failure.

The subprocess tests are the load-bearing ones, and they are deliberately
paranoid about *which* copy of ``furrow`` got imported: a bare ``import furrow``
can resolve to a different session's editable install unless ``PYTHONPATH`` and
the working directory are both pinned.  Every helper below sets both and asserts
the loaded ``furrow.__file__`` lives under this checkout, so no assertion in this
module can be satisfied by another session's copy of the tree.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from typing import Any

import pytest

import furrow
import furrow.config
import furrow.llm

__all__ = ["REPO_ROOT"]

#: This checkout.  ``tests/test_package.py`` -> ``tests`` -> repository root.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: An unknown name.  Kept in a variable so the ``getattr`` call below does not trip
#: ruff's constant-``getattr`` rule (B009); the test needs the *real* module
#: ``__getattr__`` to run, which means a genuine attribute miss.
_UNKNOWN_NAME = "Nope"

#: Generous enough that a cold interpreter on a loaded machine is never what fails,
#: bounded enough that a hang is a failure rather than a stuck suite.
_SUBPROCESS_TIMEOUT = 60.0

#: Only these variables reach the child, so an ambient ``FURROW_*`` or
#: ``ANTHROPIC_API_KEY`` in the surrounding shell can never decide a result.
_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TZ")

#: Prepended to every child program.  ``furrow.__file__`` is a real module attribute,
#: so reading it cannot re-enter the lazy ``__getattr__`` - the statements that
#: follow still exercise exactly the import behaviour under test.
_PREAMBLE = "import furrow\nprint(furrow.__file__)\n"


def _child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return a minimal, explicit environment for a child interpreter.

    Built from an allowlist instead of inherited from ``os.environ`` (and never by
    mutating it), so the child sees this repository and nothing else that could
    influence an import.
    """
    env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra:
        env.update(extra)
    return env


def _run_child(
    source: str, cwd: pathlib.Path, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run *source* in a fresh interpreter rooted at this checkout.

    Args:
        source: Python statements appended to ``_PREAMBLE``.
        cwd: Working directory.  ``Settings`` reads ``.env`` from the *current*
            directory, so this is part of the test, not just hygiene.
        extra_env: Extra environment variables for this call only.

    Returns:
        The finished child process.
    """
    return subprocess.run(
        [sys.executable, "-c", _PREAMBLE + source],
        cwd=str(cwd),
        env=_child_env(extra_env),
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
        check=False,
    )


def _loaded_module_path(result: subprocess.CompletedProcess[str]) -> pathlib.Path:
    """Return the ``furrow/__init__.py`` the child actually imported.

    Asserting that path is under :data:`REPO_ROOT` is what makes the other subprocess
    assertions trustworthy: an editable install belonging to another session would
    otherwise satisfy them with a copy of the tree this file is not testing.
    """
    lines = result.stdout.splitlines()
    assert lines, f"child printed no module path (rc={result.returncode}): {result.stderr}"
    path = pathlib.Path(lines[0]).resolve()
    assert path.is_relative_to(REPO_ROOT), (
        f"child imported {path}, which is outside {REPO_ROOT}; "
        "these assertions would be testing a different checkout"
    )
    return path


# --------------------------------------------------------------------------- in-process


def test_all_lists_both_public_names() -> None:
    assert furrow.__all__ == ["LLMClient", "Settings"]


def test_dir_is_sorted_all() -> None:
    assert dir(furrow) == sorted(furrow.__all__)


def test_from_import_resolves_the_real_classes() -> None:
    from furrow import LLMClient, Settings

    assert Settings is furrow.config.Settings
    assert LLMClient is furrow.llm.LLMClient


def test_star_import_yields_both_names() -> None:
    # ``import *`` is only legal at module level, so the statement is compiled here.
    namespace: dict[str, Any] = {}
    exec("from furrow import *", namespace)

    assert "LLMClient" in namespace
    assert "Settings" in namespace
    assert namespace["Settings"] is furrow.config.Settings
    assert namespace["LLMClient"] is furrow.llm.LLMClient


def test_unknown_attribute_raises_attribute_error() -> None:
    assert not hasattr(furrow, _UNKNOWN_NAME)

    with pytest.raises(AttributeError) as excinfo:
        getattr(furrow, _UNKNOWN_NAME)

    assert str(excinfo.value) == f"module 'furrow' has no attribute '{_UNKNOWN_NAME}'"


def test_access_caches_the_name_in_module_globals(monkeypatch: pytest.MonkeyPatch):
    module_globals: dict[str, Any] = furrow.__dict__
    # Drop any cached value first, so the test starts from the un-cached state
    # whatever order the suite ran in.  monkeypatch restores the entry on teardown.
    monkeypatch.delitem(module_globals, "Settings", raising=False)

    resolved = furrow.Settings  # the real module __getattr__ runs, and caches

    assert resolved is furrow.config.Settings
    assert module_globals["Settings"] is resolved

    # A second access must be served straight from the module globals, so the
    # resolver is never entered again.  Recording __getattr__ proves which happened.
    calls: list[str] = []

    def _record(name: str) -> Any:
        calls.append(name)
        return object()

    monkeypatch.setitem(module_globals, "__getattr__", _record)

    assert furrow.Settings is resolved
    assert calls == []


# --------------------------------------------------------------------------- subprocess


def test_bare_import_uses_this_checkout_and_loads_no_submodule(tmp_path: pathlib.Path):
    """The guard for every other subprocess test, and the laziness claim itself."""
    result = _run_child(
        "import sys\n"
        "print('furrow.config' in sys.modules, 'furrow.llm' in sys.modules)\n",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert _loaded_module_path(result).name == "__init__.py"
    assert result.stdout.splitlines()[1] == "False False"


def test_import_survives_an_out_of_range_env_var(tmp_path: pathlib.Path):
    """``import furrow`` must not touch Settings, so a bad value cannot fail here."""
    result = _run_child("pass\n", cwd=tmp_path, extra_env={"FURROW_MAX_CYCLES": "-1"})

    assert result.returncode == 0, result.stderr
    assert _loaded_module_path(result)


def test_settings_access_still_reports_the_bad_env_var(tmp_path: pathlib.Path):
    """The other half of the contrast: the value *is* rejected, loudly."""
    result = _run_child("furrow.Settings\n", cwd=tmp_path, extra_env={"FURROW_MAX_CYCLES": "-1"})

    assert result.returncode == 1, result.stdout
    _loaded_module_path(result)
    assert "max_cycles" in result.stderr
    assert "greater_than_equal" in result.stderr


def test_import_survives_a_malformed_dotenv(tmp_path: pathlib.Path):
    (tmp_path / ".env").write_text("FURROW_MAX_CYCLES=not-a-number\n", encoding="utf-8")

    result = _run_child("pass\n", cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert _loaded_module_path(result)


def test_settings_access_still_reports_the_malformed_dotenv(tmp_path: pathlib.Path):
    (tmp_path / ".env").write_text("FURROW_MAX_CYCLES=not-a-number\n", encoding="utf-8")

    result = _run_child("furrow.Settings\n", cwd=tmp_path)

    assert result.returncode == 1, result.stdout
    _loaded_module_path(result)
    assert "max_cycles" in result.stderr
    assert "int_parsing" in result.stderr