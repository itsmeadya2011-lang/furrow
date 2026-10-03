"""Load-bearing guard: make sure this suite imports *this* checkout's ``furrow``.

Why this file exists
--------------------
This machine carries a stale editable install of ``furrow`` (the
``__editable__.furrow-*.pth`` finder in site-packages) that points at a
*different* session checkout.  Any Python process whose ``sys.path[0]`` is not
this repository therefore imports that foreign tree instead of the working tree.

That failure mode is invisible: pytest still reports green, but the assertions
ran against someone else's source.  It already produced three misleading
verification results in a single session -- one agent read a symbol that does
not exist in this repo, and two others ran verification against the wrong
``furrow`` entirely.

DO NOT DELETE THIS FILE AS REDUNDANT.  It is not redundant: ``import furrow``
inside a test module resolves correctly on the machine where that test was
written and wrongly everywhere else, so nothing in the suite itself would ever
notice.  This assertion is the only thing standing between a future maintainer
and a silent false-green.
"""

import os
from pathlib import Path

import pytest

_EXPECTED_INIT = Path(__file__).resolve().parent.parent / "furrow" / "__init__.py"


def _assert_furrow_is_this_checkout() -> None:
    """Raise ``pytest.UsageError`` unless ``furrow`` resolves to this checkout."""
    expected = os.path.realpath(_EXPECTED_INIT)

    try:
        import furrow
    except ImportError as exc:
        raise pytest.UsageError(
            "Could not import `furrow` while collecting tests.\n"
            f"  expected: {expected}\n"
            f"  error:    {exc}\n"
            "Run pytest from the repository root, and/or (re)install this checkout:\n"
            "  python3 -m pip install -e .\n"
            "(tests/conftest.py is load-bearing; see its module docstring.)"
        ) from exc

    resolved = os.path.realpath(furrow.__file__)
    if resolved != expected:
        raise pytest.UsageError(
            "The `furrow` package under test is NOT this checkout.\n"
            f"  resolved: {resolved}\n"
            f"  expected: {expected}\n"
            "A stale editable install is shadowing this working tree, so this suite "
            "would silently validate foreign source code.\n"
            "Remedies:\n"
            "  - run pytest from the repository root:  python3 -m pytest\n"
            "  - and/or reinstall this checkout:        python3 -m pip install -e .\n"
            "(tests/conftest.py is load-bearing; see its module docstring.)"
        )


# Runs at conftest *import* time on purpose.  pytest imports test modules during
# collection, which happens BEFORE session fixtures run; by then the test modules
# have already imported the foreign `furrow`, so an autouse fixture alone would be
# too late.
_assert_furrow_is_this_checkout()


@pytest.fixture(scope="session", autouse=True)
def _furrow_is_this_checkout() -> None:
    """Defence in depth: re-check once the session starts, in case `furrow` was
    re-imported or re-pointed after conftest import."""
    _assert_furrow_is_this_checkout()