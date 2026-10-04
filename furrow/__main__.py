"""Entry point for ``python -m furrow``.

Mirrors the ``furrow`` console script declared in ``pyproject.toml`` so both
invocations behave identically.
"""

from __future__ import annotations

from furrow.cli.main import main

if __name__ == "__main__":
    main()
