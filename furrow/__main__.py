"""Make `python -m furrow` work, like `python -m furrow.cli` already does."""

from __future__ import annotations

from furrow.cli.main import main

if __name__ == "__main__":
    main()
