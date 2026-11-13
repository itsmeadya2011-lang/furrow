"""Allow running Furrow as a module: ``python -m furrow``.

Delegates to the Click CLI defined in :mod:`furrow.cli.main`.
"""

from furrow.cli.main import main

if __name__ == "__main__":
    main()
