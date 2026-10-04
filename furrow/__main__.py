"""Support ``python -m furrow``, which is the same entry point as ``furrow``.

Without this module ``python -m furrow`` fails with "No module named
furrow.__main__", which is the only way to run the CLI in an environment where
the ``[project.scripts]`` console script has not been installed.
"""

from furrow.cli.main import main

if __name__ == "__main__":
    main()
