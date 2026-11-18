"""Enable ``python -m furrow`` as an entry point to the Furrow CLI."""

import sys

from furrow.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
