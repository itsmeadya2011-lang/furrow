"""Command-line interface for Furrow.

This package provides the CLI entry point with commands:
- `furrow start "<goal>"` - Start a new run (or resume with --resume)
- `furrow resume` - Resume the last persisted run
- `furrow status` - Show the state of the last run
- `furrow web` - Start the web UI server

The CLI handles argument parsing, logging configuration, and orchestrator lifecycle.
"""

from furrow.cli.main import main

__all__ = ["main"]