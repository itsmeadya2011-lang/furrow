#!/bin/bash
set -e

cd /workspace/64b5f2dc-1bcb-407e-b26a-0216ac2acba8/sessions/agent_eee7de79-2d80-4116-92d7-d0d95b424264

# Install in development mode
pip install -e ".[dev]" --quiet 2>&1 | tail -5

# Run tests
python -m pytest tests/ -v