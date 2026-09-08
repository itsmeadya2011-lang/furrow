---
permission:
  bash: "allow"
  edit: "allow"
  read: "allow"
---

# AGENTS.md - Project Configuration

## Commands to run

### Install dependencies
pip install -e ".[dev]"

### Verify import
python -c "import furrow; print('Import OK')"

### Run tests
python -m pytest tests/ -v
