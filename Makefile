.PHONY: install install-dev lint lint-fix typecheck test test-cov clean format all

PYTHON ?= python3

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

# ---------------------------------------------------------------------------
# Linting & Formatting
# ---------------------------------------------------------------------------
lint:
	ruff check furrow/ tests/

lint-fix:
	ruff check --fix furrow/ tests/

format:
	ruff format furrow/ tests/

# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------
typecheck:
	mypy furrow/

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test:
	pytest

test-cov:
	pytest --cov=furrow --cov-report=term-missing

# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------
all: lint-fix format typecheck test-cov

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	rm -rf *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
