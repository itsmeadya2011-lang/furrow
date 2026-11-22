# Makefile for Furrow
#
# Developer convenience targets. Requires Python 3.10+.
# Install dev dependencies first:  pip install -e ".[dev]"

.PHONY: install test lint typecheck clean all

install: ## Install the package in editable mode with dev dependencies
	pip install -e ".[dev]"

test: ## Run the full offline test suite (no API keys required)
	python -m pytest -q

lint: ## Lint the codebase with ruff
	python -m ruff check .

typecheck: ## Type check the furrow package with mypy
	python -m mypy furrow

all: test lint typecheck ## Run test, lint, and typecheck

clean: ## Remove Python cache files and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .mypy_cache -prune -exec rm -rf {} +
	find . -type d -name .ruff_cache -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name build -prune -exec rm -rf {} +
	find . -type d -name dist -prune -exec rm -rf {} +
	find . -type d -name *.egg-info -prune -exec rm -rf {} +
	rm -rf .furrow

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
