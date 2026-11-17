.PHONY: install dev test lint typecheck clean

install:
	pip install -e ".[dev]"

dev: install

test:
	python -m pytest -q

lint:
	python -m ruff check .

typecheck:
	python -m mypy furrow

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +