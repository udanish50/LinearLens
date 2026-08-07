.PHONY: install lint format test validate synthetic

install:
	python -m pip install -e '.[dev]'

lint:
	ruff format --check .
	ruff check .

format:
	ruff format .
	ruff check . --fix

test:
	pytest

validate:
	python scripts/validate_release.py

synthetic:
	linear-lens synthetic
