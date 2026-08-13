PYTHON ?= python3

.PHONY: test integration-test lint format format-check build check-dist

test:
	$(PYTHON) -m pytest tests/unit_tests -q

integration-test:
	$(PYTHON) -m pytest tests/integration_tests -q

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests

format-check:
	$(PYTHON) -m ruff format --check src tests

build:
	$(PYTHON) -m build

check-dist:
	$(PYTHON) -m twine check dist/*
