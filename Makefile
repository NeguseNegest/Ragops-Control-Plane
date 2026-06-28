PYTHON ?= python3.12
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/python -m pip

.PHONY: setup lint test docker-up clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(BIN)/ruff check src tests

test:
	$(BIN)/pytest

docker-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
