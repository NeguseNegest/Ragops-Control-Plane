PYTHON ?= python3.12
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/python -m pip

.PHONY: setup lint test services-up docker-up ingest-dry-run ingest index index-recreate serve dashboard clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(BIN)/ruff check src tests scripts dashboard

test:
	$(BIN)/python -m pytest

services-up:
	docker compose up -d qdrant mlflow

docker-up:
	docker compose up --build

ingest-dry-run:
	PYTHONPATH=src $(BIN)/python scripts/ingest.py --dry-run

ingest:
	PYTHONPATH=src $(BIN)/python scripts/ingest.py

index:
	PYTHONPATH=src $(BIN)/python scripts/build_index.py

index-recreate:
	PYTHONPATH=src $(BIN)/python scripts/build_index.py --recreate

serve:
	PYTHONPATH=src $(BIN)/uvicorn ragops.app:app --reload

dashboard:
	PYTHONPATH=src $(BIN)/streamlit run dashboard/app.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
