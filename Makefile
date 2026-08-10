PYTHON ?= python3.12
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/python -m pip

.PHONY: setup lint test test-retrieval-metrics services-up docker-up ingest-dry-run ingest index index-recreate generate-synthetic-qa review-synthetic-qa bootstrap-retrieval-labels label-retrieval validate-retrieval-labels serve dashboard clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(BIN)/ruff check src tests scripts dashboard

test:
	$(BIN)/python -m pytest

test-retrieval-metrics:
	$(BIN)/python -m pytest tests/test_retrieval_metrics.py

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

generate-synthetic-qa:
	PYTHONPATH=src $(BIN)/python scripts/generate_synthetic_qa.py

review-synthetic-qa:
	PYTHONPATH=src $(BIN)/python scripts/review_synthetic_qa.py

bootstrap-retrieval-labels:
	PYTHONPATH=src $(BIN)/python scripts/label_retrieval.py --bootstrap-approved-synthetic --reviewer codex-source-audit

label-retrieval:
	PYTHONPATH=src $(BIN)/python scripts/label_retrieval.py

validate-retrieval-labels:
	PYTHONPATH=src $(BIN)/python scripts/label_retrieval.py --validate-only

serve:
	PYTHONPATH=src $(BIN)/uvicorn ragops.app:app --reload

dashboard:
	PYTHONPATH=src $(BIN)/streamlit run dashboard/app.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
