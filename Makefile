PYTHON ?= python3.12
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/python -m pip

.PHONY: setup lint test test-retrieval-metrics test-llm-judge test-bm25 validate-dense-evaluation evaluate-dense validate-generation-judge judge-answers review-judgments validate-day20 validate-bm25-config build-bm25-index validate-bm25-index services-up docker-up ingest-dry-run ingest index index-recreate generate-synthetic-qa review-synthetic-qa bootstrap-retrieval-labels label-retrieval validate-retrieval-labels serve dashboard clean

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

test-llm-judge:
	$(BIN)/python -m pytest tests/test_llm_judge.py

test-bm25:
	$(BIN)/python -m pytest tests/test_bm25.py

validate-dense-evaluation:
	PYTHONPATH=src $(BIN)/python scripts/evaluate.py --config configs/dense_baseline.yaml --validate-only

evaluate-dense:
	PYTHONPATH=src $(BIN)/python scripts/evaluate.py --config configs/dense_baseline.yaml

validate-generation-judge:
	PYTHONPATH=src $(BIN)/python scripts/judge_answers.py --config configs/generation_judge.yaml --validate-only

judge-answers:
	PYTHONPATH=src $(BIN)/python scripts/judge_answers.py --config configs/generation_judge.yaml

review-judgments:
	PYTHONPATH=src $(BIN)/python scripts/review_judgments.py --config configs/generation_judge.yaml

validate-day20:
	PYTHONPATH=src $(BIN)/python scripts/review_judgments.py --config configs/generation_judge.yaml --validate-only --require-reviewed

validate-bm25-config:
	PYTHONPATH=src $(BIN)/python scripts/build_bm25_index.py --config configs/bm25_baseline.yaml --validate-only

build-bm25-index:
	PYTHONPATH=src $(BIN)/python scripts/build_bm25_index.py --config configs/bm25_baseline.yaml

validate-bm25-index:
	PYTHONPATH=src $(BIN)/python scripts/build_bm25_index.py --config configs/bm25_baseline.yaml --validate-index

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
