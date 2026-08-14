PYTHON ?= python3.12
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/python -m pip
HYBRID_QUERY ?= What operation is used to quantify the similarity between the query and document vectors?
RERANK_QUERY ?= What operation is used to quantify the similarity between the query and document vectors?
MLFLOW_CONFIG ?= configs/mlflow.yaml
PIPELINE_REGISTRY_CONFIG ?= configs/pipeline_registry.yaml
TRACE_DB_PATH ?= data/traces/ragops_traces.sqlite3

.PHONY: setup lint test test-mlflow test-pipeline-registry test-tracing test-retrieval-interface test-retrieval-metrics test-llm-judge test-bm25 test-bm25-evaluation test-hybrid test-hybrid-evaluation test-reranker test-reranker-evaluation validate-mlflow log-retrieval-runs verify-retrieval-runs validate-pipeline-registry build-pipeline-registry init-trace-store validate-trace-store validate-dense-evaluation evaluate-dense validate-generation-judge judge-answers review-judgments validate-day20 validate-bm25-config build-bm25-index validate-bm25-index validate-bm25-evaluation evaluate-bm25 validate-hybrid retrieve-hybrid validate-hybrid-evaluation evaluate-hybrid validate-hybrid-rerank retrieve-hybrid-rerank validate-reranker-evaluation evaluate-reranker services-up docker-up ingest-dry-run ingest index index-recreate generate-synthetic-qa review-synthetic-qa bootstrap-retrieval-labels label-retrieval validate-retrieval-labels serve dashboard clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(BIN)/ruff check src tests scripts dashboard

test:
	$(BIN)/python -m pytest

test-mlflow:
	$(BIN)/python -m pytest tests/test_mlflow_tracking.py

test-pipeline-registry:
	$(BIN)/python -m pytest tests/test_pipeline_registry.py

test-tracing:
	$(BIN)/python -m pytest tests/test_tracing.py tests/test_api.py

test-retrieval-interface:
	$(BIN)/python -m pytest tests/test_retriever_interface.py tests/test_hybrid.py tests/test_reranking.py

test-retrieval-metrics:
	$(BIN)/python -m pytest tests/test_retrieval_metrics.py

test-llm-judge:
	$(BIN)/python -m pytest tests/test_llm_judge.py

test-bm25:
	$(BIN)/python -m pytest tests/test_bm25.py

test-bm25-evaluation:
	$(BIN)/python -m pytest tests/test_bm25_evaluation.py

test-hybrid:
	$(BIN)/python -m pytest tests/test_hybrid.py

test-hybrid-evaluation:
	$(BIN)/python -m pytest tests/test_hybrid_evaluation.py

test-reranker:
	$(BIN)/python -m pytest tests/test_reranking.py

test-reranker-evaluation:
	$(BIN)/python -m pytest tests/test_reranker_evaluation.py

validate-mlflow:
	PYTHONPATH=src $(BIN)/python scripts/log_retrieval_runs.py --config $(MLFLOW_CONFIG) --validate-only

log-retrieval-runs:
	PYTHONPATH=src $(BIN)/python scripts/log_retrieval_runs.py --config $(MLFLOW_CONFIG)

verify-retrieval-runs:
	PYTHONPATH=src $(BIN)/python scripts/log_retrieval_runs.py --config $(MLFLOW_CONFIG) --verify-only

validate-pipeline-registry:
	PYTHONPATH=src $(BIN)/python scripts/build_pipeline_registry.py --config $(PIPELINE_REGISTRY_CONFIG) --validate-only

build-pipeline-registry:
	PYTHONPATH=src $(BIN)/python scripts/build_pipeline_registry.py --config $(PIPELINE_REGISTRY_CONFIG) --overwrite

init-trace-store:
	PYTHONPATH=src $(BIN)/python scripts/init_trace_store.py --db-path $(TRACE_DB_PATH)

validate-trace-store:
	PYTHONPATH=src $(BIN)/python scripts/init_trace_store.py --db-path $(TRACE_DB_PATH) --validate-only

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

validate-bm25-evaluation:
	PYTHONPATH=src $(BIN)/python scripts/evaluate_bm25.py --config configs/bm25_baseline.yaml --validate-only

evaluate-bm25:
	PYTHONPATH=src $(BIN)/python scripts/evaluate_bm25.py --config configs/bm25_baseline.yaml --overwrite

validate-hybrid:
	PYTHONPATH=src $(BIN)/python scripts/retrieve_hybrid.py --config configs/hybrid.yaml --validate-only

retrieve-hybrid:
	PYTHONPATH=src $(BIN)/python scripts/retrieve_hybrid.py --config configs/hybrid.yaml --query "$(HYBRID_QUERY)"

validate-hybrid-evaluation:
	PYTHONPATH=src $(BIN)/python scripts/evaluate_hybrid.py --config configs/hybrid.yaml --validate-only

evaluate-hybrid:
	PYTHONPATH=src $(BIN)/python scripts/evaluate_hybrid.py --config configs/hybrid.yaml --overwrite

validate-hybrid-rerank:
	PYTHONPATH=src $(BIN)/python scripts/retrieve_hybrid_rerank.py --config configs/hybrid_rerank.yaml --validate-only

retrieve-hybrid-rerank:
	PYTHONPATH=src $(BIN)/python scripts/retrieve_hybrid_rerank.py --config configs/hybrid_rerank.yaml --query "$(RERANK_QUERY)"

validate-reranker-evaluation:
	PYTHONPATH=src $(BIN)/python scripts/evaluate_reranker.py --config configs/hybrid_rerank.yaml --validate-only

evaluate-reranker:
	PYTHONPATH=src $(BIN)/python scripts/evaluate_reranker.py --config configs/hybrid_rerank.yaml --overwrite

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
