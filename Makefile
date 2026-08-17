PYTHON ?= python3.12
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/python -m pip
HYBRID_QUERY ?= What operation is used to quantify the similarity between the query and document vectors?
RERANK_QUERY ?= What operation is used to quantify the similarity between the query and document vectors?
ROUTER_QUERY ?= What is FastAPI?
MLFLOW_CONFIG ?= configs/mlflow.yaml
PIPELINE_REGISTRY_CONFIG ?= configs/pipeline_registry.yaml
ROUTER_CONFIG ?= configs/routed.yaml
NO_ANSWER_CONFIG ?= configs/no_answer.yaml
MODEL_COST_CONFIG ?= configs/model_costs.yaml
ROUTER_EVALUATION_CONFIG ?= configs/router_evaluation.yaml
ROUTER_TUNING_CONFIG ?= configs/router_tuning.yaml
EVAL_GATE_CONFIG ?= configs/eval_gate.yaml
TRACE_DB_PATH ?= data/traces/ragops_traces.sqlite3
API_URL ?= http://127.0.0.1:8000
API_TRACE_DB_PATH ?= $(TRACE_DB_PATH)

.PHONY: setup lint test test-mlflow test-pipeline-registry test-tracing test-query-endpoint test-api-ci test-api-evaluation test-eval-gate test-routing-probe test-no-answer test-cost test-router-evaluation test-router-stabilization test-retrieval-interface test-retrieval-metrics test-llm-judge test-bm25 test-bm25-evaluation test-hybrid test-hybrid-evaluation test-reranker test-reranker-evaluation eval-gate validate-mlflow log-retrieval-runs verify-retrieval-runs validate-pipeline-registry build-pipeline-registry validate-router-config validate-no-answer evaluate-no-answer replay-no-answer validate-model-costs validate-router-evaluation evaluate-router validate-router-tuning tune-router init-trace-store validate-trace-store validate-dense-evaluation evaluate-dense evaluate-api probe-query route-query validate-generation-judge judge-answers review-judgments validate-day20 validate-bm25-config build-bm25-index validate-bm25-index validate-bm25-evaluation evaluate-bm25 validate-hybrid retrieve-hybrid validate-hybrid-evaluation evaluate-hybrid validate-hybrid-rerank retrieve-hybrid-rerank validate-reranker-evaluation evaluate-reranker services-up docker-up ingest-dry-run ingest index index-recreate generate-synthetic-qa review-synthetic-qa bootstrap-retrieval-labels label-retrieval validate-retrieval-labels serve dashboard clean

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
	$(BIN)/python -m pytest tests/test_trace_context.py tests/test_tracing.py tests/test_api.py

test-query-endpoint:
	$(BIN)/python -m pytest tests/test_query_pipelines.py tests/test_generation_cost.py tests/test_generation.py tests/test_generation_providers.py tests/test_api.py tests/test_tracing.py

test-api-ci:
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_api_integration.py tests/test_api_evaluation.py tests/test_api.py tests/test_query_pipelines.py tests/test_router.py tests/test_no_answer.py tests/test_router_evaluation.py tests/test_router_tuning.py tests/test_generation_cost.py tests/test_generation.py tests/test_generation_providers.py tests/test_trace_context.py tests/test_tracing.py tests/test_retrieval.py

test-api-evaluation:
	$(BIN)/python -m pytest tests/test_api_evaluation.py tests/test_api_integration.py tests/test_query_pipelines.py tests/test_tracing.py

test-eval-gate:
	$(BIN)/python -m pytest tests/test_eval_gate.py

eval-gate:
	PYTHONPATH=src $(BIN)/python scripts/eval_gate.py --config $(EVAL_GATE_CONFIG)

test-routing-probe:
	$(BIN)/python -m pytest tests/test_router.py tests/test_query_pipelines.py

test-no-answer:
	$(BIN)/python -m pytest tests/test_no_answer.py tests/test_router.py tests/test_api.py

test-cost:
	$(BIN)/python -m pytest tests/test_generation_cost.py tests/test_tracing.py tests/test_api.py tests/test_api_evaluation.py tests/test_api_integration.py

test-router-evaluation:
	$(BIN)/python -m pytest tests/test_router_evaluation.py tests/test_router.py tests/test_no_answer.py tests/test_generation_cost.py

test-router-stabilization:
	$(BIN)/python -m pytest tests/test_router_tuning.py tests/test_router_evaluation.py tests/test_router.py tests/test_no_answer.py

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

validate-router-config:
	PYTHONPATH=src $(BIN)/python scripts/validate_router_config.py --config $(ROUTER_CONFIG) --registry reports/pipeline_registry.json

validate-no-answer:
	PYTHONPATH=src $(BIN)/python scripts/evaluate_no_answer.py --config $(NO_ANSWER_CONFIG) --validate-only

evaluate-no-answer:
	PYTHONPATH=src $(BIN)/python scripts/evaluate_no_answer.py --config $(NO_ANSWER_CONFIG) --overwrite

replay-no-answer:
	PYTHONPATH=src $(BIN)/python scripts/evaluate_no_answer.py --config $(NO_ANSWER_CONFIG) --replay-existing --overwrite

validate-model-costs:
	PYTHONPATH=src $(BIN)/python scripts/validate_model_costs.py --config $(MODEL_COST_CONFIG)

validate-router-evaluation:
	PYTHONPATH=src $(BIN)/python scripts/evaluate_router.py --config $(ROUTER_EVALUATION_CONFIG) --validate-only

evaluate-router:
	PYTHONPATH=src $(BIN)/python scripts/evaluate_router.py --config $(ROUTER_EVALUATION_CONFIG) --overwrite

validate-router-tuning:
	PYTHONPATH=src $(BIN)/python scripts/tune_router.py --config $(ROUTER_TUNING_CONFIG) --validate-only

tune-router:
	PYTHONPATH=src $(BIN)/python scripts/tune_router.py --config $(ROUTER_TUNING_CONFIG) --overwrite

init-trace-store:
	PYTHONPATH=src $(BIN)/python scripts/init_trace_store.py --db-path $(TRACE_DB_PATH)

validate-trace-store:
	PYTHONPATH=src $(BIN)/python scripts/init_trace_store.py --db-path $(TRACE_DB_PATH) --validate-only

validate-dense-evaluation:
	PYTHONPATH=src $(BIN)/python scripts/evaluate.py --config configs/dense_baseline.yaml --validate-only

evaluate-dense:
	PYTHONPATH=src $(BIN)/python scripts/evaluate.py --config configs/dense_baseline.yaml

evaluate-api:
	PYTHONPATH=src $(BIN)/python scripts/evaluate_api.py --api-url $(API_URL) --trace-db-path $(API_TRACE_DB_PATH) --mlflow-config $(MLFLOW_CONFIG) --overwrite

probe-query:
	PYTHONPATH=src $(BIN)/python scripts/probe_query.py --query "$(ROUTER_QUERY)"

route-query:
	PYTHONPATH=src $(BIN)/python scripts/route_query.py --query "$(ROUTER_QUERY)"

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
