import csv
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ragops.evaluation.retrieval_labels import RetrievalLabel
from ragops.evaluation.runner import (
    EvaluationDatasetConfig,
    EvaluationOutputConfig,
    RetrievalEvaluationConfig,
    configured_qdrant_url,
    evaluate_dense_config,
    load_evaluation_config,
    load_evaluation_labels,
    run_retrieval_evaluation,
    write_evaluation_artifacts,
)
from ragops.evaluation.synthetic_qa import write_jsonl


def make_config(tmp_path, top_k=2, k_values=None, minimum_labels=1, qdrant_url=None):
    return RetrievalEvaluationConfig(
        name="dense_test",
        retriever={
            "type": "dense",
            "collection_name": "test_chunks",
            "embedding_model": "test-embedding-model",
            "top_k": top_k,
            "qdrant_url": qdrant_url,
        },
        evaluation=EvaluationDatasetConfig(
            labels_path=tmp_path / "labels.jsonl",
            k_values=k_values or [1, 2],
            minimum_labels=minimum_labels,
        ),
        output=EvaluationOutputConfig(directory=tmp_path / "reports"),
    )


def make_label(question_id="q1", relevant_chunk_ids=None):
    return RetrievalLabel(
        question_id=question_id,
        question=f"How does documented feature {question_id} work?",
        relevant_chunk_ids=relevant_chunk_ids or [f"relevant-{question_id}"],
        expected_source="fastapi/docs/example.md",
        metadata={"label_method": "manual", "reviewed_by": "test-reviewer"},
    )


def result(chunk_id, score):
    return SimpleNamespace(chunk_id=chunk_id, score=score)


class FakeClock:

    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class FakeQdrantClient:

    def __init__(self, collection_exists=True):
        self.has_collection = collection_exists
        self.collection_calls = []
        self.closed = False

    def collection_exists(self, collection_name):
        self.collection_calls.append(collection_name)
        return self.has_collection

    def close(self):
        self.closed = True


def test_load_evaluation_config_resolves_project_paths(tmp_path):
    config_path = tmp_path / "dense.yaml"
    config_path.write_text(
        """name: dense_test
retriever:
  type: dense
  collection_name: test_chunks
  embedding_model: test-model
  top_k: 5
evaluation:
  labels_path: data/eval/labels.jsonl
  k_values: [1, 5]
  minimum_labels: 2
output:
  directory: reports/test
""",
        encoding="utf-8",
    )

    config = load_evaluation_config(config_path, project_root=tmp_path)

    assert config.name == "dense_test"
    assert config.retriever.top_k == 5
    assert config.evaluation.labels_path == tmp_path / "data/eval/labels.jsonl"
    assert config.output.directory == tmp_path / "reports/test"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "YAML mapping"),
        ("- item\n", "YAML mapping"),
        (
            """name: Dense Test
retriever: {type: dense, top_k: 5}
evaluation: {k_values: [1, 5]}
output: {}
""",
            "lowercase",
        ),
        (
            """name: dense_test
retriever: {type: dense, top_k: 3}
evaluation: {k_values: [1, 5]}
output: {}
""",
            "largest evaluation cutoff",
        ),
        (
            """name: dense_test
retriever: {type: sparse, top_k: 5}
evaluation: {k_values: [1, 5]}
output: {}
""",
            "dense",
        ),
    ],
)
def test_load_evaluation_config_rejects_invalid_config(tmp_path, content, message):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises((ValueError, ValidationError), match=message):
        load_evaluation_config(config_path, project_root=tmp_path)


def test_load_evaluation_config_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_evaluation_config(tmp_path / "missing.yaml", project_root=tmp_path)


def test_load_evaluation_labels_enforces_configured_minimum(tmp_path):
    config = make_config(tmp_path, minimum_labels=2)
    write_jsonl([make_label()], config.evaluation.labels_path)

    with pytest.raises(ValueError, match="at least 2"):
        load_evaluation_labels(config)


def test_run_retrieval_evaluation_preserves_order_and_computes_metrics(tmp_path):
    config = make_config(tmp_path)
    labels = [make_label("q1", ["a"]), make_label("q2", ["c", "d"])]
    responses = {
        labels[0].question: [result("a", 0.9), result("x", 0.5)],
        labels[1].question: [result("x", 0.8), result("c", 0.7)],
    }
    calls = []
    progress_events = []

    def fake_retriever(**kwargs):
        calls.append(kwargs)
        return responses[kwargs["query"]]

    report = run_retrieval_evaluation(
        config,
        labels,
        client="fake-client",
        retriever=fake_retriever,
        clock=FakeClock([1.0, 1.01, 2.0, 2.02]),
        progress=progress_events.append,
    )

    assert [call["query"] for call in calls] == [labels[0].question, labels[1].question]
    assert all(call["client"] == "fake-client" for call in calls)
    assert all(call["top_k"] == 2 for call in calls)
    assert all(call["collection_name"] == "test_chunks" for call in calls)
    assert all(call["embedding_model"] == "test-embedding-model" for call in calls)
    assert report["questions"][0]["retrieved_chunk_ids"] == ["a", "x"]
    assert report["questions"][0]["retrieved_scores"] == [0.9, 0.5]
    assert report["questions"][1]["retrieved_chunk_ids"] == ["x", "c"]
    assert report["metrics"]["mrr"] == 0.75
    assert report["metrics"]["recall_at_k"] == {"1": 0.5, "2": 0.75}
    assert report["latency_ms"]["total"] == pytest.approx(30.0)
    assert report["latency_ms"]["average"] == pytest.approx(15.0)
    assert len(progress_events) == 2


def test_run_retrieval_evaluation_allows_empty_rankings(tmp_path):
    config = make_config(tmp_path)
    label = make_label()

    report = run_retrieval_evaluation(config, [label], client=None, retriever=lambda **kwargs: [], clock=FakeClock([0.0, 0.1]))

    assert report["metrics"]["mrr"] == 0.0
    assert report["metrics"]["recall_at_k"] == {"1": 0.0, "2": 0.0}


@pytest.mark.parametrize(
    ("retrieved", "message"),
    [
        ([result("duplicate", 0.9), result("duplicate", 0.8)], "duplicate chunk ID"),
        ([result("", 0.9)], "valid chunk_id"),
        ([result("chunk-1", float("nan"))], "non-finite"),
        ([result("chunk-1", 0.9), result("chunk-2", 0.8), result("chunk-3", 0.7)], "more than top_k"),
    ],
)
def test_run_retrieval_evaluation_rejects_invalid_results(tmp_path, retrieved, message):
    config = make_config(tmp_path)

    with pytest.raises(ValueError, match=message):
        run_retrieval_evaluation(config, [make_label()], client=None, retriever=lambda **kwargs: retrieved, clock=FakeClock([0.0, 0.1]))


def test_run_retrieval_evaluation_rejects_duplicate_labels_before_retrieval(tmp_path):
    calls = []

    def fake_retriever(**kwargs):
        calls.append(kwargs)
        return []

    with pytest.raises(ValueError, match="duplicate question IDs"):
        run_retrieval_evaluation(make_config(tmp_path), [make_label(), make_label()], client=None, retriever=fake_retriever)

    assert calls == []


def test_run_retrieval_evaluation_identifies_failed_question(tmp_path):
    def failed_retriever(**kwargs):
        raise ConnectionError("offline")

    with pytest.raises(RuntimeError, match="Retrieval failed for question q1.*offline"):
        run_retrieval_evaluation(make_config(tmp_path), [make_label()], client=None, retriever=failed_retriever, clock=FakeClock([0.0]))


def test_evaluate_dense_config_reuses_and_closes_one_client(tmp_path, monkeypatch):
    config = make_config(tmp_path, qdrant_url="http://configured:6333/")
    label = make_label()
    client = FakeQdrantClient()
    factory_calls = []

    def client_factory(qdrant_url):
        factory_calls.append(qdrant_url)
        return client

    report = evaluate_dense_config(
        config,
        [label],
        client_factory=client_factory,
        retriever=lambda **kwargs: [result("relevant-q1", 1.0)],
        clock=FakeClock([0.0, 0.01]),
    )

    assert report["metrics"]["mrr"] == 1.0
    assert factory_calls == ["http://configured:6333"]
    assert client.collection_calls == ["test_chunks"]
    assert client.closed

    monkeypatch.setenv("QDRANT_URL", "http://environment:6333/")
    config_without_url = make_config(tmp_path)
    assert configured_qdrant_url(config_without_url) == "http://environment:6333"


def test_evaluate_dense_config_closes_client_when_collection_missing(tmp_path):
    client = FakeQdrantClient(collection_exists=False)

    with pytest.raises(RuntimeError, match="collection does not exist"):
        evaluate_dense_config(make_config(tmp_path), [make_label()], client_factory=lambda url: client)

    assert client.closed


def test_evaluate_dense_config_closes_client_after_retrieval_failure(tmp_path):
    client = FakeQdrantClient()

    def failed_retriever(**kwargs):
        raise ConnectionError("offline")

    with pytest.raises(RuntimeError, match="offline"):
        evaluate_dense_config(
            make_config(tmp_path),
            [make_label()],
            client_factory=lambda url: client,
            retriever=failed_retriever,
            clock=FakeClock([0.0]),
        )

    assert client.closed


def test_write_evaluation_artifacts_produces_deterministic_json_and_csv(tmp_path):
    config = make_config(tmp_path)
    labels = [make_label("q1", ["a"]), make_label("q2", ["c"])]
    responses = {
        labels[0].question: [result("a", 0.9), result("x", 0.5)],
        labels[1].question: [result("x", 0.8), result("c", 0.7)],
    }
    report = run_retrieval_evaluation(
        config,
        labels,
        client=None,
        retriever=lambda **kwargs: responses[kwargs["query"]],
        clock=FakeClock([0.0, 0.01, 1.0, 1.02]),
    )

    json_path, csv_path = write_evaluation_artifacts(report)
    first_json = json_path.read_text(encoding="utf-8")
    first_csv = csv_path.read_text(encoding="utf-8")
    write_evaluation_artifacts(report)

    assert json_path.read_text(encoding="utf-8") == first_json
    assert csv_path.read_text(encoding="utf-8") == first_csv
    parsed_json = json.loads(first_json)
    assert parsed_json["metrics"]["mrr"] == 0.75

    with csv_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert len(rows) == 2
    assert rows[0]["question_id"] == "q1"
    assert json.loads(rows[0]["retrieved_chunk_ids"]) == ["a", "x"]
    assert float(rows[0]["aggregate_mrr"]) == 0.75
    assert float(rows[0]["aggregate_recall_at_2"]) == 1.0
