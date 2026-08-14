from pathlib import Path

import pytest
from pydantic import ValidationError

from ragops.evaluation.runner import load_evaluation_config
from ragops.reranking.cross_encoder import CrossEncoderRerankedRetriever, load_hybrid_rerank_config
from ragops.retrieval.base import COMMON_RETRIEVER_INTERFACE, Retriever
from ragops.retrieval.bm25 import BM25Index, BM25Retriever, build_bm25_index, load_bm25_config
from ragops.retrieval.dense import DenseRetriever, RetrievedChunk
from ragops.retrieval.factory import build_retriever
from ragops.retrieval.hybrid import FUSION_METADATA_KEY, HybridRetriever, load_hybrid_config


def make_chunk(chunk_id, rank, score=1.0):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        text=f"Evidence for {chunk_id}",
        score=score,
        rank=rank,
        metadata={"relative_path": f"docs/{chunk_id}.md"},
        source_url=f"docs/{chunk_id}.md",
    )


def make_index():
    records = [
        {
            "chunk_id": "a",
            "document_id": "document-a",
            "text": "alpha evidence",
            "token_count": 2,
            "chunk_hash": "hash-a",
            "metadata": {"relative_path": "docs/a.md"},
        },
        {
            "chunk_id": "b",
            "document_id": "document-b",
            "text": "beta evidence",
            "token_count": 2,
            "chunk_hash": "hash-b",
            "metadata": {"relative_path": "docs/b.md"},
        },
    ]
    return BM25Index(build_bm25_index(records, source_path="chunks.jsonl", source_sha256="a" * 64))


class StubRetriever(Retriever):
    def __init__(self, results, default_top_k=5):
        super().__init__(default_top_k)
        self.results = results
        self.calls = []

    def retrieve(self, query, top_k=None, timings=None):
        top_k = self.default_top_k if top_k is None else top_k
        self.calls.append((query, top_k, timings))
        return self.results[:top_k]


class FakeReranker:
    model_name = "fake-cross-encoder"

    def score(self, query, chunks):
        return [float(chunk.chunk_id == "shared") for chunk in chunks]


def test_dense_and_bm25_implement_the_common_interface():
    dense = DenseRetriever(client="client", default_top_k=3)
    sparse = BM25Retriever(make_index(), default_top_k=3)

    assert isinstance(dense, Retriever)
    assert isinstance(sparse, Retriever)
    assert dense.interface == COMMON_RETRIEVER_INTERFACE
    assert sparse.interface == COMMON_RETRIEVER_INTERFACE
    assert sparse.retrieve("alpha", top_k=1)[0].chunk_id == "a"


def test_bm25_interface_records_sparse_stage_latency():
    clock_values = iter([0.0, 0.025])
    sparse = BM25Retriever(make_index(), default_top_k=1, clock=lambda: next(clock_values))
    timings = {}

    results = sparse.retrieve("alpha", timings=timings)

    assert results[0].chunk_id == "a"
    assert timings == {"bm25_ms": pytest.approx(25.0)}


def test_hybrid_interface_runs_rrf_and_records_stage_timings():
    dense = StubRetriever([make_chunk("dense", 1), make_chunk("shared", 2)])
    sparse = StubRetriever([make_chunk("sparse", 1), make_chunk("shared", 2)])
    clock_values = iter([0.0, 0.01, 0.02, 0.04, 0.05, 0.051])
    retriever = HybridRetriever(dense, sparse, dense_top_k=2, bm25_top_k=2, default_top_k=2, clock=lambda: next(clock_values))
    timings = {}

    results = retriever.retrieve("  query  ", timings=timings)

    assert [result.chunk_id for result in results] == ["shared", "dense"]
    assert results[0].metadata[FUSION_METADATA_KEY]["sources"].keys() == {"dense", "bm25"}
    assert dense.calls == [("query", 2, {"dense_ms": pytest.approx(10.0)})]
    assert sparse.calls == [("query", 2, {"bm25_ms": pytest.approx(20.0)})]
    assert timings == {"dense_ms": 10.0, "bm25_ms": 20.0, "fusion_ms": pytest.approx(1.0)}


def test_reranked_interface_returns_candidates_and_final_results():
    candidates = [make_chunk("first", 1, 0.5), make_chunk("shared", 2, 0.4)]
    candidate_retriever = StubRetriever(candidates, default_top_k=2)
    retriever = CrossEncoderRerankedRetriever(candidate_retriever, FakeReranker(), candidate_top_k=2, default_top_k=1, clock=lambda: 0.0)
    timings = {}

    retained_candidates, results = retriever.retrieve_with_candidates("query", timings=timings)

    assert retained_candidates == candidates
    assert [result.chunk_id for result in results] == ["shared"]
    assert retriever.retrieve("query")[0].chunk_id == "shared"
    assert timings == {"reranker_ms": 0.0, "total_ms": 0.0}


def test_factory_builds_every_checked_in_pipeline_from_config():
    project_root = Path(__file__).resolve().parents[1]
    index = make_index()
    dense_config = load_evaluation_config(project_root / "configs/dense_baseline.yaml", project_root=project_root)
    bm25_config = load_bm25_config(project_root / "configs/bm25_baseline.yaml", project_root=project_root)
    hybrid_config = load_hybrid_config(project_root / "configs/hybrid.yaml", project_root=project_root)
    reranked_config = load_hybrid_rerank_config(project_root / "configs/hybrid_rerank.yaml", project_root=project_root)

    pipelines = [
        build_retriever(dense_config, client="client"),
        build_retriever(bm25_config, index=index),
        build_retriever(hybrid_config, client="client", index=index),
        build_retriever(reranked_config, client="client", index=index, reranker=FakeReranker()),
    ]

    assert [type(pipeline) for pipeline in pipelines] == [DenseRetriever, BM25Retriever, HybridRetriever, CrossEncoderRerankedRetriever]
    assert all(pipeline.interface == COMMON_RETRIEVER_INTERFACE for pipeline in pipelines)


def test_checked_in_configs_pin_the_common_interface():
    project_root = Path(__file__).resolve().parents[1]
    loaders = [
        (load_evaluation_config, "dense_baseline.yaml"),
        (load_bm25_config, "bm25_baseline.yaml"),
        (load_hybrid_config, "hybrid.yaml"),
        (load_hybrid_rerank_config, "hybrid_rerank.yaml"),
    ]

    for loader, filename in loaders:
        config = loader(project_root / "configs" / filename, project_root=project_root)
        assert config.retriever_interface == COMMON_RETRIEVER_INTERFACE


def test_config_rejects_an_unknown_retriever_interface():
    project_root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(project_root / "configs/hybrid.yaml", project_root=project_root)
    data = config.model_dump(mode="python")
    data["retriever_interface"] = "future_interface"

    with pytest.raises(ValidationError, match="retriever_interface"):
        type(config).model_validate(data)


def test_factory_requires_pipeline_resources():
    project_root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(project_root / "configs/hybrid.yaml", project_root=project_root)

    with pytest.raises(ValueError, match="client is required"):
        build_retriever(config, index=make_index())
