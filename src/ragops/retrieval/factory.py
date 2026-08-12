import time

from ragops.retrieval.base import COMMON_RETRIEVER_INTERFACE
from ragops.retrieval.bm25 import BM25Retriever
from ragops.retrieval.dense import DenseRetriever


def _require_resource(resource, name, pipeline):
    if resource is None:
        raise ValueError(f"{name} is required to build the {pipeline} retriever.")
    return resource


def build_retriever(config, client=None, index=None, reranker=None, clock=time.perf_counter):
    """Build any configured retrieval pipeline behind the common interface."""
    interface = getattr(config, "retriever_interface", None)
    if interface != COMMON_RETRIEVER_INTERFACE:
        raise ValueError(f"retriever_interface must be {COMMON_RETRIEVER_INTERFACE}.")

    if getattr(config, "reranker", None) is not None:
        from ragops.reranking.cross_encoder import build_hybrid_reranked_retriever

        return build_hybrid_reranked_retriever(
            config,
            _require_resource(client, "client", "reranked"),
            _require_resource(index, "index", "reranked"),
            _require_resource(reranker, "reranker", "reranked"),
            clock=clock,
        )

    if getattr(config, "fusion", None) is not None:
        from ragops.retrieval.hybrid import build_hybrid_retriever

        return build_hybrid_retriever(
            config,
            _require_resource(client, "client", "hybrid"),
            _require_resource(index, "index", "hybrid"),
            clock=clock,
        )

    retriever_config = getattr(config, "retriever", None)
    retriever_type = getattr(retriever_config, "type", None)
    if retriever_type == "dense":
        return DenseRetriever(
            _require_resource(client, "client", "dense"),
            collection_name=retriever_config.collection_name,
            embedding_model=retriever_config.embedding_model,
            default_top_k=retriever_config.top_k,
            clock=clock,
        )
    if retriever_type == "bm25":
        return BM25Retriever(index if index is not None else retriever_config.index_path, default_top_k=retriever_config.top_k, clock=clock)
    raise ValueError("Unsupported retrieval pipeline configuration.")
