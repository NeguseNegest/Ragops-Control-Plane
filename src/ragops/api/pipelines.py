import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

from ragops.evaluation.runner import load_evaluation_config
from ragops.indexing.qdrant import DEFAULT_QDRANT_URL, create_qdrant_client
from ragops.reranking.cross_encoder import build_cross_encoder_reranker, load_hybrid_rerank_config
from ragops.retrieval.bm25 import load_bm25_index, validate_bm25_index
from ragops.retrieval.dense import validate_query
from ragops.retrieval.factory import build_retriever
from ragops.retrieval.hybrid import configured_qdrant_url, load_hybrid_config
from ragops.tracing.store import PipelineIdentity

QueryConfigName = Literal["dense_baseline", "hybrid_rrf", "hybrid_rrf_cross_encoder"]
QueryRoute = Literal["dense", "hybrid", "reranked"]

DEFAULT_QUERY_CONFIG = "dense_baseline"
DEFAULT_PIPELINE_CONFIG_PATHS = {
    "dense_baseline": Path("configs/dense_baseline.yaml"),
    "hybrid_rrf": Path("configs/hybrid.yaml"),
    "hybrid_rrf_cross_encoder": Path("configs/hybrid_rerank.yaml"),
}


class PipelineResourceError(RuntimeError):
    """A selected pipeline could not initialize one of its runtime resources."""


class PipelineExecutionError(RuntimeError):
    """A selected, initialized retrieval pipeline failed while serving a query."""


@dataclass(frozen=True)
class PipelineDefinition:
    """Validated executable configuration and public route metadata."""

    name: QueryConfigName
    route: QueryRoute
    config_path: Path
    config: object

    @property
    def identity(self):
        return PipelineIdentity(name=self.config.name, version=self.config.version)

    def candidate_depths(self):
        if self.route == "dense":
            return {"dense": self.config.retriever.top_k}
        depths = {
            "dense": self.config.dense.top_k,
            "bm25": self.config.bm25.top_k,
            "fusion": self.config.fusion.top_k,
        }
        if self.route == "reranked":
            depths.update(
                {
                    "reranker_candidates": self.config.reranker.candidate_top_k,
                    "reranker_output": self.config.reranker.top_k,
                }
            )
        return depths


@dataclass(frozen=True)
class PipelineExecution:
    """One retrieval result plus non-sensitive resource-cache diagnostics."""

    definition: PipelineDefinition
    chunks: list
    bm25_cache_hit: bool | None = None
    reranker_cache_hit: bool | None = None

    def cache_status(self):
        status = {}
        if self.bm25_cache_hit is not None:
            status["bm25_index"] = self.bm25_cache_hit
        if self.reranker_cache_hit is not None:
            status["reranker_model"] = self.reranker_cache_hit
        return status


def close_qdrant_client(client):
    """Close a request-scoped Qdrant client when close() is available."""
    close = getattr(client, "close", None)
    if close:
        close()


def _project_path(path, project_root):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _load_pipeline_definition(name, config_path, project_root):
    routes_and_loaders = {
        "dense_baseline": ("dense", load_evaluation_config),
        "hybrid_rrf": ("hybrid", load_hybrid_config),
        "hybrid_rrf_cross_encoder": ("reranked", load_hybrid_rerank_config),
    }
    try:
        route, loader = routes_and_loaders[name]
    except KeyError as error:
        supported = ", ".join(routes_and_loaders)
        raise ValueError(f"Unsupported query config '{name}'. Choose one of: {supported}.") from error
    resolved_path = _project_path(config_path, project_root)
    config = loader(resolved_path, project_root=project_root)
    if config.name != name:
        raise ValueError(f"Query config name {config.name!r} does not match its runtime selector {name!r}.")
    return PipelineDefinition(name=name, route=route, config_path=resolved_path, config=config)


class PipelineRuntime:
    """Construct and execute validated online retrieval pipelines."""

    def __init__(
        self,
        project_root=None,
        config_paths=None,
        client_factory=create_qdrant_client,
        index_loader=load_bm25_index,
        index_validator=validate_bm25_index,
        reranker_factory=build_cross_encoder_reranker,
        retriever_factory=build_retriever,
    ):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[3]).resolve()
        configured_paths = dict(DEFAULT_PIPELINE_CONFIG_PATHS if config_paths is None else config_paths)
        if set(configured_paths) != set(DEFAULT_PIPELINE_CONFIG_PATHS):
            raise ValueError("Query runtime must configure dense_baseline, hybrid_rrf, and hybrid_rrf_cross_encoder exactly once.")
        self._definitions = {
            name: _load_pipeline_definition(name, configured_paths[name], self.project_root)
            for name in DEFAULT_PIPELINE_CONFIG_PATHS
        }
        self.client_factory = client_factory
        self.index_loader = index_loader
        self.index_validator = index_validator
        self.reranker_factory = reranker_factory
        self.retriever_factory = retriever_factory
        self._resource_lock = Lock()
        self._bm25_indexes = {}
        self._validated_indexes = set()
        self._rerankers = {}

    @property
    def available_configs(self):
        return tuple(self._definitions)

    def select(self, name):
        try:
            return self._definitions[name]
        except KeyError as error:
            supported = ", ".join(self.available_configs)
            raise ValueError(f"Unsupported query config '{name}'. Choose one of: {supported}.") from error

    def _qdrant_url(self, definition):
        if definition.route == "dense":
            configured = definition.config.retriever.qdrant_url
            if configured:
                return configured
            environment_url = os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL).strip().rstrip("/")
            return environment_url or DEFAULT_QDRANT_URL
        return configured_qdrant_url(definition.config) or DEFAULT_QDRANT_URL

    def _bm25_index(self, definition):
        path = definition.config.bm25.index_path
        validation_key = (path, definition.name)
        with self._resource_lock:
            cache_hit = path in self._bm25_indexes
            try:
                if not cache_hit:
                    self._bm25_indexes[path] = self.index_loader(path)
                index = self._bm25_indexes[path]
                if validation_key not in self._validated_indexes:
                    self.index_validator(index, definition.config.bm25_validation_config())
                    self._validated_indexes.add(validation_key)
            except Exception as error:
                self._bm25_indexes.pop(path, None)
                self._validated_indexes.discard(validation_key)
                raise PipelineResourceError(f"Unable to load the BM25 index for {definition.name}: {error}") from error
        return index, cache_hit

    def _reranker(self, definition):
        config = definition.config.reranker
        cache_key = (config.model, config.batch_size, config.max_length, config.device)
        with self._resource_lock:
            cache_hit = cache_key in self._rerankers
            try:
                if not cache_hit:
                    self._rerankers[cache_key] = self.reranker_factory(config)
                reranker = self._rerankers[cache_key]
            except Exception as error:
                self._rerankers.pop(cache_key, None)
                raise PipelineResourceError(f"Unable to load the reranker for {definition.name}: {error}") from error
        return reranker, cache_hit

    def retrieve(self, definition, query, top_k, timings=None):
        """Run one selected pipeline with cached local models and a fresh Qdrant client."""
        query = validate_query(query)
        if not isinstance(definition, PipelineDefinition):
            raise ValueError("definition must be selected by this query runtime.")
        if self._definitions.get(definition.name) is not definition:
            raise ValueError("definition must belong to this query runtime.")

        client = None
        index = None
        reranker = None
        bm25_cache_hit = None
        reranker_cache_hit = None
        try:
            try:
                client = self.client_factory(self._qdrant_url(definition))
                if definition.route in ("hybrid", "reranked"):
                    index, bm25_cache_hit = self._bm25_index(definition)
                if definition.route == "reranked":
                    reranker, reranker_cache_hit = self._reranker(definition)
                retriever = self.retriever_factory(
                    definition.config,
                    client=client,
                    index=index,
                    reranker=reranker,
                )
            except PipelineResourceError:
                raise
            except Exception as error:
                raise PipelineResourceError(f"Unable to initialize {definition.name}: {error}") from error

            try:
                chunks = retriever.retrieve(query, top_k=top_k, timings=timings)
            except Exception as error:
                raise PipelineExecutionError(f"Retrieval failed for {definition.name}: {error}") from error
            return PipelineExecution(
                definition=definition,
                chunks=list(chunks),
                bm25_cache_hit=bm25_cache_hit,
                reranker_cache_hit=reranker_cache_hit,
            )
        finally:
            if client is not None:
                close_qdrant_client(client)
