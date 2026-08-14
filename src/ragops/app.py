import os
from datetime import UTC, datetime
from functools import partial
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from ragops import __version__
from ragops.api.pipelines import PipelineExecutionError, PipelineResourceError, PipelineRuntime
from ragops.api.schemas import (
    CitationResponse,
    QueryCostResponse,
    QueryDebugResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunkResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from ragops.generation.client import generate_answer
from ragops.generation.cost import configured_generation_pricing, estimate_generation_cost, generation_model, generation_provider
from ragops.generation.factory import create_generation_client
from ragops.indexing.qdrant import DEFAULT_QDRANT_URL, create_qdrant_client
from ragops.retrieval.dense import retrieve_dense
from ragops.tracing.context import TraceContext
from ragops.tracing.store import (
    RetrievedChunkTrace,
    TraceRecord,
    TraceStore,
    configured_pipeline_identity,
    configured_trace_db_path,
)


class HealthResponse(BaseModel):
    status: str
    version: str


def chunk_to_response(chunk):
    """Convert one retrieved chunk into the API response shape."""
    return RetrievedChunkResponse(chunk_id=chunk.chunk_id, document_id=chunk.document_id, text=chunk.text, score=chunk.score, rank=chunk.rank, metadata=chunk.metadata or {}, source_url=chunk.source_url)


def citation_to_response(citation):
    """Convert one citation dictionary into the API response shape."""
    return CitationResponse(citation_id=citation.get("citation_id", ""), document_id=citation.get("document_id", ""), title=citation.get("title", ""), url=citation.get("url"), metadata=citation.get("metadata", {}), chunk_ids=citation.get("chunk_ids", []))


def close_qdrant_client(client):
    """Close the Qdrant client when the installed client exposes close()."""
    close = getattr(client, "close", None)

    if close:
        close()


def get_qdrant_url():
    """Return the configured Qdrant URL with a stable local default."""
    qdrant_url = os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL).strip()

    if not qdrant_url:
        qdrant_url = DEFAULT_QDRANT_URL

    return qdrant_url.rstrip("/")


def retrieve_chunks(query, top_k, timings=None, client_factory=None, query_embedder=None):
    """Retrieve chunks from Qdrant and close the client afterward."""
    client_factory = client_factory or create_qdrant_client
    client = client_factory(get_qdrant_url())

    try:
        parameters = {"query": query, "client": client, "top_k": top_k, "timings": timings}
        if query_embedder is not None:
            parameters["query_embedder"] = query_embedder
        return retrieve_dense(**parameters)
    finally:
        close_qdrant_client(client)


def create_trace_store(path=None):
    """Create and validate the configured durable trace repository."""
    return TraceStore(configured_trace_db_path(path)).initialize()


def trace_chunks(chunks, used_chunk_ids=()):
    """Convert retrieval results into storage records with generation-use flags."""
    used_chunk_ids = set(used_chunk_ids)
    return [
        RetrievedChunkTrace(
            rank=chunk.rank,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            text=chunk.text,
            score=chunk.score,
            source_url=chunk.source_url,
            metadata=chunk.metadata or {},
            used_for_generation=chunk.chunk_id in used_chunk_ids,
        )
        for chunk in chunks
    ]


def persist_request_trace(
    app,
    *,
    trace_id,
    created_at,
    endpoint,
    query,
    top_k,
    chunks,
    latency_ms,
    component_latencies,
    answer=None,
    used_chunk_ids=(),
    error=None,
    pipeline_identity=None,
):
    """Persist one completed API attempt or return a trace-specific service error."""
    pipeline_identity = pipeline_identity or app.state.pipeline_identity
    status = "error" if error is not None else "success"
    error_type = type(error).__name__ if error is not None else None
    error_message = (str(error).strip() or error_type) if error is not None else None
    record = TraceRecord(
        trace_id=trace_id,
        created_at=created_at,
        completed_at=datetime.now(UTC),
        endpoint=endpoint,
        query=query,
        requested_top_k=top_k,
        pipeline_name=pipeline_identity.name,
        pipeline_version=pipeline_identity.version,
        status=status,
        retrieved_chunk_count=len(chunks),
        answer=answer,
        total_latency_ms=latency_ms,
        **component_latencies.model_dump(),
        error_type=error_type,
        error_message=error_message,
    )
    try:
        app.state.trace_store.record_trace(record, trace_chunks(chunks, used_chunk_ids=used_chunk_ids))
    except Exception as trace_error:
        raise HTTPException(status_code=503, detail="Unable to persist query trace.") from trace_error
    return trace_id


def create_app(
    generation_client=None,
    trace_store=None,
    pipeline_name=None,
    pipeline_version=None,
    pipeline_runtime=None,
    generation_pricing=None,
    retrieval_client_factory=None,
    query_embedder=None,
):
    app = FastAPI(
        title="RAGOps Control Plane",
        version=__version__,
        summary="Evaluation-gated control plane for RAG systems.",
    )
    app.state.generation_client = generation_client if generation_client is not None else create_generation_client()
    app.state.trace_store = trace_store if trace_store is not None else create_trace_store()
    app.state.pipeline_identity = configured_pipeline_identity(name=pipeline_name, version=pipeline_version)
    if pipeline_runtime is None:
        runtime_parameters = {}
        if retrieval_client_factory is not None:
            runtime_parameters["client_factory"] = retrieval_client_factory
        if query_embedder is not None:
            runtime_parameters["query_embedder"] = query_embedder
        pipeline_runtime = PipelineRuntime(**runtime_parameters)
    app.state.pipeline_runtime = pipeline_runtime
    app.state.generation_pricing = generation_pricing if generation_pricing is not None else configured_generation_pricing()
    retrieval_parameters = {}
    if retrieval_client_factory is not None:
        retrieval_parameters["client_factory"] = retrieval_client_factory
    if query_embedder is not None:
        retrieval_parameters["query_embedder"] = query_embedder
    app.state.retrieve_chunks = partial(retrieve_chunks, **retrieval_parameters) if retrieval_parameters else retrieve_chunks

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok", version=__version__)

    @app.post("/retrieve", response_model=RetrieveResponse)
    def retrieve(request: RetrieveRequest):
        """Retrieve relevant context chunks for a given query."""
        trace_context = TraceContext()
        created_at = datetime.now(UTC)
        trace_id = str(uuid4())
        chunks = []

        try:
            chunks = app.state.retrieve_chunks(request.query, request.top_k, timings=trace_context.timings)
        except ValueError as error:
            component_latencies = trace_context.snapshot()
            persist_request_trace(
                app,
                trace_id=trace_id,
                created_at=created_at,
                endpoint="retrieve",
                query=request.query,
                top_k=request.top_k,
                chunks=chunks,
                latency_ms=trace_context.total_ms(),
                component_latencies=component_latencies,
                error=error,
            )
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            component_latencies = trace_context.snapshot()
            persist_request_trace(
                app,
                trace_id=trace_id,
                created_at=created_at,
                endpoint="retrieve",
                query=request.query,
                top_k=request.top_k,
                chunks=chunks,
                latency_ms=trace_context.total_ms(),
                component_latencies=component_latencies,
                error=error,
            )
            raise HTTPException(status_code=503, detail="Unable to retrieve chunks.") from error

        response_chunks = [chunk_to_response(chunk) for chunk in chunks]
        latency_ms = trace_context.total_ms()
        component_latencies = trace_context.snapshot()
        persist_request_trace(
            app,
            trace_id=trace_id,
            created_at=created_at,
            endpoint="retrieve",
            query=request.query,
            top_k=request.top_k,
            chunks=chunks,
            latency_ms=latency_ms,
            component_latencies=component_latencies,
        )
        return RetrieveResponse(
            query=request.query,
            top_k=request.top_k,
            chunks=response_chunks,
            latency_ms=latency_ms,
            component_latencies=component_latencies,
        )

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest, response: Response):
        """Retrieve context and generate a cited answer."""
        trace_context = TraceContext()
        created_at = datetime.now(UTC)
        trace_id = str(uuid4())
        chunks = []
        definition = app.state.pipeline_runtime.select(request.config)

        try:
            execution = app.state.pipeline_runtime.retrieve(
                definition,
                request.query,
                request.top_k,
                timings=trace_context.timings,
            )
            chunks = execution.chunks
        except ValueError as error:
            component_latencies = trace_context.snapshot()
            persist_request_trace(
                app,
                trace_id=trace_id,
                created_at=created_at,
                endpoint="query",
                query=request.query,
                top_k=request.top_k,
                chunks=chunks,
                latency_ms=trace_context.total_ms(),
                component_latencies=component_latencies,
                error=error,
                pipeline_identity=definition.identity,
            )
            raise HTTPException(status_code=400, detail=str(error), headers={"X-Trace-ID": trace_id}) from error
        except PipelineResourceError as error:
            component_latencies = trace_context.snapshot()
            persist_request_trace(
                app,
                trace_id=trace_id,
                created_at=created_at,
                endpoint="query",
                query=request.query,
                top_k=request.top_k,
                chunks=chunks,
                latency_ms=trace_context.total_ms(),
                component_latencies=component_latencies,
                error=error,
                pipeline_identity=definition.identity,
            )
            raise HTTPException(
                status_code=503,
                detail="Selected query pipeline is unavailable.",
                headers={"X-Trace-ID": trace_id},
            ) from error
        except PipelineExecutionError as error:
            component_latencies = trace_context.snapshot()
            persist_request_trace(
                app,
                trace_id=trace_id,
                created_at=created_at,
                endpoint="query",
                query=request.query,
                top_k=request.top_k,
                chunks=chunks,
                latency_ms=trace_context.total_ms(),
                component_latencies=component_latencies,
                error=error,
                pipeline_identity=definition.identity,
            )
            raise HTTPException(
                status_code=503,
                detail="Unable to retrieve chunks with the selected pipeline.",
                headers={"X-Trace-ID": trace_id},
            ) from error
        except Exception as error:
            component_latencies = trace_context.snapshot()
            persist_request_trace(
                app,
                trace_id=trace_id,
                created_at=created_at,
                endpoint="query",
                query=request.query,
                top_k=request.top_k,
                chunks=chunks,
                latency_ms=trace_context.total_ms(),
                component_latencies=component_latencies,
                error=error,
                pipeline_identity=definition.identity,
            )
            raise HTTPException(
                status_code=503,
                detail="Unable to retrieve chunks with the selected pipeline.",
                headers={"X-Trace-ID": trace_id},
            ) from error

        try:
            with trace_context.measure("generation"):
                generation_result = generate_answer(query=request.query, chunks=chunks, client=app.state.generation_client)
        except ValueError as error:
            component_latencies = trace_context.snapshot()
            persist_request_trace(
                app,
                trace_id=trace_id,
                created_at=created_at,
                endpoint="query",
                query=request.query,
                top_k=request.top_k,
                chunks=chunks,
                latency_ms=trace_context.total_ms(),
                component_latencies=component_latencies,
                error=error,
                pipeline_identity=definition.identity,
            )
            raise HTTPException(status_code=400, detail=str(error), headers={"X-Trace-ID": trace_id}) from error
        except Exception as error:
            component_latencies = trace_context.snapshot()
            persist_request_trace(
                app,
                trace_id=trace_id,
                created_at=created_at,
                endpoint="query",
                query=request.query,
                top_k=request.top_k,
                chunks=chunks,
                latency_ms=trace_context.total_ms(),
                component_latencies=component_latencies,
                error=error,
                pipeline_identity=definition.identity,
            )
            raise HTTPException(
                status_code=503,
                detail="Unable to generate answer.",
                headers={"X-Trace-ID": trace_id},
            ) from error

        response_chunks = [chunk_to_response(chunk) for chunk in chunks]
        response_citations = [citation_to_response(citation) for citation in generation_result.citations]
        generation_cost = estimate_generation_cost(
            app.state.generation_client,
            generation_result.usage,
            pricing=app.state.generation_pricing,
        )
        cost_response = QueryCostResponse(**generation_cost.__dict__)
        debug_response = None
        if request.debug:
            debug_response = QueryDebugResponse(
                pipeline_id=f"{definition.name}@{definition.config.version}",
                pipeline_status=definition.config.status,
                retriever_interface=definition.config.retriever_interface,
                requested_top_k=request.top_k,
                returned_chunks=len(chunks),
                configured_depths=definition.candidate_depths(),
                generation_provider=generation_provider(app.state.generation_client),
                generation_model=generation_model(app.state.generation_client),
                resource_cache_hits=execution.cache_status(),
            )
        latency_ms = trace_context.total_ms()
        component_latencies = trace_context.snapshot()
        persist_request_trace(
            app,
            trace_id=trace_id,
            created_at=created_at,
            endpoint="query",
            query=request.query,
            top_k=request.top_k,
            chunks=chunks,
            latency_ms=latency_ms,
            component_latencies=component_latencies,
            answer=generation_result.answer,
            used_chunk_ids=generation_result.used_chunk_ids,
            pipeline_identity=definition.identity,
        )
        response.headers["X-Trace-ID"] = trace_id
        return QueryResponse(
            trace_id=trace_id,
            route=definition.route,
            config=definition.name,
            config_version=definition.config.version,
            query=request.query,
            answer=generation_result.answer,
            citations=response_citations,
            citation_text=generation_result.citation_text,
            chunks=response_chunks,
            used_chunk_ids=generation_result.used_chunk_ids,
            latency_ms=latency_ms,
            component_latencies=component_latencies,
            cost=cost_response,
            debug=debug_response,
        )

    return app


app = create_app()
