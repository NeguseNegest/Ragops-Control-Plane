from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ragops.api.pipelines import QueryConfigName, QueryRoute
from ragops.generation.no_answer import NO_ANSWER_PROMPT_VERSION, NO_ANSWER_RESPONSE
from ragops.pipeline_registry import PipelineStatus, PipelineVersion
from ragops.routing.probe import InitialRetrievalFeatures, ProbeTimings
from ragops.routing.router import RouterDecision
from ragops.tracing.context import ComponentLatencies


class StrictResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrieveRequest(BaseModel):
    """Request body for POST /retrieve."""
    query: str = Field(..., description="The user query to retrieve context for.")
    top_k: int = Field(5, ge=1, le=20, description="The number of chunks to retrieve.")


class RetrievedChunkResponse(BaseModel):
    """One retrieved chunk returned by the API."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_url: str | None = None


class RetrieveResponse(BaseModel):
    """Response body for POST /retrieve."""
    query: str
    top_k: int
    chunks: list[RetrievedChunkResponse]
    latency_ms: float
    component_latencies: ComponentLatencies


class RouteRequest(BaseModel):
    """Request body for the Day 38 decision-only routing endpoint."""

    query: str = Field(..., description="The user query to probe and classify without executing the selected route.")


class RouteProbeChunkResponse(StrictResponseModel):
    """Minimal dense evidence provenance returned with a route decision."""

    chunk_id: str
    score: float
    rank: int = Field(gt=0)


class RouteRefusalResponse(StrictResponseModel):
    """Deterministic Day 39 response for a NO_ANSWER decision."""

    answer: Literal[NO_ANSWER_RESPONSE]
    prompt_version: Literal[NO_ANSWER_PROMPT_VERSION]
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_by: Literal["deterministic_policy"]


class RouteResponse(StrictResponseModel):
    """Deterministic route decision plus the exact feature evidence used."""

    query: str
    decision: RouterDecision
    features: InitialRetrievalFeatures
    probe_chunks: list[RouteProbeChunkResponse]
    probe_timings: ProbeTimings
    refusal: RouteRefusalResponse | None = None

    @model_validator(mode="after")
    def require_refusal_for_no_answer_only(self):
        if self.decision.route == "NO_ANSWER" and self.refusal is None:
            raise ValueError("NO_ANSWER route responses must contain a deterministic refusal.")
        if self.decision.route != "NO_ANSWER" and self.refusal is not None:
            raise ValueError("Only NO_ANSWER route responses may contain a refusal.")
        return self


class QueryRequest(BaseModel):
    """Request body for POST /query."""
    query: str = Field(..., description="The user question to answer.")
    top_k: int = Field(5, ge=1, le=20, description="The number of retrieved chunks to use.")
    config: QueryConfigName = Field("dense_baseline", description="The registered retrieval config to execute.")
    debug: bool = Field(False, description="Include non-sensitive pipeline and runtime diagnostics.")


class CitationResponse(BaseModel):
    """One citation returned with a generated answer."""
    citation_id: str
    document_id: str
    title: str
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_ids: list[str]


class QueryCostResponse(StrictResponseModel):
    """Generation cost state without pretending an unavailable value is zero."""

    amount_usd: float | None = Field(default=None, ge=0)
    currency: Literal["USD"] = "USD"
    status: Literal["zero_cost", "estimated", "unavailable"]
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_cost_state(self):
        if self.status == "unavailable" and self.amount_usd is not None:
            raise ValueError("Unavailable cost must not contain an amount.")
        if self.status != "unavailable" and self.amount_usd is None:
            raise ValueError("Available cost must contain an amount.")
        token_values = (self.input_tokens, self.output_tokens, self.total_tokens)
        if any(value is not None for value in token_values) and not all(value is not None for value in token_values):
            raise ValueError("Generation token counts must be all present or all absent.")
        return self


class QueryDebugResponse(StrictResponseModel):
    """Diagnostics emitted only when the caller explicitly enables debug mode."""

    pipeline_id: str
    pipeline_status: PipelineStatus
    retriever_interface: Literal["common_v1"]
    requested_top_k: int = Field(gt=0)
    returned_chunks: int = Field(ge=0)
    configured_depths: dict[str, int]
    generation_provider: str
    generation_model: str | None = None
    resource_cache_hits: dict[str, bool]


class QueryResponse(BaseModel):
    """Response body for POST /query."""
    trace_id: UUID
    route: QueryRoute
    config: QueryConfigName
    config_version: PipelineVersion
    query: str
    answer: str
    citations: list[CitationResponse]
    citation_text: str
    chunks: list[RetrievedChunkResponse]
    used_chunk_ids: list[str]
    latency_ms: float
    component_latencies: ComponentLatencies
    cost: QueryCostResponse
    debug: QueryDebugResponse | None = None
