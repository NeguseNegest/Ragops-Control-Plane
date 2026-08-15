"""Validated policy, feature extraction, and deterministic query routing."""

from ragops.routing.config import (
    DEFAULT_ROUTER_CONFIG_PATH,
    EXPECTED_DECISION_ORDER,
    RouterConfig,
    load_router_config,
    validate_router_calibration,
    validate_router_registry_references,
)
from ragops.routing.probe import (
    INITIAL_PROBE_TOP_K,
    MAX_INITIAL_PROBE_TOP_K,
    InitialProbeResult,
    InitialRetrievalFeatures,
    LexicalComplexityFeatures,
    ProbeTimings,
    QueryLengthFeatures,
    RetrievalConfidenceFeatures,
    build_initial_retrieval_features,
    extract_lexical_complexity,
    run_initial_retrieval_probe,
    validate_probe_top_k,
)
from ragops.routing.router import ROUTE_REASON_CODES, ROUTE_REASONS, RoutedProbeResult, RouterDecision, RuleBasedRouter

__all__ = [
    "DEFAULT_ROUTER_CONFIG_PATH",
    "EXPECTED_DECISION_ORDER",
    "INITIAL_PROBE_TOP_K",
    "MAX_INITIAL_PROBE_TOP_K",
    "InitialProbeResult",
    "InitialRetrievalFeatures",
    "LexicalComplexityFeatures",
    "ProbeTimings",
    "QueryLengthFeatures",
    "RetrievalConfidenceFeatures",
    "ROUTE_REASONS",
    "ROUTE_REASON_CODES",
    "RoutedProbeResult",
    "RouterConfig",
    "RouterDecision",
    "RuleBasedRouter",
    "build_initial_retrieval_features",
    "extract_lexical_complexity",
    "load_router_config",
    "run_initial_retrieval_probe",
    "validate_probe_top_k",
    "validate_router_calibration",
    "validate_router_registry_references",
]
