"""Structured feature extraction for deterministic query routing."""

from ragops.routing.probe import (
    INITIAL_PROBE_TOP_K,
    InitialProbeResult,
    InitialRetrievalFeatures,
    LexicalComplexityFeatures,
    ProbeTimings,
    QueryLengthFeatures,
    RetrievalConfidenceFeatures,
    build_initial_retrieval_features,
    extract_lexical_complexity,
    run_initial_retrieval_probe,
)

__all__ = [
    "INITIAL_PROBE_TOP_K",
    "InitialProbeResult",
    "InitialRetrievalFeatures",
    "LexicalComplexityFeatures",
    "ProbeTimings",
    "QueryLengthFeatures",
    "RetrievalConfidenceFeatures",
    "build_initial_retrieval_features",
    "extract_lexical_complexity",
    "run_initial_retrieval_probe",
]
