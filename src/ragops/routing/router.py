from typing import Literal

from pydantic import Field, model_validator

from ragops.pipeline_registry import PipelineStatus
from ragops.routing.config import RetrievalPipelineName, RouterConfig, RouterRoute
from ragops.routing.probe import FEATURE_SCHEMA_VERSION, InitialProbeResult, InitialRetrievalFeatures, StrictFeatureModel

RouterReasonCode = Literal[
    "empty_probe",
    "top_score_below_no_answer_threshold",
    "missing_score_gap",
    "top_score_below_careful_threshold",
    "score_gap_below_careful_threshold",
    "token_count_above_careful_threshold",
    "complexity_marker_count_at_least_careful_threshold",
    "clause_marker_count_at_least_careful_threshold",
    "long_token_ratio_at_least_careful_threshold",
    "fast_conditions_satisfied",
    "standard_fallback",
]

ROUTE_REASONS = {
    "empty_probe": "The initial dense probe returned no corpus evidence.",
    "top_score_below_no_answer_threshold": "The best dense result is below the configured NO_ANSWER score threshold.",
    "missing_score_gap": "The initial dense probe returned only one result, so confidence separation is unavailable.",
    "top_score_below_careful_threshold": "The best dense result is below the configured CAREFUL score threshold.",
    "score_gap_below_careful_threshold": "The top-two dense score gap is below the configured CAREFUL threshold.",
    "token_count_above_careful_threshold": "The query token count is above the configured CAREFUL threshold.",
    "complexity_marker_count_at_least_careful_threshold": "The query contains enough complexity markers to require the CAREFUL route.",
    "clause_marker_count_at_least_careful_threshold": "The query contains enough clause markers to require the CAREFUL route.",
    "long_token_ratio_at_least_careful_threshold": "The query long-token ratio is at or above the configured CAREFUL threshold.",
    "fast_conditions_satisfied": "All configured high-confidence and simple-query conditions are satisfied.",
    "standard_fallback": "The query matches neither the earlier NO_ANSWER/CAREFUL rules nor every FAST condition.",
}

ROUTE_REASON_CODES = {
    "NO_ANSWER": frozenset({"empty_probe", "top_score_below_no_answer_threshold"}),
    "CAREFUL": frozenset(
        {
            "missing_score_gap",
            "top_score_below_careful_threshold",
            "score_gap_below_careful_threshold",
            "token_count_above_careful_threshold",
            "complexity_marker_count_at_least_careful_threshold",
            "clause_marker_count_at_least_careful_threshold",
            "long_token_ratio_at_least_careful_threshold",
        }
    ),
    "FAST": frozenset({"fast_conditions_satisfied"}),
    "STANDARD": frozenset({"standard_fallback"}),
}


class RouterDecision(StrictFeatureModel):
    """Stable route, explanation, and execution intent from one policy evaluation."""

    router_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*@[0-9]+\.[0-9]+\.[0-9]+$")
    router_status: PipelineStatus
    feature_schema_version: Literal[1] = FEATURE_SCHEMA_VERSION
    route: RouterRoute
    reason_code: RouterReasonCode
    reason: str = Field(min_length=1)
    matched_reason_codes: tuple[RouterReasonCode, ...] = Field(min_length=1)
    pipeline_config: RetrievalPipelineName | None
    maximum_top_k: int = Field(ge=0, le=20)
    reuse_probe: bool
    generate_answer: bool
    response_mode: Literal["refusal"] | None = None

    @model_validator(mode="after")
    def require_consistent_decision_shape(self):
        if self.reason_code != self.matched_reason_codes[0]:
            raise ValueError("reason_code must be the first matched_reason_codes entry.")
        if len(self.matched_reason_codes) != len(set(self.matched_reason_codes)):
            raise ValueError("matched_reason_codes must not contain duplicates.")
        if any(code not in ROUTE_REASON_CODES[self.route] for code in self.matched_reason_codes):
            raise ValueError("matched_reason_codes must describe the selected route.")
        if self.reason != ROUTE_REASONS[self.reason_code]:
            raise ValueError("reason must match the stable text for reason_code.")
        if self.route == "NO_ANSWER":
            if self.pipeline_config is not None or self.maximum_top_k != 0 or self.reuse_probe or self.generate_answer:
                raise ValueError("NO_ANSWER must not select retrieval, probe reuse, or generation.")
            if self.response_mode != "refusal":
                raise ValueError("NO_ANSWER must select refusal response mode.")
        else:
            if self.pipeline_config is None or self.maximum_top_k == 0 or not self.generate_answer:
                raise ValueError("Retrieval routes must select a pipeline, positive output depth, and generation.")
            if self.response_mode is not None:
                raise ValueError("Retrieval routes cannot select a refusal response mode.")
        return self


class RoutedProbeResult(StrictFeatureModel):
    """One initial probe and the deterministic decision derived from it."""

    probe: InitialProbeResult
    decision: RouterDecision

    @model_validator(mode="after")
    def require_matching_feature_schema(self):
        if self.probe.features.schema_version != self.decision.feature_schema_version:
            raise ValueError("Probe and route decision feature schema versions must match.")
        return self


class RuleBasedRouter:
    """Evaluate the checked-in schema-v1 policy without model calls or mutable state."""

    def __init__(self, config):
        self.config = config if isinstance(config, RouterConfig) else RouterConfig.model_validate(config)

    def _decision(self, route, reason_codes):
        reason_codes = tuple(reason_codes)
        route_config = getattr(self.config.routes, route.lower())
        return RouterDecision(
            router_id=f"{self.config.name}@{self.config.version}",
            router_status=self.config.status,
            feature_schema_version=self.config.feature_schema_version,
            route=route,
            reason_code=reason_codes[0],
            reason=ROUTE_REASONS[reason_codes[0]],
            matched_reason_codes=reason_codes,
            pipeline_config=route_config.pipeline_config,
            maximum_top_k=route_config.maximum_top_k,
            reuse_probe=route_config.reuse_probe,
            generate_answer=route_config.generate_answer,
            response_mode=getattr(route_config, "response_mode", None),
        )

    def select(self, features):
        """Return one deterministic decision using documented order and inequalities."""
        features = features if isinstance(features, InitialRetrievalFeatures) else InitialRetrievalFeatures.model_validate(features)
        if features.schema_version != self.config.feature_schema_version:
            raise ValueError(
                f"Router expects feature schema {self.config.feature_schema_version}; received {features.schema_version}."
            )

        confidence = features.retrieval_confidence
        lexical = features.lexical_complexity
        query_length = features.query_length
        no_answer = self.config.thresholds.no_answer
        careful = self.config.thresholds.careful
        fast = self.config.thresholds.fast

        if confidence.result_count == 0:
            return self._decision("NO_ANSWER", ("empty_probe",))
        if confidence.top_score < no_answer.top_score_below:
            return self._decision("NO_ANSWER", ("top_score_below_no_answer_threshold",))

        careful_reasons = []
        if confidence.score_gap is None:
            careful_reasons.append("missing_score_gap")
        if confidence.top_score < careful.top_score_below:
            careful_reasons.append("top_score_below_careful_threshold")
        if confidence.score_gap is not None and confidence.score_gap < careful.score_gap_below:
            careful_reasons.append("score_gap_below_careful_threshold")
        if query_length.token_count > careful.token_count_above:
            careful_reasons.append("token_count_above_careful_threshold")
        if lexical.complexity_marker_count >= careful.complexity_marker_count_at_least:
            careful_reasons.append("complexity_marker_count_at_least_careful_threshold")
        if lexical.clause_marker_count >= careful.clause_marker_count_at_least:
            careful_reasons.append("clause_marker_count_at_least_careful_threshold")
        if lexical.long_token_ratio >= careful.long_token_ratio_at_least:
            careful_reasons.append("long_token_ratio_at_least_careful_threshold")
        if careful_reasons:
            return self._decision("CAREFUL", careful_reasons)

        fast_matches = (
            confidence.top_score >= fast.top_score_at_least
            and confidence.score_gap >= fast.score_gap_at_least
            and query_length.token_count <= fast.token_count_at_most
            and lexical.complexity_marker_count <= fast.complexity_marker_count_at_most
            and lexical.clause_marker_count <= fast.clause_marker_count_at_most
            and lexical.long_token_ratio <= fast.long_token_ratio_at_most
        )
        if fast_matches:
            return self._decision("FAST", ("fast_conditions_satisfied",))
        return self._decision("STANDARD", ("standard_fallback",))

    def select_probe(self, probe):
        probe = probe if isinstance(probe, InitialProbeResult) else InitialProbeResult.model_validate(probe)
        return RoutedProbeResult(probe=probe, decision=self.select(probe.features))
