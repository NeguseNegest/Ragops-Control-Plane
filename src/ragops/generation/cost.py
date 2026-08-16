import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragops.pipeline_registry import PipelineVersion

INPUT_PRICE_ENV = "RAGOPS_LLM_INPUT_USD_PER_MILLION_TOKENS"
OUTPUT_PRICE_ENV = "RAGOPS_LLM_OUTPUT_USD_PER_MILLION_TOKENS"
MODEL_COST_CONFIG_ENV = "RAGOPS_MODEL_COST_CONFIG"
DEFAULT_MODEL_COST_CONFIG = Path("configs/model_costs.yaml")
TOKEN_ESTIMATOR_VERSION = "utf8_bytes_div4_ceiling_v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class GenerationPricing:
    """Operator-supplied token prices that override the checked-in model table."""

    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float


class ModelPricing(StrictModel):
    """One reviewed standard text-token price pair for an exact provider/model."""

    provider: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    model: str = Field(min_length=1)
    input_usd_per_million_tokens: float = Field(ge=0)
    output_usd_per_million_tokens: float = Field(ge=0)
    source_url: str = Field(pattern=r"^https://")
    source_checked_at: date
    notes: str = Field(min_length=1)

    @field_validator("provider", "model", "source_url", "notes")
    @classmethod
    def clean_text(cls, value):
        return value.strip()

    @field_validator("input_usd_per_million_tokens", "output_usd_per_million_tokens")
    @classmethod
    def require_finite_rate(cls, value):
        if not math.isfinite(value):
            raise ValueError("Model token prices must be finite.")
        return value


class ModelCostTable(StrictModel):
    """Versioned cost catalog and token-estimator identity used by the API."""

    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: PipelineVersion
    status: Literal["draft", "approved"]
    currency: Literal["USD"] = "USD"
    token_estimator: Literal["utf8_bytes_div4_ceiling_v1"] = TOKEN_ESTIMATOR_VERSION
    models: list[ModelPricing] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_models(self):
        identities = [(entry.provider, entry.model) for entry in self.models]
        if len(identities) != len(set(identities)):
            raise ValueError("Model cost table provider/model identities must be unique.")
        return self

    @property
    def identity(self):
        return f"{self.name}@{self.version}"

    def pricing_for(self, provider, model):
        """Return an exact model price; aliases and approximate matching are forbidden."""
        for entry in self.models:
            if entry.provider == provider and entry.model == model:
                return entry
        return None


class GenerationCost(StrictModel):
    """Auditable per-request generation cost and token-count provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    amount_usd: float | None = Field(default=None, ge=0)
    currency: Literal["USD"] = "USD"
    status: Literal["zero_cost", "estimated", "unavailable"]
    provider: str = Field(min_length=1)
    model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    token_source: Literal["provider_reported", "heuristic_estimate", "not_applicable", "unavailable"]
    token_estimator: str | None = None
    pricing_source: Literal["model_cost_table", "environment_override", "not_applicable", "unavailable"]
    price_table_id: str | None = None
    input_usd_per_million_tokens: float | None = Field(default=None, ge=0)
    output_usd_per_million_tokens: float | None = Field(default=None, ge=0)

    @field_validator("provider", "model", "token_estimator", "price_table_id")
    @classmethod
    def clean_text(cls, value):
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Cost text fields must not be empty when provided.")
        return value

    @field_validator("amount_usd", "input_usd_per_million_tokens", "output_usd_per_million_tokens")
    @classmethod
    def require_finite_number(cls, value):
        if value is not None and not math.isfinite(value):
            raise ValueError("Cost amounts and rates must be finite.")
        return value

    @model_validator(mode="after")
    def validate_cost_state(self):
        token_values = (self.input_tokens, self.output_tokens, self.total_tokens)
        rate_values = (self.input_usd_per_million_tokens, self.output_usd_per_million_tokens)
        if any(value is not None for value in token_values) and not all(value is not None for value in token_values):
            raise ValueError("Generation token counts must be all present or all absent.")
        if all(value is not None for value in token_values) and self.total_tokens < self.input_tokens + self.output_tokens:
            raise ValueError("Total generation tokens must not be smaller than input plus output tokens.")
        if any(value is not None for value in rate_values) and not all(value is not None for value in rate_values):
            raise ValueError("Generation input/output prices must be both present or both absent.")
        if self.token_source == "heuristic_estimate" and self.token_estimator != TOKEN_ESTIMATOR_VERSION:
            raise ValueError("Heuristic token counts must record the supported estimator version.")
        if self.token_source != "heuristic_estimate" and self.token_estimator is not None:
            raise ValueError("Only heuristic token counts may record a token estimator.")

        if self.status == "zero_cost":
            if self.amount_usd != 0 or token_values != (0, 0, 0):
                raise ValueError("Zero-cost generation must record zero amount and zero billable tokens.")
            if self.token_source != "not_applicable" or self.pricing_source != "not_applicable":
                raise ValueError("Zero-cost generation must mark token and pricing sources not applicable.")
            if any(value is not None for value in rate_values) or self.price_table_id is not None:
                raise ValueError("Zero-cost generation must not attach model rates or a price table.")
            return self

        if self.status == "unavailable":
            if self.amount_usd is not None or self.pricing_source != "unavailable" or any(
                value is not None for value in rate_values
            ):
                raise ValueError("Unavailable cost must not contain an amount or token prices.")
            return self

        if self.amount_usd is None or not all(value is not None for value in token_values + rate_values):
            raise ValueError("Estimated cost requires amount, token counts, and input/output prices.")
        if self.token_source not in {"provider_reported", "heuristic_estimate"}:
            raise ValueError("Estimated cost requires provider-reported or heuristic token counts.")
        if self.pricing_source not in {"model_cost_table", "environment_override"}:
            raise ValueError("Estimated cost requires a model-table or environment price source.")
        if self.pricing_source == "model_cost_table" and self.price_table_id is None:
            raise ValueError("Model-table pricing must record the table identity.")
        if self.pricing_source == "environment_override" and self.price_table_id is not None:
            raise ValueError("Environment pricing must not claim a model-table identity.")
        expected = (
            self.input_tokens * self.input_usd_per_million_tokens
            + self.output_tokens * self.output_usd_per_million_tokens
        ) / 1_000_000
        if not math.isclose(self.amount_usd, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Estimated generation amount does not match its token counts and prices.")
        return self


def _optional_rate(value, name):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number.")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite non-negative number.") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return value


def configured_generation_pricing(input_rate=None, output_rate=None):
    """Load an optional complete input/output override pair from args or environment."""
    if input_rate is None:
        input_rate = os.getenv(INPUT_PRICE_ENV)
    if output_rate is None:
        output_rate = os.getenv(OUTPUT_PRICE_ENV)
    input_rate = _optional_rate(input_rate, INPUT_PRICE_ENV)
    output_rate = _optional_rate(output_rate, OUTPUT_PRICE_ENV)
    if input_rate is None and output_rate is None:
        return None
    if input_rate is None or output_rate is None:
        raise ValueError(f"{INPUT_PRICE_ENV} and {OUTPUT_PRICE_ENV} must be configured together.")
    return GenerationPricing(input_rate, output_rate)


def configured_model_cost_path(path=None, project_root=None):
    """Resolve the checked-in model table or an explicit environment override."""
    if path is None:
        configured = os.getenv(MODEL_COST_CONFIG_ENV)
        path = configured.strip() if configured and configured.strip() else DEFAULT_MODEL_COST_CONFIG
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    if project_root is None:
        configured_root = os.getenv("RAGOPS_PROJECT_ROOT")
        project_root = configured_root.strip() if configured_root and configured_root.strip() else Path.cwd()
    return (Path(project_root).resolve() / path).resolve()


def load_model_cost_table(path=None, project_root=None):
    """Load the strict Day 40 model cost table."""
    path = configured_model_cost_path(path, project_root=project_root)
    if not path.is_file():
        raise FileNotFoundError(f"Model cost table does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid model cost YAML in {path}: {error}") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Model cost table must contain a YAML mapping: {path}")
    return ModelCostTable.model_validate(payload)


def estimate_text_tokens(text):
    """Estimate text tokens deterministically as ceil(UTF-8 bytes / 4)."""
    if not isinstance(text, str):
        raise ValueError("Token estimation input must be a string.")
    return math.ceil(len(text.encode("utf-8")) / 4)


def generation_provider(client):
    provider = getattr(client, "provider", None)
    if isinstance(provider, str) and provider.strip():
        return provider.strip().lower()
    return type(client).__name__.strip().lower()


def generation_model(client):
    model = getattr(client, "model", None)
    return model.strip() if isinstance(model, str) and model.strip() else None


def _token_counts(usage, input_text, output_text):
    if usage is not None:
        return usage.input_tokens, usage.output_tokens, usage.total_tokens, "provider_reported", None
    if isinstance(input_text, str) and isinstance(output_text, str):
        input_tokens = estimate_text_tokens(input_text)
        output_tokens = estimate_text_tokens(output_text)
        return input_tokens, output_tokens, input_tokens + output_tokens, "heuristic_estimate", TOKEN_ESTIMATOR_VERSION
    return None, None, None, "unavailable", None


def estimate_generation_cost(client, usage, pricing=None, cost_table=None, input_text=None, output_text=None):
    """Return an auditable zero, estimated, or explicitly unavailable generation cost."""
    provider = generation_provider(client)
    model = generation_model(client)
    if provider == "template":
        return GenerationCost(
            amount_usd=0.0,
            status="zero_cost",
            provider=provider,
            model=model,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            token_source="not_applicable",
            pricing_source="not_applicable",
        )

    input_tokens, output_tokens, total_tokens, token_source, token_estimator = _token_counts(
        usage,
        input_text,
        output_text,
    )
    selected_pricing = pricing
    pricing_source = "environment_override" if pricing is not None else "unavailable"
    price_table_id = None
    if selected_pricing is None and cost_table is not None:
        entry = cost_table.pricing_for(provider, model)
        price_table_id = cost_table.identity
        if entry is not None:
            selected_pricing = GenerationPricing(
                entry.input_usd_per_million_tokens,
                entry.output_usd_per_million_tokens,
            )
            pricing_source = "model_cost_table"

    if selected_pricing is None or input_tokens is None:
        return GenerationCost(
            amount_usd=None,
            status="unavailable",
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            token_source=token_source,
            token_estimator=token_estimator,
            pricing_source="unavailable",
            price_table_id=price_table_id,
        )

    amount = (
        input_tokens * selected_pricing.input_usd_per_million_tokens
        + output_tokens * selected_pricing.output_usd_per_million_tokens
    ) / 1_000_000
    return GenerationCost(
        amount_usd=amount,
        status="estimated",
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        token_source=token_source,
        token_estimator=token_estimator,
        pricing_source=pricing_source,
        price_table_id=price_table_id if pricing_source == "model_cost_table" else None,
        input_usd_per_million_tokens=selected_pricing.input_usd_per_million_tokens,
        output_usd_per_million_tokens=selected_pricing.output_usd_per_million_tokens,
    )
