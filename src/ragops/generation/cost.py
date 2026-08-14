import math
import os
from dataclasses import dataclass

INPUT_PRICE_ENV = "RAGOPS_LLM_INPUT_USD_PER_MILLION_TOKENS"
OUTPUT_PRICE_ENV = "RAGOPS_LLM_OUTPUT_USD_PER_MILLION_TOKENS"


@dataclass(frozen=True)
class GenerationPricing:
    """Operator-supplied token prices used for an explicitly labeled estimate."""

    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float


@dataclass(frozen=True)
class GenerationCost:
    amount_usd: float | None
    currency: str
    status: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


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
    """Load an optional complete input/output price pair from args or environment."""
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


def generation_provider(client):
    provider = getattr(client, "provider", None)
    if isinstance(provider, str) and provider.strip():
        return provider.strip()
    return type(client).__name__


def generation_model(client):
    model = getattr(client, "model", None)
    return model.strip() if isinstance(model, str) and model.strip() else None


def estimate_generation_cost(client, usage, pricing=None):
    """Return zero local cost, a configured token estimate, or explicit unavailability."""
    if generation_provider(client) == "template":
        return GenerationCost(0.0, "USD", "zero_cost", None, None, None)
    if usage is None:
        return GenerationCost(None, "USD", "unavailable", None, None, None)
    if pricing is None:
        return GenerationCost(None, "USD", "unavailable", usage.input_tokens, usage.output_tokens, usage.total_tokens)
    amount = (
        usage.input_tokens * pricing.input_usd_per_million_tokens
        + usage.output_tokens * pricing.output_usd_per_million_tokens
    ) / 1_000_000
    return GenerationCost(amount, "USD", "estimated", usage.input_tokens, usage.output_tokens, usage.total_tokens)
