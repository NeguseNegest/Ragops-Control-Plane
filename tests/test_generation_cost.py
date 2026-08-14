import pytest

from ragops.generation.client import GenerationUsage, LocalTemplateGenerationClient
from ragops.generation.cost import (
    GenerationPricing,
    configured_generation_pricing,
    estimate_generation_cost,
)


class ExternalClient:
    provider = "openai"
    model = "test-model"


def test_template_generation_has_explicit_zero_cost():
    cost = estimate_generation_cost(LocalTemplateGenerationClient(), usage=None)

    assert cost.amount_usd == 0.0
    assert cost.status == "zero_cost"
    assert cost.currency == "USD"


def test_generation_usage_rejects_an_inconsistent_total():
    with pytest.raises(ValueError, match="input_tokens plus output_tokens"):
        GenerationUsage(input_tokens=100, output_tokens=20, total_tokens=119)


def test_external_generation_without_usage_or_pricing_is_not_reported_as_zero():
    without_usage = estimate_generation_cost(ExternalClient(), usage=None)
    usage = GenerationUsage(input_tokens=1_000, output_tokens=200, total_tokens=1_200)
    without_pricing = estimate_generation_cost(ExternalClient(), usage=usage)

    assert without_usage.amount_usd is None
    assert without_usage.status == "unavailable"
    assert without_pricing.amount_usd is None
    assert without_pricing.input_tokens == 1_000
    assert without_pricing.output_tokens == 200


def test_configured_token_rates_produce_labeled_estimate():
    usage = GenerationUsage(input_tokens=1_000, output_tokens=200, total_tokens=1_200)
    pricing = GenerationPricing(input_usd_per_million_tokens=2.0, output_usd_per_million_tokens=10.0)

    cost = estimate_generation_cost(ExternalClient(), usage=usage, pricing=pricing)

    assert cost.amount_usd == pytest.approx(0.004)
    assert cost.status == "estimated"
    assert cost.total_tokens == 1_200


def test_pricing_environment_requires_a_complete_non_negative_pair(monkeypatch):
    monkeypatch.setenv("RAGOPS_LLM_INPUT_USD_PER_MILLION_TOKENS", " 2.5 ")
    monkeypatch.setenv("RAGOPS_LLM_OUTPUT_USD_PER_MILLION_TOKENS", "10")

    pricing = configured_generation_pricing()

    assert pricing == GenerationPricing(2.5, 10.0)

    monkeypatch.delenv("RAGOPS_LLM_OUTPUT_USD_PER_MILLION_TOKENS")
    with pytest.raises(ValueError, match="must be configured together"):
        configured_generation_pricing()

    monkeypatch.setenv("RAGOPS_LLM_OUTPUT_USD_PER_MILLION_TOKENS", "nan")
    with pytest.raises(ValueError, match="finite non-negative"):
        configured_generation_pricing()
