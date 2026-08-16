from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ragops.generation.client import GenerationUsage, LocalTemplateGenerationClient
from ragops.generation.cost import (
    TOKEN_ESTIMATOR_VERSION,
    GenerationCost,
    GenerationPricing,
    configured_generation_pricing,
    estimate_generation_cost,
    estimate_text_tokens,
    load_model_cost_table,
)


class ExternalClient:
    provider = "openai"
    model = "gpt-5-nano"


def test_template_generation_has_explicit_zero_cost():
    cost = estimate_generation_cost(LocalTemplateGenerationClient(), usage=None)

    assert cost.amount_usd == 0.0
    assert cost.status == "zero_cost"
    assert cost.currency == "USD"
    assert (cost.input_tokens, cost.output_tokens, cost.total_tokens) == (0, 0, 0)
    assert cost.token_source == "not_applicable"
    assert cost.pricing_source == "not_applicable"


def test_generation_usage_rejects_an_inconsistent_total():
    with pytest.raises(ValueError, match="input_tokens plus output_tokens"):
        GenerationUsage(input_tokens=100, output_tokens=20, total_tokens=119)


def test_external_generation_without_usage_or_pricing_is_not_reported_as_zero():
    without_usage = estimate_generation_cost(ExternalClient(), usage=None)
    usage = GenerationUsage(input_tokens=1_000, output_tokens=200, total_tokens=1_200)
    without_pricing = estimate_generation_cost(ExternalClient(), usage=usage)

    assert without_usage.amount_usd is None
    assert without_usage.status == "unavailable"
    assert without_usage.token_source == "unavailable"
    assert without_pricing.amount_usd is None
    assert without_pricing.input_tokens == 1_000
    assert without_pricing.output_tokens == 200
    assert without_pricing.token_source == "provider_reported"


def test_configured_token_rates_produce_labeled_estimate():
    usage = GenerationUsage(input_tokens=1_000, output_tokens=200, total_tokens=1_200)
    pricing = GenerationPricing(input_usd_per_million_tokens=2.0, output_usd_per_million_tokens=10.0)

    cost = estimate_generation_cost(ExternalClient(), usage=usage, pricing=pricing)

    assert cost.amount_usd == pytest.approx(0.004)
    assert cost.status == "estimated"
    assert cost.total_tokens == 1_200
    assert cost.token_source == "provider_reported"
    assert cost.pricing_source == "environment_override"


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


def test_checked_in_model_cost_table_has_reviewed_default_model_rates():
    table = load_model_cost_table("configs/model_costs.yaml", project_root=Path.cwd())

    assert table.identity == "generation_model_costs@1.0.0"
    assert table.status == "approved"
    assert table.token_estimator == TOKEN_ESTIMATOR_VERSION
    assert table.pricing_for("openai", "gpt-5-nano").model_dump(mode="json") == {
        "provider": "openai",
        "model": "gpt-5-nano",
        "input_usd_per_million_tokens": 0.05,
        "output_usd_per_million_tokens": 0.4,
        "source_url": "https://developers.openai.com/api/docs/models/gpt-5-nano",
        "source_checked_at": "2026-08-16",
        "notes": "Standard uncached text input and output pricing; tool calls and cached input are excluded.",
    }
    assert table.pricing_for("gemini", "gemini-3.6-flash").input_usd_per_million_tokens == 1.5
    assert table.pricing_for("openai", "unknown") is None


def test_token_estimator_is_deterministic_utf8_byte_ceiling():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcde") == 2
    assert estimate_text_tokens("å") == 1
    with pytest.raises(ValueError, match="must be a string"):
        estimate_text_tokens(None)


def test_known_model_without_provider_usage_uses_heuristic_and_model_table():
    table = load_model_cost_table("configs/model_costs.yaml", project_root=Path.cwd())

    cost = estimate_generation_cost(
        ExternalClient(),
        usage=None,
        cost_table=table,
        input_text="abcd",
        output_text="abcdefgh",
    )

    assert cost.status == "estimated"
    assert (cost.input_tokens, cost.output_tokens, cost.total_tokens) == (1, 2, 3)
    assert cost.token_source == "heuristic_estimate"
    assert cost.token_estimator == TOKEN_ESTIMATOR_VERSION
    assert cost.pricing_source == "model_cost_table"
    assert cost.price_table_id == table.identity
    assert cost.amount_usd == pytest.approx((1 * 0.05 + 2 * 0.4) / 1_000_000)


def test_provider_usage_precedes_heuristic_and_environment_prices_precede_table():
    table = load_model_cost_table("configs/model_costs.yaml", project_root=Path.cwd())
    usage = GenerationUsage(input_tokens=10, output_tokens=4, total_tokens=15)
    override = GenerationPricing(2.0, 3.0)

    cost = estimate_generation_cost(
        ExternalClient(),
        usage=usage,
        pricing=override,
        cost_table=table,
        input_text="x" * 400,
        output_text="y" * 400,
    )

    assert (cost.input_tokens, cost.output_tokens, cost.total_tokens) == (10, 4, 15)
    assert cost.token_source == "provider_reported"
    assert cost.token_estimator is None
    assert cost.pricing_source == "environment_override"
    assert cost.price_table_id is None
    assert cost.amount_usd == pytest.approx((10 * 2.0 + 4 * 3.0) / 1_000_000)


def test_unknown_model_retains_token_estimate_but_never_invents_a_price():
    table = load_model_cost_table("configs/model_costs.yaml", project_root=Path.cwd())
    client = ExternalClient()
    client.model = "unknown-model"

    cost = estimate_generation_cost(client, None, cost_table=table, input_text="question", output_text="answer")

    assert cost.status == "unavailable"
    assert cost.amount_usd is None
    assert cost.token_source == "heuristic_estimate"
    assert cost.input_tokens is not None
    assert cost.pricing_source == "unavailable"
    assert cost.price_table_id == table.identity


def test_model_cost_table_rejects_duplicate_models_unknown_fields_and_bad_yaml(tmp_path):
    source = yaml.safe_load(Path("configs/model_costs.yaml").read_text(encoding="utf-8"))
    duplicate_path = tmp_path / "duplicate.yaml"
    source["models"].append(dict(source["models"][0]))
    duplicate_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValidationError, match="identities must be unique"):
        load_model_cost_table(duplicate_path)

    source["models"].pop()
    source["unexpected"] = True
    unknown_path = tmp_path / "unknown.yaml"
    unknown_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_model_cost_table(unknown_path)

    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("models: [", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid model cost YAML"):
        load_model_cost_table(invalid_path)


def test_generation_cost_rejects_inconsistent_provenance_and_arithmetic():
    with pytest.raises(ValidationError, match="must record the supported estimator"):
        GenerationCost(
            amount_usd=None,
            status="unavailable",
            provider="openai",
            model="unknown",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            token_source="heuristic_estimate",
            pricing_source="unavailable",
        )
    with pytest.raises(ValidationError, match="does not match"):
        GenerationCost(
            amount_usd=1.0,
            status="estimated",
            provider="openai",
            model="gpt-5-nano",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            token_source="provider_reported",
            pricing_source="environment_override",
            input_usd_per_million_tokens=1.0,
            output_usd_per_million_tokens=1.0,
        )
