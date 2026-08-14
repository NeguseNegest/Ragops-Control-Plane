import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import scripts.build_pipeline_registry as registry_cli
from ragops.evaluation.runner import load_evaluation_config
from ragops.pipeline_registry import (
    PipelineRegistry,
    PipelineVersionMetadata,
    build_pipeline_registry,
    load_pipeline_registry,
    load_pipeline_registry_config,
    registry_json,
    validate_registry_matches_sources,
    write_pipeline_registry,
)
from ragops.reranking.cross_encoder import load_hybrid_rerank_config
from ragops.retrieval.bm25 import load_bm25_config
from ragops.retrieval.hybrid import load_hybrid_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_CONFIG_PATH = PROJECT_ROOT / "configs/pipeline_registry.yaml"


def checked_registry():
    config = load_pipeline_registry_config(REGISTRY_CONFIG_PATH, project_root=PROJECT_ROOT)
    return config, build_pipeline_registry(config, PROJECT_ROOT)


def test_checked_registry_builds_four_versioned_pipelines_and_aliases():
    _, registry = checked_registry()

    assert [pipeline.pipeline_type for pipeline in registry.pipelines] == ["dense", "bm25", "hybrid", "reranked"]
    assert [pipeline.version for pipeline in registry.pipelines] == ["1.0.0"] * 4
    assert [pipeline.status for pipeline in registry.pipelines] == ["approved", "approved", "rejected", "evaluated"]
    assert registry.aliases.model_dump() == {
        "baseline": "bm25_baseline@1.0.0",
        "candidate": "hybrid_rrf_cross_encoder@1.0.0",
        "production": "dense_baseline@1.0.0",
    }
    assert registry.pipelines[-1].evaluation.metrics.mrr_at_5 == pytest.approx(0.6888888888888889)
    assert all(pipeline.evaluation.question_count == 45 for pipeline in registry.pipelines)


def test_checked_registry_artifact_exactly_matches_current_sources():
    config, expected = checked_registry()

    actual = validate_registry_matches_sources(config, PROJECT_ROOT)

    assert actual == expected
    assert registry_json(actual) == config.output_path.read_text(encoding="utf-8")
    for pipeline in actual.pipelines:
        config_path = PROJECT_ROOT / pipeline.config.path
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == pipeline.config.sha256


def test_every_registered_yaml_explicitly_declares_version_and_status():
    cases = [
        ("dense_baseline.yaml", load_evaluation_config, "approved"),
        ("bm25_baseline.yaml", load_bm25_config, "approved"),
        ("hybrid.yaml", load_hybrid_config, "rejected"),
        ("hybrid_rerank.yaml", load_hybrid_rerank_config, "evaluated"),
    ]

    for filename, loader, expected_status in cases:
        path = PROJECT_ROOT / "configs" / filename
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        loaded = loader(path, project_root=PROJECT_ROOT)
        assert raw["version"] == loaded.version == "1.0.0"
        assert raw["status"] == loaded.status == expected_status


@pytest.mark.parametrize("version", ["1", "1.0", "01.0.0", "1.0.0.0", "v1.0.0", "1.0.0+"])
def test_pipeline_version_metadata_rejects_non_semver_versions(version):
    with pytest.raises(ValidationError):
        PipelineVersionMetadata(version=version, status="draft")


def test_registry_rejects_dangling_alias():
    _, registry = checked_registry()
    payload = registry.model_dump(mode="json")
    payload["aliases"]["candidate"] = "missing@1.0.0"

    with pytest.raises(ValidationError, match="unknown pipeline ID"):
        PipelineRegistry.model_validate(payload)


def test_registry_rejects_alias_to_rejected_pipeline():
    _, registry = checked_registry()
    payload = registry.model_dump(mode="json")
    payload["aliases"]["candidate"] = "hybrid_rrf@1.0.0"

    with pytest.raises(ValidationError, match="cannot point to status 'rejected'"):
        PipelineRegistry.model_validate(payload)


def test_registry_requires_common_top_five_comparison(tmp_path):
    config = load_pipeline_registry_config(REGISTRY_CONFIG_PATH, project_root=PROJECT_ROOT)
    comparison = json.loads(config.comparison_path.read_text(encoding="utf-8"))
    comparison["comparison_depth"] = 10
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(ValueError, match="comparison depth of 5"):
        build_pipeline_registry(config.model_copy(update={"comparison_path": comparison_path}), PROJECT_ROOT)


def test_registry_validation_detects_stale_checked_artifact(tmp_path):
    config, registry = checked_registry()
    stale = registry.model_dump(mode="json")
    stale["pipelines"][0]["evaluation"]["metrics"]["mrr_at_5"] = 0.0
    output_path = tmp_path / "pipeline_registry.json"
    output_path.write_text(json.dumps(stale), encoding="utf-8")

    with pytest.raises(ValueError, match="stale"):
        validate_registry_matches_sources(config.model_copy(update={"output_path": output_path}), PROJECT_ROOT)


def test_registry_writer_protects_existing_file_and_overwrites_atomically(tmp_path):
    _, registry = checked_registry()
    output_path = tmp_path / "pipeline_registry.json"

    assert write_pipeline_registry(registry, output_path) == output_path
    with pytest.raises(FileExistsError, match="already exists"):
        write_pipeline_registry(registry, output_path)
    write_pipeline_registry(registry, output_path, overwrite=True)

    assert load_pipeline_registry(output_path) == registry
    assert not list(tmp_path.glob("*.tmp"))


def test_validation_only_cli_reports_aliases(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["build_pipeline_registry.py", "--config", str(REGISTRY_CONFIG_PATH), "--validate-only"])

    registry_cli.main()

    output = capsys.readouterr().out
    assert "Valid pipeline registry 'ragops-retrieval': 4 versions" in output
    assert "candidate=hybrid_rrf_cross_encoder@1.0.0" in output
