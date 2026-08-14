# Pipeline Registry

## Purpose

Day 30 treats each retrieval configuration as a named pipeline version and separates three decisions that were previously implicit:

- `version` identifies the executable configuration.
- `status` records its lifecycle decision.
- an alias points a human-facing role such as `production` at one exact `name@version` ID.

The registry is intentionally a checked-in control-plane artifact, not a deployment mechanism. Updating `production` records the selected version; a separate reviewed deployment change must make the online service load that version.

## Sources and generated artifact

[`configs/pipeline_registry.yaml`](../configs/pipeline_registry.yaml) declares:

- registry schema and name;
- the Day 29 MLflow evidence catalog;
- the Day 27 common-depth comparison;
- the generated JSON path;
- `baseline`, `candidate`, and `production` targets.

[`reports/pipeline_registry.json`](../reports/pipeline_registry.json) is generated from those sources. Each entry contains:

- immutable ID in `name@semantic-version` form;
- lifecycle status and pipeline type;
- `common_v1` retriever-interface version;
- repository-relative config path and SHA256;
- evaluation report, comparison, and benchmark paths;
- question count and the common MRR@5, Recall, Hit Rate, nDCG, and historical average latency evidence;
- the Day 29 evidence digest;
- MLflow experiment and run names.

The generated JSON contains no timestamps or machine-specific absolute paths, so an unchanged source tree produces byte-for-byte identical output.

## Version schema

Versions use `MAJOR.MINOR.PATCH` semantic-version form, with an optional prerelease suffix. The initial registry snapshot assigns `1.0.0` to all four pipelines.

- Increase `MAJOR` for an incompatible pipeline contract, schema, or retriever-interface change.
- Increase `MINOR` for a backward-compatible capability or material pipeline composition/model change.
- Increase `PATCH` for a smaller executable configuration adjustment that still needs independent evaluation and rollback identity.
- Do not bump the executable version solely for a reviewed status or alias transition. The config checksum and registry diff still make that metadata change visible.

Once evaluation evidence is attached to a version, its executable settings must not be edited in place. Copy the configuration, bump its version, run evaluation, add the resulting artifact bundle to the catalog, and then update aliases explicitly.

## Lifecycle statuses

| Status | Meaning | Alias eligibility |
| --- | --- | --- |
| `draft` | Config exists but has not passed the required evaluation. | None |
| `evaluated` | Evidence is complete, but promotion approval is pending. | `candidate` |
| `approved` | Evidence and review permit selection as a control or deployed version. | `baseline`, `candidate`, or `production` |
| `rejected` | Evidence failed a quality, latency, safety, or operational decision. | None |
| `retired` | Previously useful version is no longer selectable. | None |

The registry validator rejects dangling aliases and aliases to `draft`, `rejected`, or `retired` entries. It also requires `baseline` and `production` to be approved, while `candidate` may be evaluated or approved.

## Current assignments

| Pipeline | Status | Alias | Evidence-based reason |
| --- | --- | --- | --- |
| `dense_baseline@1.0.0` | `approved` | `production` | Dense retrieval is the only retrieval path currently wired into FastAPI. |
| `bm25_baseline@1.0.0` | `approved` | `baseline` | It is the strongest practical low-latency measured control and beats dense and unweighted RRF on the current labels. |
| `hybrid_rrf@1.0.0` | `rejected` | none | It improves over dense but trails BM25 and loses one BM25 top-ten hit. |
| `hybrid_rrf_cross_encoder@1.0.0` | `evaluated` | `candidate` | It has the best measured MRR@5, but warmed end-to-end latency is about 4.48 seconds. |

`baseline` and `production` intentionally differ. Baseline identifies the comparison control; production identifies the online implementation actually served today.

## Promotion workflow

1. Create or copy a pipeline YAML with a new semantic version whenever executable settings change. Start it as `draft`.
2. Run the appropriate validation and live evaluation. Preserve JSON, CSV, comparison, and Markdown evidence.
3. Add the evidence bundle to `configs/mlflow.yaml`, import it to MLflow, and confirm the run is complete.
4. Change the config status to `evaluated`, point `candidate` at the exact `name@version`, rebuild the registry, and review the checksum, metrics, latency, failures, and diff.
5. If the candidate passes the project’s quality, latency, safety, and operational gates, change its status to `approved` and move the `production` alias in the same reviewed change.
6. Deploy the selected config separately and verify the running service. The registry alias alone has no runtime side effect.
7. Keep the previous approved version registered so rollback remains an explicit alias and deployment reversal rather than reconstruction from memory.

Later evaluation-gate and canary milestones can automate parts of steps 4–6, but Day 30 deliberately keeps promotion reviewable and local.

## Rollback workflow

1. Identify the previous approved pipeline ID from version control or the earlier registry snapshot.
2. Move `production` back to that exact ID.
3. Rebuild and validate the registry.
4. Redeploy the corresponding config through the runtime deployment process.
5. Record why the newer version was rejected or retired; do not overwrite its evidence.

## Commands

```bash
# Unit and policy tests
make test-pipeline-registry

# Regenerate the deterministic artifact after an intentional reviewed change
make build-pipeline-registry

# Fail if config metadata, evidence, hashes, aliases, or checked-in JSON drift
make validate-pipeline-registry
```

The builder refuses to overwrite an existing artifact unless overwrite is explicit. The Make target supplies that flag for deliberate regeneration, and writes atomically so an interrupted process cannot leave a partial registry.
