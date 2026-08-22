# Pipeline registry

The registry binds an executable config to its version, lifecycle status, evaluation evidence, checksum, and alias.

It records promotion decisions. It does not deploy them.

## Current assignments

| Pipeline | Status | Alias | Reason |
| --- | --- | --- | --- |
| `dense_baseline@1.0.0` | `approved` | `production` | Default `/query` config |
| `bm25_baseline@1.0.0` | `approved` | `baseline` | Strong low-latency lexical control |
| `hybrid_rrf@1.0.0` | `rejected` | None | Lost Recall@5 to BM25 |
| `hybrid_rrf_cross_encoder@1.0.0` | `evaluated` | `candidate` | Best quality; high latency |

`baseline` is the comparison control. `production` is the default served config. They do not have to match.

Sources:

- [`configs/pipeline_registry.yaml`](../configs/pipeline_registry.yaml)
- [`configs/mlflow.yaml`](../configs/mlflow.yaml)
- [`reports/pipeline_registry.json`](../reports/pipeline_registry.json)

## Lifecycle

| Status | Meaning | Alias eligibility |
| --- | --- | --- |
| `draft` | Not fully evaluated | None |
| `evaluated` | Evidence complete | `candidate` |
| `approved` | Reviewed and selectable | Any alias |
| `rejected` | Failed a promotion decision | None |
| `retired` | No longer selectable | None |

`baseline` and `production` must point to approved versions. `candidate` may point to evaluated or approved.

## Versioning

Versions use semantic `MAJOR.MINOR.PATCH` form. Do not edit evaluated executable settings in place:

1. copy the config;
2. bump the version;
3. evaluate it;
4. attach the new evidence; and
5. move aliases explicitly.

Status or alias-only changes do not require a version bump, but remain visible in the registry diff and checksum.

## Promote

1. Evaluate a new `draft` version.
2. Log and verify its MLflow artifacts.
3. Mark it `evaluated` and move `candidate`.
4. Review quality, latency, safety, and failures.
5. Mark it `approved` and move `production`.
6. Deploy the config separately.

Rollback moves `production` to the previous approved ID, rebuilds the registry, and redeploys that config. Existing evidence is never overwritten.

## Commands

```bash
make test-pipeline-registry
make build-pipeline-registry
make validate-pipeline-registry
```

The generated JSON is deterministic and rejects stale configs, evidence, hashes, or aliases.
