#!/usr/bin/env python3
"""Validate and summarize the Day 40 generation model cost table."""

import argparse
import json
from pathlib import Path

from ragops.generation.cost import load_model_cost_table


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/model_costs.yaml"))
    return parser.parse_args()


def main():
    args = parse_args()
    table = load_model_cost_table(args.config, project_root=Path.cwd())
    print(
        json.dumps(
            {
                "cost_table_id": table.identity,
                "status": table.status,
                "currency": table.currency,
                "token_estimator": table.token_estimator,
                "models": [
                    {
                        "provider": entry.provider,
                        "model": entry.model,
                        "input_usd_per_million_tokens": entry.input_usd_per_million_tokens,
                        "output_usd_per_million_tokens": entry.output_usd_per_million_tokens,
                        "source_checked_at": entry.source_checked_at.isoformat(),
                    }
                    for entry in table.models
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
