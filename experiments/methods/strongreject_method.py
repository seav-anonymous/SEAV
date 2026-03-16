from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from baselines.strongreject_pkg import strongreject_pkg_evaluate
from jades.dataset_loaders import DatasetRow
from seav.config import DEFAULT_MAX_OUTPUT_TOKENS

from experiments.methods.tqdm_compat import tqdm
from experiments.run_methods_common import ensure_openai_client


def run_strongreject_pkg(
    *,
    rows: List[DatasetRow],
    evaluator: str,
    models: List[str] | None,
    temperature: float,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    out_dir: Path,
) -> Dict[str, Any]:
    ensure_openai_client()

    out_path = out_dir / "results.jsonl"
    summary: Dict[str, Any] = {
        "method": "strongreject_pkg",
        "evaluator": evaluator,
        "models": models,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "total_rows": len(rows),
        "errors": 0,
    }

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in enumerate(tqdm(rows, total=len(rows), desc="Evaluating (StrongREJECT pkg)"), start=1):
            obj: Dict[str, Any] = {
                "dataset": row.dataset,
                "row_id": row.row_id,
                "query": row.query,
                "response": row.response,
                "meta": row.meta,
            }
            try:
                out = strongreject_pkg_evaluate(
                    row.query,
                    row.response,
                    evaluator=evaluator,
                    models=models,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                obj["strongreject_output"] = out
            except Exception as e:
                summary["errors"] += 1
                obj["error"] = str(e)

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
