from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from baselines.harmscore_official import harmscore_evaluate_batch
from jades.dataset_loaders import DatasetRow

from experiments.methods.tqdm_compat import tqdm


def run_harmscore(
    *,
    rows: List[DatasetRow],
    out_dir: Path,
) -> Dict[str, Any]:
    out_path = out_dir / "results.jsonl"
    summary: Dict[str, Any] = {
        "method": "harmscore",
        "total_rows": len(rows),
        "errors": 0,
    }

    chunk_size = 32

    with out_path.open("w", encoding="utf-8") as f:
        pbar = tqdm(total=len(rows), desc="Evaluating (HarmScore)")
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            prompts = [r.query for r in chunk]
            responses = [r.response for r in chunk]

            try:
                results = harmscore_evaluate_batch(
                    prompts=prompts,
                    responses=responses,
                )

                for r, res in zip(chunk, results):
                    score = res.get("score", 0.0)

                    obj = {
                        "dataset": r.dataset,
                        "row_id": r.row_id,
                        "query": r.query,
                        "response": r.response,
                        "meta": r.meta,
                        "harmscore_output": res,
                        "total_score": score,
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            except Exception as e:
                summary["errors"] += len(chunk)
                err_msg = str(e)
                for r in chunk:
                    obj = {
                        "dataset": r.dataset,
                        "row_id": r.row_id,
                        "query": r.query,
                        "response": r.response,
                        "meta": r.meta,
                        "error": err_msg,
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            pbar.update(len(chunk))
        pbar.close()

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
