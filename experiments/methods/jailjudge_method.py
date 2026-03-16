from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from jades.dataset_loaders import DatasetRow

from experiments.methods.tqdm_compat import tqdm

try:
    from baselines.jailjudge import JailJudge
except Exception:
    JailJudge = None


def run_jailjudge(
    *,
    rows: List[DatasetRow],
    model_id: str,
    out_dir: Path,
) -> Dict[str, Any]:
    if JailJudge is None:
        raise SystemExit(
            "Missing dependencies for --method jailjudge.\n"
            "Install inside your env:\n"
            "  pip install -U torch transformers tqdm"
        )
    out_path = out_dir / "results.jsonl"
    summary: Dict[str, Any] = {
        "method": "jailjudge",
        "model": model_id,
        "total_rows": len(rows),
        "errors": 0,
        "label_counts": {"JAILBROKEN": 0, "NOT": 0, "UNKNOWN": 0},
    }

    judge = JailJudge(model_id=model_id)
    batch_size = 8

    with out_path.open("w", encoding="utf-8") as f:
        pbar = tqdm(total=len(rows), desc="Evaluating (JailJudge)")

        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]
            queries = [r.query for r in batch_rows]
            responses = [r.response for r in batch_rows]

            try:
                parsed_results, raw_outputs = judge.evaluate_batch(queries, responses)

                for row, parsed, raw in zip(batch_rows, parsed_results, raw_outputs):
                    label_str = "UNKNOWN"
                    if parsed["binary_label"] is True:
                        label_str = "JAILBROKEN"
                    elif parsed["binary_label"] is False:
                        label_str = "NOT"

                    summary["label_counts"][label_str] = summary["label_counts"].get(label_str, 0) + 1

                    obj = {
                        "dataset": row.dataset,
                        "row_id": row.row_id,
                        "query": row.query,
                        "response": row.response,
                        "meta": row.meta,
                        "jailjudge_score": parsed["score"],
                        "jailjudge_reason": parsed["reason"],
                        "label": label_str,
                        "raw_output": raw,
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            except Exception as e:
                err_msg = str(e)
                summary["errors"] += len(batch_rows)
                for row in batch_rows:
                    obj = {
                        "dataset": row.dataset,
                        "row_id": row.row_id,
                        "query": row.query,
                        "response": row.response,
                        "meta": row.meta,
                        "error": err_msg,
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            pbar.update(len(batch_rows))

        pbar.close()

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
