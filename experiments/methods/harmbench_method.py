from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from baselines.harmbench import harmbench_judge_no_web
from baselines.hf_local import is_local_hf_model
from jades.dataset_loaders import DatasetRow

from experiments.methods.tqdm_compat import tqdm
from experiments.run_methods_common import ensure_openai_client


def run_harmbench(
    *,
    rows: List[DatasetRow],
    model: str,
    max_output_tokens: int,
    temperature: float,
    reasoning_effort: Optional[str],
    out_dir: Path,
) -> Dict[str, Any]:
    client = ensure_openai_client() if not is_local_hf_model(model) else None
    out_path = out_dir / "results.jsonl"
    summary: Dict[str, Any] = {
        "method": "harmbench",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "total_rows": len(rows),
        "errors": 0,
        "label_counts": {"JAILBROKEN": 0, "NOT": 0, "UNKNOWN": 0},
    }
    abort_due_to_quota = False

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in enumerate(tqdm(rows, total=len(rows), desc="Evaluating (harmbench)"), start=1):
            obj: Dict[str, Any] = {
                "dataset": row.dataset,
                "row_id": row.row_id,
                "query": row.query,
                "response": row.response,
                "meta": row.meta,
            }
            try:
                label, raw = harmbench_judge_no_web(
                    client,
                    model,
                    row.query,
                    row.response,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                )
                obj.update({"label": label, "raw_output": raw})
                summary["label_counts"][label] = summary["label_counts"].get(label, 0) + 1
            except Exception as e:
                summary["errors"] += 1
                err = str(e)
                obj["error"] = err
                if "insufficient_quota" in err or "You exceeded your current quota" in err:
                    abort_due_to_quota = True

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if abort_due_to_quota:
                break

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
