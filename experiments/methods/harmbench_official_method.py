from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from jades.dataset_loaders import DatasetRow

from experiments.methods.tqdm_compat import tqdm

try:
    from baselines.harmbench_official import harmbench_official_classify_batch
except Exception:
    harmbench_official_classify_batch = None


def run_harmbench_official(
    *,
    rows: List[DatasetRow],
    model_id: str,
    batch_size: int,
    max_new_tokens: int,
    max_input_tokens: Optional[int],
    out_dir: Path,
) -> Dict[str, Any]:
    if harmbench_official_classify_batch is None:
        raise SystemExit(
            "Missing dependencies for --method harmbench_official.\n"
            "Install inside your env:\n"
            "  pip install -U torch transformers"
        )
    out_path = out_dir / "results.jsonl"
    summary: Dict[str, Any] = {
        "method": "harmbench_official",
        "model": model_id,
        "batch_size": int(batch_size),
        "max_new_tokens": int(max_new_tokens),
        "max_input_tokens": max_input_tokens,
        "total_rows": len(rows),
        "errors": 0,
        "label_counts": {"JAILBROKEN": 0, "NOT": 0, "UNKNOWN": 0},
    }

    bs = max(1, int(batch_size))
    total = len(rows)

    with out_path.open("w", encoding="utf-8") as f:
        pbar = tqdm(total=total, desc="Evaluating (HarmBench official)")
        for start in range(0, total, bs):
            batch = rows[start : start + bs]
            behaviors = [r.query for r in batch]
            generations = [r.response for r in batch]

            def _write_one(*, row: DatasetRow, label_int: Optional[int], raw_output: str) -> None:
                label = "UNKNOWN"
                if label_int == 1:
                    label = "JAILBROKEN"
                elif label_int == 0:
                    label = "NOT"

                meta = row.meta
                if not isinstance(meta, dict):
                    meta = {}
                meta = dict(meta)
                meta["harmbench_official_model_id"] = model_id
                meta["harmbench_official_label_int"] = label_int

                obj: Dict[str, Any] = {
                    "dataset": row.dataset,
                    "row_id": row.row_id,
                    "query": row.query,
                    "response": row.response,
                    "meta": meta,
                    "label": label,
                    "raw_output": raw_output,
                }
                summary["label_counts"][label] = summary["label_counts"].get(label, 0) + 1
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            try:
                label_ints, raw_outputs = harmbench_official_classify_batch(
                    model_id=model_id,
                    behaviors=behaviors,
                    generations=generations,
                    max_new_tokens=int(max_new_tokens),
                    max_input_tokens=max_input_tokens,
                )
                if len(label_ints) != len(batch) or len(raw_outputs) != len(batch):
                    raise RuntimeError(
                        f"Unexpected HarmBench output lengths: {len(label_ints)}/{len(raw_outputs)} for batch={len(batch)}"
                    )
                for row, li, raw in zip(batch, label_ints, raw_outputs):
                    _write_one(row=row, label_int=li, raw_output=raw)
            except Exception as e:
                batch_err = str(e)
                for row in batch:
                    try:
                        li1, raw1 = harmbench_official_classify_batch(
                            model_id=model_id,
                            behaviors=[row.query],
                            generations=[row.response],
                            max_new_tokens=int(max_new_tokens),
                            max_input_tokens=max_input_tokens,
                        )
                        _write_one(row=row, label_int=(li1[0] if li1 else None), raw_output=(raw1[0] if raw1 else ""))
                    except Exception as e2:
                        summary["errors"] += 1
                        obj: Dict[str, Any] = {
                            "dataset": row.dataset,
                            "row_id": row.row_id,
                            "query": row.query,
                            "response": row.response,
                            "meta": row.meta,
                            "error": f"batch_error={batch_err} | per_example_error={e2}",
                        }
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            pbar.update(len(batch))
        pbar.close()

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
