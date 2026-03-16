from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from baselines.jailbreakbench_official import jailbreakbench_evaluate_batch
from jades.dataset_loaders import DatasetRow

from experiments.methods.tqdm_compat import tqdm


def run_jailbreakbench(
    *,
    rows: List[DatasetRow],
    api_key: str | None,
    model_name: Optional[str],
    out_dir: Path,
) -> Dict[str, Any]:
    if not api_key:
        if model_name is None:
            api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        elif str(model_name).startswith("openai/"):
            api_key = os.environ.get("OPENAI_API_KEY")
        elif str(model_name).startswith("gemini/") or str(model_name).startswith("gemini-"):
            api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
            if not api_key:
                _key_path = None  # Set corresponding API key env var
                if _key_path is not None and _key_path.exists():
                    api_key = _key_path.read_text(encoding="utf-8").strip()
        else:
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TOGETHER_API_KEY")

    if not api_key:
        print("Warning: No API key found for JailbreakBench (checked OPENAI_API_KEY, TOGETHER_API_KEY, GEMINI_API_KEY).")
        api_key = "missing-key"

    out_path = out_dir / "results.jsonl"
    summary: Dict[str, Any] = {
        "method": "jailbreakbench",
        "model": model_name or "official_default",
        "total_rows": len(rows),
        "errors": 0,
        "label_counts": {"JAILBROKEN": 0, "NOT": 0, "UNKNOWN": 0},
    }

    chunk_size = 32

    with out_path.open("w", encoding="utf-8") as f:
        pbar = tqdm(total=len(rows), desc="Evaluating (JailbreakBench)")
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            prompts = [r.query for r in chunk]
            responses = [r.response for r in chunk]

            try:
                results = jailbreakbench_evaluate_batch(
                    prompts=prompts,
                    responses=responses,
                    api_key=api_key,
                    model_name=model_name,
                )

                for r, res in zip(chunk, results):
                    label = "JAILBROKEN" if res else "NOT"
                    summary["label_counts"][label] = summary["label_counts"].get(label, 0) + 1

                    obj = {
                        "dataset": r.dataset,
                        "row_id": r.row_id,
                        "query": r.query,
                        "response": r.response,
                        "meta": r.meta,
                        "label": label,
                        "jailbreakbench_unsafe": res,
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
