from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from baselines.llm_judge import is_anthropic_model
from jades import JADES, JADESConfig
from jades.dataset_loaders import DatasetRow

from experiments.methods.tqdm_compat import tqdm
from experiments.run_methods_common import ensure_openai_client
from seav.nodes.bedrock_api import is_bedrock_model
from seav.nodes.gemini_api import is_gemini_model as _is_gemini_model_for_jades
from seav.nodes.openrouter_api import is_openrouter_model


def run_jades(
    *,
    rows: List[DatasetRow],
    model: str,
    temperature: float,
    jades_variant: str,
    disable_clean_node: bool,
    enable_fact_checking: bool,
    fact_check_source: str,
    tavily_api_key: str | None,
    bias_unknown_facts_as_wrong: bool,
    out_dir: Path,
) -> Dict[str, Any]:
    client: Any
    if is_bedrock_model(model):
        class _BedrockClientSentinel:
            pass

        client = _BedrockClientSentinel()
    elif is_openrouter_model(model):
        class _OpenRouterClientSentinel:
            pass

        client = _OpenRouterClientSentinel()
    elif _is_gemini_model_for_jades(model):
        class _GeminiClientSentinel:
            pass

        client = _GeminiClientSentinel()
    elif is_anthropic_model(model):
        class _AnthropicClientSentinel:
            pass

        client = _AnthropicClientSentinel()
    else:
        client = ensure_openai_client()
    fact_check_log_path = str(out_dir / "factcheck_trace.jsonl") if enable_fact_checking else None
    if fact_check_log_path:
        Path(fact_check_log_path).write_text("", encoding="utf-8")
    config = JADESConfig(
        jades_variant=jades_variant,
        llm_model=model,
        llm_temperature=temperature,
        enable_fact_checking=enable_fact_checking,
        fact_check_source=fact_check_source,
        tavily_api_key=tavily_api_key,
        fact_check_log_path=fact_check_log_path,
        disable_clean_node=disable_clean_node,
        bias_unknown_facts_as_wrong=bool(bias_unknown_facts_as_wrong),
        verbose=False,
        memory_enabled=False,
    )
    jades = JADES(config=config, llm_client=client)

    out_path = out_dir / "results.jsonl"
    summary: Dict[str, Any] = {
        "method": "jades_web" if enable_fact_checking else "jades",
        "model": model,
        "total_rows": len(rows),
        "errors": 0,
    }

    # Resume support: skip rows already in results.jsonl
    done_row_ids: set[str] = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        with out_path.open("r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if line:
                    try:
                        existing = json.loads(line)
                        rid = existing.get("row_id", "")
                        ds = existing.get("dataset", "")
                        done_row_ids.add(f"{ds}:{rid}")
                    except json.JSONDecodeError:
                        pass
        print(f"[RESUME] Found {len(done_row_ids)} already-completed rows, skipping them.")

    with out_path.open("a" if done_row_ids else "w", encoding="utf-8") as f:
        for _, row in enumerate(tqdm(rows, total=len(rows), desc="Evaluating (JADES)"), start=1):
            row_key = f"{row.dataset}:{row.row_id}"
            if row_key in done_row_ids:
                continue
            obj: Dict[str, Any] = {
                "dataset": row.dataset,
                "row_id": row.row_id,
                "query": row.query,
                "response": row.response,
                "meta": row.meta,
            }
            try:
                if enable_fact_checking:
                    config.fact_check_trace_context = {
                        "dataset": row.dataset,
                        "row_id": row.row_id,
                    }
                result = jades.evaluate(row.query, row.response)
                sub_items = [sq.question for sq in result.sub_questions]
                sub_weights = [sq.weight for sq in result.sub_questions]
                sub_scores = [ss.score for ss in result.sub_scores]
                sub_reasons = [ss.reason for ss in result.sub_scores]

                extra: Dict[str, Any] = {}
                if jades_variant == "answer_only":
                    extra = {
                        "answer_parts": sub_items,
                        "answer_part_weights": sub_weights,
                        "answer_part_scores": sub_scores,
                        "answer_part_reasons": sub_reasons,
                    }
                obj.update(
                    {
                        "total_score": result.total_score,
                        "binary_label": bool(result.binary_label),
                        "ternary_label": result.ternary_label.value,
                        "sub_questions": sub_items,
                        "sub_weights": sub_weights,
                        "sub_scores": sub_scores,
                        "sub_reasons": sub_reasons,
                        **extra,
                    }
                )
            except Exception as e:
                summary["errors"] += 1
                obj["error"] = str(e)

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Post-run contamination check
    _FAILURE_MARKER = "_JADES_LLM_PARSE_FAILED"
    contaminated_rows = []
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    row_obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                reasons = row_obj.get("sub_reasons") or []
                if any(_FAILURE_MARKER in str(r) for r in reasons):
                    contaminated_rows.append(row_obj.get("row_id", "unknown"))
                if "error" in row_obj:
                    contaminated_rows.append(row_obj.get("row_id", "unknown"))

    summary["contaminated_rows"] = contaminated_rows
    summary["contaminated_count"] = len(contaminated_rows)

    if contaminated_rows:
        print(f"\n??  WARNING: {len(contaminated_rows)} rows had LLM parse failures (fallback to score=0.0).")
        print(f"   Affected row_ids: {contaminated_rows}")
        print(f"   These samples should be re-run for accurate results.")
        contam_path = out_dir / "contaminated_rows.txt"
        contam_path.write_text("\n".join(str(r) for r in contaminated_rows) + "\n", encoding="utf-8")
        print(f"   Contaminated row IDs saved to: {contam_path}")

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
