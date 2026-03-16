from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from baselines.hf_local import is_local_hf_model
from baselines.llm_judge import is_anthropic_model, is_gemini_model, is_openrouter_model, judge_pair_force_web, judge_pair_no_web
from jades.dataset_loaders import DatasetRow

from experiments.methods.tqdm_compat import tqdm
from experiments.run_methods_common import ensure_openai_client


def run_judge(
    *,
    rows: List[DatasetRow],
    model: str,
    force_web: bool,
    max_output_tokens: int,
    temperature: float,
    reasoning_effort: Optional[str],
    cot_prompting: bool,
    check_factuality_order: bool,
    out_dir: Path,
    openrouter_provider: Optional[str] = None,
) -> Dict[str, Any]:
    if force_web and is_local_hf_model(model):
        raise ValueError("Local HF models do not support judge_web (forced web_search). Use judge_no_web.")

    needs_openai_client = (not is_local_hf_model(model)) and (not is_anthropic_model(model)) and (not is_gemini_model(model)) and (not is_openrouter_model(model))
    client = ensure_openai_client() if needs_openai_client else None
    out_path = out_dir / "results.jsonl"
    m = str(model)
    cot_applies = bool(cot_prompting and (not force_web) and (m.startswith("gpt-4.1") or ("claude-3-haiku" in m)))
    summary: Dict[str, Any] = {
        "method": "judge_web" if force_web else "judge_no_web",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "cot_prompting": cot_applies,
        "check_factuality_order": bool(check_factuality_order),
        "total_rows": len(rows),
        "errors": 0,
        "label_counts": {"JAILBROKEN": 0, "NOT": 0, "UNKNOWN": 0},
    }
    abort_due_to_quota = False

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in enumerate(
            tqdm(rows, total=len(rows), desc="Evaluating (judge_web)" if force_web else "Evaluating (judge_no_web)"),
            start=1,
        ):
            obj: Dict[str, Any] = {
                "dataset": row.dataset,
                "row_id": row.row_id,
                "query": row.query,
                "response": row.response,
                "meta": row.meta,
            }
            try:
                if force_web:
                    # Gemini override: use SEAL grounding instead of OpenAI Responses API
                    if is_gemini_model(model):
                        from baselines.llm_judge import _build_system_prompt, JUDGE_USER_TEMPLATE, _normalize_label
                        from seav.nodes.gemini_api import (
                            call_gemini_grounding_with_openrouter_fallback,
                            _call_gemini_api_with_tavily_search,
                            extract_grounding_info,
                            _GeminiAPIError,
                        )
                        
                        system_prompt = _build_system_prompt(
                            model=model,
                            cot_prompting=False,
                            check_factuality_order=check_factuality_order,
                        )
                        user_prompt = JUDGE_USER_TEMPLATE.format(query=row.query, response=row.response)
                        
                        # Try grounding first
                        raw, metadata = None, None
                        grounding_error = None
                        try:
                            raw, metadata = call_gemini_grounding_with_openrouter_fallback(
                                model=model,
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                search_query=None,
                                max_output_tokens=max_output_tokens,
                                temperature=temperature,
                                require_web_search=True,
                            )
                        except (_GeminiAPIError, Exception) as e:
                            grounding_error = str(e)
                            # Check if error is "No web search performed"
                            if "No web search performed" in grounding_error or "web_search_queries" in grounding_error:
                                # Fallback to Tavily search: derive search_query from row.query
                                fallback_query = (row.query or "").strip()[:200]
                                if fallback_query:
                                    try:
                                        raw, metadata = _call_gemini_api_with_tavily_search(
                                            model=model,
                                            system_prompt=system_prompt,
                                            user_prompt=user_prompt,
                                            search_query=fallback_query,
                                            max_output_tokens=max_output_tokens,
                                            temperature=temperature,
                                        )
                                    except Exception as tavily_error:
                                        # Tavily fallback failed; re-raise original grounding error
                                        raise _GeminiAPIError(grounding_error) from tavily_error
                                else:
                                    # No query to search; re-raise original error
                                    raise _GeminiAPIError(grounding_error)
                            else:
                                # Different error; re-raise
                                raise
                        
                        label = _normalize_label(raw)
                        _, search_query = extract_grounding_info(metadata)
                        obj.update({"label": label, "raw_output": raw, "web_search_query": search_query})
                    else:
                        label, raw, search_query = judge_pair_force_web(
                            client,
                            model,
                            row.query,
                            row.response,
                            max_output_tokens=max_output_tokens,
                            temperature=temperature,
                            reasoning_effort=reasoning_effort,
                            check_factuality_order=check_factuality_order,
                        )
                        obj.update({"label": label, "raw_output": raw, "web_search_query": search_query})
                else:
                    label, raw = judge_pair_no_web(
                        client,
                        model,
                        row.query,
                        row.response,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                        reasoning_effort=reasoning_effort,
                        cot_prompting=cot_prompting,
                        check_factuality_order=check_factuality_order,
                        openrouter_provider=openrouter_provider,
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
