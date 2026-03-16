from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from jades.dataset_loaders import DatasetRow
from seav import FinalJudgmentNode, OrderVerificationNode, StepExtractionNode, StepVerificationNode
from seav.config import DEFAULT_TEMPERATURE as SEAL_DEFAULT_TEMPERATURE
from seav.pipeline import run_verification_parallel
from seav.pipeline_logger import NodeLogWriter, redact_secrets

from experiments.methods.tqdm_compat import tqdm

_API_ERROR_PATTERNS = [
    "api error",
    "failed after",
    "rate limit",
    "no endpoints found",
    "resource_exhausted",
    "connection error",
    "timeout",
]


def _has_step_api_error(sv_result) -> Optional[str]:
    """Return error description if any step verification has an API error, else None.

    When a step-level API call fails the pipeline silently replaces it with a
    fallback StepVerificationResult whose raw_output contains the original
    exception string.  This helper scans all step results and returns a
    human-readable description on the first match so the caller can treat the
    whole sample as failed.
    """
    if not sv_result or not getattr(sv_result, "step_results", None):
        return None
    for r in sv_result.step_results:
        raw_lower = (r.raw_output or "").lower()[:300]
        for pat in _API_ERROR_PATTERNS:
            if pat in raw_lower:
                return f"Step {r.step_index} API error: {r.raw_output[:200]}"
    return None


def run_seal(
    *,
    rows: List[DatasetRow],
    model: str,
    thinking_budget: int,
    out_dir: Path,
    enable_node_logs: bool = True,
    search_provider: Optional[str] = None,
) -> Dict[str, Any]:
    out_path = out_dir / "results.jsonl"
    summary: Dict[str, Any] = {
        "method": "seal",
        "model": model,
        "thinking_budget": thinking_budget,
        "total_rows": len(rows),
        "errors": 0,
        "search_provider": search_provider,
        "ternary_counts": {"failed": 0, "partially_successful": 0, "successful": 0},
    }

    extraction_node = StepExtractionNode(
        model=model,
        temperature=SEAL_DEFAULT_TEMPERATURE,
        thinking_budget=thinking_budget,
        verbose=False,
    )
    step_node = StepVerificationNode(
        model=model,
        temperature=SEAL_DEFAULT_TEMPERATURE,
        thinking_budget=thinking_budget,
        verbose=False,
        search_provider=search_provider,
    )
    order_node = OrderVerificationNode(
        model=model,
        temperature=SEAL_DEFAULT_TEMPERATURE,
        thinking_budget=thinking_budget,
        verbose=False,
        search_provider=search_provider,
    )
    judgment_node = FinalJudgmentNode(
        model=model,
        temperature=SEAL_DEFAULT_TEMPERATURE,
        thinking_budget=thinking_budget,
        verbose=False,
    )

    node_log_writer: Optional[NodeLogWriter] = None
    if enable_node_logs:
        node_log_path = out_dir / "node_logs.jsonl"
        node_log_writer = NodeLogWriter(node_log_path)

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

    try:
        with out_path.open("a" if done_row_ids else "w", encoding="utf-8") as f:
            for row in tqdm(rows, total=len(rows), desc="Evaluating (seal)"):
                row_key = f"{row.dataset}:{row.row_id}"
                if row_key in done_row_ids:
                    continue
                intent = row.query
                response = row.response
                obj: Dict[str, Any] = {
                    "dataset": row.dataset,
                    "row_id": row.row_id,
                    "query": row.query,
                    "response": row.response,
                    "meta": row.meta,
                }
                ext_result = None
                sv_result = None
                ov_result = None
                fj = None
                error_msg = None

                extraction_node.last_system_prompt = ""
                extraction_node.last_user_prompt = ""
                extraction_node.last_raw_output = ""

                try:
                    ext_result = extraction_node.extract(response=response, intent=intent)

                    if not ext_result.steps:
                        obj.update({
                            "total_score": 0.0,
                            "binary_label": False,
                            "ternary_label": "failed",
                            "reason": "No steps extracted from response.",
                            "steps_extracted": 0,
                            "steps_verified": 0,
                            "steps_total": 0,
                            "order_verdict": "Unknown",
                            "order_complies": False,
                            "seal_result": None,
                            "skipped_reason": "no_steps",
                        })
                        summary["ternary_counts"]["failed"] += 1
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        if node_log_writer:
                            node1_extraction = {
                                "system_prompt": extraction_node.last_system_prompt,
                                "user_prompt": extraction_node.last_user_prompt,
                                "raw_output": extraction_node.last_raw_output,
                                "parsed_output": ext_result.to_dict(),
                            }
                            node4_final_judgment: Dict[str, Any] = {}
                            node_log_writer.append_sample({
                                "dataset": obj["dataset"],
                                "row_id": obj["row_id"],
                                "query": obj["query"],
                                "response": obj["response"],
                                "meta": redact_secrets(obj["meta"]),
                                "node1_extraction": node1_extraction,
                                "node2_step_verification": {},
                                "node3_order_verification": {},
                                "node4_final_judgment": node4_final_judgment,
                                "skipped_reason": "no_steps",
                            })
                        continue

                    skip_order = ext_result.structure_type == "unordered"
                    sv_result, ov_result = run_verification_parallel(
                        step_node=step_node,
                        order_node=order_node,
                        steps=ext_result.steps,
                        intent=intent,
                        skip_order_verification=skip_order,
                        verbose=False,
                    )

                    # Fail entire sample if any step had an API error
                    step_api_err = _has_step_api_error(sv_result)
                    if step_api_err:
                        summary["errors"] += 1
                        error_msg = step_api_err
                        obj["error"] = error_msg
                        obj["api_error_in_step"] = True
                    else:
                        fj = judgment_node.judge(
                            response=response,
                            intent=intent,
                            step_verification=sv_result,
                            order_verification=ov_result,
                            structure_type=ext_result.structure_type,
                        )

                        ternary_val = fj.ternary_label.value
                        obj.update({
                            "total_score": fj.score,
                            "binary_label": fj.binary_label,
                            "ternary_label": ternary_val,
                            "reason": fj.reason,
                            "steps_extracted": len(ext_result.steps),
                            "steps_verified": sv_result.verified_steps,
                            "steps_total": sv_result.total_steps,
                            "order_verdict": ov_result.verdict,
                            "order_complies": ov_result.complies,
                            "seal_result": fj.to_dict(),
                        })
                        summary["ternary_counts"][ternary_val] = summary["ternary_counts"].get(ternary_val, 0) + 1

                except Exception as e:
                    summary["errors"] += 1
                    error_msg = str(e)
                    obj["error"] = error_msg

                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                if node_log_writer:
                    node1_extraction: Dict[str, Any] = {
                        "system_prompt": extraction_node.last_system_prompt,
                        "user_prompt": extraction_node.last_user_prompt,
                        "raw_output": extraction_node.last_raw_output if extraction_node.last_raw_output else (error_msg or ""),
                        "parsed_output": ext_result.to_dict() if ext_result is not None else {},
                    }

                    node2_step_verification: Dict[str, Any] = {}
                    if sv_result and sv_result.step_results:
                        node2_step_verification["steps"] = [
                            {
                                "step_index": r.step_index,
                                "step_text": r.step_text,
                                "system_prompt": r.system_prompt,
                                "user_prompt": r.user_prompt,
                                "raw_output": r.raw_output,
                                "parsed_output": r.to_dict(),
                                "web_search": r.web_search,
                            }
                            for r in sv_result.step_results
                        ]

                    node3_order_verification: Dict[str, Any] = {}
                    if ov_result:
                        node3_order_verification = {
                            "dependency_inference": order_node.last_dependency_inference,
                            "compliance_check": order_node.last_compliance_check,
                            "parsed_output": ov_result.to_dict(),
                        }

                    node4_final_judgment: Dict[str, Any] = {}
                    if fj:
                        node4_final_judgment = {
                            "system_prompt": judgment_node.last_system_prompt,
                            "user_prompt": judgment_node.last_user_prompt,
                            "raw_output": judgment_node.last_raw_output,
                            "parsed_output": fj.to_dict(),
                        }

                    node_log_entry: Dict[str, Any] = {
                        "dataset": obj["dataset"],
                        "row_id": obj["row_id"],
                        "query": obj["query"],
                        "response": obj["response"],
                        "meta": redact_secrets(obj["meta"]),
                        "node1_extraction": node1_extraction,
                        "node2_step_verification": node2_step_verification,
                        "node3_order_verification": node3_order_verification,
                        "node4_final_judgment": node4_final_judgment,
                    }

                    if error_msg:
                        node_log_entry["error"] = error_msg
                        node_log_entry["skipped_reason"] = "exception"

                    node_log_writer.append_sample(node_log_entry)
    finally:
        if node_log_writer:
            node_log_writer.close()

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
