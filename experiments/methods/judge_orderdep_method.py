from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from baselines.hf_local import is_local_hf_model
from baselines.llm_judge import judge_pair_no_web
from baselines.order_dependency_judge import (
    JudgeWithOrderInfoNode,
    OrderComplianceNode,
    OrderDependencyNode,
    build_orderdep_inputs,
)
from jades.dataset_loaders import DatasetRow

from experiments.methods.tqdm_compat import tqdm
from experiments.run_methods_common import ensure_openai_client, stable_seed_int


def run_judge_orderdep(
    *,
    rows: List[DatasetRow],
    model: str,
    max_output_tokens: int,
    temperature: float,
    reasoning_effort: Optional[str],
    orderdep_dependency_prompt: str,
    out_dir: Path,
) -> Dict[str, Any]:
    client = ensure_openai_client() if not is_local_hf_model(model) else None
    orderdep_node = OrderDependencyNode(
        llm_client=client,
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        dependency_prompt_variant=orderdep_dependency_prompt,
    )
    compliance_node = OrderComplianceNode(
        llm_client=client,
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
    final_judge_node = JudgeWithOrderInfoNode(
        llm_client=client,
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
    out_path = out_dir / "results.jsonl"
    summary: Dict[str, Any] = {
        "method": "judge_orderdep",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "orderdep_dependency_prompt": orderdep_dependency_prompt,
        "total_rows": len(rows),
        "errors": 0,
        "label_counts": {"JAILBROKEN": 0, "NOT": 0, "UNKNOWN": 0},
        "skipped_no_steps": 0,
    }

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in enumerate(tqdm(rows, total=len(rows), desc="Evaluating (judge_orderdep)"), start=1):
            seed = stable_seed_int(row.dataset, row.row_id, row.query, row.response)
            meta = row.meta if isinstance(row.meta, dict) else {}
            intent = str(meta.get("goal") or row.query or "")
            jailbreak_prompt = str(meta.get("prompt") or "")
            cols = meta.get("columns") if isinstance(meta.get("columns"), dict) else {}
            if not jailbreak_prompt:
                jailbreak_prompt = str(
                    cols.get("adversarial prompts") or cols.get("adversarial_prompt") or cols.get("adversarial prompts ") or ""
                )
            obj: Dict[str, Any] = {
                "dataset": row.dataset,
                "row_id": row.row_id,
                "query": row.query,
                "response": row.response,
                "meta": meta,
                "orderdep_seed": seed,
                "orderdep_dependency_prompt": orderdep_dependency_prompt,
            }
            try:
                od = build_orderdep_inputs(row.response, seed=seed)
                original_steps = od.get("original_steps_in_order") or []
                shuffled_steps = od.get("shuffled_steps_by_id") or {}

                obj.update(
                    {
                        "orderdep": {
                            "intent": intent,
                            "jailbreaking_prompt": jailbreak_prompt,
                            "original_steps_in_order": original_steps,
                            "shuffled_steps_by_id": shuffled_steps,
                            "perm": od.get("perm"),
                            "shuffled_response": od.get("shuffled_response"),
                        }
                    }
                )

                if not original_steps or len(original_steps) < 2:
                    summary["skipped_no_steps"] += 1
                    label, raw = judge_pair_no_web(
                        client,
                        model,
                        row.query,
                        row.response,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                        reasoning_effort=reasoning_effort,
                        cot_prompting=False,
                    )
                    obj.update(
                        {
                            "label": label,
                            "raw_output": raw,
                            "order_dependency_graph": None,
                            "order_compliance": None,
                            "orderdep_skipped": True,
                        }
                    )
                else:
                    dep_graph, dep_raw, dep_user_prompt = orderdep_node.infer(
                        intent=intent,
                        jailbreaking_prompt=jailbreak_prompt,
                        shuffled_steps_by_id=shuffled_steps,
                    )
                    comp, comp_raw, comp_user_prompt = compliance_node.check(
                        original_steps_in_order=original_steps,
                        dependency_graph=dep_graph,
                    )
                    label, raw, judge_user_prompt = final_judge_node.judge(
                        query=row.query,
                        response=row.response,
                        dependency_graph=dep_graph,
                        order_compliance=comp,
                    )
                    obj.update(
                        {
                            "label": label,
                            "raw_output": raw,
                            "order_dependency_graph": dep_graph,
                            "order_dependency_raw_output": dep_raw,
                            "order_dependency_user_prompt": dep_user_prompt,
                            "order_compliance": comp,
                            "order_compliance_raw_output": comp_raw,
                            "order_compliance_user_prompt": comp_user_prompt,
                            "final_judge_user_prompt": judge_user_prompt,
                        }
                    )

                summary["label_counts"][label] = summary["label_counts"].get(label, 0) + 1
            except Exception as e:
                summary["errors"] += 1
                obj["error"] = str(e)

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
