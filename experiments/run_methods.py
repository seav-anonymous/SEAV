#!/usr/bin/env python3
"""
Unified runner for JADES and simple LLM-judge baselines.

Methods:
- jades: JADES without fact-checking (no web-search extension)
- jades_web: JADES with fact-checking extension (Tavily Wikipedia search)
- judge_no_web: LLM judge baseline without tools
- judge_web: LLM judge baseline with forced web_search tool (Responses API)
- judge_orderdep: LLM judge with inferred step ordering dependencies + order-compliance check (no tools)
- harmbench: HarmBench prompt-based classifier baseline (no tools)
- harmbench_official: HarmBench official open-weight classifier (local HF; e.g., cais/HarmBench-Llama-2-13b-cls)
- strongreject_pkg: StrongREJECT official package evaluator (defaults not tied to this repo's model settings)
- seal: SEAL (ours) multi-step verification pipeline (step extraction -> verification -> order check -> final judgment)

Datasets:
- jailbreakqr: local JSON (datasets/jailbreakqr.json)
- gptfuzz: local CSV (datasets/GPTFuzz/*.csv)
- jbb_behaviors: HF JailbreakBench/JBB-Behaviors judge_comparison
- labeling_xlsx: combined small subset XLSX (datasets/labeling_subsets/labeling_subsets_subset25.xlsx)
- mcq_paired: paired A/C JSONL from dishonesty/mcq_evaluation (sample_mcq_A_C_one_per_query.py output)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import List, Optional

try:
    from baselines.harmbench_official import DEFAULT_HARMBENCH_OFFICIAL_MODEL_ID as _DEFAULT_HARMBENCH_OFFICIAL_MODEL_ID
except Exception:
    _DEFAULT_HARMBENCH_OFFICIAL_MODEL_ID = "cais/HarmBench-Llama-2-13b-cls"

from baselines.hf_local import is_local_hf_model, set_hf_gpu_device
from seav.config import DEFAULT_MODEL as SEAL_DEFAULT_MODEL, DEFAULT_MAX_OUTPUT_TOKENS
from experiments.methods.harmbench_method import run_harmbench
from experiments.methods.harmbench_official_method import run_harmbench_official
from experiments.methods.harmscore_method import run_harmscore
from experiments.methods.jades_method import run_jades
from experiments.methods.jailbreakbench_method import run_jailbreakbench
from experiments.methods.jailjudge_method import run_jailjudge
from experiments.methods.judge_method import run_judge
from experiments.methods.judge_orderdep_method import run_judge_orderdep
from experiments.methods.seal_method import run_seal
from experiments.methods.strongreject_method import run_strongreject_pkg
from experiments.run_methods_common import (
    load_rows,
    now_stamp,
    redact_run_args,
    slugify_run_name_suffix,
)
from experiments.run_methods_args import parse_args
from experiments.run_postprocess import postprocess_run_dir


def main() -> None:
    args = parse_args()

    if args.analyze_run_dir:
        run_dir = Path(args.analyze_run_dir)
        run_args_path = run_dir / "run_args.json"
        method = args.method
        dataset = args.dataset
        if run_args_path.exists():
            try:
                run_args = json.loads(run_args_path.read_text(encoding="utf-8"))
                method = method or run_args.get("method")
                dataset = dataset or run_args.get("dataset")
            except Exception:
                pass
        if not method or not dataset:
            raise SystemExit("For --analyze-run-dir, supply --method and --dataset (or ensure run_args.json exists).")
        postprocess_run_dir(run_dir=run_dir, method=method, dataset=dataset)
        print(f"Wrote figures/tables to {run_dir}")
        return

    if not args.method or not args.dataset:
        raise SystemExit("Missing required args: --method and --dataset")

    # Configure GPU device for local HF models (if specified)
    if args.gpu_device is not None:
        set_hf_gpu_device(args.gpu_device)
        print(f"[config] Using GPU device cuda:{args.gpu_device} for local HF models")

    if args.method == "jades_web":
        # Prefer Tavily when configured, but allow a no-key fallback (Wikipedia API) so
        # jades_web can still run in minimal environments.
        if args.fact_check_source == "full_web" and not args.tavily_api_key:
            raise SystemExit(
                "--method jades_web with --fact-check-source full_web requires a Tavily API key. "
                "Set `TAVILY_API_KEY` or pass `--tavily-api-key`."
            )
        if args.tavily_api_key and importlib.util.find_spec("tavily") is None:
            raise SystemExit(
                "Missing dependency for --method jades_web with Tavily enabled: `tavily-python`.\n"
                "Install inside your env:\n"
                "  pip install -U tavily-python"
            )

    model: str | None = args.model
    if not model:
        if args.method == "strongreject_pkg":
            model = "pkg_default"
        elif args.method == "harmbench_official":
            model = _DEFAULT_HARMBENCH_OFFICIAL_MODEL_ID
        elif args.method == "jailbreakbench":
            model = None
        elif args.method == "seal":
            model = SEAL_DEFAULT_MODEL
        else:
            model = "gpt-4o-2024-08-06" if args.method.startswith("jades") else "gpt-5.2-2025-12-11"

    reasoning_effort: Optional[str] = None if args.judge_reasoning_effort == "default" else args.judge_reasoning_effort

    # Opus 4.5: default to the strongest reasoning setting (unless explicitly overridden).
    m = str(model or "")
    is_opus_45 = m.strip().lower() in {"opus 4.5", "opus-4.5", "opus_4.5"}
    if args.judge_reasoning_effort == "default" and args.method in {"judge_no_web", "judge_web"} and is_opus_45:
        reasoning_effort = "xhigh"

    # Gemini 3 Pro: default to the strongest thinking level (HIGH via xhigh mapping).
    is_gemini_3_pro = m.strip().lower() in {
        "gemini-3-pro-preview", "gemini3-pro-preview", "gemini-3-pro", "gemini3-pro", "gemini 3 pro", "gemini3pro"
    }
    if args.judge_reasoning_effort == "default" and args.method in {"judge_no_web", "judge_web"} and is_gemini_3_pro:
        reasoning_effort = "xhigh"

    # GPT-5.2 via OpenRouter: default to xhigh reasoning effort.
    is_gpt52 = m.strip().lower() in {"gpt-5.2", "gpt5.2", "gpt 5.2", "openai/gpt-5.2"}
    if args.judge_reasoning_effort == "default" and args.method in {"judge_no_web", "judge_web"} and is_gpt52:
        reasoning_effort = "xhigh"

    # Kimi K2.5 via OpenRouter: default to xhigh reasoning effort.
    is_kimi_k25 = m.strip().lower() in {"kimi-k2.5", "kimi k2.5", "kimik2.5", "moonshotai/kimi-k2.5"}
    if args.judge_reasoning_effort == "default" and args.method in {"judge_no_web", "judge_web"} and is_kimi_k25:
        reasoning_effort = "xhigh"

    # gpt-4.1 is a non-reasoning model family; ignore reasoning control.
    is_haiku = m.startswith("claude-3-haiku")
    # Gemini 2.5 Pro doesn't support thinking, uses CoT prompting like gpt-4.1.
    # Gemini models with NO thinking support (2.0 Flash, 1.5/1.0 Pro), uses CoT prompting like gpt-4.1.
    is_gemini_no_thinking = m.strip().lower() in {
        "gemini-2.0-flash", "gemini2.0-flash", "gemini 2.0 flash",
        "gemini-pro", "gemini-pro-latest", "gemini-1.5-pro", "gemini-1.0-pro", "gemini 1.5 pro", "gemini 1.0 pro", "gemini pro"
    }
    if reasoning_effort and (m.startswith("gpt-4.1") or is_haiku or is_gemini_no_thinking):
        reasoning_effort = None
    # Local HF models don't support API-level reasoning knobs.
    if reasoning_effort and is_local_hf_model(m):
        reasoning_effort = None

    # Claude Haiku: only supported for judge_no_web, and uses external-CoT prompting by default.
    # Gemini no-thinking models: uses external-CoT prompting like gpt-4.1 (NO thinking support at all).
    cot_prompting = bool(args.judge_cot) or is_haiku or is_gemini_no_thinking
    if is_haiku and args.method != "judge_no_web":
        raise SystemExit("claude-3-haiku-20240307 is only supported for --method judge_no_web")

    # External-CoT can be verbose; bump max tokens for gpt-4.1/haiku/gemini-no-thinking if needed to avoid truncation.
    if args.method == "judge_no_web" and cot_prompting and (m.startswith("gpt-4.1") or is_haiku or is_gemini_no_thinking):
        if args.judge_max_output_tokens < 4096:
            print(
                f"[config] Bumping --judge-max-output-tokens from {args.judge_max_output_tokens} to 4096 "
                "for external CoT"
            )
            args.judge_max_output_tokens = 4096

    # Kimi K2.5: reasoning tokens may eat into completion budget; bump to 8192 to avoid truncation.
    if args.method == "judge_no_web" and is_kimi_k25 and args.judge_max_output_tokens < 8192:
        print(
            f"[config] Bumping --judge-max-output-tokens from {args.judge_max_output_tokens} to 8192 "
            "for Kimi K2.5 (reasoning may consume completion budget)"
        )
        args.judge_max_output_tokens = 8192

    rows = load_rows(args)

    if getattr(args, "jailbreakqr_exclude_partial", False) and args.dataset == "jailbreakqr":
        before = len(rows)
        rows = [r for r in rows if r.meta.get("human") != "partial"]
        print(f"[FILTER] Excluded 'partial' label: {before} -> {len(rows)} rows")

    if args.limit is not None:
        rows = rows[: args.limit]

    # SEAL A-side deterministic sampling (mcq_paired only)
    seal_sample_a_n = getattr(args, "seal_sample_a_n", None)
    seal_sample_seed = getattr(args, "seal_sample_seed", 42)
    if seal_sample_a_n is not None:
        if args.method != "seal":
            raise SystemExit("--seal-sample-a-n requires --method seal")
        if args.dataset != "mcq_paired":
            raise SystemExit("--seal-sample-a-n requires --dataset mcq_paired")
        
        # Filter to A-side only
        a_rows = [r for r in rows if r.meta.get("paired_choice") == "A"]
        if len(a_rows) < seal_sample_a_n:
            raise SystemExit(
                f"Requested {seal_sample_a_n} A-side samples but only {len(a_rows)} available"
            )
        
        # Deterministic sampling using local RNG
        import random
        rng = random.Random(seal_sample_seed)
        selected_rows = rng.sample(a_rows, seal_sample_a_n)
        
        # Print selected row_ids for traceability
        selected_ids = [r.row_id for r in selected_rows]
        print(f"[SEAL A-side sampling] Selected {len(selected_ids)} rows with seed={seal_sample_seed}")
        print(f"[SEAL A-side sampling] Selected row_ids: {selected_ids}")
        
        # Store sampling metadata in args for run_args.json
        args.seal_sample_a_n_actual = seal_sample_a_n
        args.seal_sample_seed_actual = seal_sample_seed
        args.seal_sample_selected_ids = selected_ids
        
        rows = selected_rows

    print(f"Loaded {len(rows)} rows (dataset={args.dataset})")
    if args.dry_run:
        return

    suffix = ""
    if args.method in ("jades", "jades_web") and args.disable_clean_node:
        suffix = "_noclean"

    fact_suffix = ""
    if args.method == "jades_web":
        src = str(getattr(args, "fact_check_source", "wikipedia") or "wikipedia")
        src_map = {
            "wikipedia": "wiki",
            "full_web": "fullweb",
            "custom": "custom",
        }
        src_slug = src_map.get(src, slugify_run_name_suffix(src))
        if src_slug:
            fact_suffix = f"_{src_slug}"

    run_name_suffix = ""
    if args.run_name_suffix:
        slug = slugify_run_name_suffix(args.run_name_suffix)
        if slug:
            run_name_suffix = f"_{slug}"

    model_dir = slugify_run_name_suffix(model or "")
    if not model_dir:
        model_dir = "model"

    # Auto-select dataset-specific output root if user didn't override --out-root
    effective_out_root = args.out_root
    if args.out_root == "runs":  # default value, auto-select based on dataset
        dataset_out_roots = {
            "jailbreakqr": "results_jailbreakqr_runs",
            "mcq_paired": "results_mcq_paired_runs",
            "jbb_behaviors": "results_jbb_behaviors_runs",
        }
        effective_out_root = dataset_out_roots.get(args.dataset, "runs")

    if args.resume_run_dir:
        out_dir = Path(args.resume_run_dir)
        if not out_dir.exists():
            raise SystemExit(f"--resume-run-dir does not exist: {out_dir}")
        print(f"Resuming from {out_dir}")
    else:
        out_dir = Path(effective_out_root) / f"{now_stamp()}_{args.method}_{args.dataset}_{model_dir}{suffix}{fact_suffix}{run_name_suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)

    if not args.resume_run_dir:
        (out_dir / "run_args.json").write_text(
            json.dumps(redact_run_args({k: v for k, v in vars(args).items()}), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def require_model_str(value: str | None, method_name: str) -> str:
        if value is None:
            raise SystemExit(f"Method {method_name} requires a concrete model string")
        return value

    if args.method == "jades":
        run_jades(
            rows=rows,
            model=require_model_str(model, "jades"),
            temperature=args.jades_temperature,
            jades_variant=args.jades_variant,
            disable_clean_node=bool(args.disable_clean_node),
            enable_fact_checking=False,
            fact_check_source=args.fact_check_source,
            tavily_api_key=None,
            bias_unknown_facts_as_wrong=bool(args.jades_bias_unknown_facts_as_wrong),
            out_dir=out_dir,
        )
    elif args.method == "jades_web":
        run_jades(
            rows=rows,
            model=require_model_str(model, "jades_web"),
            temperature=args.jades_temperature,
            jades_variant=args.jades_variant,
            disable_clean_node=bool(args.disable_clean_node),
            enable_fact_checking=True,
            fact_check_source=args.fact_check_source,
            tavily_api_key=args.tavily_api_key,
            bias_unknown_facts_as_wrong=bool(args.jades_bias_unknown_facts_as_wrong),
            out_dir=out_dir,
        )
    elif args.method == "judge_no_web":
        run_judge(
            rows=rows,
            model=require_model_str(model, "judge_no_web"),
            force_web=False,
            max_output_tokens=args.judge_max_output_tokens,
            temperature=args.judge_temperature,
            reasoning_effort=reasoning_effort,
            cot_prompting=cot_prompting,
            check_factuality_order=bool(args.judge_check_factuality_order),
            out_dir=out_dir,
            openrouter_provider=getattr(args, "openrouter_provider", None),
        )
    elif args.method == "judge_web":
        run_judge(
            rows=rows,
            model=require_model_str(model, "judge_web"),
            force_web=True,
            max_output_tokens=args.judge_max_output_tokens,
            temperature=args.judge_temperature,
            reasoning_effort=reasoning_effort,
            cot_prompting=False,
            check_factuality_order=bool(args.judge_check_factuality_order),
            out_dir=out_dir,
            openrouter_provider=getattr(args, "openrouter_provider", None),
        )
    elif args.method == "harmbench":
        run_harmbench(
            rows=rows,
            model=require_model_str(model, "harmbench"),
            max_output_tokens=args.judge_max_output_tokens,
            temperature=args.judge_temperature,
            reasoning_effort=reasoning_effort,
            out_dir=out_dir,
        )
    elif args.method == "harmbench_official":
        run_harmbench_official(
            rows=rows,
            model_id=require_model_str(model, "harmbench_official"),
            batch_size=args.harmbench_official_batch_size,
            max_new_tokens=args.harmbench_official_max_new_tokens,
            max_input_tokens=args.harmbench_official_max_input_tokens,
            out_dir=out_dir,
        )
    elif args.method == "judge_orderdep":
        run_judge_orderdep(
            rows=rows,
            model=require_model_str(model, "judge_orderdep"),
            max_output_tokens=args.judge_max_output_tokens,
            temperature=args.judge_temperature,
            reasoning_effort=reasoning_effort,
            orderdep_dependency_prompt=args.orderdep_dependency_prompt,
            out_dir=out_dir,
        )
    elif args.method == "strongreject_pkg":
        models: List[str] | None = None
        if args.strongreject_pkg_models.strip():
            models = [m.strip() for m in args.strongreject_pkg_models.split(",") if m.strip()]
        elif args.model and args.model != "pkg_default":
            models = [args.model]

        run_strongreject_pkg(
            rows=rows,
            evaluator=args.strongreject_pkg_evaluator,
            models=models,
            temperature=args.strongreject_pkg_temperature,
            max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            out_dir=out_dir,
        )
    elif args.method == "jailbreakbench":
        run_jailbreakbench(
            rows=rows,
            api_key=None,  # Will be resolved inside
            model_name=model,
            out_dir=out_dir,
        )
    elif args.method == "harmscore":
        run_harmscore(
            rows=rows,
            out_dir=out_dir,
        )
    elif args.method == "jailjudge":
        # Default model if not specified
        model_id = args.model or "usail-hkust/JailJudge-guard"
        run_jailjudge(
            rows=rows,
            model_id=model_id,
            out_dir=out_dir,
        )

    elif args.method == "seal":
        run_seal(
            rows=rows,
            model=require_model_str(model, "seal"),
            thinking_budget=args.thinking_budget,
            out_dir=out_dir,
            enable_node_logs=not args.no_node_logs,
            search_provider=getattr(args, "seal_search_provider", None),
        )

    else:
        raise SystemExit(f"Unknown method: {args.method}")

    postprocess_run_dir(run_dir=out_dir, method=args.method, dataset=args.dataset)
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
