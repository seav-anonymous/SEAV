from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run JADES / baselines on a dataset.")
    p.add_argument(
        "--method",
        required=False,
        choices=(
            "jades",
            "jades_web",
            "judge_no_web",
            "judge_web",
            "judge_orderdep",
            "harmbench",
            "harmbench_official",
            "strongreject_pkg",
            "jailbreakbench",
            "harmscore",
            "jailjudge",
            "seal",
        ),
        help="Which method to run.",
    )
    p.add_argument(
        "--dataset",
        required=False,
        choices=("jailbreakqr", "gptfuzz", "jbb_behaviors", "labeling_xlsx", "mcq_paired", "order"),
        help="Which dataset to evaluate.",
    )
    p.add_argument(
        "--analyze-run-dir",
        default=None,
        help="Analyze an existing run directory (generate plots/tables) without calling any models.",
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            "Model to use. Default depends on method (JADES: gpt-4o-2024-08-06; judge: gpt-5.2-2025-12-11; SEAL: gemini-3-flash-preview). "
            "For SEAL, gpt-4o-2024-08-06 is also supported via OpenRouter (pass as bare name or openai/gpt-4o-2024-08-06). "
            "Local HF models are supported for judge_no_web via Transformers (e.g., openai/gpt-oss-20b or alias gpt-oss-20b)."
        ),
    )
    p.add_argument("--limit", type=int, default=None, help="Only run first N rows.")
    p.add_argument("--dry-run", action="store_true", help="Only load dataset and print counts.")
    p.add_argument(
        "--out-root",
        default="runs",
        help="Root directory for outputs (default: runs).",
    )
    p.add_argument(
        "--resume-run-dir",
        default=None,
        help="Resume a previous run from this directory. Skips rows already in results.jsonl and appends new results.",
    )
    p.add_argument(
        "--run-name-suffix",
        default="",
        help="Optional suffix appended to the run directory name (useful to distinguish dataset variants).",
    )

    p.add_argument("--jailbreakqr-path", default="datasets/jailbreakqr/jailbreakqr.json")
    p.add_argument(
        "--jailbreakqr-exclude-partial",
        action="store_true",
        help="Exclude rows with human label 'partial' from jailbreakqr dataset (400 -> 262 rows).",
    )
    p.add_argument("--gptfuzz-csv", default="datasets/GPTFuzz/all_labeled_gpt-3.5-turbo.csv")
    p.add_argument("--gptfuzz-question-list", default="datasets/GPTFuzz/question_list.csv")
    p.add_argument("--xlsx-path", default="datasets/labeling_subsets/labeling_subsets_subset25.xlsx")
    p.add_argument("--xlsx-subset", default="all", help="Sheet name to load (or 'all').")
    p.add_argument("--mcq-paired-path", default="", help="Path to paired.jsonl for dataset=mcq_paired.")
    p.add_argument("--order-variant", default="original", choices=("original", "alt_correct", "wrong_order"))
    p.add_argument("--order-jsonl-path", default="datasets/how2_dep_labels/order_variants_with_responses.jsonl")

    p.add_argument("--tavily-api-key", default=os.environ.get("TAVILY_API_KEY"))
    p.add_argument(
        "--fact-check-source",
        choices=("wikipedia", "full_web", "custom"),
        default="wikipedia",
        help=(
            "Source scope for JADES fact-checking extension (jades_web). "
            "'wikipedia' restricts Tavily queries to Wikipedia; "
            "'full_web' searches the open web."
        ),
    )
    p.add_argument("--jades-temperature", type=float, default=0.0, help="JADES temperature (default: 0.0).")
    p.add_argument(
        "--jades-variant",
        choices=("standard", "answer_only"),
        default="standard",
        help="JADES variant (default: standard).",
    )
    p.add_argument(
        "--disable-clean-node",
        action="store_true",
        help="Disable JADES clean node so downstream nodes see the full response.",
    )
    p.add_argument(
        "--jades-bias-unknown-facts-as-wrong",
        action="store_true",
        help=(
            "In JADES scoring, bias toward treating fact-check verdicts of 'Unknown' as false/unsupported (penalize). "
            "Only affects prompts when fact-checking is enabled (jades_web)."
        ),
    )

    p.add_argument(
        "--thinking-budget",
        type=int,
        default=10000,
        help="SEAL thinking budget for Gemini 2.5 models (default: 10000). Gemini 3 uses API default (HIGH) and ignores this.",
    )
    p.add_argument(
        "--no-node-logs",
        action="store_true",
        help="Disable per-node logging for SEAL (default: logging enabled). When disabled, node_logs.jsonl is not created.",
    )
    p.add_argument(
        "--seal-sample-a-n",
        type=int,
        default=None,
        help="(SEAL + mcq_paired only) Sample exactly N A-side rows deterministically. Requires --dataset mcq_paired.",
    )
    p.add_argument(
        "--seal-sample-seed",
        type=int,
        default=42,
        help="(SEAL + mcq_paired only) Random seed for deterministic A-side sampling (default: 42).",
    )
    p.add_argument(
        "--seal-search-provider",
        type=str,
        default=None,
        choices=["tavily"],
        help=(
            "Search provider for SEAL verification nodes. "
            "Default: use model's built-in search (Gemini grounding). "
            "tavily: use Tavily web search + inject results into prompt."
        ),
    )

    p.add_argument("--judge-max-output-tokens", type=int, default=8192)
    p.add_argument("--judge-temperature", type=float, default=0.0)
    p.add_argument(
        "--judge-reasoning-effort",
        choices=("default", "none", "minimal", "low", "medium", "high", "xhigh"),
        default="default",
        help="Reasoning strength for judge baselines (Responses: reasoning.effort; Chat: reasoning_effort).",
    )
    p.add_argument(
        "--judge-cot",
        action="store_true",
        help=(
            "Enable EXTERNAL chain-of-thought prompting for judge_no_web, but only for non-reasoning models "
            "(currently: gpt-4.1*). This changes the judge output format to include a THOUGHT section."
        ),
    )
    p.add_argument(
        "--judge-check-factuality-order",
        action="store_true",
        default=True,
        help=(
            "When judging jailbreak success, also check response factuality and (if it provides steps) whether the "
            "step order is correct. If it is materially incorrect or the order is wrong such that it would not work, "
            "label NOT. (Default: enabled)"
        ),
    )
    p.add_argument(
        "--no-judge-check-factuality-order",
        dest="judge_check_factuality_order",
        action="store_false",
        help="Disable factuality/order checking in judge prompt.",
    )
    p.add_argument(
        "--openrouter-provider",
        type=str,
        default="Azure",
        help="OpenRouter provider to use (e.g. 'Azure', 'OpenAI'). Default: Azure.",
    )
    p.add_argument(
        "--orderdep-dependency-prompt",
        choices=("default", "conservative"),
        default="conservative",
        help=(
            "Prompt variant for judge_orderdep dependency inference. "
            "'conservative' adds edges only for strict prerequisites and allows {} for order-invariant procedures."
        ),
    )

    p.add_argument(
        "--harmbench-official-batch-size",
        type=int,
        default=8,
        help="Batch size for --method harmbench_official (default: 8).",
    )
    p.add_argument(
        "--harmbench-official-max-new-tokens",
        type=int,
        default=1,
        help="Max new tokens for --method harmbench_official generation (default: 1).",
    )
    p.add_argument(
        "--harmbench-official-max-input-tokens",
        type=int,
        default=None,
        help=(
            "Optional input truncation length for --method harmbench_official. "
            "Defaults to (model max_position_embeddings - max_new_tokens)."
        ),
    )

    p.add_argument(
        "--strongreject-pkg-evaluator",
        default="strongreject_rubric",
        help="Evaluator name from the strong_reject package (default: strongreject_rubric).",
    )
    p.add_argument(
        "--strongreject-pkg-models",
        default="",
        help=(
            "Comma-separated model list for strong_reject (e.g., 'openai/gpt-4o-mini,openai/gpt-3.5-turbo'). "
            "If empty, strong_reject uses its package defaults."
        ),
    )
    p.add_argument(
        "--strongreject-pkg-temperature",
        type=float,
        default=0.0,
        help="Temperature passed to strong_reject generate() (default: 0.0).",
    )

    p.add_argument(
        "--gpu-device",
        type=int,
        default=None,
        help=(
            "GPU device index for local HF models (e.g., 3 for cuda:3). "
            "If not specified, uses 'auto' device mapping. "
            "Can also be set via JADES_HF_GPU_DEVICE env var."
        ),
    )

    return p.parse_args()
