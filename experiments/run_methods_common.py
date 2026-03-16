from __future__ import annotations

import argparse
import hashlib
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from jades.dataset_loaders import (
    DatasetRow,
    load_gptfuzz,
    load_jailbreakqr_json,
    load_jbb_behaviors_judge_comparison,
    load_labeling_subsets_xlsx,
    load_mcq_paired_jsonl,
    load_order_variants_jsonl,
)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify_run_name_suffix(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80]


def load_rows(args: argparse.Namespace) -> list[DatasetRow]:
    if args.dataset == "jailbreakqr":
        return load_jailbreakqr_json(Path(args.jailbreakqr_path))
    if args.dataset == "gptfuzz":
        return load_gptfuzz(
            all_labeled_csv=Path(args.gptfuzz_csv),
            question_list_csv=Path(args.gptfuzz_question_list),
        )
    if args.dataset == "jbb_behaviors":
        return load_jbb_behaviors_judge_comparison()
    if args.dataset == "labeling_xlsx":
        return load_labeling_subsets_xlsx(Path(args.xlsx_path), subset=args.xlsx_subset)
    if args.dataset == "mcq_paired":
        if not args.mcq_paired_path:
            raise ValueError("--mcq-paired-path is required when --dataset mcq_paired")
        return load_mcq_paired_jsonl(Path(args.mcq_paired_path))
    if args.dataset == "order":
        return load_order_variants_jsonl(
            Path(args.order_jsonl_path),
            variant=args.order_variant,
        )
    raise ValueError(f"Unknown dataset: {args.dataset}")


def ensure_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def stable_seed_int(*parts: Any) -> int:
    h = hashlib.md5()
    for p in parts:
        h.update(str(p or "").encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:4], "big", signed=False)


def redact_run_args(data: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(data)
    for key in list(redacted.keys()):
        if key.endswith("api_key") or key.endswith("_key"):
            if redacted.get(key):
                redacted[key] = "***REDACTED***"
    return redacted
