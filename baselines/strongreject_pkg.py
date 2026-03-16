"""
Wrapper for running the official `strong_reject` package evaluators.

Use this when you want StrongREJECT's *package defaults* (e.g., its default judge
model fallback list) rather than the local reimplementation in
`baselines/strongreject.py`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def strongreject_pkg_evaluate(
    forbidden_prompt: str,
    response: str,
    *,
    evaluator: str = "strongreject_rubric",
    models: Optional[List[str]] = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run a StrongREJECT evaluator via the `strong_reject` package.

    If `models` is None, StrongREJECT uses its internal default model list.
    """
    try:
        from strong_reject.evaluate import evaluate as sr_evaluate  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency `strong_reject` (and its backend, typically `litellm`).\n"
            "Install inside your env:\n"
            "  pip install -U \"git+https://github.com/dsbowen/strong_reject.git\" litellm\n"
            f"Original import error: {e}"
        )

    outputs = sr_evaluate(
        forbidden_prompt,
        response,
        [evaluator],
        models=models,
        temperature=temperature,
        **kwargs,
    )
    out = outputs[0] if outputs else {}

    # Normalize NaN score to None (JSON friendly).
    score = out.get("score")
    if isinstance(score, float) and math.isnan(score):
        out["score"] = None

    return out
