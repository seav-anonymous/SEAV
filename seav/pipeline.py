"""
Parallel verification pipeline for OURS jailbreak evaluation.

Runs all step verifications (Node 2) and order verification (Node 3) in a
single ThreadPoolExecutor so that N steps + 1 order check = N+1 concurrent tasks.

Usage:
    from seav.pipeline import run_verification_parallel

    sv_result, ov_result = run_verification_parallel(
        step_node=step_node,
        order_node=order_node,
        steps=extracted_steps,
        intent=intent,
    )
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict, List, Optional, Tuple

from seav.nodes.step_extraction import ExtractedStep
from seav.nodes.step_verification import (
    AllStepsVerificationResult,
    StepVerificationNode,
    StepVerificationResult,
)
from seav.nodes.order_verification import (
    OrderVerificationNode,
    OrderVerificationResult,
)


def run_verification_parallel(
    *,
    step_node: StepVerificationNode,
    order_node: OrderVerificationNode,
    steps: List[ExtractedStep],
    intent: str,
    skip_order_verification: bool = False,
    skip_reason: Optional[str] = None,
    max_workers: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[AllStepsVerificationResult, OrderVerificationResult]:
    """Run step verification and order verification concurrently.

    For N steps this spawns N+1 tasks in a single thread pool:
      - N  tasks: ``step_node.verify_step()`` (one per step)
      - 1  task:  ``order_node.verify_order()``

    Args:
        step_node: Configured StepVerificationNode instance.
        order_node: Configured OrderVerificationNode instance.
        steps: Extracted steps from Node 1.
        intent: The harmful intent string.
        skip_order_verification: If True, skip order verification.
            Use when content is unordered. Defaults to False.
        skip_reason: Optional reason for skipping verification.
            Only used when skip_order_verification=True and len(steps) >= 2.
        max_workers: Thread pool size (default: N+1).
        verbose: Print progress messages.

    Returns:
        Tuple of (AllStepsVerificationResult, OrderVerificationResult).
    """
    n = len(steps)
    skip_order = n < 2 or skip_order_verification
    if max_workers is None:
        max_workers = n + 1 if not skip_order else max(1, n)

    if verbose:
        print(f"[PIPELINE] Launching {n} step verifications + 1 order verification "
              f"in parallel (max_workers={max_workers})")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Submit all step verifications
        step_futures: Dict[Future[StepVerificationResult], ExtractedStep] = {}
        for step in steps:
            fut: Future[StepVerificationResult] = pool.submit(
                step_node.verify_step, step=step, intent=intent,
            )
            step_futures[fut] = step

        # Submit order verification
        order_future: Optional[Future[OrderVerificationResult]] = None
        if not skip_order:
            order_future = pool.submit(
                order_node.verify_order, steps=steps, intent=intent,
            )

        # Collect step results (unordered as they complete)
        step_results: List[StepVerificationResult] = []
        for fut in step_futures:
            step = step_futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                if verbose:
                    print(f"[PIPELINE] Step {step.index} failed: {e}")
                result = StepVerificationResult(
                    step_index=step.index,
                    step_text=step.text,
                    is_factual=False,
                    factuality_confidence="low",
                    factuality_evidence=f"Verification failed: {str(e)[:200]}",
                    is_relevant=False,
                    relevance_score="low",
                    relevance_explanation=f"Verification failed: {str(e)[:200]}",
                    search_query="",
                    search_performed=False,
                    raw_output=str(e),
                )
            step_results.append(result)
            if verbose:
                print(f"[PIPELINE] Step {result.step_index}: {result.get_verdict()}")

        # Sort by step index to match original order
        step_results.sort(key=lambda r: r.step_index)

        # Build aggregated step verification result
        factual_count = sum(1 for r in step_results if r.is_factual)
        relevant_count = sum(1 for r in step_results if r.is_relevant)
        verified_count = sum(1 for r in step_results if r.is_factual and r.is_relevant)

        all_steps_result = AllStepsVerificationResult(
            step_results=step_results,
            total_steps=n,
            factual_steps=factual_count,
            relevant_steps=relevant_count,
            verified_steps=verified_count,
        )

        # Collect order result
        if order_future is not None:
            order_result = order_future.result()
        else:
            # n < 2: pass actual steps (verify_order early-returns, no LLM)
            # skip_order_verification=True and n >= 2: pass [] to trigger early-return + skipped traces
            sync_steps = [] if (skip_order_verification and n >= 2) else steps
            skip_reason_to_pass = None
            if skip_order_verification and n >= 2:
                skip_reason_to_pass = (
                    "Unordered structure; no sequential order to verify."
                )
            order_result = order_node.verify_order(
                steps=sync_steps,
                intent=intent,
                skip_reason=skip_reason_to_pass,
            )
        if verbose:
            print(f"[PIPELINE] Order: {order_result.verdict} "
                  f"(complies={order_result.complies}, "
                  f"violations={len(order_result.violations)})")

    return all_steps_result, order_result
