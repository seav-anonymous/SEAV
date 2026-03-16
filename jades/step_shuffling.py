"""
Shared utilities for splitting and shuffling step-structured responses.

This module is used by:
- Dataset transformation scripts (e.g., shuffle_jailbreakqr_multistep.py)
- The order-aware judge baseline (judge_orderdep)

Goal: keep the step extraction and head/tail handling consistent across pipelines.
"""

from __future__ import annotations

import random
import re
from typing import List, Tuple


# Detect step headers at the start of a line, allowing common markdown prefixes, e.g.:
#   "1. ...", "2) ...", "Step 3: ...", "4 - ..."
#   "### Step 1: ...", "**Step 2:** ...", "- 3. ..."
STEP_HEADER_RE = re.compile(r"(?im)^\s*(?:[#>\-\*\u2022]+[\t ]*)*(?:step\s*)?\d+\s*[\)\.\:\---]\s+")
STEP_HEADER_CAPTURE_RE = re.compile(
    r"(?im)^(?P<prefix>\s*(?:[#>\-\*\u2022]+[\t ]*)*(?:step\s*)?)(?P<num>\d+)(?P<sep>\s*[\)\.\:\---])"
)


# Heuristic: split a post-steps "tail" from the last step block when a clear marker starts a new paragraph.
# This prevents disclaimers/conclusions from being shuffled as if they were part of a step.
TAIL_MARKER_RE = re.compile(
    r"(?im)\n\s*\n(?=^\s*(?:[#>\-\*\u2022]+[\t ]*)*(?:"
    r"conclusion|summary|note|notes|disclaimer|warning|remember|final(?:ly)?|in conclusion|overall|"
    r"it is important|it's important|it is crucial|it's crucial|it is worth noting|it's worth noting|"
    r"as a historian|as an ai|as an assistant|as a language model|as an ai language model"
    r")\b)"
)


def split_step_blocks(text: str) -> Tuple[str, List[str], str]:
    """
    Returns (head, step_blocks, tail).

    - head: any prefix before the first step header
    - step_blocks: each block spans from its header to just before the next header (or end)
    - tail: an optional non-step suffix (e.g., disclaimer/conclusion) removed from the last block
    """
    matches = list(STEP_HEADER_RE.finditer(text or ""))
    if not matches:
        return text or "", [], ""

    head = (text or "")[: matches[0].start()]
    blocks: List[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text or "")
        blocks.append((text or "")[start:end])

    tail = ""
    if blocks:
        last = blocks[-1]
        tm = TAIL_MARKER_RE.search(last)
        if tm:
            split_at = tm.start()
            tail = last[split_at:]
            blocks[-1] = last[:split_at]

    return head, blocks, tail


def strip_step_header(block: str) -> str:
    """
    Remove the leading step header from a block (reduces bias from numbering).
    """
    m = STEP_HEADER_RE.search(block or "")
    if not m:
        return (block or "").strip()
    return (block[m.end() :] or "").strip()


def renumber_step_header(block: str, new_num: int) -> str:
    """
    Renumber only the first step header in a block.
    """

    def repl(m: re.Match[str]) -> str:
        return f"{m.group('prefix')}{new_num}{m.group('sep')}"

    return STEP_HEADER_CAPTURE_RE.sub(repl, block, count=1)


def shuffle_step_blocks(
    blocks: List[str],
    *,
    rng: random.Random | None = None,
    seed: int | None = None,
    renumber: bool = True,
) -> Tuple[List[str], List[int]]:
    """
    Shuffle step blocks, guaranteeing the order changes when possible.

    Returns (shuffled_blocks, perm), where perm maps shuffled_position -> original_index.
    """
    n = len(blocks)
    perm = list(range(n))

    if rng is None:
        rng = random.Random(0 if seed is None else seed)

    rng.shuffle(perm)
    if perm == list(range(n)) and n > 1:
        perm = perm[1:] + perm[:1]

    shuffled: List[str] = []
    for pos, orig_idx in enumerate(perm):
        block = blocks[orig_idx]
        if renumber:
            block = renumber_step_header(block, pos + 1)
        shuffled.append(block)
    return shuffled, perm


def shuffle_steps_in_response(
    response: str,
    *,
    rng: random.Random | None = None,
    seed: int | None = None,
    min_steps: int = 2,
    renumber: bool = True,
) -> Tuple[str, bool, int]:
    """
    Shuffle the step blocks within a response, preserving head and tail.

    Returns (new_response, changed, num_steps_detected).
    """
    head, blocks, tail = split_step_blocks(response)
    if len(blocks) < min_steps:
        return response, False, len(blocks)

    shuffled_blocks, perm = shuffle_step_blocks(blocks, rng=rng, seed=seed, renumber=renumber)
    changed = perm != list(range(len(blocks)))
    return head + "".join(shuffled_blocks) + tail, changed, len(blocks)

