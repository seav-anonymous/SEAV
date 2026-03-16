"""
Official HarmScore evaluation wrapper.
Uses the implementation from SpeakEasy_repo.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add SpeakEasy_repo to path so we can import modules from it
SPEAKEASY_ROOT = Path(__file__).resolve().parent.parent / "SpeakEasy_repo"
if str(SPEAKEASY_ROOT) not in sys.path:
    sys.path.append(str(SPEAKEASY_ROOT))

# We need to make sure 'utils' can be imported by harmscore
# The harmscore init imports 'utils' directly: `from utils import contains_refusal_words`
# This likely expects SpeakEasy_repo to be the CWD or in python path.
# We added SpeakEasy_repo to path, so `import utils` might work if utils.py is in SpeakEasy_repo root.

try:
    from eval_models.harmscore import HarmScore
except ImportError:
    HarmScore = None

def is_harmscore_available() -> bool:
    return HarmScore is not None

def harmscore_evaluate_batch(
    *,
    prompts: List[str],
    responses: List[str],
) -> List[Dict[str, Any]]:
    """
    Run HarmScore evaluation on a batch of prompt/response pairs.
    Returns a list of dicts with keys: score, actionable_score, informative_score.
    """
    if not is_harmscore_available():
        raise ImportError("HarmScore not found. Ensure SpeakEasy_repo is cloned.")

    evaluator = HarmScore()
    
    qa_pairs = [
        {"query": q, "response": r} 
        for q, r in zip(prompts, responses)
    ]
    
    # compute_scores modifies the list in-place
    results = evaluator.compute_scores(qa_pairs)
    
    return results
