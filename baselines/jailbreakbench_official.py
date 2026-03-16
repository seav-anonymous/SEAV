"""
Official JailbreakBench evaluation wrapper.
Uses the `jailbreakbench` python package.
"""

import os
from typing import List, Optional, Tuple, Any

try:
    from jailbreakbench.classifier import Llama3JailbreakJudge
    import litellm
    litellm.suppress_debug_info = True
except ImportError:
    Llama3JailbreakJudge = None  # type: ignore

def is_jailbreakbench_available() -> bool:
    return Llama3JailbreakJudge is not None

if Llama3JailbreakJudge is not None:
    class CustomModelJailbreakJudge(Llama3JailbreakJudge):
        """
        Subclass of Llama3JailbreakJudge that allows overriding the model name.
        """

        def __init__(self, api_key: str, model_name: Optional[str] = None):
            # We need to initialize the parent.
            # The parent __init__ just sets self._api_key.
            super().__init__(api_key)
            if model_name:
                self.LITELLM_MODEL = model_name
else:  # pragma: no cover
    CustomModelJailbreakJudge = None  # type: ignore[assignment]

def jailbreakbench_evaluate_batch(
    *,
    prompts: List[str],
    responses: List[str],
    api_key: str,
    model_name: Optional[str] = None,
) -> List[bool]:
    """
    Run JailbreakBench evaluation on a batch of prompt/response pairs.
    Returns a list of booleans (True = jailbroken/unsafe, False = safe).
    """
    if not is_jailbreakbench_available():
        raise ImportError("jailbreakbench package not found. Install with `pip install jailbreakbench`.")
    if CustomModelJailbreakJudge is None:  # pragma: no cover
        raise ImportError("jailbreakbench dependency missing; CustomModelJailbreakJudge unavailable.")

    # Initialize judge
    # If model_name is provided, use custom subclass to override LITELLM_MODEL
    judge = CustomModelJailbreakJudge(api_key=api_key, model_name=model_name)

    # The judge call is judge(prompts, responses)
    # It returns a list of bools.
    results = judge(prompts, responses)
    return results
