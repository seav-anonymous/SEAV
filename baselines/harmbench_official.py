"""baselines.harmbench_official

Run the HarmBench *official* open-weight classifier locally via Hugging Face.

This corresponds to the common HarmBench setup where a dedicated classifier
checkpoint (e.g. `cais/HarmBench-Llama-2-13b-cls`) is used with HarmBench's
public evaluation prompt template.

Source:
  https://github.com/centerforaisafety/HarmBench
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

from baselines.harmbench import HARM_BENCH_CLS_PROMPT, _normalize_yes_no


DEFAULT_HARMBENCH_OFFICIAL_MODEL_ID = "cais/HarmBench-Llama-2-13b-cls"


@lru_cache(maxsize=2)
def _load_harmbench_classifier(model_id: str) -> tuple[LlamaTokenizer, AutoModelForCausalLM, int]:
    # Use LlamaTokenizer explicitly. In some envs, AutoTokenizer may load an
    # incompatible fast tokenizer for this checkpoint.
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    # Avoid using <unk> as padding; eos padding + attention_mask is standard for Llama.
    tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto")
    model.eval()

    max_pos = getattr(model.config, "max_position_embeddings", None)
    if not isinstance(max_pos, int) or max_pos <= 0:
        # Conservative fallback.
        max_pos = 2048

    return tokenizer, model, max_pos


def harmbench_official_classify_batch(
    *,
    model_id: str,
    behaviors: List[str],
    generations: List[str],
    max_new_tokens: int = 1,
    max_input_tokens: Optional[int] = None,
) -> Tuple[List[Optional[int]], List[str]]:
    """Classify a batch of (behavior, generation) pairs.

    Returns:
      - label_int: 1 (yes / harmful), 0 (no), or None (unparseable)
      - raw_text: decoded model output (new tokens only)
    """
    if len(behaviors) != len(generations):
        raise ValueError(f"behaviors/generations length mismatch: {len(behaviors)} vs {len(generations)}")
    if not behaviors:
        return [], []

    tokenizer, model, max_pos = _load_harmbench_classifier(model_id)

    max_new_tokens_i = int(max_new_tokens)
    if max_new_tokens_i <= 0:
        raise ValueError(f"max_new_tokens must be >= 1 (got {max_new_tokens})")

    if max_input_tokens is None:
        # Ensure there is room to generate at least max_new_tokens.
        max_input = max(1, int(max_pos) - max_new_tokens_i)
    else:
        max_input = int(max_input_tokens)
        max_input = max(1, min(max_input, int(max_pos) - max_new_tokens_i))

    prompts = [
        HARM_BENCH_CLS_PROMPT.format(behavior=str(b or ""), generation=str(g or ""))
        for b, g in zip(behaviors, generations)
    ]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens_i,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[:, input_ids.shape[1] :]
    raw_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    labels: List[Optional[int]] = []
    for raw in raw_texts:
        yn = _normalize_yes_no(raw)
        if yn == "YES":
            labels.append(1)
        elif yn == "NO":
            labels.append(0)
        else:
            labels.append(None)

    return labels, raw_texts
