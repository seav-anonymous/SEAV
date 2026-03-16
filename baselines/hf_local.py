"""JADES local Hugging Face (Transformers) backend.

This module is used by the judge baselines to run open-weight models locally (GPU)
via Transformers.

We primarily support the OpenAI open-weight series on the HF Hub, e.g.:
- `openai/gpt-oss-20b`
- `gpt-oss-20b` (convenience alias -> `openai/gpt-oss-20b`)

Environment variables:
- JADES_HF_GPU_DEVICE: Specify GPU device index (e.g., "3" for cuda:3). Default: auto.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

# Global GPU device setting (can be set via set_hf_gpu_device)
_HF_GPU_DEVICE: Optional[int] = None


def set_hf_gpu_device(device_idx: Optional[int]) -> None:
    """Set the GPU device index for local HF model loading."""
    global _HF_GPU_DEVICE
    _HF_GPU_DEVICE = device_idx


def get_hf_gpu_device() -> Optional[int]:
    """Get the configured GPU device index."""
    global _HF_GPU_DEVICE
    if _HF_GPU_DEVICE is not None:
        return _HF_GPU_DEVICE
    env_val = os.environ.get("JADES_HF_GPU_DEVICE", "").strip()
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            pass
    return None


def is_local_hf_model(model: str) -> bool:
    """
    Return True if `model` should be executed locally via Hugging Face Transformers.

    Supported:
      - "hf:<repo_id>" (generic escape hatch)
      - "openai/gpt-oss-*" (OpenAI open-weight series on the HF Hub)
      - "gpt-oss-*" (alias for "openai/gpt-oss-*")
    """
    m = (model or "").strip()
    if not m:
        return False
    if m.startswith("hf:"):
        return True
    low = m.lower()
    if low.startswith("openai/gpt-oss-"):
        return True
    if low.startswith("gpt-oss-"):
        return True
    return False


def resolve_hf_model_id(model: str) -> str:
    m = (model or "").strip()
    if m.startswith("hf:"):
        return m.split("hf:", 1)[1].strip()
    low = m.lower()
    if low.startswith("openai/gpt-oss-"):
        # Canonicalize the org name for robustness.
        return "openai/" + low.split("/", 1)[1]
    if low.startswith("gpt-oss-"):
        # Convenience alias.
        return "openai/" + low
    return m


@lru_cache(maxsize=2)
def _load_hf_chat_model(model_id: str, gpu_device: Optional[int] = None) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Determine device mapping
    if gpu_device is not None:
        device_map = {"": f"cuda:{gpu_device}"}
    else:
        device_map = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()
    return tokenizer, model


def hf_chat_generate(
    *,
    model_id: str,
    messages: List[Dict[str, str]],
    max_output_tokens: int,
    temperature: float,
) -> str:
    """
    Generate text from a local HF chat model.
    """
    import torch

    gpu_device = get_hf_gpu_device()
    tokenizer, model = _load_hf_chat_model(model_id, gpu_device)

    # apply_chat_template may return a tensor or BatchEncoding depending on tokenizer version
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    
    # Handle both tensor and BatchEncoding returns
    if hasattr(encoded, "input_ids"):
        # BatchEncoding object
        input_ids = encoded.input_ids.to(model.device)
        attention_mask = encoded.attention_mask.to(model.device) if hasattr(encoded, "attention_mask") else None
    else:
        # Plain tensor
        input_ids = encoded.to(model.device)
        attention_mask = None
    
    # Create attention mask if not provided
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=model.device)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_output_tokens),
        "do_sample": float(temperature) > 0.0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if float(temperature) > 0.0:
        gen_kwargs["temperature"] = float(temperature)

    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

    text = tokenizer.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True)
    return text or ""
