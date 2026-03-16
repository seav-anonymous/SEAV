"""
OpenRouter API utilities for OURS jailbreak evaluation method.

Calls OpenAI-compatible models (e.g. GPT-5.2) through OpenRouter,
with optional provider routing (e.g. Azure).

The API is OpenAI-chat-completions compatible, so we use raw `requests`
(consistent with the existing Anthropic integration pattern).

Usage:
    from seav.nodes.openrouter_api import (
        is_openrouter_model,
        _call_openrouter_api,
        _call_openrouter_api_no_search,
    )
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# - constants -

_OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_DEFAULT_KEY_PATH = None  # Set OPENROUTER_API_KEY env var
    None  # Set OPENROUTER_API_KEY env var
)

# Models that should be routed through OpenRouter
_OPENROUTER_MODEL_PREFIXES = (
    "openai/",
    "anthropic/",
    "google/",
    "meta-llama/",
    "mistralai/",
    "deepseek/",
    "moonshotai/",
    "z-ai/",
)

# Convenience alias -> official OpenRouter model id
_OPENROUTER_ALIASES: Dict[str, str] = {
    "gpt-5.2":              "openai/gpt-5.2",
    "gpt5.2":               "openai/gpt-5.2",
    "gpt 5.2":              "openai/gpt-5.2",
    "openai/gpt-5.2":       "openai/gpt-5.2",
    "gpt-4.1":              "openai/gpt-4.1",
    "gpt-4.1-mini":         "openai/gpt-4.1-mini",
    "gpt-4.1-nano":         "openai/gpt-4.1-nano",
    "gpt-4o-2024-08-06":        "openai/gpt-4o-2024-08-06",
    "openai/gpt-4o-2024-08-06": "openai/gpt-4o-2024-08-06",
}


class _OpenRouterAPIError(RuntimeError):
    """Custom exception for OpenRouter API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(f"OpenRouter API error: {message}")
        self.status_code = status_code
        self.message = str(message)


# - detection / resolution -

def is_openrouter_model(model: str) -> bool:
    """Return True when *model* should be routed through OpenRouter."""
    m = (model or "").strip().lower()
    if not m:
        return False
    if m in _OPENROUTER_ALIASES:
        return True
    for prefix in _OPENROUTER_MODEL_PREFIXES:
        if m.startswith(prefix):
            return True
    return False


def _resolve_openrouter_model_id(model: str) -> str:
    m = (model or "").strip()
    low = m.lower()
    if low in _OPENROUTER_ALIASES:
        return _OPENROUTER_ALIASES[low]
    return m  # already in "provider/model" form


# - key loading -

def _load_openrouter_api_key() -> str:
    # Priority 1: file
    try:
        if _OPENROUTER_DEFAULT_KEY_PATH is not None and _OPENROUTER_DEFAULT_KEY_PATH.exists():
            key = _OPENROUTER_DEFAULT_KEY_PATH.read_text(encoding="utf-8").strip()
            if key:
                return key
    except Exception:
        pass
    # Priority 2: env
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if key:
        return key
    raise SystemExit(
        f"Missing OpenRouter API key.  Either:\n"
        f"  1. Create file: {_OPENROUTER_DEFAULT_KEY_PATH}\n"
        f"  2. Set OPENROUTER_API_KEY environment variable"
    )


# - core call (no web search) -

def _call_openrouter_api(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
    provider: Optional[str] = None,  # None = let OpenRouter pick
    timeout_s: float = 180.0,
) -> str:
    """
    Call an OpenAI-compatible model through OpenRouter.

    Args:
        model:             OpenRouter model id (e.g. "openai/gpt-5.2")
        system_prompt:     System message
        user_prompt:       User message
        max_output_tokens: max_tokens for the completion
        temperature:       sampling temperature
        provider:          upstream provider to prefer (default "Azure")
        timeout_s:         HTTP timeout
    Returns:
        The assistant's text reply.
    """
    import requests as _req

    api_key = _load_openrouter_api_key()
    model_id = _resolve_openrouter_model_id(model)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/anonymous",
        "X-Title": "SEAV",
    }

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        # OpenRouter rejects max_tokens=65536 (2^16) with "No endpoints found"
        # for Google models - use 65535 as a safe upper bound.
        "max_tokens": min(int(max_output_tokens), 65535),
        "temperature": float(temperature),
    }

    # - Provider routing -
    # For Moonshot models, force the official Moonshot AI endpoint
    # (non-quantized) and use default reasoning/thinking level.
    if provider:
        payload["provider"] = {"allow_fallbacks": False, "order": [provider]}
    elif model_id.startswith("google/"):
        payload["provider"] = {"allow_fallbacks": False, "order": ["Google"]}
    elif model_id.startswith("moonshotai/"):
        payload["provider"] = {"allow_fallbacks": False, "order": ["Moonshot AI"], "quantizations": ["int4"]}
        # Kimi K2.5 supports reasoning - use high effort (matches ours/config.py thinking_level=HIGH).
        # temperature=0.0 is now supported by Moonshot AI (verified 2026-03).
        payload["reasoning"] = {"effort": "high"}
    elif model_id.startswith("z-ai/"):
        # Z.AI is the official provider running BF16 (original precision, no quantization).
        payload["provider"] = {"allow_fallbacks": False, "order": ["Z.AI"]}

    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = _req.post(
                _OPENROUTER_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=timeout_s,
            )
            if resp.status_code >= 400:
                msg = ""
                try:
                    body = resp.json()
                    err = body.get("error") or {}
                    msg = str(err.get("message") or str(body)[:500])
                except Exception:
                    msg = (resp.text or "")[:500]
                raise _OpenRouterAPIError(msg, resp.status_code)

            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                return ""
            msg = choices[0].get("message") or {}
            text = msg.get("content") or ""
            if not text.strip():
                # Reasoning models (e.g., GLM-5) may put output in reasoning field
                text = msg.get("reasoning") or ""
            return text.strip()

        except _OpenRouterAPIError as e:
            if e.status_code == 429:
                last_err = e
                wait = 20.0 + 20.0 * attempt
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1.0 + 2.0 * attempt)
                continue
            raise _OpenRouterAPIError(f"Failed after 3 attempts: {e}")

    raise _OpenRouterAPIError(f"Failed after 3 attempts: {last_err}")


# - Tavily key loading -

_TAVILY_DEFAULT_KEY_PATH = (
    None  # Set TAVILY_API_KEY env var
)


def _load_tavily_api_key() -> str:
    # Priority 1: file
    try:
        if _TAVILY_DEFAULT_KEY_PATH is not None and _TAVILY_DEFAULT_KEY_PATH.exists():
            key = _TAVILY_DEFAULT_KEY_PATH.read_text(encoding="utf-8").strip()
            if key:
                return key
    except Exception:
        pass
    # Priority 2: env
    key = (os.environ.get("TAVILY_API_KEY") or "").strip()
    if key:
        return key
    raise SystemExit(
        f"Missing Tavily API key.  Either:\n"
        f"  1. Create file: {_TAVILY_DEFAULT_KEY_PATH}\n"
        f"  2. Set TAVILY_API_KEY environment variable"
    )


# - Tavily search helper -

def _tavily_search(query: str, *, max_results: int = 3) -> str:
    """
    Search the open internet via Tavily and return a formatted evidence
    string.  Mirrors the pattern in jades/fact_checking.py but searches
    the *full web* (no site: restriction) and returns up to *max_results*.

    Returns a string like:
        [1] Title: ?\n?content?\n\n[2] Title: ?\n?
    or "No results found".
    """
    from tavily import TavilyClient

    api_key = _load_tavily_api_key()
    tavily = TavilyClient(api_key=api_key)

    response = tavily.search(query=query, max_results=max_results)

    results = (response or {}).get("results") or [] if isinstance(response, dict) else []
    if not results:
        return "No results found"

    parts: List[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        content = r.get("content", "")
        parts.append(f"[{i}] Title: {title}\n{content}")
    return "\n\n".join(parts)


def _tavily_search_with_raw(query: str, *, max_results: int = 3) -> Tuple[str, Dict[str, Any]]:
    """
    Search the open internet via Tavily and return both the formatted evidence
    string and the raw structured response.

    Returns (evidence_text, raw_response) where:
        evidence_text: string like "[1] Title: ?\n?content?\n\n[2] Title: ?\n?"
                       or "No results found"
        raw_response: dict with full Tavily API response (including "results" list)
    """
    from tavily import TavilyClient

    api_key = _load_tavily_api_key()
    tavily = TavilyClient(api_key=api_key)

    response = tavily.search(query=query, max_results=max_results)

    results = (response or {}).get("results") or [] if isinstance(response, dict) else []
    if not results:
        return "No results found", response or {}

    parts: List[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        content = r.get("content", "")
        parts.append(f"[{i}] Title: {title}\n{content}")
    return "\n\n".join(parts), response or {}


# - call with Tavily search + OpenRouter LLM -
#    Two-step approach (same idea as Anthropic path in judge_pair_force_web):
#      1. Search with Tavily -> get evidence text
#      2. Inject evidence into prompt -> call LLM via OpenRouter
#    This guarantees at least one real web search per call.

def _call_openrouter_api_with_tavily_search(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
    search_query: Optional[str] = None,
    provider: Optional[str] = None,
    tavily_max_results: int = 5,
    timeout_s: float = 180.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Search with Tavily, inject results into prompt, then call GPT via
    OpenRouter.

    Args:
        search_query: Explicit search query for Tavily.  When *None* (the
            default), the caller did not supply one, so we extract one from
            the user_prompt using a simple heuristic (step text only).
        tavily_max_results: Number of Tavily results to fetch (default 5).

    Returns (text, metadata) compatible with the Gemini grounding and
    Anthropic web_search return shapes.
    """
    # - Step 1: determine search query -
    if not search_query:
        # Fallback: try to extract just the step text from the template.
        # The template has "STEP TO VERIFY:\n<text>\n\nPlease verify"
        import re
        m = re.search(r"STEP TO VERIFY:\s*\n(.*?)(?:\n\nPlease verify|\Z)", user_prompt, re.S)
        if m:
            search_query = m.group(1).strip()[:300]
        else:
            # Last resort: first 200 chars, but skip the INTENT header
            search_query = user_prompt[:200].replace("\n", " ").strip()

    # - Step 2: Tavily search -
    search_evidence = ""
    tavily_raw_response: Dict[str, Any] = {}
    try:
        search_evidence, tavily_raw_response = _tavily_search_with_raw(search_query, max_results=tavily_max_results)
    except Exception as e:
        search_evidence = f"Web search failed: {e}"
        tavily_raw_response = {}

    # - Step 3: augment the prompt with evidence -
    augmented_prompt = (
        f"{user_prompt}\n\n"
        f"--- Web search evidence (from internet search) ---\n"
        f"{search_evidence[:4000]}\n"
        f"--- End of web search evidence ---\n\n"
        f"Use the search evidence above to help verify factuality. "
        f"If the evidence supports the claim, mark it factual. "
        f"If the evidence contradicts it, mark it not factual."
    )

    # - Step 4: call LLM with enriched prompt -
    text = _call_openrouter_api(
        model=model,
        system_prompt=system_prompt,
        user_prompt=augmented_prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        provider=provider,
        timeout_s=timeout_s,
    )

    metadata: Dict[str, Any] = {
        "grounding_metadata": {
            "web_search_queries": [search_query],
            "tavily_raw_response": tavily_raw_response,
        }
    }
    return text, metadata
