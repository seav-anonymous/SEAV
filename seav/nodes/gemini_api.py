"""
Gemini API utilities for OURS jailbreak evaluation method.

This module provides shared functions for calling Google's Gemini API
using the new google-genai SDK (replaces deprecated google-generativeai).

Usage:
    from seav.nodes.gemini_api import (
        is_gemini_model,
        _call_gemini_api,
        _call_gemini_api_with_grounding,
    )
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Default model configuration
DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"

# Gemini API key file path (same directory level as api-claude)
_GEMINI_DEFAULT_KEY_PATH = None  # Set GEMINI_API_KEY env var

# Model aliases for Gemini 3 Pro Preview
_GEMINI_3_PRO_ALIASES = {
    "gemini-3-pro-preview", "gemini 3 pro preview", "gemini-3-pro",
    "gemini 3 pro", "gemini3pro", "gemini-3", "gemini 3",
    "google/gemini-3-pro-preview", "google/gemini-3-pro",
}

# Model aliases for Gemini 3 Flash Preview
_GEMINI_3_FLASH_PREVIEW_ALIASES = {
    "gemini-3-flash-preview", "gemini 3 flash preview", "gemini-3-flash",
    "gemini 3 flash", "gemini3flash",
    "google/gemini-3-flash-preview", "google/gemini-3-flash",
}

# Model aliases for other Gemini models
_GEMINI_2_5_PRO_ALIASES = {
    "gemini-2.5-pro", "gemini 2.5 pro", "gemini-2.5-pro-latest",
    "google/gemini-2.5-pro", "gemini-2-5-pro",
}

_GEMINI_2_FLASH_ALIASES = {
    "gemini-2.0-flash", "gemini 2.0 flash", "gemini-2-flash",
    "google/gemini-2.0-flash", "gemini-flash",
}


class _GeminiAPIError(RuntimeError):
    """Custom exception for Gemini API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(f"Gemini API error: {message}")
        self.status_code = status_code
        self.message = str(message)


# Sticky fallback: once Gemini grounding rate limit is hit, route ALL subsequent calls to OpenRouter
_gemini_grounding_rate_limited: bool = False


def _gemini_model_to_openrouter(model: str) -> str:
    """Convert Gemini model name to OpenRouter format (prepend 'google/' if needed)."""
    m = (model or "").strip()
    if m.startswith("google/"):
        return m
    return f"google/{m}"


def is_gemini_model(model: str) -> bool:
    """Check if the model is a Gemini model."""
    m = (model or "").strip().lower()
    if not m:
        return False
    
    if m in _GEMINI_3_PRO_ALIASES:
        return True
    if m in _GEMINI_3_FLASH_PREVIEW_ALIASES:
        return True
    if m in _GEMINI_2_5_PRO_ALIASES:
        return True
    if m in _GEMINI_2_FLASH_ALIASES:
        return True
    
    return m.startswith("gemini") or m.startswith("google/gemini")


def _resolve_gemini_model_id(model: str) -> str:
    """Resolve model name to official Gemini model ID."""
    m = (model or "").strip()
    if not m:
        return DEFAULT_GEMINI_MODEL
    
    low = m.lower()
    
    if low in _GEMINI_3_PRO_ALIASES:
        env_id = os.environ.get("GEMINI_3_PRO_MODEL_ID", "").strip()
        if env_id:
            return env_id
        return "gemini-3-pro-preview"

    if low in _GEMINI_3_FLASH_PREVIEW_ALIASES:
        env_id = (
            os.environ.get("GEMINI_3_FLASH_MODEL_ID")
            or os.environ.get("GEMINI_3_FLASH_PREVIEW_MODEL_ID")
            or ""
        ).strip()
        if env_id:
            return env_id
        return "gemini-3-flash-preview"
    
    if low in _GEMINI_2_5_PRO_ALIASES:
        return "gemini-2.5-pro-preview-05-06"
    
    if low in _GEMINI_2_FLASH_ALIASES:
        return "gemini-2.0-flash"
    
    if low.startswith("google/"):
        return m.split("/", 1)[1].strip()
    
    return m


def _is_gemini_3_flash_preview_model(model: str) -> bool:
    """Return True if model resolves to Gemini 3 Flash Preview family."""
    m = (model or "").strip().lower()
    if not m:
        return False
    if m in _GEMINI_3_FLASH_PREVIEW_ALIASES:
        return True
    return m.startswith("gemini-3-flash-preview") or m.startswith("google/gemini-3-flash-preview")


def _is_gemini_3_model(model_id: str) -> bool:
    """Check if model is any Gemini 3 variant (Flash or Pro)."""
    return "gemini-3-" in model_id.lower()


def _load_gemini_api_key() -> str:
    """Load Gemini API key from file or environment variable."""
    # Priority 1: File
    try:
        if _GEMINI_DEFAULT_KEY_PATH is not None and _GEMINI_DEFAULT_KEY_PATH.exists():
            key = _GEMINI_DEFAULT_KEY_PATH.read_text(encoding="utf-8").strip()
            if key:
                return key
    except Exception:
        pass
    
    # Priority 2: GOOGLE_API_KEY
    key = (os.environ.get("GOOGLE_API_KEY") or "").strip()
    if key:
        return key
    
    # Priority 3: GEMINI_API_KEY
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if key:
        return key
    
    raise SystemExit(
        f"Missing Gemini API key. Either:\n"
        f"  1. Create file: {_GEMINI_DEFAULT_KEY_PATH}\n"
        f"  2. Set GOOGLE_API_KEY environment variable\n"
        f"  3. Set GEMINI_API_KEY environment variable"
    )


def _extract_text_from_response(response: Any) -> str:
    """Extract text from Gemini response object."""
    # Try direct text attribute
    try:
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
    except Exception:
        pass
    
    # Try candidates
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            # Check content.parts with safety
            try:
                if hasattr(candidate, 'content') and candidate.content:
                    content = candidate.content
                    if hasattr(content, 'parts') and content.parts:
                        parts_text = []
                        for part in content.parts:
                            if hasattr(part, 'text') and part.text:
                                parts_text.append(part.text)
                        if parts_text:
                            return "".join(parts_text).strip()
            except Exception:
                continue
    
    return ""


def _check_response_blocked(response: Any) -> Optional[str]:
    """Check if response was blocked by safety filters. Returns error message if blocked."""
    # Check prompt feedback
    if hasattr(response, 'prompt_feedback'):
        feedback = response.prompt_feedback
        if hasattr(feedback, 'block_reason') and feedback.block_reason:
            return f"Prompt blocked: {feedback.block_reason}"
    
    # Check candidates finish reason
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'finish_reason'):
                reason = str(candidate.finish_reason)
                # SAFETY = blocked by safety
                if "SAFETY" in reason.upper() or reason == "2":
                    return f"Content blocked by safety filter: {reason}"
    
    return None


def _call_gemini_api(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
    thinking_budget: int = 10000,  # Default same as Claude config
    timeout_s: float = 180.0,
) -> str:
    """
    Call Gemini API using the new google-genai SDK.
    
    Args:
        model: Gemini model name
        system_prompt: System instruction
        user_prompt: User message
        max_output_tokens: Maximum output tokens
        temperature: Sampling temperature (0.0-2.0)
        thinking_budget: Budget for extended thinking in tokens (default: 10000)
        timeout_s: Request timeout in seconds
        
    Returns:
        Text response from the model
    """
    from google import genai
    from google.genai import types
    
    api_key = _load_gemini_api_key()
    model_id = _resolve_gemini_model_id(model)
    
    # Create client
    client = genai.Client(api_key=api_key)
    
    # Configure temperature + thinking.
    # Gemini 3 (Flash/Pro preview): use thinking_level (HIGH/MEDIUM/LOW/MINIMAL);
    #   thinking_budget is NOT supported and will cause an API error.
    # Gemini 2.5 and older: use thinking_budget (0=disabled, -1=dynamic, N=fixed tokens);
    #   thinking_level is NOT supported.
    effective_temperature = float(temperature)
    if _is_gemini_3_model(model_id):
        thinking_config = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.HIGH
        )
    elif thinking_budget > 0:
        thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)
    else:
        thinking_config = None

    # Configure generation
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=int(max_output_tokens),
        temperature=effective_temperature,
        thinking_config=thinking_config,
    )
    
    max_retries = 5
    last_err: Optional[Exception] = None
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=user_prompt,
                config=config,
            )
            
            # Check for blocking
            block_reason = _check_response_blocked(response)
            if block_reason:
                raise _GeminiAPIError(block_reason)
            
            # Extract text
            text = _extract_text_from_response(response)
            if text:
                return text
            
            # Empty response - not necessarily an error
            return ""
            
        except _GeminiAPIError:
            raise
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            
            if "api_key" in err_msg or "authentication" in err_msg or "invalid" in err_msg:
                raise _GeminiAPIError(f"Authentication failed: {e}")
            if "quota" in err_msg or "rate" in err_msg:
                if attempt < max_retries - 1:
                    wait = min(5.0 * (2 ** attempt), 120.0)  # 5, 10, 20, 40, 80
                    time.sleep(wait)
                    continue
            
            if attempt < max_retries - 1:
                time.sleep(1.0 + 1.5 * attempt)
                continue
            raise _GeminiAPIError(f"API call failed after {max_retries} attempts: {e}")
    
    raise _GeminiAPIError(f"API call failed: {last_err}")


def _call_gemini_api_with_grounding(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    search_query: Optional[str] = None,
    max_output_tokens: int,
    temperature: float,
    thinking_budget: int = 10000,  # Default same as Claude config
    require_web_search: bool = False,
    two_stage_fallback: bool = False,
    timeout_s: float = 180.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Call Gemini API with Google Search grounding enabled.
    
    Args:
        model: Gemini model name
        system_prompt: System instruction
        user_prompt: User message
        search_query: Optional web search query to suggest to the model
        max_output_tokens: Maximum output tokens
        temperature: Sampling temperature
        thinking_budget: Budget for extended thinking in tokens (default: 10000)
        require_web_search: If True, enforce web search grounding
        two_stage_fallback: If False (default), send forced prompt directly when
            require_web_search=True (single API call). If True, use legacy two-stage
            behavior: try original prompt first, fall back to forced prompt only if
            no search was detected.
        timeout_s: Request timeout in seconds
        
    Returns:
        Tuple of (text_response, metadata_dict)
    """
    from google import genai
    from google.genai import types
    
    api_key = _load_gemini_api_key()
    model_id = _resolve_gemini_model_id(model)
    
    # Create client
    client = genai.Client(api_key=api_key)
    
    # Configure temperature + thinking.
    # Gemini 3 (Flash/Pro preview): use thinking_level (HIGH/MEDIUM/LOW/MINIMAL);
    #   thinking_budget is NOT supported and will cause an API error.
    # Gemini 2.5 and older: use thinking_budget (0=disabled, -1=dynamic, N=fixed tokens);
    #   thinking_level is NOT supported.
    effective_temperature = float(temperature)
    if _is_gemini_3_model(model_id):
        thinking_config = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.HIGH
        )
    elif thinking_budget > 0:
        thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)
    else:
        thinking_config = None

    # Configure with Google Search grounding
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=int(max_output_tokens),
        temperature=effective_temperature,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        thinking_config=thinking_config,
    )
    
    def _build_forced_prompt(base_prompt: str) -> str:
        extra = (
            "\n\nIMPORTANT: You MUST use web search grounding at least once before answering. "
            "Perform at least one web search query (use the suggested query if provided), "
            "then answer ONLY in the required JSON format. "
            "If you did not use web search, your answer is invalid; run a web search and try again."
        )
        if search_query and isinstance(search_query, str) and search_query.strip():
            return (
                f"{base_prompt}\n\nSUGGESTED_WEB_SEARCH_QUERY:\n{search_query.strip()}" + extra
            )
        return base_prompt + extra

    if require_web_search and not two_stage_fallback:
        # Default: single forced call
        prompts_to_try: List[str] = [_build_forced_prompt(user_prompt)]
    elif require_web_search and two_stage_fallback:
        # Legacy two-stage: original first, forced as fallback
        prompts_to_try: List[str] = [user_prompt, _build_forced_prompt(user_prompt)]
    else:
        # No search required
        prompts_to_try: List[str] = [user_prompt]

    max_retries = 5
    last_err: Optional[Exception] = None

    for prompt_idx, prompt in enumerate(prompts_to_try):
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=config,
                )

                # Check for blocking
                block_reason = _check_response_blocked(response)
                if block_reason:
                    raise _GeminiAPIError(block_reason)

                # Extract text
                text = _extract_text_from_response(response)

                # Extract grounding metadata
                metadata: Dict[str, Any] = {"grounding_metadata": {}}

                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]

                    if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                        gm = candidate.grounding_metadata
                        gm_dict: Dict[str, Any] = {}

                        # Extract web search queries
                        if hasattr(gm, "web_search_queries") and gm.web_search_queries:
                            gm_dict["web_search_queries"] = list(gm.web_search_queries)

                        # Extract grounding chunks (sources)
                        if hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
                            chunks = []
                            for chunk in gm.grounding_chunks:
                                chunk_info = {}
                                if hasattr(chunk, "web") and chunk.web:
                                    chunk_info["uri"] = getattr(chunk.web, "uri", "")
                                    chunk_info["title"] = getattr(chunk.web, "title", "")
                                chunks.append(chunk_info)
                            gm_dict["grounding_chunks"] = chunks

                        # Extract grounding supports
                        if hasattr(gm, "grounding_supports") and gm.grounding_supports:
                            supports = []
                            for support in gm.grounding_supports:
                                support_info = {}
                                if hasattr(support, "segment") and support.segment:
                                    support_info["text"] = getattr(support.segment, "text", "")
                                if hasattr(support, "grounding_chunk_indices"):
                                    support_info["chunk_indices"] = list(support.grounding_chunk_indices)
                                supports.append(support_info)
                            gm_dict["grounding_supports"] = supports

                        metadata["grounding_metadata"] = gm_dict

                # Enforce at least one web search (if requested)
                queries = (metadata.get("grounding_metadata") or {}).get("web_search_queries") or []
                if require_web_search and not queries:
                    is_last_prompt_variant = prompt_idx == (len(prompts_to_try) - 1)
                    if not is_last_prompt_variant:
                        # Try the forced prompt variant.
                        break
                    # Already on forced prompt; retry.
                    if attempt < max_retries - 1:
                        time.sleep(1.0 + 1.5 * attempt)
                        continue
                    raise _GeminiAPIError(
                        "No web search performed (missing grounding_metadata.web_search_queries)."
                    )

                return text, metadata

            except _GeminiAPIError:
                raise
            except Exception as e:
                last_err = e
                err_msg = str(e).lower()

                if "api_key" in err_msg or "authentication" in err_msg:
                    raise _GeminiAPIError(f"Authentication failed: {e}")
                if "quota" in err_msg or "rate" in err_msg:
                    if attempt < max_retries - 1:
                        wait = min(5.0 * (2 ** attempt), 120.0)  # 5, 10, 20, 40, 80
                        time.sleep(wait)
                        continue

                if attempt < max_retries - 1:
                    time.sleep(1.0 + 1.5 * attempt)
                    continue
                raise _GeminiAPIError(f"API call failed after {max_retries} attempts: {e}")

    if require_web_search:
        raise _GeminiAPIError("No web search performed (missing grounding_metadata.web_search_queries).")

    raise _GeminiAPIError(f"API call failed: {last_err}")


def _call_gemini_api_with_tavily_search(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    search_query: Optional[str] = None,
    max_output_tokens: int,
    temperature: float,
    thinking_budget: int = 10000,
    tavily_max_results: int = 5,
    augmentation_suffix: str = "Use the search evidence above to inform your analysis.",
    timeout_s: float = 180.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Two-step Tavily search + plain Gemini API call (no grounding).

    1. If search_query is provided, search with Tavily and inject results into prompt.
    2. Call _call_gemini_api() (without grounding) with the (possibly augmented) prompt.

    When search_query is None, skips Tavily and calls _call_gemini_api() directly
    with no evidence injection.

    Returns (text, metadata) compatible with _build_web_search_metadata().
    """
    from seav.nodes.openrouter_api import _tavily_search_with_raw  # lazy import

    # No search query -> skip Tavily, call plain Gemini
    if not search_query:
        text = _call_gemini_api(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            thinking_budget=thinking_budget,
            timeout_s=timeout_s,
        )
        return text, {}

    # Step 1: Tavily search
    search_evidence: str = ""
    tavily_raw_response: Dict[str, Any] = {}
    try:
        search_evidence, tavily_raw_response = _tavily_search_with_raw(
            search_query, max_results=tavily_max_results
        )
    except Exception as e:
        search_evidence = f"Web search failed: {e}"
        tavily_raw_response = {}

    # Step 2: Augment prompt with evidence
    augmented_prompt = (
        f"{user_prompt}\n\n"
        f"--- Web search evidence (from internet search) ---\n"
        f"{search_evidence[:4000]}\n"
        f"--- End of web search evidence ---\n\n"
        f"{augmentation_suffix}"
    )

    # Step 3: Call Gemini (no grounding)
    text = _call_gemini_api(
        model=model,
        system_prompt=system_prompt,
        user_prompt=augmented_prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        thinking_budget=thinking_budget,
        timeout_s=timeout_s,
    )

    # Step 4: Build metadata compatible with _build_web_search_metadata()
    # (Tavily detection: has grounding_metadata.web_search_queries but no grounding_chunks)
    metadata: Dict[str, Any] = {
        "grounding_metadata": {
            "web_search_queries": [search_query],
            "tavily_raw_response": tavily_raw_response,
        }
    }
    return text, metadata


def extract_grounding_info(metadata: Dict[str, Any]) -> Tuple[bool, str]:
    """Extract grounding information from response metadata."""
    gm = metadata.get("grounding_metadata", {})
    
    queries = gm.get("web_search_queries", [])
    search_performed = len(queries) > 0
    
    search_query = ""
    if queries:
        search_query = queries[0] if isinstance(queries[0], str) else str(queries[0])
    
    return search_performed, search_query


def call_gemini_grounding_with_openrouter_fallback(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    search_query: Optional[str] = None,
    max_output_tokens: int,
    temperature: float,
    thinking_budget: int = 10000,
    require_web_search: bool = True,
    two_stage_fallback: bool = False,
    timeout_s: float = 180.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Call Gemini API with grounding, with sticky OpenRouter fallback on rate limit.
    
    Once a rate limit is hit, all subsequent calls route to OpenRouter (sticky).
    
    Args:
        model: Gemini model name
        system_prompt: System instruction
        user_prompt: User message
        search_query: Optional web search query
        max_output_tokens: Maximum output tokens
        temperature: Sampling temperature
        thinking_budget: Budget for extended thinking in tokens
        require_web_search: If True, enforce web search grounding
        two_stage_fallback: If False (default), send forced prompt directly
        timeout_s: Request timeout in seconds
        
    Returns:
        Tuple of (text_response, metadata_dict)
    """
    global _gemini_grounding_rate_limited
    
    # Lazy import to avoid circular dependencies
    from seav.nodes.openrouter_api import _call_openrouter_api_with_tavily_search
    
    # Convert model to OpenRouter format
    or_model = _gemini_model_to_openrouter(model)
    
    # If already rate limited, skip Gemini and go straight to OpenRouter
    if _gemini_grounding_rate_limited:
        return _call_openrouter_api_with_tavily_search(
            model=or_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            search_query=search_query,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            provider="Google",
            timeout_s=timeout_s,
        )
    
    # Try Gemini first
    try:
        return _call_gemini_api_with_grounding(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            search_query=search_query,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            thinking_budget=thinking_budget,
            require_web_search=require_web_search,
            two_stage_fallback=two_stage_fallback,
            timeout_s=timeout_s,
        )
    except _GeminiAPIError as e:
        err_msg = str(e).lower()
        if "quota" in err_msg or "rate" in err_msg or "resource" in err_msg or "429" in err_msg:
            # Rate limit hit: set sticky flag and fallback to OpenRouter
            _gemini_grounding_rate_limited = True
            print(f"[GEMINI->OPENROUTER] Rate limit hit - switching ALL subsequent grounding calls to OpenRouter ({or_model})")
            return _call_openrouter_api_with_tavily_search(
                model=or_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                search_query=search_query,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                provider="Google",
                timeout_s=timeout_s,
            )
        # Non-quota error: re-raise
        raise

