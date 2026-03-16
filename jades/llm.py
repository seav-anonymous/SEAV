"""jades.llm

Helpers for calling LLMs from within JADES nodes.

Originally this repo only supported OpenAI models (Chat Completions / Responses).
We also support Anthropic Claude models (Messages API) for running JADES with
Opus 4.5.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from .config import JADESConfig


# Sticky fallback: once Gemini rate limit is hit, route ALL subsequent calls to OpenRouter
_gemini_rate_limited: bool = False


LLMApi = Literal["chat_completions", "responses", "anthropic_messages", "gemini", "bedrock", "openrouter"]


# OpenRouter GPT-4o max output tokens (official OpenAI limit)
_OPENROUTER_GPT4O_MAX_OUTPUT_TOKENS = 16384


_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION") or "2023-06-01"
_ANTHROPIC_DEFAULT_KEY_PATH = None  # Set ANTHROPIC_API_KEY env var


# - Gemini model detection -
_GEMINI_MODEL_PREFIXES = ("gemini-", "gemini ", "google/gemini")
_GEMINI_3_FLASH_PREVIEW_NAMES = {
    "gemini-3-flash-preview", "gemini 3 flash preview",
    "gemini-3-flash", "gemini 3 flash", "gemini3flash",
    "google/gemini-3-flash-preview", "google/gemini-3-flash",
}


def _is_gemini_model(model: str) -> bool:
    """Return True when model should be routed to the Gemini API."""
    m = (model or "").strip().lower()
    if not m:
        return False
    if m in _GEMINI_3_FLASH_PREVIEW_NAMES:
        return True
    for prefix in _GEMINI_MODEL_PREFIXES:
        if m.startswith(prefix):
            return True
    return False


def _is_anthropic_model(model: str) -> bool:
    m = (model or "").strip().lower()
    if not m:
        return False
    if m in {"opus 4.5", "opus-4.5", "opus_4.5", "opus45", "opus 45", "opus-45"}:
        return True
    if m.startswith("anthropic/"):
        return True
    return m.startswith("claude-")


def _resolve_anthropic_model_ids(model: str) -> list[str]:
    m = (model or "").strip()
    if not m:
        return []
    low = m.lower()

    # Support anthropic/<model> as a convenience alias.
    if low.startswith("anthropic/"):
        return [m.split("/", 1)[1].strip()]

    if low in {"opus 4.5", "opus-4.5", "opus_4.5", "opus45", "opus 45", "opus-45"}:
        env_id = (
            os.environ.get("OPUS_45_MODEL_ID")
            or os.environ.get("ANTHROPIC_OPUS_45_MODEL_ID")
            or ""
        ).strip()
        candidates = [
            env_id,
            # Prefer a pinned snapshot for reproducibility; fall back to the moving alias.
            "claude-opus-4-5-20251101",
            "claude-opus-4-5",
            m,
        ]
        out: list[str] = []
        seen: set[str] = set()
        for c in candidates:
            c = (c or "").strip()
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    return [m]


def _load_anthropic_api_key() -> str:
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if key:
        return key
    try:
        if _ANTHROPIC_DEFAULT_KEY_PATH is not None and _ANTHROPIC_DEFAULT_KEY_PATH.exists():
            key = _ANTHROPIC_DEFAULT_KEY_PATH.read_text(encoding="utf-8").strip()
            if key:
                return key
    except Exception:
        pass
    raise SystemExit(
        "Missing Anthropic API key. Set ANTHROPIC_API_KEY or add an api-claude file in the repo root."
    )


class _AnthropicHTTPError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Anthropic API error ({status_code}): {message}")
        self.status_code = int(status_code)
        self.message = str(message)


def _anthropic_error_looks_like_unknown_model(e: Exception) -> bool:
    if not isinstance(e, _AnthropicHTTPError):
        return False
    if int(e.status_code) not in {400, 404}:
        return False
    msg = (e.message or "").lower()
    return ("model" in msg) and (
        "not found" in msg
        or "does not exist" in msg
        or "unknown" in msg
        or "invalid model" in msg
        or "unsupported model" in msg
    )


def _anthropic_messages_create_text(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
    thinking: Optional[Dict[str, Any]] = None,
    timeout_s: float = 120.0,
) -> str:
    import requests

    api_key = _load_anthropic_api_key()
    headers = {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_output_tokens),
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": float(temperature),
    }
    if thinking is not None:
        payload["thinking"] = thinking

    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = requests.post(_ANTHROPIC_MESSAGES_URL, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code >= 400:
                msg = ""
                try:
                    data = resp.json()
                    err = data.get("error") or {}
                    msg = str(err.get("message") or "")
                    if not msg:
                        msg = str(data)[:400]
                except Exception:
                    msg = (resp.text or "")[:400]
                raise _AnthropicHTTPError(resp.status_code, msg)

            data = resp.json()
            blocks = data.get("content")
            out_parts: list[str] = []
            if isinstance(blocks, list):
                for b in blocks:
                    if isinstance(b, dict) and b.get("type") == "text":
                        out_parts.append(str(b.get("text") or ""))
            out = "".join(out_parts).strip()
            if out:
                return out
            if isinstance(data.get("completion"), str):
                return str(data.get("completion") or "").strip()
            return ""
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1.0 + 1.5 * attempt)
                continue
            raise
    raise last_err or RuntimeError("Anthropic call failed")


def _resolve_llm_api(config: JADESConfig, llm_client: Any) -> LLMApi:
    """
    Decide which API surface to use.

    - If config.llm_api is set explicitly, respect it.
    - If set to "auto", prefer Responses for GPT-5.* models when available.
    - Otherwise fall back to whichever interface exists on the client.
    """
    llm_api = getattr(config, "llm_api", "auto")
    if llm_api in ("chat_completions", "responses"):
        return llm_api  # type: ignore[return-value]

    model = str(getattr(config, "llm_model", "") or "")

    from seav.nodes.bedrock_api import is_bedrock_model as _is_bedrock_model
    if _is_bedrock_model(model):
        return "bedrock"

    from seav.nodes.openrouter_api import is_openrouter_model as _is_openrouter_model
    if _is_openrouter_model(model):
        return "openrouter"

    if _is_gemini_model(model):
        return "gemini"

    if _is_anthropic_model(model):
        return "anthropic_messages"

    has_responses = hasattr(llm_client, "responses") and hasattr(llm_client.responses, "create")
    has_chat = hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions")

    if has_responses and model.startswith("gpt-5"):
        return "responses"
    if has_chat:
        return "chat_completions"
    if has_responses:
        return "responses"
    raise RuntimeError("LLM client does not expose chat.completions or responses.create")


def call_llm_text(
    *,
    llm_client: Any,
    config: JADESConfig,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
) -> str:
    """
    Make a single text-only LLM call and return the model's text output.

    Notes:
    - No tools are used here (this is the "no web-search extension" path).
    - `max_output_tokens` is mapped to the relevant API parameter.
    """
    api = _resolve_llm_api(config, llm_client)

    if api == "bedrock":
        from seav.nodes.bedrock_api import _call_bedrock_api

        model = str(getattr(config, "llm_model", "") or "")
        _temp = float(getattr(config, "llm_temperature", 0.0) or 0.0)
        return _call_bedrock_api(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=int(max_output_tokens),
            temperature=_temp,
        )

    if api == "openrouter":
        from seav.nodes.openrouter_api import _resolve_openrouter_model_id, _call_openrouter_api
        from seav.config import DEFAULT_MAX_OUTPUT_TOKENS as _DEFAULT_MAX_TOKENS
        model = str(getattr(config, "llm_model", "") or "")
        resolved_model_id = _resolve_openrouter_model_id(model)
        provider = "OpenAI" if resolved_model_id.startswith("openai/") else None
        
        # Bump max_output_tokens for Google Gemini models to avoid truncation with thinking
        if resolved_model_id.startswith("google/"):
            effective_max_out = max(int(max_output_tokens), _DEFAULT_MAX_TOKENS)
        else:
            effective_max_out = min(int(max_output_tokens), _OPENROUTER_GPT4O_MAX_OUTPUT_TOKENS)
        
        temp = float(getattr(config, "llm_temperature", 0.0) or 0.0)
        return _call_openrouter_api(
            model=resolved_model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=effective_max_out,
            temperature=temp,
            provider=provider,
        )

    if api == "gemini":
        global _gemini_rate_limited
        from seav.nodes.gemini_api import _call_gemini_api as _gemini_call  # lazy import
        from seav.nodes.gemini_api import _is_gemini_3_model  # lazy import
        from seav.nodes.openrouter_api import _call_openrouter_api  # lazy import
        from seav.config import DEFAULT_MAX_OUTPUT_TOKENS as _GEMINI_MAX_TOKENS
        _temp = float(getattr(config, "llm_temperature", 0.0) or 0.0)
        
        # Bump max_output_tokens for Gemini 3 models to avoid truncation with thinking
        model_id = str(config.llm_model or "")
        if _is_gemini_3_model(model_id):
            _max_out = max(int(max_output_tokens), _GEMINI_MAX_TOKENS)
        else:
            _max_out = min(int(max_output_tokens), _GEMINI_MAX_TOKENS)
        
        _or_model = "google/gemini-3-flash-preview"

        if _gemini_rate_limited:
            return _call_openrouter_api(
                model=_or_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=_max_out,
                temperature=_temp,
                provider="Google",
            )

        try:
            return _gemini_call(
                model=config.llm_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=_max_out,
                temperature=_temp,
                thinking_budget=0,
            )
        except Exception as _gemini_err:
            _err_msg = str(_gemini_err).lower()
            if "quota" in _err_msg or "rate" in _err_msg or "resource" in _err_msg or "429" in _err_msg:
                _gemini_rate_limited = True
                print(f"[GEMINI->OPENROUTER] Rate limit hit - switching ALL subsequent calls to OpenRouter ({_or_model})")
                return _call_openrouter_api(
                    model=_or_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=_max_out,
                    temperature=_temp,
                    provider="Google",
                )
            raise

    if api == "anthropic_messages":
        # Claude (Anthropic Messages API)
        # Opus 4.5 supports extended thinking, but Anthropic currently requires `temperature=1`
        # when thinking is enabled. Since this repo defaults all model temperatures to 0.0,
        # we keep thinking disabled by default for JADES runs.
        thinking: Optional[Dict[str, Any]] = None
        model = str(getattr(config, "llm_model", "") or "")
        resolved_ids = _resolve_anthropic_model_ids(model)

        base_max_out = int(max_output_tokens)
        max_tokens = base_max_out

        # To enable thinking explicitly, set JADES_ANTHROPIC_THINKING_BUDGET_TOKENS>=1024 AND
        # run with --jades-temperature 1.0.
        is_opus_45 = model.strip().lower() in {"opus 4.5", "opus-4.5", "opus_4.5", "opus45", "opus 45", "opus-45"}
        thinking_budget = int(os.environ.get("JADES_ANTHROPIC_THINKING_BUDGET_TOKENS") or 0)
        temp = float(getattr(config, "llm_temperature", 0.0) or 0.0)
        enable_thinking = is_opus_45 and thinking_budget >= 1024 and base_max_out >= 800 and abs(temp - 1.0) < 1e-9
        if enable_thinking:
            thinking = {"type": "enabled", "budget_tokens": thinking_budget}
            max_tokens = base_max_out + thinking_budget

        last_err: Optional[Exception] = None
        for model_id in resolved_ids:
            try:
                return _anthropic_messages_create_text(
                    model=model_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=max_tokens,
                    temperature=temp,
                    thinking=thinking,
                )
            except Exception as e:
                last_err = e
                if _anthropic_error_looks_like_unknown_model(e):
                    continue
                break
        if last_err is not None:
            raise last_err
        raise RuntimeError("Anthropic call failed")

    if api == "responses":
        # Some models enforce a minimum max_output_tokens value on the Responses API.
        # Clamp low values upward to avoid hard failures for short classification calls.
        effective_max_output_tokens = max(16, int(max_output_tokens))
        kwargs: dict[str, Any] = {
            "model": config.llm_model,
            "input": user_prompt,
            "temperature": config.llm_temperature,
            "max_output_tokens": effective_max_output_tokens,
        }
        if system_prompt:
            kwargs["instructions"] = system_prompt
        resp = llm_client.responses.create(**kwargs)
        text = getattr(resp, "output_text", None)
        if text is None:
            raise RuntimeError("Responses API returned no output_text")
        return text

    # chat_completions
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    resp = llm_client.chat.completions.create(
        model=config.llm_model,
        messages=messages,
        temperature=config.llm_temperature,
        max_tokens=max_output_tokens,
    )
    choice0 = resp.choices[0]
    content = getattr(getattr(choice0, "message", None), "content", None)
    if content is None:
        raise RuntimeError("Chat Completions API returned empty message content")
    return content
