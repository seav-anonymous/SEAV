"""
Local OpenAI-compatible API utilities for OURS jailbreak evaluation method.

Calls models served by llama.cpp, vLLM, or any OpenAI-compatible local server.

Usage:
    from seav.nodes.local_api import (
        is_local_model,
        _call_local_api,
    )
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Dict, Optional


# - constants -

_LOCAL_PREFIX = "local/"
_DEFAULT_LOCAL_URL = os.environ.get(
    "LOCAL_LLM_API_URL", "http://localhost:8090/v1/chat/completions"
)


class _LocalAPIError(RuntimeError):
    """Custom exception for local API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(f"Local API error: {message}")
        self.status_code = status_code
        self.message = str(message)


# - detection / resolution -


def is_local_model(model: str) -> bool:
    """Return True when *model* should be routed to a local server.

    Convention: model names starting with ``local/`` are local models.
    Examples: ``local/glm-4.7``, ``local/qwen3-235b``.
    """
    m = (model or "").strip().lower()
    return m.startswith(_LOCAL_PREFIX) if m else False


def _resolve_local_model_id(model: str) -> str:
    """Strip the ``local/`` prefix and return the bare model id."""
    m = (model or "").strip()
    if m.lower().startswith(_LOCAL_PREFIX):
        return m[len(_LOCAL_PREFIX):]
    return m


# - core call -


def _call_local_api(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
    api_url: Optional[str] = None,
    timeout_s: float = 600.0,
) -> str:
    """Call a local OpenAI-compatible server.

    Args:
        model:             Model name (``local/glm-4.7`` or bare id).
        system_prompt:     System message.
        user_prompt:       User message.
        max_output_tokens: ``max_tokens`` for the completion.
        temperature:       Sampling temperature.
        api_url:           Override API endpoint (default: ``LOCAL_LLM_API_URL``
                           env var or ``http://localhost:8090/v1/chat/completions``).
        timeout_s:         HTTP timeout in seconds (default 600 - local models
                           can be slow on long inputs).

    Returns:
        The assistant's text reply.  Falls back to ``reasoning_content`` when
        ``content`` is empty (common with thinking-mode models).
    """
    url = api_url or _DEFAULT_LOCAL_URL
    model_id = _resolve_local_model_id(model)

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": int(max_output_tokens),
        "temperature": float(temperature),
        "stream": False,
    }

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
    )

    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read())

            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            if not content:
                # Thinking-mode models sometimes put everything in
                # reasoning_content (e.g. Qwen3 with --reasoning-format
                # deepseek).  The leading ``{`` of a JSON object may be
                # swallowed by the reasoning token boundary, so we
                # heuristically restore it when the text looks like a
                # truncated JSON object.
                rc = (msg.get("reasoning_content") or "").strip()
                if rc:
                    # Fix truncated JSON: starts with a key like
                    # ``"has_steps"`` but missing the opening brace.
                    if rc.startswith('"') and not rc.startswith('"{'):
                        rc = "{" + rc
                    content = rc
            return content

        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1.0 + 1.5 * attempt)

    raise _LocalAPIError(f"Failed after 3 retries: {last_err}")
