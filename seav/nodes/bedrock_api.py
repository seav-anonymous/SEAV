from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_BEDROCK_API_URL_TEMPLATE = "https://bedrock-runtime.us-east-2.amazonaws.com/model/{model_id}/converse"
_BEDROCK_DEFAULT_KEY_PATH = None  # Set AWS_BEDROCK_API_KEY env var
_TAVILY_DEFAULT_KEY_PATH = None  # Set TAVILY_API_KEY env var


class _BedrockAPIError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(f"Bedrock API error: {message}")
        self.status_code = status_code
        self.message = str(message)


def is_bedrock_model(model: str) -> bool:
    return (model or "").strip().lower().startswith("bedrock/")


def _resolve_bedrock_model_id(model: str) -> str:
    m = (model or "").strip()
    if m.lower().startswith("bedrock/"):
        return m.split("/", 1)[1].strip()
    return m


def _load_bedrock_api_key() -> str:
    try:
        if _BEDROCK_DEFAULT_KEY_PATH is not None and _BEDROCK_DEFAULT_KEY_PATH.exists():
            key = _BEDROCK_DEFAULT_KEY_PATH.read_text(encoding="utf-8").strip()
            if key:
                return key
    except Exception:
        pass
    key = (os.environ.get("BEDROCK_API_KEY") or "").strip()
    if key:
        return key
    raise SystemExit(
        f"Missing Bedrock API key.  Either:\n"
        f"  1. Create file: {_BEDROCK_DEFAULT_KEY_PATH}\n"
        f"  2. Set BEDROCK_API_KEY environment variable"
    )


def _call_bedrock_api(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
    timeout_s: float = 180.0,
) -> str:
    import requests as _req

    api_key = _load_bedrock_api_key()
    model_id = _resolve_bedrock_model_id(model)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
        "system": [{"text": system_prompt}],
        "inferenceConfig": {
            "maxTokens": min(int(max_output_tokens), 65536),
            "temperature": float(temperature),
        },
    }

    last_err: Optional[Exception] = None
    url = _BEDROCK_API_URL_TEMPLATE.format(model_id=model_id)
    for attempt in range(3):
        try:
            resp = _req.post(url, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code >= 400:
                msg = ""
                try:
                    body = resp.json()
                    msg = str(body.get("message") or body.get("error") or str(body)[:500])
                except Exception:
                    msg = (resp.text or "")[:500]
                raise _BedrockAPIError(msg, resp.status_code)

            data = resp.json()
            content = (((data.get("output") or {}).get("message") or {}).get("content") or [])
            if not content:
                return ""
            first = content[0] if isinstance(content[0], dict) else {}
            return str(first.get("text") or "").strip()

        except _BedrockAPIError as e:
            if e.status_code == 429:
                last_err = e
                time.sleep(30.0 + 20.0 * attempt)
                continue
            if e.status_code is not None and 400 <= e.status_code < 500:
                raise
            last_err = e
            if attempt < 2:
                time.sleep(1.0 + 2.0 * attempt)
                continue
            raise _BedrockAPIError(f"Failed after 3 attempts: {e}")
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1.0 + 2.0 * attempt)
                continue
            raise _BedrockAPIError(f"Failed after 3 attempts: {e}")

    raise _BedrockAPIError(f"Failed after 3 attempts: {last_err}")


def _load_tavily_api_key() -> str:
    try:
        if _TAVILY_DEFAULT_KEY_PATH is not None and _TAVILY_DEFAULT_KEY_PATH.exists():
            key = _TAVILY_DEFAULT_KEY_PATH.read_text(encoding="utf-8").strip()
            if key:
                return key
    except Exception:
        pass
    key = (os.environ.get("TAVILY_API_KEY") or "").strip()
    if key:
        return key
    raise SystemExit(
        f"Missing Tavily API key.  Either:\n"
        f"  1. Create file: {_TAVILY_DEFAULT_KEY_PATH}\n"
        f"  2. Set TAVILY_API_KEY environment variable"
    )


def _tavily_search_with_raw(query: str, *, max_results: int = 3) -> Tuple[str, Dict[str, Any]]:
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


def _call_bedrock_api_with_tavily_search(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
    search_query: Optional[str] = None,
    tavily_max_results: int = 5,
    timeout_s: float = 180.0,
) -> Tuple[str, Dict[str, Any]]:
    if not search_query:
        import re
        m = re.search(r"STEP TO VERIFY:\s*\n(.*?)(?:\n\nPlease verify|\Z)", user_prompt, re.S)
        if m:
            search_query = m.group(1).strip()[:300]
        else:
            search_query = user_prompt[:200].replace("\n", " ").strip()

    search_evidence = ""
    tavily_raw_response: Dict[str, Any] = {}
    try:
        search_evidence, tavily_raw_response = _tavily_search_with_raw(search_query or "", max_results=tavily_max_results)
    except Exception as e:
        search_evidence = f"Web search failed: {e}"
        tavily_raw_response = {}

    augmented_prompt = (
        f"{user_prompt}\n\n"
        f"--- Web search evidence (from internet search) ---\n"
        f"{search_evidence[:4000]}\n"
        f"--- End of web search evidence ---\n\n"
        f"Use the search evidence above to help verify factuality. "
        f"If the evidence supports the claim, mark it factual. "
        f"If the evidence contradicts it, mark it not factual."
    )

    text = _call_bedrock_api(
        model=model,
        system_prompt=system_prompt,
        user_prompt=augmented_prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
    )

    metadata: Dict[str, Any] = {
        "grounding_metadata": {
            "web_search_queries": [search_query],
            "tavily_raw_response": tavily_raw_response,
        }
    }
    return text, metadata
