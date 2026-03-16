"""
Per-node logging schema and thread-safe JSONL writer for SEAL pipeline.

This module provides a NodeLogWriter that writes one JSON object per sample
to a JSONL file, with nested placeholders for Node1-Node4 outputs.

Design:
  - Writer is NOT thread-safe for concurrent writes. Call append_sample()
    from the main thread only, once per sample.
  - File is opened in append mode with per-write flush for durability.
  - Secrets (API keys, tokens) are recursively redacted from metadata blobs.
  - No truncation: all strings are preserved as-is.

Usage:
    from seav.pipeline_logger import NodeLogWriter
    
    writer = NodeLogWriter("node_logs.jsonl")
    writer.append_sample({
        "row_id": "sample_123",
        "node1_extraction": {...},
        "node2_step_verification": {...},
        "node3_order_verification": {...},
        "node4_final_judgment": {...},
    })
    writer.close()
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


# - Secret redaction patterns -

SECRET_KEY_PATTERNS = {
    "api_key",
    "apikey",
    "authorization",
    "openai_api_key",
    "tavily_api_key",
    "anthropic_api_key",
    "openrouter_api_key",
    "gemini_api_key",
    "together_api_key",
}

# Regex for token-like strings (simple heuristic)
TOKEN_PATTERNS = [
    r"^tvly-[a-zA-Z0-9_-]+$",  # Tavily tokens
    r"^sk-[a-zA-Z0-9_-]{20,}$",  # OpenAI-like tokens
    r"^Bearer\s+[a-zA-Z0-9_.-]+$",  # Bearer tokens
]


def _is_secret_key(key: str) -> bool:
    """Check if a key name matches known secret patterns (case-insensitive)."""
    return key.lower() in SECRET_KEY_PATTERNS


def _is_secret_value(value: str) -> bool:
    """Check if a string value looks like a secret token."""
    if not isinstance(value, str):
        return False
    for pattern in TOKEN_PATTERNS:
        if re.match(pattern, value):
            return True
    return False


def redact_secrets(obj: Any) -> Any:
    """
    Recursively redact obvious secrets from a data structure.
    
    Replaces values for keys matching SECRET_KEY_PATTERNS and strings
    matching TOKEN_PATTERNS with "[REDACTED]".
    
    Args:
        obj: Any JSON-serializable object (dict, list, str, etc.)
    
    Returns:
        A deep copy of obj with secrets replaced.
    """
    if isinstance(obj, dict):
        return {
            key: (
                "[REDACTED]"
                if _is_secret_key(key)
                else redact_secrets(value)
            )
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [redact_secrets(item) for item in obj]
    elif isinstance(obj, str):
        return "[REDACTED]" if _is_secret_value(obj) else obj
    else:
        return obj


class NodeLogWriter:
    """
    Thread-safe-by-design JSONL writer for per-node pipeline logs.
    
    Writes one JSON object per sample to a JSONL file. Each object contains
    nested placeholders for Node1-Node4 outputs plus metadata.
    
    IMPORTANT: This writer is NOT thread-safe for concurrent writes.
    Call append_sample() from the main thread only, once per sample.
    
    File size guardrail: Warns (prints) when the JSONL file grows beyond
    200MB, but does NOT truncate. Warning is issued at most once per writer
    instance.
    """

    # File size threshold for guardrail warning (200 MB)
    FILE_SIZE_THRESHOLD_BYTES = 200 * 1024 * 1024

    def __init__(self, filepath: str | Path) -> None:
        """
        Initialize the JSONL writer.
        
        Args:
            filepath: Path to the output JSONL file (will be created or appended).
        """
        self.filepath = Path(filepath)
        self.file = open(self.filepath, "a", encoding="utf-8")
        self._size_warning_issued = False  # Track if warning has been issued

    def append_sample(self, sample: Dict[str, Any]) -> None:
        """
        Append one sample (JSON object) to the JSONL file.
        
        The sample should contain:
          - row_id: unique identifier for the sample
          - node1_extraction: output from Node 1 (step extraction)
          - node2_step_verification: output from Node 2 (step verification)
          - node3_order_verification: output from Node 3 (order verification)
          - node4_final_judgment: output from Node 4 (final judgment)
        
        Secrets in metadata blobs are recursively redacted before writing.
        
        Args:
            sample: Dictionary with sample data and node outputs.
        """
        # Redact secrets from the entire sample
        redacted_sample = redact_secrets(sample)
        
        # Write as pretty-printed JSON (human-readable) followed by a record separator
        json_block = json.dumps(redacted_sample, ensure_ascii=False, indent=2)
        self.file.write(json_block + "\n\n")
        self.file.flush()
        
        # Check file size and warn if threshold exceeded (once per instance)
        self._check_file_size_guardrail()

    def _check_file_size_guardrail(self) -> None:
        """
        Check if file size exceeds threshold and warn once per instance.
        
        Does NOT truncate or rotate the file; only warns to stdout.
        """
        if self._size_warning_issued:
            return
        
        try:
            file_size = self.filepath.stat().st_size
            if file_size > self.FILE_SIZE_THRESHOLD_BYTES:
                size_mb = file_size / (1024 * 1024)
                print(
                    f"[GUARDRAIL] JSONL file {self.filepath} has grown to {size_mb:.1f} MB "
                    f"(threshold: 200 MB). Consider archiving or splitting logs."
                )
                self._size_warning_issued = True
        except OSError:
            pass

    def close(self) -> None:
        """Close the JSONL file."""
        if self.file and not self.file.closed:
            self.file.close()

    def __enter__(self) -> NodeLogWriter:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
