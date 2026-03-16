"""Centralized LLM configuration for the OURS evaluation pipeline."""
from __future__ import annotations

# - Default LLM configuration -
# Change these values to switch the default model/params for ALL nodes.
#
# Thinking: Gemini 3 Flash/Pro defaults to HIGH thinking (API default).
#   No thinking_config needed - just don't pass it.
#   For Gemini 2.5, nodes still have thinking_budget param (default 10000).

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 65536   # API maximum for Gemini 3 Flash Preview
