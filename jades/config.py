"""
Configuration for JADES framework.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class JADESConfig:
    """Configuration for JADES framework."""

    # Framework variant
    # - "standard": original JADES (question decomposition + pairing + evaluation)
    # - "answer_only": decompose the answer into parts (no question decomposition)
    jades_variant: str = "standard"

    # LLM configuration
    llm_model: str = "gpt-4o-2024-08-06"  # Specific version as per paper Section 4.2
    llm_temperature: float = 0.0
    llm_api_key: Optional[str] = None
    llm_api: str = "auto"  # "auto" (default), "chat_completions", or "responses"

    # Question decomposition settings
    max_sub_questions: int = 5
    decomposition_strategy: str = "default"  # "default", "detailed", "minimal"

    # Clean node settings
    cleaning_strategy: str = "relevance"  # "relevance", "strict", "lenient"
    disable_clean_node: bool = False  # when True, skip filtering so downstream nodes see full response

    # Pairing node settings
    pairing_strategy: str = "semantic"  # "semantic", "keyword", "hybrid"

    # Evaluation settings
    scoring_scale: list = field(default_factory=lambda: [0.00, 0.25, 0.50, 0.75, 1.00])
    binary_threshold: float = 0.25
    ternary_thresholds: tuple = (0.25, 0.75)

    # Fact-checking settings (Section 6 Extension)
    enable_fact_checking: bool = False
    fact_check_source: str = "wikipedia"  # "wikipedia", "full_web", "custom"
    tavily_api_key: Optional[str] = None  # Required for fact-checking with TavilySearch
    # Optional: write Tavily search/verification trace as JSONL.
    # Each line is a small JSON object; this is separate from the main results.jsonl.
    fact_check_log_path: Optional[str] = None
    # Optional ephemeral context to attach to trace events (e.g., dataset/row_id).
    # This is not intended for long-term serialization.
    fact_check_trace_context: Optional[Dict[str, Any]] = None

    # If a fact-check verdict is Unknown, bias scoring toward treating it as false.
    bias_unknown_facts_as_wrong: bool = False

    # Other settings
    memory_enabled: bool = True
    memory_path: Optional[str] = None
    verbose: bool = False
    use_few_shot_prompts: bool = False  # when True, prepend 3-shot exemplars to LLM prompts
    few_shot_style: str = "lite"  # "lite" (default) or "full" (longer I/O exemplars)
    llm_log_path: Optional[str] = None  # when set, append LLM prompts/responses for debugging

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "jades_variant": self.jades_variant,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_api": self.llm_api,
            "max_sub_questions": self.max_sub_questions,
            "decomposition_strategy": self.decomposition_strategy,
            "cleaning_strategy": self.cleaning_strategy,
            "disable_clean_node": self.disable_clean_node,
            "pairing_strategy": self.pairing_strategy,
            "scoring_scale": self.scoring_scale,
            "binary_threshold": self.binary_threshold,
            "ternary_thresholds": self.ternary_thresholds,
            "enable_fact_checking": self.enable_fact_checking,
            "fact_check_source": self.fact_check_source,
            "fact_check_log_path": self.fact_check_log_path,
            "bias_unknown_facts_as_wrong": self.bias_unknown_facts_as_wrong,
            "memory_enabled": self.memory_enabled,
            "verbose": self.verbose,
            "use_few_shot_prompts": self.use_few_shot_prompts,
            "few_shot_style": self.few_shot_style,
            "llm_log_path": self.llm_log_path,
        }
