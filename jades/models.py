"""
Data models for JADES framework.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class JailbreakLabel(Enum):
    """Jailbreak evaluation labels."""
    FAILED = "failed"
    PARTIALLY_SUCCESSFUL = "partially_successful"
    SUCCESSFUL = "successful"


@dataclass
class SubQuestion:
    """Represents a sub-question with its weight."""
    question: str
    weight: float
    index: int = 0

    def __post_init__(self):
        """Validate sub-question data."""
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")


@dataclass
class SubAnswer:
    """Represents a sub-answer (selected sentences from cleaned response)."""
    sentences: List[str]
    sub_question_index: int
    reason: str = ""


@dataclass
class SubScore:
    """Represents a score for a sub-question."""
    sub_question_index: int
    score: float
    reason: str = ""
    contribution: float = 0.0  # weight * score


@dataclass
class CleanedResponse:
    """Represents a cleaned jailbreak response."""
    sentences: List[str]
    original_response: str
    removed_sentences: List[str] = field(default_factory=list)


@dataclass
class FactCheckResult:
    """Represents fact-checking results."""
    fact: str
    verdict: str  # "Right", "Wrong", "Unknown"
    evidence: str = ""


@dataclass
class JADESResult:
    """Final JADES evaluation result."""
    question: str
    response: str
    sub_questions: List[SubQuestion]
    cleaned_response: CleanedResponse
    sub_answers: List[SubAnswer]
    sub_scores: List[SubScore]
    total_score: float
    binary_label: bool  # True = successful, False = failed
    ternary_label: JailbreakLabel
    fact_check_results: Optional[List[FactCheckResult]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "question": self.question,
            "response": self.response,
            "sub_questions": [
                {"question": sq.question, "weight": sq.weight, "index": sq.index}
                for sq in self.sub_questions
            ],
            "cleaned_sentences": self.cleaned_response.sentences,
            "sub_scores": [
                {
                    "index": ss.sub_question_index,
                    "score": ss.score,
                    "contribution": ss.contribution,
                    "reason": ss.reason,
                }
                for ss in self.sub_scores
            ],
            "total_score": self.total_score,
            "binary_label": self.binary_label,
            "ternary_label": self.ternary_label.value,
            "fact_check_results": [
                {"fact": fc.fact, "verdict": fc.verdict, "evidence": fc.evidence}
                for fc in (self.fact_check_results or [])
            ],
            "metadata": self.metadata,
        }

    def to_audit_trail(self) -> str:
        """Generate human-readable audit trail."""
        lines = ["=" * 80]
        lines.append("JADES EVALUATION AUDIT TRAIL")
        lines.append("=" * 80)
        lines.append(f"\nOriginal Question:\n{self.question}\n")
        lines.append(f"Original Response:\n{self.response[:200]}...\n")

        lines.append("\n" + "-" * 80)
        lines.append("SUB-QUESTIONS & WEIGHTS:")
        lines.append("-" * 80)
        for sq in self.sub_questions:
            lines.append(f"[{sq.index}] Weight: {sq.weight:.2f} | {sq.question}")

        lines.append("\n" + "-" * 80)
        lines.append("CLEANED RESPONSE SENTENCES:")
        lines.append("-" * 80)
        for i, sent in enumerate(self.cleaned_response.sentences, 1):
            lines.append(f"[{i}] {sent}")

        lines.append("\n" + "-" * 80)
        lines.append("SUB-SCORES:")
        lines.append("-" * 80)
        for ss in self.sub_scores:
            sq = self.sub_questions[ss.sub_question_index]
            lines.append(f"\nSub-Question [{ss.sub_question_index}]: {sq.question}")
            lines.append(f"Score: {ss.score:.2f} | Weight: {sq.weight:.2f} | Contribution: {ss.contribution:.4f}")
            lines.append(f"Reason: {ss.reason}")

        lines.append("\n" + "=" * 80)
        lines.append(f"TOTAL SCORE: {self.total_score:.4f}")
        lines.append(f"BINARY LABEL: {'SUCCESSFUL' if self.binary_label else 'FAILED'}")
        lines.append(f"TERNARY LABEL: {self.ternary_label.value.upper()}")
        lines.append("=" * 80)

        return "\n".join(lines)
