"""
JADES: Jailbreak Assessment via Decompositional Scoring

A universal framework for evaluating jailbreak attacks on Large Language Models.

Based on the paper:
"JADES: A Universal Framework for Jailbreak Assessment via Decompositional Scoring"
by Chu et al., 2025
"""

__version__ = "0.1.0"

from .core import JADES
from .config import JADESConfig
from .models import (
    JADESResult,
    JailbreakLabel,
    SubQuestion,
    SubAnswer,
    SubScore,
    CleanedResponse,
    FactCheckResult,
)
from .nodes import (
    QuestionDecompositionNode,
    AnswerDecompositionNode,
    CleanNode,
    PairingNode,
    EvaluationNode,
    AnswerPartEvaluationNode,
)
from .fact_checking import FactCheckingNode

__all__ = [
    "JADES",
    "JADESConfig",
    "JADESResult",
    "JailbreakLabel",
    "SubQuestion",
    "SubAnswer",
    "SubScore",
    "CleanedResponse",
    "FactCheckResult",
    "QuestionDecompositionNode",
    "AnswerDecompositionNode",
    "CleanNode",
    "PairingNode",
    "EvaluationNode",
    "AnswerPartEvaluationNode",
    "FactCheckingNode",
]
