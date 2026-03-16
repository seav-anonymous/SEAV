"""
JADES Core Framework
Main framework that orchestrates all four nodes for jailbreak assessment.
"""

from typing import Optional, Any, Callable, Dict
import json
from pathlib import Path

from .config import JADESConfig
from .models import JADESResult, JailbreakLabel, SubAnswer, SubQuestion, CleanedResponse
from .nodes import (
    QuestionDecompositionNode,
    AnswerDecompositionNode,
    CleanNode,
    PairingNode,
    EvaluationNode,
    AnswerPartEvaluationNode,
)
from .fact_checking import FactCheckingNode


class JADES:
    """
    JADES: Jailbreak Assessment via Decompositional Scoring

    Main framework that orchestrates the JADES pipeline.

    Variants:
    - "standard" (default): question decomposition -> clean -> pairing -> evaluation
    - "answer_only": clean -> answer decomposition -> answer-part evaluation

    Optionally includes a Fact-Checking extension module.
    """

    def __init__(
        self,
        config: Optional[JADESConfig] = None,
        llm_client: Optional[Any] = None,
        search_tool: Optional[Any] = None,
        custom_decompose_fn: Optional[Callable] = None,
        custom_clean_fn: Optional[Callable] = None,
        custom_pair_fn: Optional[Callable] = None,
        custom_score_fn: Optional[Callable] = None,
    ):
        """
        Initialize the JADES framework.

        Args:
            config: JADES configuration (uses defaults if not provided)
            llm_client: LLM client for all nodes (e.g., OpenAI client)
            search_tool: Search tool for fact-checking (e.g., Wikipedia API)
            custom_decompose_fn: Custom question decomposition function
            custom_clean_fn: Custom response cleaning function
            custom_pair_fn: Custom sub-question pairing function
            custom_score_fn: Custom scoring function
        """
        self.config = config if config is not None else JADESConfig()
        self.llm_client = llm_client
        self.search_tool = search_tool

        # Initialize memory for question decomposition
        self.memory: Dict = {}
        if self.config.memory_enabled and self.config.memory_path:
            self._load_memory()

        # Initialize all nodes
        self.clean_node = CleanNode(
            config=self.config,
            llm_client=llm_client,
            custom_clean_fn=custom_clean_fn,
        )
        self.decomposition_node = None
        self.pairing_node = None
        self.evaluation_node = None
        self.answer_decomposition_node = None
        self.answer_evaluation_node = None

        if self.config.jades_variant == "answer_only":
            self.answer_decomposition_node = AnswerDecompositionNode(
                config=self.config,
                llm_client=llm_client,
                memory=self.memory,
            )
            self.answer_evaluation_node = AnswerPartEvaluationNode(
                config=self.config,
                llm_client=llm_client,
                custom_score_fn=custom_score_fn,
            )
        else:
            self.decomposition_node = QuestionDecompositionNode(
                config=self.config,
                llm_client=llm_client,
                memory=self.memory,
                custom_decompose_fn=custom_decompose_fn,
            )
            self.pairing_node = PairingNode(
                config=self.config,
                llm_client=llm_client,
                custom_pair_fn=custom_pair_fn,
            )
            self.evaluation_node = EvaluationNode(
                config=self.config,
                llm_client=llm_client,
                custom_score_fn=custom_score_fn,
            )

        # Initialize optional fact-checking node
        self.fact_checking_node = None
        if self.config.enable_fact_checking:
            self.fact_checking_node = FactCheckingNode(
                config=self.config,
                llm_client=llm_client,
                search_tool=search_tool,
            )

    def evaluate(
        self,
        question: str,
        response: str,
        return_details: bool = True,
    ) -> JADESResult:
        """
        Evaluate a jailbreak attempt.

        Args:
            question: The harmful question
            response: The jailbreak response to evaluate
            return_details: Whether to return detailed results

        Returns:
            JADESResult object containing the full evaluation
        """
        if self.config.verbose:
            print("=" * 80)
            print("Starting JADES Evaluation")
            print("=" * 80)

        sub_questions: list[SubQuestion]
        sub_answers: list[SubAnswer]

        if self.config.jades_variant != "answer_only":
            if self.decomposition_node is None:
                raise RuntimeError("Standard nodes not initialized (missing decomposition_node)")

            # Node 1: Question Decomposition
            if self.config.verbose:
                print("\n[1/4] Decomposing question into sub-questions...")

            sub_questions = self.decomposition_node.decompose(question)

            if self.config.verbose:
                print(f"Generated {len(sub_questions)} sub-questions")
                for sq in sub_questions:
                    print(f"  - [{sq.index}] (w={sq.weight:.2f}) {sq.question}")

        # Node 1 (answer_only) / Node 2 (standard): Clean
        if self.config.verbose:
            step = "1/?" if self.config.jades_variant == "answer_only" else "2/4"
            if self.config.disable_clean_node:
                print(f"\n[{step}] Skipping clean node (using full response)...")
            else:
                print(f"\n[{step}] Cleaning response...")

        if self.config.disable_clean_node:
            # Keep all sentences so downstream nodes see the full response.
            sentences = self.clean_node._segment_into_sentences(response)
            cleaned_response = CleanedResponse(
                sentences=sentences,
                original_response=response,
                removed_sentences=[],
            )
        else:
            cleaned_response = self.clean_node.clean(response, question)

        if self.config.verbose:
            print(f"Kept {len(cleaned_response.sentences)} sentences, "
                  f"removed {len(cleaned_response.removed_sentences)} sentences")
            if cleaned_response.sentences:
                print("\n  Kept sentences:")
                for i, sent in enumerate(cleaned_response.sentences, 1):
                    print(f"    {i}. {sent}")

        if self.config.jades_variant == "answer_only":
            if self.answer_decomposition_node is None or self.answer_evaluation_node is None:
                raise RuntimeError("Answer-only nodes not initialized")

            # Node 2: Answer decomposition (response -> weighted parts)
            if self.config.verbose:
                print("\n[2/?] Decomposing answer into weighted parts...")

            if self.config.disable_clean_node:
                response_for_decomp = response
            else:
                response_for_decomp = (
                    "\n".join(cleaned_response.sentences) if cleaned_response.sentences else cleaned_response.original_response
                )
            parts = self.answer_decomposition_node.decompose(question=question, response=response_for_decomp)

            sub_questions = []
            sub_answers = []
            for i, part in enumerate(parts):
                text = str(part.get("text") or "").strip()
                weight = float(part.get("weight") or 0.0)
                reason = str(part.get("reason") or "").strip()
                sub_questions.append(SubQuestion(question=text, weight=weight, index=i))
                sub_answers.append(SubAnswer(sentences=[text] if text else [], sub_question_index=i, reason=reason))

            if self.config.verbose:
                print(f"Generated {len(sub_questions)} answer parts")
                for sq in sub_questions:
                    preview = (sq.question[:120] + "...") if len(sq.question) > 120 else sq.question
                    print(f"  - [{sq.index}] (w={sq.weight:.2f}) {preview}")
        else:
            if self.decomposition_node is None or self.pairing_node is None or self.evaluation_node is None:
                raise RuntimeError("Standard nodes not initialized")

            # Node 3: Sub-Question Pairing
            if self.config.verbose:
                print("\n[3/4] Pairing sub-questions with response sentences...")

            sub_answers = self.pairing_node.pair(sub_questions, cleaned_response)

            if self.config.verbose:
                for sa in sub_answers:
                    sq = sub_questions[sa.sub_question_index]
                    print(f"\n  Sub-question [{sa.sub_question_index}]: {sq.question}")
                    print(f"    Matched {len(sa.sentences)} sentences:")
                    if sa.sentences:
                        for i, sent in enumerate(sa.sentences, 1):
                            print(f"      {i}. {sent}")
                    else:
                        print(f"      (no sentences matched)")

        # Optional: Fact-Checking
        fact_check_results = None
        if self.fact_checking_node is not None:
            if self.config.verbose:
                print("\n[Extension] Running fact-checking...")

            fact_check_results = self.fact_checking_node.check_facts(
                cleaned_response, question
            )

            if self.config.verbose:
                print(f"\nChecked {len(fact_check_results)} facts:")
                for i, fc in enumerate(fact_check_results, 1):
                    print(f"\n  [{i}] {fc.verdict}: {fc.fact}")
                    if fc.evidence:
                        print(f"      Evidence: {fc.evidence}")

        # Node 4: Evaluation
        if self.config.verbose:
            print("\n[4/?] Scoring and aggregating...")

        if self.config.jades_variant == "answer_only":
            sub_scores, total_score = self.answer_evaluation_node.evaluate(
                question, sub_questions, sub_answers, fact_check_results
            )
            binary_label = self.answer_evaluation_node.get_binary_label(total_score)
            ternary_label = self.answer_evaluation_node.get_ternary_label(total_score)
        else:
            sub_scores, total_score = self.evaluation_node.evaluate(
                sub_questions, sub_answers, fact_check_results
            )
            binary_label = self.evaluation_node.get_binary_label(total_score)
            ternary_label = self.evaluation_node.get_ternary_label(total_score)

        if self.config.verbose:
            print(f"\nTotal Score: {total_score:.4f}")
            for ss in sub_scores:
                sq = sub_questions[ss.sub_question_index]
                print(f"  - [{ss.sub_question_index}] "
                      f"score={ss.score:.2f}, weight={sq.weight:.2f}, "
                      f"contribution={ss.contribution:.4f}")

        if self.config.verbose:
            print(f"\nBinary Label: {'SUCCESSFUL' if binary_label else 'FAILED'}")
            print(f"Ternary Label: {ternary_label.value.upper()}")
            print("=" * 80)

        # Create result object
        result = JADESResult(
            question=question,
            response=response,
            sub_questions=sub_questions,
            cleaned_response=cleaned_response,
            sub_answers=sub_answers,
            sub_scores=sub_scores,
            total_score=total_score,
            binary_label=binary_label,
            ternary_label=ternary_label,
            fact_check_results=fact_check_results,
            metadata={
                "config": self.config.to_dict(),
            },
        )

        return result

    def evaluate_batch(
        self,
        questions: list[str],
        responses: list[str],
    ) -> list[JADESResult]:
        """
        Evaluate multiple jailbreak attempts.

        Args:
            questions: List of harmful questions
            responses: List of jailbreak responses

        Returns:
            List of JADESResult objects
        """
        if len(questions) != len(responses):
            raise ValueError(
                f"Number of questions ({len(questions)}) must match "
                f"number of responses ({len(responses)})"
            )

        results = []
        for i, (q, r) in enumerate(zip(questions, responses)):
            if self.config.verbose:
                print(f"\n\nEvaluating pair {i + 1}/{len(questions)}")

            result = self.evaluate(q, r)
            results.append(result)

        return results

    def save_memory(self):
        """Save decomposition memory to disk."""
        if not self.config.memory_enabled or not self.config.memory_path:
            return

        memory_path = Path(self.config.memory_path)
        memory_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert SubQuestion objects to dicts for JSON serialization
        serializable_memory = {}
        for key, sub_questions in self.memory.items():
            serializable_memory[key] = [
                {
                    "question": sq.question,
                    "weight": sq.weight,
                    "index": sq.index,
                }
                for sq in sub_questions
            ]

        with open(memory_path, "w") as f:
            json.dump(serializable_memory, f, indent=2)

        if self.config.verbose:
            print(f"Saved memory to {memory_path}")

    def _load_memory(self):
        """Load decomposition memory from disk."""
        if not self.config.memory_path:
            return

        memory_path = Path(self.config.memory_path)
        if not memory_path.exists():
            return

        try:
            with open(memory_path, "r") as f:
                serializable_memory = json.load(f)

            # Convert dicts back to SubQuestion objects
            from .models import SubQuestion

            for key, sub_questions_data in serializable_memory.items():
                self.memory[key] = [
                    SubQuestion(
                        question=sq["question"],
                        weight=sq["weight"],
                        index=sq["index"],
                    )
                    for sq in sub_questions_data
                ]

            if self.config.verbose:
                print(f"Loaded memory from {memory_path}")

        except Exception as e:
            if self.config.verbose:
                print(f"Error loading memory: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save memory if enabled."""
        if self.config.memory_enabled and self.config.memory_path:
            self.save_memory()
