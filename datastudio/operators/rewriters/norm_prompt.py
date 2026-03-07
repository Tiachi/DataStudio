"""Normalize hint/instruction prompt formatting rewriter."""

from typing import Any, List, Optional, Tuple

from datastudio.utils.registry import OPERATORS

from ..core import DataItem, Result, Rewriter

# Common hint/instruction patterns appended or prepended to the question.
# These prompts guide the model's output format (e.g. "Answer: xxx", "step by step"),
# which interfere with MLLM answer rewriting. Removing them from the question before
# rewriting allows the model to generate cleaner, higher-quality answers.
HEAD_TAIL_PROMPT = [
    "请先给出解析，最后再给出答案。",
    "Describe this image in detail with GPT-T-COCO format.",
    "Solve the problem step-by-step before stating the result.",
    "Break down the problem and analyze it carefully before giving the answer. End your response with Answer: xxx at the end if possible.",
    "Please think step by step before answering.",
    "Conclude with your final answer in the format: Answer: xxx at the end if possible.",
    "Describe the visual elements first, then reason through them, and finally answer.",
    "Think out loud before giving the final conclusion.",
    "Let's think step by step and conclude your answer as Answer: xxx at the end if possible.",
    "Deliberate carefully through reasoning, then state your final response in the format: Answer: xxx at the end if possible.",
    "Your final answer should be in the form: Answer: xxx at the end if possible.",
    "Let's think step by step, answer the question, and provide the final answer at the end in the format: Answer: xxx at the end if possible.",
    "请用一个词组或一个短语回答问题。",
    "Please answer the question based on the options mentioned before.",
    "Answer the question with GPT-T-COCO format.",
    "Answer yes or no.",
    "Ensure brevity in your answer.",
    "Your answer should be very brief.",
    "Keep it short and to the point.",
    "Give a very brief answer.",
    "Your answer should be compact.",
    "Offer a terse response.",
    "Provide a succinct answer.",
    "Provide a short and direct response.",
    "Make the answer very short.",
    "Straight to the answer.",
    "Provide the answer directly.",
    "Answer only, no process needed.",
    "Directly provide the answer.",
    "Simply give me the answer, bypass the process.",
    "I want the answer only, no need to explain how.",
    "Just give me the answer, no need for the process.",
    "Answer with detailed steps.",
    "Offer a very short reply.",
    "According to the question shown in the image, please first conduct reasoning, and then answer the question and provide the final value, e.g., The answer is xxx",
    "According to the question shown in the image, please first perform reasoning, then finally select the right answer from the choices, e.g., Answer: xxx.",
    # old add
    "Provide a single-word or phrase answer.",
    "Reply using only a word or a phrase.",
    "Answer the question using a single word or phrase.",
    "Answer concisely with one word or phrase.",
    "Give your answer in one word or a phrase.",
    "Respond with just one word or a short phrase.",
    "Answer with the given options directly.",
    "Answer with the option's letter from the given choices directly.",
    "Please answer with a single word.",
    "Answer the question with Yes or No.",
    "Answer the question with a short phrase.",
    "Based on the image, directly select the correct answer for the following question: ",
    "Answer the mathematical geometry problem and directly provide the correct option letter.",
    "Answer the question by selecting only one option from the given options.",
    "Please try to answer the question with short words or phrases if possible.",
    "Please answer yes or no.",
    "Please select the correct answer from the options above.",
    "Please conclude your answer as Answer: xxx at the end if possible.",
    "Please provide your answer by stating the letter followed by the full option.",
    "Answer the question with a single word (or phrase).",
    "Please answer the question and provide the final answer at the end.",
    "Please answer the question and provide the final answer at the end.",
    "Answer with the given letter directly",
    "Answer the question with Yes or No.",
    "Provide the requested information directly.",
    "Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.",
    "Answer with the option's letter from the given choices directly.",
    "Ensure brevity in your answer.",
    "Give a very brief answer.",
    "Keep it short and to the point.",
    "Make the answer very short.",
    "Offer a terse response.",
    "Offer a very short reply.",
    "Provide a short and direct response.",
    "Provide a succinct answer.",
    "Your answer should be compact.",
    "Your answer should be very brief.",
    "Your response must be concise.",
]


@OPERATORS.register_module()
class NormPromptRewriter(Rewriter):
    """Normalize hint/instruction prompt formatting.

    Ensures hints like ``"Answer concisely"`` are properly separated
    with newlines from the main question text.
    """

    def __init__(
        self,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the rewriter.

        Args:
            logger: Logger instance.
        """
        super().__init__(logger=logger, **kwargs)

        if logger:
            logger.info(f"[{self.name}] Initialized")

    def _detect_patterns(self, text: str) -> List[Tuple[int, int]]:
        """Detect hint patterns and their positions.

        Returns:
            List of (start_pos, end_pos) tuples for found patterns,
            sorted by position with overlapping patterns deduplicated.
        """
        found_patterns = []
        for pattern in HEAD_TAIL_PROMPT:
            start_pos = text.find(pattern)
            if start_pos != -1:
                end_pos = start_pos + len(pattern)
                found_patterns.append((pattern, start_pos, end_pos))

        # Sort by position
        patterns = sorted(found_patterns, key=lambda x: x[1])

        # Remove duplicates and keep longer patterns
        final_patterns = []
        for i, (pattern_text, start_pos, end_pos) in enumerate(patterns):
            should_keep = True
            for j, (other_pattern, other_start, other_end) in enumerate(patterns):
                if i != j:
                    # Check if current is contained in another
                    if other_start <= start_pos and end_pos <= other_end:
                        if other_start != start_pos or other_end != end_pos:
                            should_keep = False
                            break
            if should_keep:
                final_patterns.append((start_pos, end_pos))

        return sorted(set(final_patterns), key=lambda x: x[0])

    def _fix_hint_format(self, text: str) -> Tuple[str, bool]:
        """
        Fix hint formatting by adding proper newlines.

        Returns:
            Tuple of (fixed_text, was_fixed).
        """
        patterns = self._detect_patterns(text)
        if not patterns:
            return text, False

        is_fixed = False
        result = text

        # Process from back to front to preserve positions
        for start_pos, end_pos in reversed(patterns):
            # Check if newline needed before pattern
            text_before = result[:start_pos].strip()
            meaningful_before = text_before.replace("<image>", "").strip()

            if meaningful_before:
                char_before = result[start_pos - 1] if start_pos > 0 else ""
                if char_before not in ["\n", "\r"]:
                    result = result[:start_pos] + "\n" + result[start_pos:]
                    is_fixed = True
                    end_pos += 1

            # Check if newline needed after pattern
            if end_pos < len(result):
                text_after = result[end_pos:].strip()
                meaningful_after = text_after.replace("<image>", "").strip()

                if meaningful_after:
                    char_after = result[end_pos]
                    if char_after not in ["\n", "\r"]:
                        result = result[:end_pos] + "\n" + result[end_pos:]
                        is_fixed = True

        return result.strip(), is_fixed

    def process(self, item: DataItem) -> Result:
        """Normalize prompt formatting in questions (modifies question)."""
        result = Result(item_idx=item.idx)

        for qa in item.qa_pairs:
            fixed_text, was_fixed = self._fix_hint_format(qa.question)
            if was_fixed:
                result.add_rewrite(
                    qa_idx=qa.idx,
                    new_question=fixed_text,
                    message="normalized hint/instruction format",
                )

        return result

    def rewrite(self, item: DataItem, qa_idx: int) -> Optional[str]:
        """Not used - this rewriter modifies questions."""
        return None
