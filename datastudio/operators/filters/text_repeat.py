"""Text repetition filter."""

from typing import Any, Optional, Tuple

from datastudio.utils.registry import OPERATORS
from datastudio.utils.repetition import detect_consecutive_repetition

from ..core import DataItem, Filter


@OPERATORS.register_module()
class TextRepeatFilter(Filter):
    """
    Filter data items with repetitive text patterns.

    Detects and filters QA pairs containing consecutive repetitive patterns.
    Supports per-QA-pair filtering (partial filtering).
    """

    def __init__(
        self,
        check_question: bool = True,
        check_answer: bool = True,
        check_ori_answer: bool = True,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the filter.

        Args:
            check_question: Whether to check questions.
            check_answer: Whether to check answers.
            check_ori_answer: Whether to check original answers.
            logger: Logger instance.
        """
        super().__init__(logger=logger, **kwargs)
        self.check_question = check_question
        self.check_answer = check_answer
        self.check_ori_answer = check_ori_answer

        if logger:
            logger.info(f"[{self.name}] Initialized")

    def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
        """Check for repetitive patterns in a QA pair."""
        qa = item.get_qa(qa_idx)
        ori_answer = item.data.get("ori_answer", {})

        # Check question
        if self.check_question:
            has_repeat, match = detect_consecutive_repetition(qa.question)
            if has_repeat:
                return True, f"Repetition in question: {match}"

        # Check answer
        if self.check_answer:
            has_repeat, match = detect_consecutive_repetition(qa.answer)
            if has_repeat:
                return True, f"Repetition in answer: {match}"

        # Check original answer
        if self.check_ori_answer:
            ori_text = ori_answer.get(str(qa_idx), "")
            if ori_text:
                has_repeat, match = detect_consecutive_repetition(ori_text)
                if has_repeat:
                    return True, f"Repetition in original answer: {match}"

        return False, ""
