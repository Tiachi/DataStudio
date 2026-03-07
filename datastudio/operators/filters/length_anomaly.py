"""Text length anomaly filter."""

from typing import Any, Optional, Tuple

from datastudio.utils.registry import OPERATORS

from ..core import DataItem, Filter


@OPERATORS.register_module()
class LengthAnomalyFilter(Filter):
    """
    Filter data items based on text length anomalies.

    Detects and filters QA pairs with abnormally short or long text.
    Supports per-QA-pair filtering (partial filtering).
    """

    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 8192,
        check_question: bool = False,
        check_answer: bool = False,
        check_ori_answer: bool = False,
        use_tokenizer: bool = False,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the filter.

        Args:
            min_length: Minimum allowed length.
            max_length: Maximum allowed length.
            check_question: Whether to check question length.
            check_answer: Whether to check answer length.
            check_ori_answer: Whether to check original answer length.
            use_tokenizer: If True, use tiktoken for length calculation.
            logger: Logger instance.
        """
        super().__init__(logger=logger, **kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.check_question = check_question
        self.check_answer = check_answer
        self.check_ori_answer = check_ori_answer
        self.use_tokenizer = use_tokenizer
        self._tokenizer = None

        if use_tokenizer:
            import tiktoken

            self._tokenizer = tiktoken.encoding_for_model("gpt-4o")

        if logger:
            logger.info(
                f"[{self.name}] Initialized: range=[{min_length}, {max_length}], "
                f"tokenizer={use_tokenizer}"
            )

    def _get_length(self, text: str) -> int:
        """Get text length (character or token count)."""
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text, disallowed_special=()))
            except Exception:
                return len(text)
        return len(text)

    def _is_anomaly(self, length: int) -> bool:
        """Check if length is out of bounds."""
        return length < self.min_length or length > self.max_length

    def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
        """Check for text length anomalies in a QA pair."""
        qa = item.get_qa(qa_idx)
        ori_answer = item.data.get("ori_answer", {})

        # Check question
        if self.check_question:
            length = self._get_length(qa.question)
            if self._is_anomaly(length):
                return (
                    True,
                    f"Question length {length} out of range [{self.min_length}, {self.max_length}]",
                )

        # Check answer
        if self.check_answer:
            length = self._get_length(qa.answer)
            if self._is_anomaly(length):
                return (
                    True,
                    f"Answer length {length} out of range [{self.min_length}, {self.max_length}]",
                )

        # Check original answer
        if self.check_ori_answer:
            ori_text = ori_answer.get(str(qa_idx), "")
            if ori_text:
                length = self._get_length(ori_text)
                if self._is_anomaly(length):
                    return True, f"Original answer length {length} out of range"

        return False, ""
