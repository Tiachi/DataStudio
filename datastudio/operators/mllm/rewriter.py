"""MLLM-based rewriter operator."""

from typing import Any, Callable, Dict, List, Optional

from datastudio.utils.registry import OPERATORS

from ..core.data_item import DataItem
from ..core.result import Result
from .base import MLLMOperator


@OPERATORS.register_module()
class MLLMRewriter(MLLMOperator):
    """Rewrite data using MLLM.

    Supports both structured (JSON dict) and plain text responses.
    Set ``key_templates=None`` in the request builder for plain text mode.

    Example config::

        request_builder = dict(
            type="RequestBuilder",
            prompt="prompts/rewriter/xxx.txt",
            key_templates={"result": "q{idx}_answer"},
            with_image=True,
            with_answer=True,
        )
    """

    def __init__(
        self,
        model: Any,
        request_builder: Optional[Dict] = None,
        rewrite_type: str = "answer",
        batch_qa: bool = False,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the MLLM rewriter.

        Args:
            model: MLLM model instance.
            request_builder: RequestBuilder config dict.
            rewrite_type: What to rewrite - "answer" or "question".
            batch_qa: If True, combine all QA pairs into one request.
            logger: Logger instance.
        """
        super().__init__(
            model=model,
            request_builder=request_builder,
            batch_qa=batch_qa,
            logger=logger,
            **kwargs,
        )

        self.rewrite_type = rewrite_type

        if logger:
            logger.info(f"[{self.name}] Initialized (rewrite_type={rewrite_type})")

    def _add_decision(self, result: Result, qa_idx: int, parsed: Dict):
        """
        Add rewrite decision based on parsed response.

        Args:
            result: Result to add decision to.
            qa_idx: QA pair index.
            parsed: Parsed response with "result".
        """
        # Handle parse error - skip rewriting
        if parsed.get("parse_error"):
            if self.logger:
                self.logger.warning(
                    f"[{self.name}] Parse error for qa_idx={qa_idx}, skipping rewrite"
                )
            return

        content = parsed.get("result")
        if content is None:
            return

        if self.rewrite_type == "answer":
            result.add_rewrite(qa_idx, new_answer=content, message="rewritten answer")
        else:
            result.add_rewrite(
                qa_idx, new_question=content, message="rewritten question"
            )


@OPERATORS.register_module()
class SelectiveMLLMRewriter(MLLMRewriter):
    """MLLM rewriter that only processes items matching a condition.

    Example::

        rewriter = SelectiveMLLMRewriter(
            model=model,
            request_builder=dict(type="RequestBuilder", prompt="prompts/translate.txt"),
            should_rewrite_fn=is_mixed_language,
        )
    """

    def __init__(
        self,
        should_rewrite_fn: Optional[Callable[[str, str], bool]] = None,
        **kwargs,
    ):
        """
        Initialize selective rewriter.

        Args:
            should_rewrite_fn: Function(question, answer) -> bool.
                               Returns True if this QA should be rewritten.
            **kwargs: Arguments passed to MLLMRewriter.
        """
        super().__init__(**kwargs)
        self.should_rewrite_fn = should_rewrite_fn

    def _build_requests(self, items: List[DataItem]) -> List[Dict]:
        """Build requests with only QA pairs needing rewrite."""
        all_requests = []

        for item in items:
            if self._should_skip(item):
                continue

            # Collect QA indices that need rewriting
            qa_indices = [
                qa.idx
                for qa in item.qa_pairs
                if self._should_rewrite(qa.question, qa.answer)
            ]

            if not qa_indices:
                continue

            # Use RequestBuilder's selective method for proper image handling
            requests = self.request_builder.build_requests_selective(item, qa_indices)
            all_requests.extend(requests)

        return all_requests

    def _should_rewrite(self, question: str, answer: str) -> bool:
        """Check if this QA pair should be rewritten."""
        if self.should_rewrite_fn is None:
            return True
        return self.should_rewrite_fn(question, answer)
