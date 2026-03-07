"""MLLM-based filter operator."""

from typing import Any, Dict, Optional

from datastudio.utils.registry import OPERATORS

from ..core.result import Result
from .base import MLLMOperator


@OPERATORS.register_module()
class MLLMFilter(MLLMOperator):
    """Filter data items using MLLM quality assessment.

    Expected response format: ``{"q0": true/false, "q0_reason": "..."}``
    where ``true`` means reject and ``false`` means keep.

    Example config::

        request_builder = dict(
            type="RequestBuilder",
            prompt="prompts/filter/xxx.txt",
            key_templates={"result": "q{idx}", "reason": "q{idx}_reason"},
            with_image=True,
        )
    """

    def __init__(
        self,
        model: Any,
        request_builder: Optional[Dict] = None,
        batch_qa: bool = False,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            request_builder=request_builder,
            batch_qa=batch_qa,
            logger=logger,
            **kwargs,
        )

        if logger and self.request_builder:
            logger.info(
                f"[{self.name}] Initialized "
                f"(with_image={self.request_builder.with_image})"
            )

    def _add_decision(self, result: Result, qa_idx: int, parsed: Dict):
        """
        Add filter decision based on parsed response.

        Args:
            result: Result to add decision to.
            qa_idx: QA pair index.
            parsed: Parsed response with "result" and "reason".
        """
        # Handle parse error - default to not rejecting
        if parsed.get("parse_error"):
            if self.logger:
                self.logger.warning(
                    f"[{self.name}] Parse error for qa_idx={qa_idx}, defaulting to keep"
                )
            result.add_filter(qa_idx, rejected=False, reason="parse_error")
            return

        raw_result = parsed.get("result", False)

        # Handle string "true"/"false" from model output
        if isinstance(raw_result, str):
            rejected = raw_result.lower().strip() in ("true", "yes", "1")
        else:
            rejected = bool(raw_result)

        reason = str(parsed.get("reason", ""))
        result.add_filter(qa_idx, rejected=rejected, reason=reason)
