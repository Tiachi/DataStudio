"""Base MLLM operator for model-powered filtering and rewriting."""

import gc
import json
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from datastudio.utils.registry import REQUESTS, build_from_cfg

from ..core.data_item import DataItem
from ..core.operator import Operator
from ..core.result import Result


class MLLMOperator(Operator):
    """Base class for MLLM-powered operators.

    Builds requests via :class:`RequestBuilder`, calls ``model.generate()``,
    and delegates response parsing to subclasses via :meth:`_add_decision`.
    """

    def __init__(
        self,
        model: Any,
        request_builder: Optional[Dict] = None,
        batch_qa: bool = False,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the MLLM operator.

        Args:
            model: MLLM model with generate() method.
            request_builder: RequestBuilder config dict.
            batch_qa: If True, combine all QA pairs into one request.
                      Model should return {q0: ..., q1: ..., ...}.
                      If False (default), each QA pair is a separate request.
            logger: Logger instance.
        """
        super().__init__(logger=logger, **kwargs)

        if model is None:
            raise ValueError("model is required")

        self.model = model
        self.batch_qa = batch_qa

        # Build RequestBuilder from config
        self.request_builder = build_from_cfg(request_builder, REQUESTS)
        if self.request_builder is None:
            raise ValueError("request_builder is required and must be a valid config")

    # === Main processing ===

    def process(self, item: DataItem) -> Result:
        """Process a single item (delegates to process_batch)."""
        return self.process_batch([item])[0]

    def process_batch(self, items: List[DataItem]) -> List[Result]:
        """
        Process a batch of items.

        Flow:
        1. Build requests using RequestBuilder
        2. Execute model
        3. Parse responses and aggregate into Results
        """
        # Initialize results for all items
        results: Dict[int, Result] = {
            item.idx: Result(item_idx=item.idx) for item in items
        }

        # Step 1: Build requests
        all_requests = self._build_requests(items)

        if not all_requests:
            return [results[item.idx] for item in items]

        # Step 2: Execute model
        self._log_example_request(all_requests)
        payloads = [r["payload"] for r in all_requests]
        responses = self.model.generate(payloads)
        self._log_example_response(responses)

        self._cleanup_requests(all_requests, payloads)

        # Step 3: Parse responses and add decisions
        # Track which items were actually processed by the model
        processed_item_idxs = set()

        for req, resp in zip(all_requests, responses):
            item: DataItem = req["item"]
            qa_idx: int = req["qa_idx"]
            processed_item_idxs.add(item.idx)

            # Handle batch_qa mode vs single mode
            if qa_idx == -1:
                # Batch mode: parse all QA pairs from response
                for i in range(item.qa_count):
                    parsed = self.request_builder.parse_response(resp, i, self.logger)
                    self._add_decision(results[item.idx], i, parsed)
            else:
                # Single mode: response has q0, maps to actual qa_idx
                parsed = self.request_builder.parse_response(resp, 0, self.logger)
                self._add_decision(results[item.idx], qa_idx, parsed)

        # Step 4: Record which model processed each item
        model_name = getattr(self.model, "model", None)
        if model_name:
            for item in items:
                if item.idx in processed_item_idxs:
                    item.add_model_record(self.name, model_name)

        return [results[item.idx] for item in items]

    # === Build requests ===

    def _build_requests(self, items: List[DataItem]) -> List[Dict]:
        """Build all requests for all items."""
        all_requests = []

        for item in items:
            if self._should_skip(item):
                continue

            requests = self.request_builder.build_requests(item, self.batch_qa)
            all_requests.extend(requests)

        return all_requests

    def _should_skip(self, item: DataItem) -> bool:
        """Check if this item should be skipped."""
        return self.request_builder.with_image and not item.has_image

    @staticmethod
    def _cleanup_requests(all_requests: List[Dict], payloads: List[Any]) -> None:
        del payloads
        for req in all_requests:
            req.pop("payload", None)
        gc.collect()

    # === Abstract method for subclasses ===

    @abstractmethod
    def _add_decision(self, result: Result, qa_idx: int, parsed: Dict):
        """
        Add decision based on parsed response.

        Subclasses implement this to add FilterDecision or RewriteDecision.

        Args:
            result: Result to add decision to.
            qa_idx: QA pair index.
            parsed: Parsed response dict with "result", "reason", etc.
                    May contain "parse_error": True if parsing failed.
        """
        pass

    # === Logging ===

    def _log_example_request(self, requests: List[Dict]):
        """Log a sample request for debugging."""
        if not self.logger or not requests:
            return

        # Pick middle request
        idx = len(requests) // 2
        request = requests[idx]
        payload = request["payload"]

        # Make loggable (replace image with placeholder)
        loggable = []
        for item in payload:
            if item.get("type") == "image" and item.get("value") is not None:
                value = item["value"]
                if isinstance(value, list):
                    placeholder = f"<List of {len(value)} PIL.Image>"
                else:
                    placeholder = "<PIL.Image>"
                loggable.append({"type": "image", "value": placeholder})
            else:
                loggable.append(item.copy())

        self.logger.info(
            f"\n[{self.name}] Example request:\n"
            f"{json.dumps(loggable, indent=2, ensure_ascii=False)}\n"
        )

    def _log_example_response(self, responses: List[Any]):
        """Log a sample response for debugging."""
        if not self.logger or not responses:
            return

        idx = len(responses) // 2
        response = responses[idx]

        try:
            if isinstance(response, str):
                response = json.loads(response)
            if isinstance(response, dict):
                self.logger.info(
                    f"\n[{self.name}] Example response:\n"
                    f"{json.dumps(response, indent=2, ensure_ascii=False)}\n"
                )
            else:
                self.logger.info(f"\n[{self.name}] Example response:\n{response}\n")
        except (json.JSONDecodeError, TypeError):
            self.logger.warning(
                f"\n[{self.name}] Example response (not JSON):\n{response}\n"
            )
