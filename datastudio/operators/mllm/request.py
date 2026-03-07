"""Request builder for MLLM operators."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from datastudio.utils.registry import REQUESTS

if TYPE_CHECKING:
    from ..core.data_item import DataItem


def copy_image(image: Any) -> Any:
    """Return image reference directly (images are pre-encoded to base64)."""
    return image


@REQUESTS.register_module()
class RequestBuilder:
    """
    MLLM request builder.

    Responsibilities:
    1. Store request_builder configuration
    2. Build requests (build_request, build_requests)
    3. Format content (format_content)
    4. Parse model responses (parse_response)

    Config format:
        request_builder = dict(
            type="RequestBuilder",
            prompt="prompts/filter/xxx.txt",
            system_prompt="prompts/grounding/grounding_system.txt",
            key_templates={"result": "q{idx}", "reason": "q{idx}_reason"},
            with_image=True,
            with_answer=False,
        )

    Attributes:
        prompt: Prompt text or path to .txt file (user message).
        system_prompt: System prompt text or path to .txt file (system message).
        key_templates: Dict mapping field names to key templates with {idx}.
        with_image/with_question/with_answer/with_original: Request flags.
    """

    # Class-level defaults (can be overridden by subclass or instance)
    prompt: str = ""
    system_prompt: str = ""
    key_templates: Dict[str, str] = {"result": "q{idx}", "reason": "q{idx}_reason"}
    with_image: bool = True
    with_question: bool = True
    with_answer: bool = False
    with_original: bool = False

    # Sentinel to distinguish "not provided" from "explicitly None"
    _NOT_PROVIDED = object()

    def __init__(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        key_templates: Union[Dict[str, str], None, object] = _NOT_PROVIDED,
        with_image: Optional[bool] = None,
        with_question: Optional[bool] = None,
        with_answer: Optional[bool] = None,
        with_original: Optional[bool] = None,
        **kwargs,
    ):
        """
        Initialize request builder.

        Instance values override class defaults.
        For key_templates, explicitly passing None disables JSON parsing.
        """
        # prompt: None or "" both mean no prompt
        if prompt:
            self.prompt = prompt
        if system_prompt:
            self.system_prompt = system_prompt
        # key_templates: need to distinguish "not provided" vs "explicitly None"
        if key_templates is not self._NOT_PROVIDED:
            self.key_templates = key_templates

        # Boolean flags: only override if provided (not None)
        if with_image is not None:
            self.with_image = with_image
        if with_question is not None:
            self.with_question = with_question
        if with_answer is not None:
            self.with_answer = with_answer
        if with_original is not None:
            self.with_original = with_original

        # Lazy-loaded prompt text
        self._prompt_text: Optional[str] = None
        self._system_prompt_text: Optional[str] = None

    # === Prompt loading ===

    @property
    def prompt_text(self) -> str:
        """Load and return the prompt text (lazy)."""
        if self._prompt_text is None:
            self._prompt_text = self._load_prompt(self.prompt)
        return self._prompt_text

    @property
    def system_prompt_text(self) -> str:
        """Load and return the system prompt text (lazy)."""
        if self._system_prompt_text is None:
            self._system_prompt_text = self._load_prompt(self.system_prompt)
        return self._system_prompt_text

    def _load_prompt(self, prompt: str) -> str:
        """Load prompt from file or return as-is."""
        if not prompt:
            return ""
        if prompt.endswith(".txt"):
            path = Path(prompt)
            if not path.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt}")
            return path.read_text(encoding="utf-8")
        return prompt

    # === Build requests ===

    def build_request(
        self,
        item: "DataItem",
        qa_idx: int,
        copy_img: bool = False,
    ) -> Dict:
        """
        Build a single request.

        Args:
            item: DataItem to build request for.
            qa_idx: QA pair index (-1 for batch_qa mode).
            copy_img: Whether to deep copy the image (for multiple requests per item).

        Returns:
            Dict with keys: payload, item, qa_idx
        """
        content = self.format_content(item, qa_idx)
        image = item.image if self.with_image else None
        if copy_img and image is not None:
            image = copy_image(image)
        payload = self._build_payload(content, image)
        return {
            "payload": payload,
            "item": item,
            "qa_idx": qa_idx,
        }

    def build_requests(
        self,
        item: "DataItem",
        batch_qa: bool = False,
    ) -> List[Dict]:
        """
        Build all requests for a DataItem.

        Args:
            item: DataItem to build requests for.
            batch_qa: If True, combine all QA pairs into one request.

        Returns:
            List of request dicts.
        """
        if batch_qa:
            return [self.build_request(item, qa_idx=-1)]

        requests = []
        for qa_idx in range(item.qa_count):
            # First request uses original image, subsequent ones use copies
            req = self.build_request(item, qa_idx, copy_img=(qa_idx > 0))
            requests.append(req)
        return requests

    def build_requests_selective(
        self,
        item: "DataItem",
        qa_indices: List[int],
    ) -> List[Dict]:
        """
        Build requests for specific QA indices only.

        Used by SelectiveMLLMRewriter for conditional rewriting.

        Args:
            item: DataItem to build requests for.
            qa_indices: List of QA indices to build requests for.

        Returns:
            List of request dicts.
        """
        requests = []
        for i, qa_idx in enumerate(qa_indices):
            # First request uses original image, subsequent ones use copies
            req = self.build_request(item, qa_idx, copy_img=(i > 0))
            requests.append(req)
        return requests

    # === Format content ===

    def format_content(self, item: "DataItem", qa_idx: int) -> str:
        """
        Format request content.

        Args:
            item: DataItem to format.
            qa_idx: QA pair index (-1 for all QA).

        Returns:
            Formatted prompt string.
        """
        # Simple mode: no prompt and single QA, return raw question only
        if not self.prompt_text and qa_idx != -1:
            qa = item.qa_pairs[qa_idx]
            parts = []
            if self.with_question:
                parts.append(qa.question)
            if self.with_answer:
                parts.append(qa.answer)
            if self.with_original and qa.ori_answer:
                parts.append(qa.ori_answer)
            return "\n".join(parts)

        if qa_idx == -1:
            qa_content = item.format_all_qa(
                with_question=self.with_question,
                with_answer=self.with_answer,
                with_original=self.with_original,
            )
        else:
            qa_content = item.format_qa(
                qa_idx,
                with_question=self.with_question,
                with_answer=self.with_answer,
                with_original=self.with_original,
            )

        if self.prompt_text:
            return f"{self.prompt_text}\n{qa_content}"
        return qa_content

    def _build_payload(self, content: str, image: Any = None) -> List[Dict]:
        """Build payload for model.generate()."""
        payload = []
        if self.system_prompt_text:
            payload.append({"type": "system", "value": self.system_prompt_text})
        payload.append({"type": "text", "value": content})
        if image is not None:
            payload.append({"type": "image", "value": image})
        return payload

    # === Parse response ===

    def parse_response(
        self,
        response: Any,
        qa_idx: int,
        logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Parse model response.

        Args:
            response: Raw response from model.
            qa_idx: QA index (used to format key_templates).
            logger: Optional logger for warnings.

        Returns:
            Dict with parsed fields, e.g. {"result": ..., "reason": ...}
            Returns {"result": None, "parse_error": True} if parsing fails.
        """
        if not isinstance(response, dict) or self.key_templates is None:
            return {"result": response}

        parsed = {}
        missing_keys = []
        for field_name, template in self.key_templates.items():
            key = template.format(idx=qa_idx)
            if key in response:
                parsed[field_name] = response[key]
            else:
                missing_keys.append(key)

        # Warn if expected keys are missing
        if missing_keys and logger:
            logger.warning(
                f"Response missing expected keys: {missing_keys}. "
                f"Response keys: {list(response.keys())}"
            )

        # If no fields were parsed, mark as parse error
        if not parsed and self.key_templates:
            return {"result": None, "parse_error": True}

        return parsed


__all__ = ["RequestBuilder", "copy_image"]
