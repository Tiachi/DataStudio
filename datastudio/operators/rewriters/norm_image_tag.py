"""Normalize image tag positions rewriter."""

import re
from typing import Any, Optional

from datastudio.utils.registry import OPERATORS

from ..core import DataItem, Result, Rewriter


@OPERATORS.register_module()
class NormImageTagRewriter(Rewriter):
    """
    Normalize image tag positions in conversations.

    Moves all <image> tags to the beginning of the first human message,
    ensuring consistent format across all data items.

    .. warning::
        This rewriter is designed for single-image or multi-image scenarios
        where all images logically belong to the first question. It does NOT
        support interleaved image-text conversations (e.g. images scattered
        across multiple turns), as it will forcibly move all <image> tags to
        the first message and break the original image-text correspondence.
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

    def _count_image_tags(self, text: str) -> int:
        """Count <image> tags in text."""
        return text.count("<image>")

    def _count_images_in_item(self, item: DataItem) -> int:
        """Count actual images in item."""
        image = item.data.get("image")
        if image is None:
            image = item.data.get("image_pil")
        if image is None:
            return 0
        if isinstance(image, list):
            return len(image)
        return 1

    def process(self, item: DataItem) -> Result:
        """Normalize image tags - move all to first question (item-level)."""
        result = Result(item_idx=item.idx)

        image_count = self._count_images_in_item(item)
        if image_count == 0:
            return result

        first_question = item.get_question(0)

        # Count original tags across all questions
        original_tag_count = sum(
            self._count_image_tags(qa.question) for qa in item.qa_pairs
        )

        # Remove all image tags from all questions and track changes
        questions_changed = False
        new_questions = []
        for qa in item.qa_pairs:
            cleaned = re.sub(r"<image>\s*", "", qa.question)
            new_questions.append(cleaned)
            if cleaned != qa.question:
                questions_changed = True

        if not questions_changed and original_tag_count == image_count:
            return result

        # Add correct number of tags to first question
        image_tags = "<image>\n" * image_count
        new_first_question = image_tags + new_questions[0].strip()

        # First question gets the image tags
        if new_first_question != first_question:
            result.add_rewrite(
                qa_idx=0,
                new_question=new_first_question,
                message=f"normalized image tags {original_tag_count}->{image_count}",
            )

        # Other questions just get cleaned
        for i, (qa, new_q) in enumerate(
            zip(item.qa_pairs[1:], new_questions[1:]), start=1
        ):
            if new_q != qa.question:
                result.add_rewrite(
                    qa_idx=i,
                    new_question=new_q.strip(),
                    message="removed stray image tags",
                )

        return result

    def rewrite(self, item: DataItem, qa_idx: int) -> Optional[str]:
        """Not used - this rewriter operates at item level."""
        return None
