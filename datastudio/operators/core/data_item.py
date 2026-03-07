"""Data item abstraction for conversation data."""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class QA:
    """Immutable view of a single QA pair.

    Attributes:
        idx: QA pair index (0-based).
        question: Question text.
        answer: Answer text.
        ori_answer: Original answer before any rewriting.
        ori_question: Original question before any rewriting.
    """

    idx: int
    question: str
    answer: str
    ori_answer: Optional[str] = None
    ori_question: Optional[str] = None


class DataItem:
    """Wrapper around a raw data dict with typed QA pair access.

    Example::

        item = DataItem(raw_data, idx=0)
        for qa in item.qa_pairs:
            print(qa.question, qa.answer)
        item.set_answer(0, "new answer")
    """

    def __init__(self, data: Dict, idx: int = 0):
        """
        Initialize a DataItem.

        Args:
            data: Raw data dict (will be modified in place for rewrites).
            idx: Index in the batch (for tracking).
        """
        self._data = data
        self.idx = idx
        self._qa_pairs: Optional[List[QA]] = None

    @property
    def data(self) -> Dict:
        """Access the underlying data dict."""
        return self._data

    @property
    def conversations(self) -> List[Dict]:
        """Get the conversations list."""
        return self._data.get("conversations", [])

    @property
    def qa_pairs(self) -> List[QA]:
        """Get all QA pairs (cached)."""
        if self._qa_pairs is None:
            self._qa_pairs = self._parse_qa_pairs()
        return self._qa_pairs

    @property
    def qa_count(self) -> int:
        """Number of QA pairs."""
        return len(self.qa_pairs)

    @property
    def image(self) -> Optional[Any]:
        """Get the image (PIL Image or None)."""
        return self._data.get("image_pil")

    @property
    def has_image(self) -> bool:
        """Whether this item has an image."""
        return "image_pil" in self._data or "image" in self._data

    @property
    def is_rejected(self) -> bool:
        """Whether this item is marked as rejected."""
        return self._data.get("rejected", False)

    def _parse_qa_pairs(self) -> List[QA]:
        """Parse conversations into QA objects."""
        conversations = self.conversations
        ori_answers = self._data.get("ori_answer", {})
        ori_questions = self._data.get("ori_question", {})

        pairs = []
        for i in range(0, len(conversations), 2):
            if i + 1 >= len(conversations):
                break
            qa_idx = i // 2
            pairs.append(
                QA(
                    idx=qa_idx,
                    question=conversations[i].get("value", ""),
                    answer=conversations[i + 1].get("value", ""),
                    ori_answer=ori_answers.get(str(qa_idx)),
                    ori_question=ori_questions.get(str(qa_idx)),
                )
            )
        return pairs

    def get_qa(self, qa_idx: int) -> QA:
        """Get a specific QA pair.

        Args:
            qa_idx: Index of the QA pair.

        Returns:
            QA object at the specified index.

        Raises:
            IndexError: If qa_idx is out of range.
        """
        if qa_idx < 0 or qa_idx >= len(self.qa_pairs):
            raise IndexError(
                f"QA index {qa_idx} out of range [0, {len(self.qa_pairs)})"
            )
        return self.qa_pairs[qa_idx]

    def get_question(self, qa_idx: int) -> str:
        """Get question at index."""
        return self.qa_pairs[qa_idx].question

    def get_answer(self, qa_idx: int) -> str:
        """Get answer at index."""
        return self.qa_pairs[qa_idx].answer

    # === Modification methods ===

    def set_question(self, qa_idx: int, value: str, save_original: bool = True):
        """Set question at index.

        Args:
            qa_idx: Index of the QA pair.
            value: New question value.
            save_original: Whether to save the original value.

        Raises:
            IndexError: If qa_idx is out of range.
        """
        conv_idx = qa_idx * 2
        if conv_idx >= len(self.conversations):
            raise IndexError(
                f"QA index {qa_idx} out of range (conv_idx={conv_idx}, len={len(self.conversations)})"
            )
        if save_original:
            self._save_original(qa_idx, "question")
        self.conversations[conv_idx]["value"] = value
        self._qa_pairs = None  # Invalidate cache

    def set_answer(self, qa_idx: int, value: str, save_original: bool = True):
        """Set answer at index.

        Args:
            qa_idx: Index of the QA pair.
            value: New answer value.
            save_original: Whether to save the original value.

        Raises:
            IndexError: If qa_idx is out of range.
        """
        conv_idx = qa_idx * 2 + 1
        if conv_idx >= len(self.conversations):
            raise IndexError(
                f"QA index {qa_idx} out of range (conv_idx={conv_idx}, len={len(self.conversations)})"
            )
        if save_original:
            self._save_original(qa_idx, "answer")
        self.conversations[conv_idx]["value"] = value
        self._qa_pairs = None  # Invalidate cache

    def _save_original(self, qa_idx: int, field: str = "answer"):
        """
        Save original content before rewriting.

        Args:
            qa_idx: QA pair index.
            field: 'answer' or 'question'.
        """
        key = f"ori_{field}"
        if key not in self._data:
            self._data[key] = {}

        if str(qa_idx) not in self._data[key]:
            if field == "answer":
                self._data[key][str(qa_idx)] = self.get_answer(qa_idx)
            else:
                self._data[key][str(qa_idx)] = self.get_question(qa_idx)

    # === Formatting (for RequestBuilder) ===

    def format_qa(
        self,
        qa_idx: int,
        with_question: bool = True,
        with_answer: bool = False,
        with_original: bool = False,
    ) -> str:
        """
        Format a single QA pair for prompt.

        Display index is always 0 (standard for single-QA prompts).
        """
        qa = self.qa_pairs[qa_idx]
        lines = []
        if with_question:
            lines.append(f"[QUESTION 0]: {qa.question}")
        if with_answer:
            lines.append(f"[ANSWER 0]: {qa.answer}")
        if with_original and qa.ori_answer:
            lines.append(f"[ORI_ANSWER 0]: {qa.ori_answer}")
        return "\n".join(lines)

    def format_all_qa(
        self,
        with_question: bool = True,
        with_answer: bool = False,
        with_original: bool = False,
    ) -> str:
        """Format all QA pairs for prompt."""
        lines = []
        for qa in self.qa_pairs:
            if with_question:
                lines.append(f"[QUESTION {qa.idx}]: {qa.question}")
            if with_answer:
                lines.append(f"[ANSWER {qa.idx}]: {qa.answer}")
            if with_original and qa.ori_answer:
                lines.append(f"[ORI_ANSWER {qa.idx}]: {qa.ori_answer}")
        return "\n".join(lines)

    def mark_rejected(self):
        """Mark this item as rejected."""
        self._data["rejected"] = True

    def mark_kept(self):
        """Mark this item as not rejected (kept)."""
        self._data["rejected"] = False

    # === Record management ===

    def add_filter_record(self, op_name: str, qa_idx: int, reason: str):
        """Add a filter record (for rejected items)."""
        if "filter_ops" not in self._data:
            self._data["filter_ops"] = {}
        if op_name not in self._data["filter_ops"]:
            self._data["filter_ops"][op_name] = {}
        self._data["filter_ops"][op_name][qa_idx] = reason

    def add_keep_record(self, op_name: str, qa_idx: int, reason: str):
        """Add a keep record (for kept items with reason)."""
        if not reason:
            return
        if "keep_ops" not in self._data:
            self._data["keep_ops"] = {}
        if op_name not in self._data["keep_ops"]:
            self._data["keep_ops"][op_name] = {}
        self._data["keep_ops"][op_name][qa_idx] = reason

    def add_rewrite_record(self, op_name: str, qa_idx: int, message: str):
        """Add a rewrite record."""
        if "rewrite_ops" not in self._data:
            self._data["rewrite_ops"] = {}
        if op_name not in self._data["rewrite_ops"]:
            self._data["rewrite_ops"][op_name] = {}

        existing = self._data["rewrite_ops"][op_name].get(qa_idx)
        if existing:
            self._data["rewrite_ops"][op_name][qa_idx] = f"{message} | {existing}"
        else:
            self._data["rewrite_ops"][op_name][qa_idx] = message

    def add_full_filter_record(self, op_name: str, reason: str):
        """Add a filter record for the entire item (not per-QA)."""
        if "filter_ops" not in self._data:
            self._data["filter_ops"] = {}
        self._data["filter_ops"][op_name] = reason

    def add_model_record(self, op_name: str, model_name: str):
        """Record which model was used by an operator to process this item.

        Stored as data["model"] = {op_name: model_name, ...}.

        Args:
            op_name: Operator name (e.g., "MLLMFilter").
            model_name: Model identifier (e.g., "Qwen2-VL-72B").
        """
        if "model" not in self._data:
            self._data["model"] = {}
        self._data["model"][op_name] = model_name

    # === Split operation ===

    def split(
        self,
        kept_indices: List[int],
        rejected_indices: List[int],
    ) -> Tuple["DataItem", "DataItem"]:
        """
        Split this item into kept and rejected parts.

        Creates two new DataItems:
        - kept_item: Contains only the kept QA pairs
        - rejected_item: Contains only the rejected QA pairs

        Both items have their metadata (filter_ops, rewrite_ops, etc.)
        properly re-indexed.

        Args:
            kept_indices: Original indices of QA pairs to keep.
            rejected_indices: Original indices of QA pairs to reject.

        Returns:
            Tuple of (kept_item, rejected_item).
        """
        # Exclude image_pil from deepcopy to avoid PIL serialization errors;
        # both splits can safely share the same image reference.
        image_pil = self._data.pop("image_pil", None)
        kept_data = copy.deepcopy(self._data)
        rejected_data = copy.deepcopy(self._data)
        if image_pil is not None:
            self._data["image_pil"] = image_pil
            kept_data["image_pil"] = image_pil
            rejected_data["image_pil"] = image_pil

        # Build new conversations
        kept_data["conversations"] = self._extract_conversations(kept_indices)
        rejected_data["conversations"] = self._extract_conversations(rejected_indices)

        # Mark states
        kept_data["rejected"] = False
        rejected_data["rejected"] = True

        # Re-index metadata
        kept_map = {old: new for new, old in enumerate(kept_indices)}
        rejected_map = {old: new for new, old in enumerate(rejected_indices)}

        self._reindex_metadata(kept_data, kept_map)
        self._reindex_metadata(rejected_data, rejected_map)

        return DataItem(kept_data, self.idx), DataItem(rejected_data, self.idx)

    def _extract_conversations(self, indices: List[int]) -> List[Dict]:
        """Extract conversations at specified QA indices."""
        result = []
        for qa_idx in indices:
            q_idx = qa_idx * 2
            a_idx = q_idx + 1
            if a_idx < len(self.conversations):
                result.append(copy.deepcopy(self.conversations[q_idx]))
                result.append(copy.deepcopy(self.conversations[a_idx]))
        return result

    def _parse_qa_index(self, key) -> int:
        """Parse QA index from various key formats (int, '0', 'qa_pair_0', etc.)."""
        if isinstance(key, int):
            return key
        if isinstance(key, str):
            # Handle 'qa_pair_X' format
            if key.startswith("qa_pair_"):
                return int(key.split("_")[-1])
            # Handle plain numeric string
            return int(key)
        raise ValueError(f"Cannot parse QA index from: {key}")

    def _reindex_metadata(self, data: Dict, index_map: Dict[int, int]):
        """Re-index per-QA metadata fields based on new indices."""
        metadata_fields = [
            "ori_answer",
            "ori_question",
            "rewrite_ops",
            "filter_ops",
            "score",
        ]

        for field_name in metadata_fields:
            if field_name not in data:
                continue

            original = data[field_name]
            if not isinstance(original, dict):
                continue

            # Check if it's a nested dict (like filter_ops: {op_name: {qa_idx: reason}})
            sample_value = next(iter(original.values()), None) if original else None

            if isinstance(sample_value, dict):
                # Nested structure: {op_name: {qa_idx: value}}
                new_outer = {}
                for op_name, qa_records in original.items():
                    if isinstance(qa_records, dict):
                        new_inner = {}
                        for qa_idx, value in qa_records.items():
                            old_idx = self._parse_qa_index(qa_idx)
                            if old_idx in index_map:
                                new_inner[str(index_map[old_idx])] = value
                        if new_inner:
                            new_outer[op_name] = new_inner
                    else:
                        # Not per-QA (e.g., full-item filter reason)
                        new_outer[op_name] = qa_records
                data[field_name] = new_outer
            else:
                # Flat structure: {qa_idx: value}
                new_dict = {}
                for qa_idx, value in original.items():
                    old_idx = self._parse_qa_index(qa_idx)
                    if old_idx in index_map:
                        new_dict[str(index_map[old_idx])] = value
                data[field_name] = new_dict


def wrap_data_list(datas: List[Dict]) -> List[DataItem]:
    """Wrap a list of raw dicts into DataItems."""
    return [DataItem(data=d, idx=i) for i, d in enumerate(datas)]


def unwrap_data_items(items: List[DataItem]) -> List[Dict]:
    """Extract raw dicts from DataItems."""
    return [item.data for item in items]
