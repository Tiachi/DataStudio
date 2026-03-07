# Development Guide

This document provides detailed instructions for developing custom operators in DataStudio.

## Operator Architecture

DataStudio provides two base operator types:

| Base Class | Purpose | Key Method |
|------------|---------|------------|
| `Filter` | Decide whether to keep or reject QA pairs | `check(item, qa_idx) -> (rejected, reason)` |
| `Rewriter` | Transform or modify QA pair content | `rewrite(item, qa_idx) -> str` |

All operators are registered via `@OPERATORS.register_module()` and can be referenced by class name in pipeline configs.

---

## Creating a Filter

Filters implement `check()` which returns `(rejected: bool, reason: str)`. If `rejected=True`, the QA pair is filtered out.

```python
# datastudio/operators/filters/my_filter.py
from typing import Any, Optional, Tuple
from datastudio.utils.registry import OPERATORS
from datastudio.operators.core import Filter, DataItem, Result

@OPERATORS.register_module()
class MyCustomFilter(Filter):
    """Filter QA pairs based on answer length.

    Args:
        min_length: Minimum answer length to keep.
        max_length: Maximum answer length to keep.
    """

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 8192,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(logger=logger, **kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
        """Check if the QA pair should be rejected.

        Args:
            item: The data item containing the QA pairs.
            qa_idx: Index of the QA pair to check.

        Returns:
            Tuple of (rejected, reason).
        """
        answer = item.get_answer(qa_idx)
        length = len(answer)

        if length < self.min_length:
            return True, f"Answer too short: {length} < {self.min_length}"
        if length > self.max_length:
            return True, f"Answer too long: {length} > {self.max_length}"
        return False, ""
```

### Item-level vs QA-level Filtering

- **QA-level** (default): Override `check()`. The base `process()` calls `check()` for each QA pair.
- **Item-level**: Override `process()` directly for conditions that apply to the entire item (e.g., conversation length, image format).

```python
@OPERATORS.register_module()
class ItemLevelFilter(Filter):
    """Example of an item-level filter."""

    def process(self, item: DataItem) -> Result:
        result = Result(item_idx=item.idx)
        if item.qa_count > 20:
            result.add_filter(qa_idx=-1, rejected=True, reason="Too many QA pairs")
        return result

    def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
        return False, ""  # Not used
```

---

## Creating a Rewriter

Rewriters implement `rewrite()` to transform the answer text of a QA pair.

```python
# datastudio/operators/rewriters/my_rewriter.py
from typing import Any, Optional
from datastudio.utils.registry import OPERATORS
from datastudio.operators.core import Rewriter, DataItem

@OPERATORS.register_module()
class MyCustomRewriter(Rewriter):
    """Strip leading/trailing whitespace from answers."""

    def __init__(self, logger: Optional[Any] = None, **kwargs):
        super().__init__(logger=logger, **kwargs)

    def rewrite(self, item: DataItem, qa_idx: int) -> Optional[str]:
        """Rewrite the answer of a QA pair.

        Args:
            item: The data item.
            qa_idx: QA pair index.

        Returns:
            The rewritten answer string, or None if no change.
        """
        answer = item.get_answer(qa_idx)
        stripped = answer.strip()
        return stripped if stripped != answer else None
```

---

## Registration and Export

### 1. Register with the decorator

```python
from datastudio.utils.registry import OPERATORS

@OPERATORS.register_module()
class MyCustomFilter(Filter):
    ...
```

### 2. Export in `__init__.py`

```python
# datastudio/operators/filters/__init__.py
from .my_filter import MyCustomFilter

__all__ = [
    # ... existing exports
    "MyCustomFilter",
]
```

---

## Using in Config

Once registered, reference operators by class name in config files:

```python
# configs/my_pipeline.py
_base_ = ["@/_base_/dataset.py"]

dataset_yaml = "path/to/dataset.yaml"

sub_pipelines = dict(
    filters=dict(
        cfg=dict(type="SubPipeline", operators=[
            dict(type="MyCustomFilter", min_length=10, max_length=4096),
        ]),
        priority=0,
    ),
    rewriters=dict(
        cfg=dict(type="SubPipeline", operators=[
            dict(type="MyCustomRewriter"),
        ]),
        priority=1,
    ),
)
```

---

## Writing Tests

Place test files in `tests/` with the `test_` prefix:

```python
# tests/test_my_filter.py
import pytest
from datastudio.operators.core import DataItem, Result
from datastudio.operators.filters.my_filter import MyCustomFilter


class TestMyCustomFilter:
    def test_reject_short_answer(self):
        """Should reject answers shorter than min_length."""
        f = MyCustomFilter(min_length=10, max_length=100)
        item = DataItem(
            {"messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "short"},
            ]},
            idx=0,
        )
        rejected, reason = f.check(item, qa_idx=0)
        assert rejected is True
        assert "too short" in reason.lower()

    def test_keep_valid_answer(self):
        """Should keep answers within length bounds."""
        f = MyCustomFilter(min_length=1, max_length=1000)
        item = DataItem(
            {"messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "This is a valid answer."},
            ]},
            idx=0,
        )
        rejected, reason = f.check(item, qa_idx=0)
        assert rejected is False

    @pytest.mark.parametrize("min_len,max_len", [(1, 5), (10, 100), (50, 500)])
    def test_various_bounds(self, min_len, max_len):
        """Should initialize with various configurations."""
        f = MyCustomFilter(min_length=min_len, max_length=max_len)
        assert f.min_length == min_len
        assert f.max_length == max_len
```

Run tests:

```bash
python -m pytest tests/test_my_filter.py -v
```
