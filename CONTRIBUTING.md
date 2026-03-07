# Contributing to DataStudio

Thank you for considering contributing to DataStudio! This guide will help you get started.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Code Style](#code-style)
- [Adding New Operators](#adding-new-operators)

---

## Getting Started

### Prerequisites

- Python 3.10+
- Git

### Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/DataStudio.git
cd DataStudio
git remote add upstream https://github.com/Open-Bee/DataStudio.git
```

---

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify
python -m pytest tests/
```

---

## Contribution Workflow

1. Create a branch from `main`:
   ```bash
   git checkout main && git pull upstream main
   git checkout -b feature/your-feature-name
   ```
2. Make changes, write tests, format code
3. Commit with clear messages (conventional commit style):
   ```
   feat(operators): add TextQualityFilter
   fix(dataloader): handle empty image paths
   docs(readme): update MLLM examples
   ```
4. Push and create a Pull Request

---

## Code Style

We follow [PEP 8](https://peps.python.org/pep-0008/):

- **Line length**: 88 (Black default)
- **Formatting**: `black datastudio/ tests/`
- **Import sorting**: `isort datastudio/ tests/`
- **Linting**: `flake8 datastudio/ tests/`

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `ConvLengthFilter` |
| Functions / Variables | snake_case | `process_batch` |
| Constants | UPPER_SNAKE | `MAX_RETRY_COUNT` |

### Docstrings

Use Google-style docstrings:

```python
def process_data(items: List[DataItem], threshold: float = 0.5) -> List[Result]:
    """Process a batch of data items.

    Args:
        items: List of DataItem objects to process.
        threshold: Minimum score threshold. Defaults to 0.5.

    Returns:
        List of Result objects.

    Raises:
        ValueError: If items list is empty.
    """
```

---

## Adding New Operators

A brief overview of how to add a custom operator. For the full guide with detailed examples, see [docs/guide/development.md](docs/guide/development.md).

### 1. Create the Operator

```python
# datastudio/operators/filters/my_filter.py
from typing import Tuple
from datastudio.utils.registry import OPERATORS
from datastudio.operators.core import Filter, DataItem

@OPERATORS.register_module()
class MyCustomFilter(Filter):
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
        answer = item.get_answer(qa_idx)
        if len(answer) < self.threshold:
            return True, "answer too short"
        return False, ""
```

### 2. Register in `__init__.py`

```python
# datastudio/operators/filters/__init__.py
from .my_filter import MyCustomFilter
```

### 3. Add Tests

```python
# tests/test_my_filter.py
from datastudio.operators.filters.my_filter import MyCustomFilter
```

### 4. Update Documentation

Add the new operator to the operator table in docs.

---

## Getting Help

- **Questions**: [GitHub Discussions](https://github.com/Open-Bee/DataStudio/discussions)
- **Bugs**: [GitHub Issues](https://github.com/Open-Bee/DataStudio/issues)
