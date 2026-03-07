"""
Shared pytest fixtures and configurations for DataStudio tests.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Common Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_conversation():
    """Sample conversation data for testing."""
    return [
        {"from": "human", "value": "What is 2+2?"},
        {"from": "gpt", "value": "2+2 equals 4."},
    ]


@pytest.fixture
def sample_multi_turn_conversation():
    """Sample multi-turn conversation data."""
    return [
        {"from": "human", "value": "Hello"},
        {"from": "gpt", "value": "Hi there!"},
        {"from": "human", "value": "What is AI?"},
        {"from": "gpt", "value": "AI stands for Artificial Intelligence."},
    ]


@pytest.fixture
def sample_data_item(sample_conversation):
    """Sample data item dictionary without image."""
    return {"conversations": sample_conversation, "source_file": "test.json"}


@pytest.fixture
def sample_data_item_with_image(sample_conversation):
    """Sample data item with image."""
    return {
        "conversations": sample_conversation,
        "image": "test_image.jpg",
        "source_file": "test.json",
    }


@pytest.fixture
def sample_data_list(sample_data_item):
    """List of sample data items."""
    return [
        sample_data_item,
        {
            "conversations": [
                {"from": "human", "value": "Question 2"},
                {"from": "gpt", "value": "Answer 2"},
            ],
            "source_file": "test.json",
        },
        {
            "conversations": [
                {"from": "human", "value": "Question 3"},
                {"from": "gpt", "value": "Answer 3"},
            ],
            "source_file": "test2.json",
        },
    ]


# =============================================================================
# File System Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def temp_json_file(temp_dir, sample_data_list):
    """Create a temporary JSON file with sample data."""
    filepath = os.path.join(temp_dir, "test_data.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(sample_data_list, f, ensure_ascii=False, indent=2)
    return filepath


@pytest.fixture
def temp_jsonl_file(temp_dir, sample_data_list):
    """Create a temporary JSONL file with sample data."""
    filepath = os.path.join(temp_dir, "test_data.jsonl")
    with open(filepath, "w", encoding="utf-8") as f:
        for item in sample_data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return filepath


@pytest.fixture
def temp_yaml_file(temp_dir):
    """Create a temporary YAML config file."""
    import yaml

    filepath = os.path.join(temp_dir, "test_config.yaml")
    config = {
        "datasets": [
            {"file_path": "data1.json", "source": "TestSource1"},
            {"file_path": "data2.json", "source": "TestSource2"},
        ]
    }
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return filepath


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.debug = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.critical = MagicMock()
    return logger


@pytest.fixture
def mock_model():
    """Mock model for MLLM operators."""
    model = MagicMock()
    model.generate = MagicMock(return_value=[{"response": "test response"}])
    return model


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {"choices": [{"message": {"content": "This is a test response."}}]}


@pytest.fixture
def mock_image():
    """Mock PIL Image for testing."""
    mock_img = MagicMock()
    mock_img.size = (100, 100)
    mock_img.mode = "RGB"
    mock_img.format = "JPEG"
    return mock_img


# =============================================================================
# DataItem Fixtures
# =============================================================================


@pytest.fixture
def data_item(sample_data_item):
    """Create a DataItem instance for testing."""
    from datastudio.operators.core.data_item import DataItem

    return DataItem(sample_data_item)


@pytest.fixture
def data_item_with_image(sample_data_item_with_image):
    """Create a DataItem with image for testing."""
    from datastudio.operators.core.data_item import DataItem

    return DataItem(sample_data_item_with_image)


@pytest.fixture
def data_item_multi_turn(sample_multi_turn_conversation):
    """Create a multi-turn DataItem for testing."""
    from datastudio.operators.core.data_item import DataItem

    return DataItem(
        {"conversations": sample_multi_turn_conversation, "source_file": "test.json"}
    )


# =============================================================================
# Registry Fixtures
# =============================================================================


@pytest.fixture
def clean_registry():
    """Provide a clean registry for testing."""
    from datastudio.utils.registry import Registry

    return Registry(name="test_registry")


# =============================================================================
# Config Fixtures
# =============================================================================


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        "data_root": "/data/test",
        "output_dir": "/output/test",
        "batch_size": 32,
        "operators": {
            "filter": {"type": "ConvLengthFilter", "min_length": 1},
            "rewriter": {"type": "RemoveThinkRewriter"},
        },
    }


# =============================================================================
# Operator Fixtures
# =============================================================================


@pytest.fixture
def sample_result():
    """Create a sample Result for testing."""
    from datastudio.operators.core.result import Result

    return Result(item_idx=0)


# =============================================================================
# Helper Functions
# =============================================================================


def create_mock_file_content(content: str):
    """Helper to create mock file content."""
    return mock_open(read_data=content)


def assert_json_equal(actual: Any, expected: Any):
    """Assert two JSON-serializable objects are equal."""
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require external API"
    )
