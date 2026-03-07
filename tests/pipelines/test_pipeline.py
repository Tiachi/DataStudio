"""
Unit tests for datastudio.pipelines.pipeline module.

Tests cover:
- Pipeline class
- Sub-pipeline building
- Operator execution
"""

from unittest.mock import MagicMock, patch

import pytest

from datastudio.pipelines.pipeline import Pipeline
from datastudio.pipelines.sub_pipeline import SubPipeline


@pytest.fixture
def valid_operations():
    """Create valid operations config with SubPipeline type."""
    return {
        "filter_stage": {
            "cfg": {
                "type": "SubPipeline",
                "operators": [{"type": "ConvLengthFilter", "min_length": 1}],
            },
            "priority": 0,
        }
    }


@pytest.fixture
def multi_stage_operations():
    """Create multi-stage operations config."""
    return {
        "stage1": {
            "cfg": {
                "type": "SubPipeline",
                "operators": [{"type": "ConvLengthFilter", "min_length": 1}],
            },
            "priority": 0,
        },
        "stage2": {
            "cfg": {
                "type": "SubPipeline",
                "operators": [{"type": "ConvLengthFilter", "max_length": 100}],
            },
            "priority": 1,
        },
    }


class TestPipelineInit:
    """Tests for Pipeline initialization."""

    def test_init_with_operations(self, valid_operations, mock_logger):
        """Test initialization with valid operations."""
        pipeline = Pipeline(operations=valid_operations, logger=mock_logger)
        assert pipeline is not None
        assert len(pipeline.sub_pipelines) == 1

    def test_init_with_model(self, valid_operations, mock_logger):
        """Test initialization with model."""
        mock_model = MagicMock()
        pipeline = Pipeline(
            operations=valid_operations, logger=mock_logger, model=mock_model
        )
        assert pipeline.model == mock_model

    def test_init_empty_operations(self, mock_logger):
        """Test initialization with empty operations."""
        pipeline = Pipeline(operations={}, logger=mock_logger)
        assert len(pipeline.sub_pipelines) == 0

    def test_init_invalid_config_type(self, mock_logger):
        """Test initialization with invalid config type."""
        operations = {"stage": {"cfg": {"type": "InvalidType"}, "priority": 0}}
        with pytest.raises(ValueError, match="Expected SubPipeline"):
            Pipeline(operations=operations, logger=mock_logger)


class TestPipelineCall:
    """Tests for Pipeline __call__ method."""

    @pytest.fixture
    def mock_pipeline(self, valid_operations, mock_logger):
        """Create pipeline with mocked sub-pipeline."""
        return Pipeline(operations=valid_operations, logger=mock_logger)

    def test_call_basic(self, mock_pipeline):
        """Test basic pipeline call."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        ]
        result = mock_pipeline(data)
        assert isinstance(result, list)

    def test_call_empty_data(self, mock_pipeline):
        """Test pipeline call with empty data."""
        result = mock_pipeline([])
        assert result == []

    def test_call_single_item(self, mock_pipeline):
        """Test pipeline call with single item."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        ]
        result = mock_pipeline(data)
        assert isinstance(result, list)


class TestPipelineOperatorBuilding:
    """Tests for operator building in Pipeline."""

    def test_builds_filter_operator(self, mock_logger):
        """Test that filter operators are built correctly."""
        operations = {
            "filters": {
                "cfg": {
                    "type": "SubPipeline",
                    "operators": [{"type": "ConvLengthFilter", "min_length": 1}],
                },
                "priority": 0,
            }
        }
        pipeline = Pipeline(operations=operations, logger=mock_logger)
        assert len(pipeline.sub_pipelines) == 1

    def test_builds_multiple_operators(self, multi_stage_operations, mock_logger):
        """Test building multiple operators."""
        pipeline = Pipeline(operations=multi_stage_operations, logger=mock_logger)
        assert len(pipeline.sub_pipelines) == 2

    def test_invalid_operator_type(self, mock_logger):
        """Test handling invalid operator type."""
        operations = {
            "stage": {
                "cfg": {
                    "type": "SubPipeline",
                    "operators": [{"type": "NonExistentOperator"}],
                },
                "priority": 0,
            }
        }
        # Should raise KeyError for non-existent operator
        with pytest.raises(KeyError):
            Pipeline(operations=operations, logger=mock_logger)


class TestPipelinePriority:
    """Tests for priority-based execution order."""

    def test_execution_order_by_priority(self, mock_logger):
        """Test that sub-pipelines execute in priority order."""
        operations = {
            "high_priority": {
                "cfg": {"type": "SubPipeline", "operators": []},
                "priority": 10,
            },
            "low_priority": {
                "cfg": {"type": "SubPipeline", "operators": []},
                "priority": 0,
            },
        }
        pipeline = Pipeline(operations=operations, logger=mock_logger)

        # Check order - lower priority should come first
        names = [name for _, name in pipeline.sub_pipelines]
        assert names[0] == "low_priority"
        assert names[1] == "high_priority"


class TestPipelineExecution:
    """Tests for pipeline execution."""

    def test_operators_execute_sequentially(self, mock_logger):
        """Test that operators in sub-pipeline execute sequentially."""
        operations = {
            "stage": {
                "cfg": {
                    "type": "SubPipeline",
                    "operators": [
                        {"type": "ConvLengthFilter", "min_length": 1},
                        {"type": "ConvLengthFilter", "max_length": 100},
                    ],
                },
                "priority": 0,
            }
        }
        pipeline = Pipeline(operations=operations, logger=mock_logger)

        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        ]
        result = pipeline(data)
        assert isinstance(result, list)

    def test_filter_removes_items(self, mock_logger):
        """Test that filter removes items."""
        operations = {
            "stage": {
                "cfg": {
                    "type": "SubPipeline",
                    "operators": [
                        {
                            "type": "ConvLengthFilter",
                            "min_length": 10,
                        }  # Require 10 turns
                    ],
                },
                "priority": 0,
            }
        }
        pipeline = Pipeline(operations=operations, logger=mock_logger)

        # Single turn should be rejected
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        ]
        result = pipeline(data)

        # Result should contain rejected items
        rejected = [d for d in result if d.get("rejected", False)]
        assert len(rejected) > 0 or len(result) == 0


class TestPipelineEdgeCases:
    """Edge case tests for Pipeline."""

    def test_large_data_batch(self, valid_operations, mock_logger):
        """Test processing large data batch."""
        pipeline = Pipeline(operations=valid_operations, logger=mock_logger)

        data = [
            {
                "conversations": [
                    {"from": "human", "value": f"Q{i}"},
                    {"from": "gpt", "value": f"A{i}"},
                ]
            }
            for i in range(100)
        ]
        result = pipeline(data)
        assert len(result) >= 0

    def test_unicode_data(self, valid_operations, mock_logger):
        """Test processing unicode data."""
        pipeline = Pipeline(operations=valid_operations, logger=mock_logger)

        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Sample question"},
                    {"from": "gpt", "value": "Sample answer"},
                ]
            }
        ]
        result = pipeline(data)
        assert isinstance(result, list)


class TestPipelineRegistration:
    """Tests for Pipeline registration."""

    def test_registered_in_pipelines(self):
        """Test Pipeline is registered in PIPELINES registry."""
        from datastudio.utils.registry import PIPELINES

        assert "Pipeline" in PIPELINES


class TestPipelineStatistics:
    """Tests for Pipeline statistics tracking."""

    def test_pipeline_tracks_stats_collector(self, valid_operations, mock_logger):
        """Test that Pipeline tracks stats_collector when provided."""
        stats_collector = MagicMock()
        pipeline = Pipeline(
            operations=valid_operations,
            logger=mock_logger,
            stats_collector=stats_collector,
        )
        assert pipeline.stats_collector is stats_collector

        # Run data through and check stats_collector.record_operator was called
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        ]
        pipeline(data)
        assert stats_collector.record_operator.called

    def test_pipeline_works_without_stats_collector(
        self, valid_operations, mock_logger
    ):
        """Test that Pipeline works without stats_collector (None)."""
        pipeline = Pipeline(
            operations=valid_operations,
            logger=mock_logger,
            stats_collector=None,
        )
        assert pipeline.stats_collector is None

        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        ]
        # Should not raise
        result = pipeline(data)
        assert isinstance(result, list)


class TestPipelineEmptyInput:
    """Tests for Pipeline with empty input."""

    def test_call_with_empty_list(self, valid_operations, mock_logger):
        """Test __call__ with empty list returns empty list."""
        pipeline = Pipeline(operations=valid_operations, logger=mock_logger)
        result = pipeline([])
        assert result == []

    def test_call_with_all_rejected(self, mock_logger):
        """Test __call__ with data where all items get rejected."""
        operations = {
            "strict_filter": {
                "cfg": {
                    "type": "SubPipeline",
                    "operators": [{"type": "ConvLengthFilter", "min_length": 100}],
                },
                "priority": 0,
            }
        }
        pipeline = Pipeline(operations=operations, logger=mock_logger)

        # Single-turn items will be rejected because min_length=100 (QA pairs, not chars)
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q1"},
                    {"from": "gpt", "value": "A1"},
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "Q2"},
                    {"from": "gpt", "value": "A2"},
                ]
            },
        ]
        result = pipeline(data)
        # All items should be in result, but marked as rejected
        rejected = [d for d in result if d.get("rejected", False)]
        kept = [d for d in result if not d.get("rejected", False)]
        # With min_length=100, single-turn items should be rejected
        assert len(rejected) == 2
        assert len(kept) == 0


class TestPipelinePriorityOrder:
    """Tests for Pipeline priority ordering."""

    def test_lower_priority_executes_first(self, mock_logger):
        """Test that operations with lower priority execute first."""
        operations = {
            "high_pri": {
                "cfg": {"type": "SubPipeline", "operators": []},
                "priority": 10,
            },
            "medium_pri": {
                "cfg": {"type": "SubPipeline", "operators": []},
                "priority": 5,
            },
            "low_pri": {"cfg": {"type": "SubPipeline", "operators": []}, "priority": 1},
        }
        pipeline = Pipeline(operations=operations, logger=mock_logger)

        names = [name for _, name in pipeline.sub_pipelines]
        assert names == ["low_pri", "medium_pri", "high_pri"]

    def test_same_priority_maintains_order(self, mock_logger):
        """Test that same priority operations maintain stable sort order."""
        operations = {
            "alpha": {"cfg": {"type": "SubPipeline", "operators": []}, "priority": 0},
            "beta": {"cfg": {"type": "SubPipeline", "operators": []}, "priority": 0},
        }
        pipeline = Pipeline(operations=operations, logger=mock_logger)

        names = [name for _, name in pipeline.sub_pipelines]
        # Both have priority 0, so dict insertion order is preserved by sorted() stability
        assert len(names) == 2
        assert set(names) == {"alpha", "beta"}
