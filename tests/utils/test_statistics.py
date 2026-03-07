"""
Unit tests for datastudio.utils.statistics module.

Tests cover:
- OperatorStats dataclass
- DatasetStats dataclass
- PipelineStats dataclass
- StatsCollector class
- get_stats_collector function
- reset_stats_collector function
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from datastudio.utils.statistics import (
    DatasetStats,
    OperatorStats,
    PipelineStats,
    StatsCollector,
    get_stats_collector,
    reset_stats_collector,
)


class TestOperatorStats:
    """Tests for OperatorStats dataclass."""

    def test_create_basic(self):
        """Test creating basic OperatorStats."""
        stats = OperatorStats(
            name="TestOperator",
            input_count=100,
            output_count=90,
            rejected_count=10,
            rewritten_count=5,
            execution_time=1.5,
        )

        assert stats.name == "TestOperator"
        assert stats.input_count == 100
        assert stats.output_count == 90

    def test_reject_rate_property(self):
        """Test reject_rate property."""
        stats = OperatorStats(
            name="Test",
            input_count=100,
            output_count=90,
            rejected_count=10,
            rewritten_count=0,
            execution_time=1.0,
        )

        assert stats.reject_rate == 0.1  # 10/100

    def test_reject_rate_zero_input(self):
        """Test reject_rate with zero input."""
        stats = OperatorStats(
            name="Test",
            input_count=0,
            output_count=0,
            rejected_count=0,
            rewritten_count=0,
            execution_time=0.0,
        )

        # Should handle division by zero
        assert stats.reject_rate == 0.0 or stats.reject_rate is None

    def test_rewrite_rate_property(self):
        """Test rewrite_rate property."""
        stats = OperatorStats(
            name="Test",
            input_count=100,
            output_count=100,
            rejected_count=0,
            rewritten_count=20,
            execution_time=1.0,
        )

        assert stats.rewrite_rate == 0.2  # 20/100

    def test_with_details(self):
        """Test OperatorStats with details."""
        stats = OperatorStats(
            name="Test",
            input_count=100,
            output_count=90,
            rejected_count=10,
            rewritten_count=0,
            execution_time=1.0,
            details={"custom_metric": 42},
        )

        assert stats.details["custom_metric"] == 42


class TestDatasetStats:
    """Tests for DatasetStats dataclass."""

    def test_create_basic(self):
        """Test creating basic DatasetStats."""
        stats = DatasetStats(
            path="/data/test.json",
            source="TestSource",
            total_count=1000,
            rejected_count=100,
        )

        assert stats.path == "/data/test.json"
        assert stats.total_count == 1000

    def test_kept_count_property(self):
        """Test kept_count property."""
        stats = DatasetStats(
            path="/data/test.json", source="Test", total_count=1000, rejected_count=100
        )

        assert stats.kept_count == 900  # 1000 - 100


class TestPipelineStats:
    """Tests for PipelineStats dataclass."""

    def test_create_basic(self):
        """Test creating basic PipelineStats."""
        stats = PipelineStats(
            start_time=time.time(),
            end_time=time.time() + 10,
            total_input=1000,
            total_output=900,
            total_rejected=100,
            total_rewritten=50,
        )

        assert stats.total_input == 1000
        assert stats.total_output == 900

    def test_execution_time_property(self):
        """Test execution_time property."""
        start = time.time()
        end = start + 10.5

        stats = PipelineStats(
            start_time=start,
            end_time=end,
            total_input=100,
            total_output=90,
            total_rejected=10,
            total_rewritten=0,
        )

        assert abs(stats.execution_time - 10.5) < 0.01

    def test_reject_rate_property(self):
        """Test reject_rate property."""
        stats = PipelineStats(
            start_time=0,
            end_time=10,
            total_input=1000,
            total_output=900,
            total_rejected=100,
            total_rewritten=0,
        )

        assert stats.reject_rate == 0.1

    def test_throughput_property(self):
        """Test throughput property."""
        stats = PipelineStats(
            start_time=0,
            end_time=10,
            total_input=1000,
            total_output=1000,
            total_rejected=0,
            total_rewritten=0,
        )

        assert stats.throughput == 100.0  # 1000/10

    def test_with_operator_stats(self):
        """Test PipelineStats with operator stats."""
        op_stats = [
            OperatorStats("Op1", 100, 90, 10, 0, 1.0),
            OperatorStats("Op2", 90, 85, 5, 10, 0.5),
        ]

        stats = PipelineStats(
            start_time=0,
            end_time=10,
            total_input=100,
            total_output=85,
            total_rejected=15,
            total_rewritten=10,
            operator_stats=op_stats,
        )

        assert len(stats.operator_stats) == 2


class TestStatsCollectorInit:
    """Tests for StatsCollector initialization."""

    def test_init_default(self):
        """Test initialization with defaults."""
        collector = StatsCollector()
        assert collector is not None

    def test_init_with_logger(self, mock_logger):
        """Test initialization with logger."""
        collector = StatsCollector(logger=mock_logger)
        assert collector is not None

    def test_init_with_work_dir(self, temp_dir):
        """Test initialization with work_dir."""
        collector = StatsCollector(work_dir=temp_dir)
        assert collector is not None


class TestStatsCollectorPipeline:
    """Tests for StatsCollector pipeline tracking."""

    @pytest.fixture
    def collector(self):
        """Create collector for testing."""
        return StatsCollector()

    def test_start_pipeline(self, collector):
        """Test start_pipeline."""
        collector.start_pipeline()
        # Should record start time

    def test_end_pipeline(self, collector):
        """Test end_pipeline."""
        collector.start_pipeline()
        time.sleep(0.1)
        collector.end_pipeline()
        # Should record end time


class TestStatsCollectorOperator:
    """Tests for StatsCollector operator tracking."""

    @pytest.fixture
    def collector(self):
        """Create collector for testing."""
        return StatsCollector()

    def test_start_operator(self, collector):
        """Test start_operator."""
        collector.start_operator("TestOperator")

    def test_end_operator(self, collector):
        """Test end_operator."""
        collector.start_operator("TestOperator")
        collector.end_operator(
            input_count=100, output_count=90, rejected_count=10, rewritten_count=5
        )

    def test_record_operator(self, collector):
        """Test record_operator."""
        collector.record_operator(
            name="TestOperator",
            input_count=100,
            output_count=90,
            rejected_count=10,
            rewritten_count=5,
            execution_time=1.5,
        )

    def test_record_multiple_operators(self, collector):
        """Test recording multiple operators."""
        collector.record_operator("Op1", 100, 90, 10, 0, 1.0)
        collector.record_operator("Op2", 90, 85, 5, 10, 0.5)
        collector.record_operator("Op3", 85, 80, 5, 0, 0.3)


class TestStatsCollectorDataset:
    """Tests for StatsCollector dataset tracking."""

    @pytest.fixture
    def collector(self):
        """Create collector for testing."""
        return StatsCollector()

    def test_record_dataset(self, collector):
        """Test record_dataset."""
        collector.record_dataset(
            path="/data/test.json", source="TestSource", total=1000, rejected=100
        )

    def test_record_dataset_merge_by_basename(self, collector):
        """Test record_dataset merges by basename."""
        # Record kept part
        collector.record_dataset(
            path="/output/source/test.json", source="TestSource", total=900, rejected=0
        )
        # Record rejected part (same basename)
        collector.record_dataset(
            path="/output/source/rejected/test.json",
            source="TestSource",
            total=100,
            rejected=100,
        )

        # Should have merged into one entry
        assert len(collector._file_stats) == 1
        stats = collector._file_stats["test.json"]
        assert stats.total_count == 1000
        assert stats.rejected_count == 100
        assert stats.kept_count == 900


class TestStatsCollectorSummary:
    """Tests for StatsCollector summary methods."""

    @pytest.fixture
    def collector_with_data(self):
        """Create collector with recorded data."""
        collector = StatsCollector()
        collector.start_pipeline()
        collector.record_operator("Op1", 100, 90, 10, 0, 1.0)
        collector.record_operator("Op2", 90, 85, 5, 10, 0.5)
        collector.record_dataset("/data/test.json", "Test", 100, 15)
        collector.end_pipeline()
        return collector

    def test_get_summary(self, collector_with_data):
        """Test get_summary."""
        summary = collector_with_data.get_summary()

        assert isinstance(summary, dict)
        assert "total_input" in summary or "operators" in summary

    def test_print_summary(self, collector_with_data, capsys):
        """Test print_summary."""
        collector_with_data.print_summary()

        captured = capsys.readouterr()
        # Should output something


class TestStatsCollectorExport:
    """Tests for StatsCollector export methods."""

    @pytest.fixture
    def collector_with_data(self, temp_dir):
        """Create collector with recorded data."""
        collector = StatsCollector(work_dir=temp_dir)
        collector.start_pipeline()
        collector.record_operator("Op1", 100, 90, 10, 0, 1.0)
        collector.record_dataset("/data/test.json", "Test", 100, 10)
        collector.end_pipeline()
        return collector

    def test_export_csv(self, collector_with_data, temp_dir):
        """Test export_csv."""
        output_path = collector_with_data.export_csv()

        assert output_path is not None
        assert os.path.exists(output_path)

    def test_export_csv_custom_path(self, collector_with_data, temp_dir):
        """Test export_csv with custom path."""
        custom_path = os.path.join(temp_dir, "custom_stats.csv")
        output_path = collector_with_data.export_csv(output_path=custom_path)

        assert output_path == custom_path
        assert os.path.exists(custom_path)

    def test_export_json(self, collector_with_data, temp_dir):
        """Test export_json."""
        output_path = collector_with_data.export_json()

        assert output_path is not None
        assert os.path.exists(output_path)

    def test_export_json_custom_path(self, collector_with_data, temp_dir):
        """Test export_json with custom path."""
        custom_path = os.path.join(temp_dir, "custom_stats.json")
        output_path = collector_with_data.export_json(output_path=custom_path)

        assert output_path == custom_path


class TestGetStatsCollector:
    """Tests for get_stats_collector function."""

    def test_get_stats_collector_default(self):
        """Test get_stats_collector with defaults."""
        collector = get_stats_collector()
        assert collector is not None
        assert isinstance(collector, StatsCollector)

    def test_get_stats_collector_with_work_dir(self, temp_dir):
        """Test get_stats_collector with work_dir."""
        collector = get_stats_collector(work_dir=temp_dir)
        assert collector is not None

    def test_get_stats_collector_singleton(self):
        """Test get_stats_collector returns same instance."""
        collector1 = get_stats_collector()
        collector2 = get_stats_collector()

        assert collector1 is collector2


class TestResetStatsCollector:
    """Tests for reset_stats_collector function."""

    def test_reset_stats_collector(self):
        """Test reset_stats_collector."""
        collector1 = get_stats_collector()
        collector1.record_operator("Test", 100, 90, 10, 0, 1.0)

        reset_stats_collector()

        collector2 = get_stats_collector()
        # Should be a fresh collector


class TestStatsCollectorEdgeCases:
    """Edge case tests for StatsCollector."""

    def test_end_operator_without_start(self):
        """Test end_operator without start_operator."""
        collector = StatsCollector()

        # Should handle gracefully
        collector.end_operator(input_count=100, output_count=90)

    def test_end_pipeline_without_start(self):
        """Test end_pipeline without start_pipeline."""
        collector = StatsCollector()

        # Should handle gracefully
        collector.end_pipeline()

    def test_get_summary_empty(self):
        """Test get_summary with no data."""
        collector = StatsCollector()
        summary = collector.get_summary()

        assert isinstance(summary, dict)

    def test_export_empty_collector(self, temp_dir):
        """Test exporting empty collector."""
        collector = StatsCollector(work_dir=temp_dir)

        # Should handle gracefully
        collector.export_csv()
        collector.export_json()
