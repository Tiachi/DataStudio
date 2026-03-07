"""Unified statistics collection for pipeline execution.

Collects, aggregates, and exports per-operator and per-dataset statistics.
"""

import csv
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .logging import get_logger


@dataclass
class OperatorStats:
    """Statistics for a single operator execution."""

    name: str
    input_count: int = 0
    output_count: int = 0
    rejected_count: int = 0
    rewritten_count: int = 0
    execution_time: float = 0.0
    # QA-level statistics
    input_qa_count: int = 0
    output_qa_count: int = 0
    rejected_qa_count: int = 0
    # Split statistics (partial rejection causes item split)
    split_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def reject_rate(self) -> float:
        """Calculate reject rate (item level)."""
        if self.input_count == 0:
            return 0.0
        return self.rejected_count / self.input_count

    @property
    def rewrite_rate(self) -> float:
        """Calculate rewrite rate."""
        if self.input_count == 0:
            return 0.0
        return self.rewritten_count / self.input_count

    @property
    def qa_reject_rate(self) -> float:
        """Calculate QA-level reject rate."""
        if self.input_qa_count == 0:
            return 0.0
        return self.rejected_qa_count / self.input_qa_count


@dataclass
class DatasetStats:
    """Statistics for a dataset."""

    path: str
    source: str = "unknown"
    total_count: int = 0
    rejected_count: int = 0

    @property
    def kept_count(self) -> int:
        return self.total_count - self.rejected_count


@dataclass
class PipelineStats:
    """Overall pipeline statistics."""

    start_time: float = 0.0
    end_time: float = 0.0
    total_input: int = 0
    total_output: int = 0
    total_rejected: int = 0
    total_rewritten: int = 0
    operator_stats: List[OperatorStats] = field(default_factory=list)
    dataset_stats: List[DatasetStats] = field(default_factory=list)

    @property
    def execution_time(self) -> float:
        return self.end_time - self.start_time

    @property
    def reject_rate(self) -> float:
        if self.total_input == 0:
            return 0.0
        return self.total_rejected / self.total_input

    @property
    def throughput(self) -> float:
        """Items per second."""
        if self.execution_time == 0:
            return 0.0
        return self.total_input / self.execution_time


class StatsCollector:
    """Collects and aggregates pipeline statistics.

    Tracks per-operator and per-dataset metrics, and exports to CSV/JSON.

    Example:
        >>> collector = StatsCollector()
        >>> collector.record_operator('filter_mllm', input_count=100, rejected_count=10)
        >>> collector.record_dataset('data.json', source='train', total=100, rejected=10)
        >>> collector.print_summary()
        >>> collector.export_csv('stats.csv')
    """

    def __init__(self, logger=None, work_dir: Optional[str] = None):
        self.logger = logger or get_logger()
        self.work_dir = work_dir or os.getcwd()
        self._pipeline_stats = PipelineStats()
        self._operator_counters: Dict[str, Counter] = defaultdict(Counter)
        self._file_stats: Dict[str, DatasetStats] = {}
        self._current_operator: Optional[str] = None
        self._operator_start_time: float = 0.0

    def start_pipeline(self) -> None:
        """Mark pipeline start."""
        self._pipeline_stats.start_time = time.time()
        self.logger.info("Pipeline started")

    def end_pipeline(self) -> None:
        """Mark pipeline end."""
        self._pipeline_stats.end_time = time.time()
        self._pipeline_stats.dataset_stats = list(self._file_stats.values())
        self.logger.info(
            f"Pipeline completed in {self._pipeline_stats.execution_time:.2f}s"
        )

    def start_operator(self, name: str) -> None:
        """Mark operator start."""
        self._current_operator = name
        self._operator_start_time = time.time()

    def end_operator(
        self,
        name: Optional[str] = None,
        input_count: int = 0,
        output_count: int = 0,
        rejected_count: int = 0,
        rewritten_count: int = 0,
        **details,
    ) -> None:
        """
        Record operator execution statistics.

        Args:
            name: Operator name (uses current if not specified)
            input_count: Number of input items
            output_count: Number of output items
            rejected_count: Number of rejected items
            rewritten_count: Number of rewritten items
            **details: Additional details to record
        """
        op_name = name or self._current_operator
        if not op_name:
            return

        exec_time = (
            time.time() - self._operator_start_time
            if self._operator_start_time
            else 0.0
        )

        stats = OperatorStats(
            name=op_name,
            input_count=input_count,
            output_count=output_count,
            rejected_count=rejected_count,
            rewritten_count=rewritten_count,
            execution_time=exec_time,
            details=details,
        )

        self._pipeline_stats.operator_stats.append(stats)
        self._operator_counters[op_name]["input"] += input_count
        self._operator_counters[op_name]["rejected"] += rejected_count
        self._operator_counters[op_name]["rewritten"] += rewritten_count

        self._current_operator = None
        self._operator_start_time = 0.0

        # Log operator stats
        self.logger.info(
            f"[{op_name}] input={input_count}, output={output_count}, "
            f"rejected={rejected_count}, rewritten={rewritten_count}, time={exec_time:.2f}s"
        )

    def record_operator(
        self,
        name: str,
        input_count: int = 0,
        output_count: int = 0,
        rejected_count: int = 0,
        rewritten_count: int = 0,
        execution_time: float = 0.0,
        input_qa_count: int = 0,
        output_qa_count: int = 0,
        rejected_qa_count: int = 0,
        split_count: int = 0,
        **details,
    ) -> None:
        """Record operator statistics directly (called from Pipeline).

        Args:
            name: Operator name
            input_count: Number of input items
            output_count: Number of kept items (not rejected)
            rejected_count: Number of rejected items (including split-off rejected parts)
            rewritten_count: Number of rewritten items
            execution_time: Execution time in seconds
            input_qa_count: Number of input QA pairs
            output_qa_count: Number of kept QA pairs
            rejected_qa_count: Number of rejected QA pairs
            split_count: Number of items that were partially rejected and split
            **details: Additional details
        """
        stats = OperatorStats(
            name=name,
            input_count=input_count,
            output_count=output_count,
            rejected_count=rejected_count,
            rewritten_count=rewritten_count,
            execution_time=execution_time,
            input_qa_count=input_qa_count,
            output_qa_count=output_qa_count,
            rejected_qa_count=rejected_qa_count,
            split_count=split_count,
            details=details,
        )

        self._pipeline_stats.operator_stats.append(stats)
        self._operator_counters[name]["input"] += input_count
        self._operator_counters[name]["output"] += output_count
        self._operator_counters[name]["rejected"] += rejected_count
        self._operator_counters[name]["rewritten"] += rewritten_count
        self._operator_counters[name]["input_qa"] += input_qa_count
        self._operator_counters[name]["output_qa"] += output_qa_count
        self._operator_counters[name]["rejected_qa"] += rejected_qa_count
        self._operator_counters[name]["split"] += split_count

    def record_dataset(
        self,
        path: str,
        source: str = "unknown",
        total: int = 0,
        rejected: int = 0,
    ) -> None:
        """
        Record dataset statistics (called from DataSaver).

        Statistics are merged by filename (basename), so kept and rejected parts
        of the same source file are combined into one record.

        Note: Dataset statistics reflect actual saved file counts, which may
        differ from operator statistics due to multi-turn conversation splits.

        Args:
            path: Dataset file path
            source: Data source name
            total: Total item count (for this save operation)
            rejected: Rejected item count
        """
        # Use basename as key to merge kept and rejected parts of the same source file
        basename = os.path.basename(path)

        if basename in self._file_stats:
            # Merge with existing stats for this file
            old_stats = self._file_stats[basename]

            # Subtract old values from pipeline totals first
            self._pipeline_stats.total_input -= old_stats.total_count
            self._pipeline_stats.total_output -= old_stats.kept_count
            self._pipeline_stats.total_rejected -= old_stats.rejected_count

            # Merge counts
            total = old_stats.total_count + total
            rejected = old_stats.rejected_count + rejected

        stats = DatasetStats(
            path=basename,
            source=source,
            total_count=total,
            rejected_count=rejected,
        )

        self._file_stats[basename] = stats

        # Update totals
        self._pipeline_stats.total_input += total
        self._pipeline_stats.total_output += total - rejected
        self._pipeline_stats.total_rejected += rejected

    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary as dictionary."""
        stats = self._pipeline_stats

        # Aggregate operator stats
        operator_summary = {}
        for name, counts in self._operator_counters.items():
            input_count = counts["input"]
            output_count = counts["output"]
            rejected_count = counts["rejected"]
            split_count = counts.get("split", 0)
            input_qa = counts.get("input_qa", 0)
            output_qa = counts.get("output_qa", 0)
            rejected_qa = counts.get("rejected_qa", 0)

            operator_summary[name] = {
                # Item-level stats
                "input": input_count,
                "output": output_count,
                "rejected": rejected_count,
                "split": split_count,
                "reject_rate": rejected_count / input_count if input_count > 0 else 0,
                "rewritten": counts["rewritten"],
                # QA-level stats
                "input_qa": input_qa,
                "output_qa": output_qa,
                "rejected_qa": rejected_qa,
                "qa_reject_rate": rejected_qa / input_qa if input_qa > 0 else 0,
            }

        return {
            "pipeline": {
                "total_input": stats.total_input,
                "total_output": stats.total_output,
                "total_rejected": stats.total_rejected,
                "total_rewritten": stats.total_rewritten,
                "reject_rate": stats.reject_rate,
                "execution_time": stats.execution_time,
                "throughput": stats.throughput,
            },
            "operators": operator_summary,
            "datasets": {
                ds.path: {
                    "source": ds.source,
                    "total": ds.total_count,
                    "rejected": ds.rejected_count,
                    "kept": ds.kept_count,
                }
                for ds in stats.dataset_stats
            },
        }

    def print_summary(self) -> None:
        """Print formatted statistics summary."""
        summary = self.get_summary()

        self.logger.info("\n" + "=" * 60)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("=" * 60)

        if summary["operators"]:
            self.logger.info("\n--- Operator Statistics (Processing View) ---")
            for name, stats in summary["operators"].items():
                has_qa_stats = stats.get("input_qa", 0) > 0
                split_info = (
                    f", {stats['split']} splits" if stats.get("split", 0) > 0 else ""
                )

                if has_qa_stats:
                    self.logger.info(
                        f"  [{name}]\n"
                        f"    Items: {stats['input']} input -> {stats['output']} kept, "
                        f"{stats['rejected']} rejected{split_info}\n"
                        f"    QA:    {stats['input_qa']} input -> {stats['output_qa']} kept, "
                        f"{stats['rejected_qa']} rejected\n"
                        f"    Rate:  item {stats['reject_rate']:.1%}, QA {stats['qa_reject_rate']:.1%}"
                    )
                else:
                    self.logger.info(
                        f"  [{name}] input={stats['input']}, output={stats['output']}, "
                        f"rejected={stats['rejected']}, rewritten={stats['rewritten']}"
                    )

        if summary["datasets"]:
            self.logger.info(
                f"\n--- Dataset Statistics (Saved Files View, {len(summary['datasets'])} files) ---"
            )
            self.logger.info(
                "  Note: Item counts may be higher than operator input due to partial rejection splits"
            )
            for path, stats in list(summary["datasets"].items())[:10]:
                self.logger.info(
                    f"  {path}: {stats['kept']} kept + {stats['rejected']} rejected = {stats['total']} total"
                )
            if len(summary["datasets"]) > 10:
                self.logger.info(
                    f"  ... and {len(summary['datasets']) - 10} more files"
                )

        self.logger.info("\n--- Overall Statistics ---")
        p = summary["pipeline"]
        self.logger.info(
            f"  Saved items: {p['total_output']} kept, {p['total_rejected']} rejected, {p['total_input']} total"
        )
        self.logger.info(f"  Reject rate: {p['reject_rate']:.2%}")
        self.logger.info(f"  Execution time: {p['execution_time']:.2f}s")
        self.logger.info(f"  Throughput: {p['throughput']:.2f} items/s")
        self.logger.info("=" * 60)

    def export_csv(self, output_path: Optional[str] = None) -> str:
        """
        Export statistics to CSV file.

        Args:
            output_path: Output file path (default: work_dir/stats.csv)

        Returns:
            Path to exported CSV file
        """
        if output_path is None:
            output_path = os.path.join(self.work_dir, "pipeline_stats.csv")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Collect all operator names
        all_operators = sorted(self._operator_counters.keys())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["file_path", "source", "total", "rejected", "kept"] + list(
                all_operators
            )
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Write per-dataset stats
            for ds in self._file_stats.values():
                row = {
                    "file_path": ds.path,
                    "source": ds.source,
                    "total": ds.total_count,
                    "rejected": ds.rejected_count,
                    "kept": ds.kept_count,
                }
                for op in all_operators:
                    row[op] = 0
                writer.writerow(row)

            # Write global summary row
            summary_row = {
                "file_path": "TOTAL",
                "source": "ALL",
                "total": self._pipeline_stats.total_input,
                "rejected": self._pipeline_stats.total_rejected,
                "kept": self._pipeline_stats.total_output,
            }
            for op in all_operators:
                summary_row[op] = self._operator_counters[op]["rejected"]
            writer.writerow(summary_row)

        self.logger.info(f"Statistics exported to: {output_path}")
        return output_path

    def export_json(self, output_path: Optional[str] = None) -> str:
        """
        Export statistics to JSON file.

        Args:
            output_path: Output file path (default: work_dir/stats.json)

        Returns:
            Path to exported JSON file
        """
        if output_path is None:
            output_path = os.path.join(self.work_dir, "pipeline_stats.json")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        summary = self.get_summary()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Statistics exported to: {output_path}")
        return output_path


# Global instance
_stats_collector: Optional[StatsCollector] = None


def get_stats_collector(work_dir: Optional[str] = None, logger=None) -> StatsCollector:
    """Get or create global statistics collector."""
    global _stats_collector
    if _stats_collector is None:
        _stats_collector = StatsCollector(logger=logger, work_dir=work_dir)
    return _stats_collector


def reset_stats_collector() -> None:
    """Reset global statistics collector."""
    global _stats_collector
    _stats_collector = None
