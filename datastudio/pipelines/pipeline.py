"""Main pipeline that composes sub-pipelines with priority-based ordering.

Example config::

    pipeline = dict(
        type='Pipeline',
        operations={
            'basic_filters': dict(
                cfg=dict(type='SubPipeline', operators=[
                    dict(type='ConvLengthFilter', min_length=1),
                    dict(type='RemoveThinkRewriter'),
                ]),
                priority=0,
            ),
            'mllm_filter': dict(
                cfg=dict(type='SubPipeline', operators=[
                    dict(type='MLLMFilter', ...),
                ]),
                priority=1,
            ),
        }
    )
"""

from typing import Any, Dict, List, Optional, Tuple

from datastudio.utils.qa_utils import get_qa_count
from datastudio.utils.registry import MODELS, OPERATORS, PIPELINES, build_from_cfg
from datastudio.utils.statistics import StatsCollector

from .sub_pipeline import SubPipeline


@PIPELINES.register_module()
class Pipeline:
    """Composes sub-pipelines and executes them by priority (lower = earlier).

    Operators are resolved from the unified ``OPERATORS`` registry, so
    sub-pipelines can freely mix filters and rewriters.

    Attributes:
        sub_pipelines: List of (sub_pipeline, name) tuples, sorted by priority.
    """

    def __init__(
        self,
        operations: Dict[str, Dict],
        logger: Any,
        model: Optional[Any] = None,
        stats_collector: Optional[StatsCollector] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            operations: Dict of operation configs with cfg and priority.
            logger: Logger instance.
            model: Optional model instance for MLLM operators.
            stats_collector: Optional statistics collector for tracking operator stats.
        """
        self.logger = logger
        self.model = model
        self.stats_collector = stats_collector

        logger.info(f"Building Pipeline with {len(operations)} sub-pipelines...")

        # Sort by priority (lower = earlier)
        sorted_ops = sorted(operations.items(), key=lambda x: x[1].get("priority", 0))

        logger.info(f"Execution order: {[op[0] for op in sorted_ops]}")

        # Build sub-pipelines
        self.sub_pipelines: List[Tuple[SubPipeline, str]] = []
        for op_name, op in sorted_ops:
            op_cfg = op.get("cfg", {})
            priority = op.get("priority", 0)

            # Build the sub-pipeline using unified OPERATORS registry
            sub_pipeline = self._build_sub_pipeline(
                op_cfg, op_name, priority, logger, model
            )

            self.sub_pipelines.append((sub_pipeline, op_name))
            logger.info(f"  [{priority}] {op_name}")

    def _build_sub_pipeline(
        self,
        cfg: Dict,
        name: str,
        priority: int,
        logger: Any,
        model: Optional[Any],
    ) -> SubPipeline:
        """
        Build a sub-pipeline from config.

        Args:
            cfg: Sub-pipeline config dict.
            name: Name of the sub-pipeline.
            priority: Execution priority.
            logger: Logger instance.
            model: Optional model for MLLM operators.

        Returns:
            SubPipeline instance.
        """
        cfg_type = cfg.get("type", "") if isinstance(cfg, dict) else ""

        if cfg_type != "SubPipeline":
            raise ValueError(f"Expected SubPipeline config, got '{cfg_type}'")

        operators = []
        for op_cfg in cfg.get("operators", []):
            # Support per-operator model override: if an operator config contains
            # a "model" dict, build a dedicated model instance for that operator
            # instead of using the global one.
            op_model = model
            if isinstance(op_cfg, dict) and isinstance(op_cfg.get("model"), dict):
                op_model = build_from_cfg(op_cfg["model"], MODELS, logger=logger)
                op_cfg = {k: v for k, v in op_cfg.items() if k != "model"}
            op = build_from_cfg(op_cfg, OPERATORS, logger=logger, model=op_model)
            if op is not None:
                operators.append(op)

        return SubPipeline(operators, name=name, priority=priority, logger=logger)

    def __call__(self, datas: List[Dict]) -> List[Dict]:
        """
        Execute all sub-pipelines in priority order.

        Args:
            datas: List of raw data dicts to process.

        Returns:
            List of all processed data dicts (both kept and rejected).
            Rejected items have 'rejected=True' in their dict.
        """
        all_rejected: List[Dict] = []

        for sub_pipeline, name in self.sub_pipelines:
            before_count = len(datas)

            # Calculate QA counts before processing
            before_qa_count = sum(
                get_qa_count(d.get("conversations", [])) for d in datas
            )

            # SubPipeline returns merged list, split for counting
            result = sub_pipeline(datas)
            datas = [d for d in result if not d.get("rejected", False)]
            newly_rejected = [d for d in result if d.get("rejected", False)]
            all_rejected.extend(newly_rejected)

            after_count = len(datas)
            newly_rejected_count = len(newly_rejected)

            # Calculate QA counts after processing
            after_qa_count = sum(
                get_qa_count(d.get("conversations", [])) for d in datas
            )
            qa_rejected_from_input = before_qa_count - after_qa_count

            # Count items rewritten by this sub_pipeline
            operator_names = [op.name for op in sub_pipeline.operators]
            rewritten_count = sum(
                1
                for d in datas
                if any(
                    op_name in d.get("rewrite_ops", {}) for op_name in operator_names
                )
            )

            # Calculate split stats: when partial rejection happens,
            # one input item becomes two output items (kept + rejected)
            total_output_items = after_count + newly_rejected_count
            split_count = (
                total_output_items - before_count
            )  # Number of items that got split

            if self.logger:
                # Log both item-level and QA-level statistics
                log_msg = f"{name}: items {before_count}->{after_count} kept, {newly_rejected_count} rejected"
                if split_count > 0:
                    log_msg += f" ({split_count} partial splits)"
                log_msg += f" | QA {before_qa_count}->{after_qa_count} kept, {qa_rejected_from_input} rejected"
                self.logger.info(log_msg)

            # Record operator statistics
            if self.stats_collector:
                self.stats_collector.record_operator(
                    name=name,
                    input_count=before_count,
                    output_count=after_count,
                    rejected_count=newly_rejected_count,
                    rewritten_count=rewritten_count,
                    input_qa_count=before_qa_count,
                    output_qa_count=after_qa_count,
                    rejected_qa_count=qa_rejected_from_input,
                    split_count=split_count,
                )

        return datas + all_rejected

    def __del__(self):
        try:
            if hasattr(self, "logger") and self.logger:
                self.logger.info("Pipeline completed.")
        except Exception:
            pass  # Ignore errors during cleanup
