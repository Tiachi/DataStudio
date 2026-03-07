"""Example 1: Rule-based filtering only (no MLLM model required)

Use case: Quick data cleaning for text data. Filters out samples with format
anomalies, abnormal lengths, or repetitive content.
No inference service needed, runs on CPU.

Inherits from:
    - _base_/dataset.py: default dataloader/datasaver parameters
    - _base_/filters/filter_rule_base_for_answer.py: question + answer rule filters (priority=50)

Usage:
    python run.py -c configs/examples/rule_filter_only.py
"""

_base_ = [
    "@/_base_/dataset.py",
    "@/_base_/filters/filter_rule_base_for_answer.py",
]

work_dir = "./work_dirs/examples/rule_filter_only"

logger = dict(type="Logger", log_file="logs/rule_filter_only.log")

# ---- Dataset ----
# Use demo data included in the repository
dataloader = dict(
    dataset="demo_data.yaml",
    data_root="configs/examples/demo_data",  # Base path for resolving relative paths
    batch_size=10000,
    use_image=False,  # Text-only processing, no image loading
)

datasaver = dict(
    dataset="configs/examples/demo_data/demo_data.yaml",
    output_dir="./output/examples/rule_filter_only",
    save_yaml_name="rule_filtered",
)

# Pipeline is inherited from _base_/filters/filter_rule_base_for_answer.py
# Includes: ImageSizeFilter, ImageAspectRatioFilter, ImageExtFilter,
#           ConvLengthFilter, LengthAnomalyFilter (question + answer), TextRepeatFilter
# Note: Image filters are no-ops when use_image=False (they simply pass through)
