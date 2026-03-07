"""Example 5: Text normalization (rule-based rewriters only)

Use case: Clean up and standardize existing data formats without any quality
filtering. For example, removing think tags, standardizing image tags, and
cleaning up extra formatting. No MLLM model required.

The demo data contains <think>...</think> tags in all answers,
making it a good test case for this normalization pipeline.

Usage:
    python run.py -c configs/examples/text_normalization.py
"""

_base_ = ["@/_base_/dataset.py"]

work_dir = "./work_dirs/examples/text_normalization"

logger = dict(type="Logger", log_file="logs/text_normalization.log")

# ---- Dataset ----
dataloader = dict(
    dataset="demo_data.yaml",
    data_root="configs/examples/demo_data",
    batch_size=10000,
    use_image=False,
)

datasaver = dict(
    dataset="configs/examples/demo_data/demo_data.yaml",
    output_dir="./output/examples/text_normalization",
    save_yaml_name="normalized",
)

# ---- Pipeline ----
pipeline = dict(
    type="Pipeline",
    operations={
        "rewriters": dict(
            cfg=dict(
                type="SubPipeline",
                operators=[
                    # Remove <think>...</think> tags and their content
                    dict(type="RemoveThinkRewriter"),
                    # Standardize <image> tag placement (move to beginning of question)
                    dict(type="NormImageTagRewriter"),
                    # Standardize prompt format
                    dict(type="NormPromptRewriter"),
                    # Remove reasoning prefix content
                    dict(type="RemoveReasonRewriter"),
                ],
            ),
            priority=0,
        ),
    },
)
