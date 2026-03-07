"""Example 2: MLLM quality filtering

Use case: Use a large language model for deep quality assessment. Rule-based
filters run first to quickly remove obviously abnormal data, then the MLLM
performs semantic-level quality evaluation on the remaining samples.

Inherits from:
    - _base_/dataset.py: default dataloader/datasaver parameters
    - _base_/models/local_api_model.py: default model parameters
    - _base_/filters/filter_rule_base_for_question.py: question rule filters (priority=0)

Prerequisites:
    Deploy an OpenAI-compatible inference service, e.g.:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-VL-30B-A3B-Instruct \
        --port 8000 --tensor-parallel-size 4

    Tip: If you have multiple LLM backends, use the bundled LLMRouter
    (tools/LLMRouter/) as a reverse proxy for load balancing and auto
    failover, then point ``api_base`` to the router address.

Usage:
    python run.py -c configs/examples/mllm_quality_filter.py
"""

_base_ = [
    "@/_base_/models/local_api_model.py",
    "@/_base_/dataset.py",
    "@/_base_/filters/filter_rule_base_for_question.py",
]

work_dir = "./work_dirs/examples/mllm_quality_filter"

logger = dict(type="Logger", log_file="logs/mllm_quality_filter.log")

# ---- Dataset ----
dataloader = dict(
    dataset="demo_data.yaml",
    data_root="configs/examples/demo_data",
    batch_size=5000,
    use_image=True,
    cache_dir="./cache/demo_lmdb",
    resize_image_size=2048,
)

datasaver = dict(
    dataset="configs/examples/demo_data/demo_data.yaml",
    output_dir="./output/examples/mllm_quality_filter",
    save_yaml_name="mllm_filtered",
)

# ---- Model ----
model = dict(
    model="Qwen3-VL-30B-A3B-Instruct",
    api_base="http://127.0.0.1",
    port=8000,
    thread_num=512,
    return_dict=True,   # MLLM filter requires JSON output
    max_tokens=2048,
)

# ---- MLLM Filter (priority=10, runs after inherited rule filters at priority=0) ----
# Question-image consistency check: determine if the question is relevant to the image
qi_consist_request = dict(
    type="RequestBuilder",
    prompt="prompts/filter/question_image_consist_v2.txt",
    key_templates={"result": "q{idx}", "reason": "q{idx}_reason"},
    with_image=True,
    with_question=True,
    with_answer=False,
)

mllm_qi_filter = dict(
    cfg=dict(
        type="SubPipeline",
        operators=[
            dict(
                type="MLLMFilter",
                request_builder=qi_consist_request,
            ),
        ],
    ),
    priority=10,  # Runs after inherited rule-based filters (priority=0)
)

# Pipeline: rule filters inherited from _base_ (priority=0) + MLLM filter (priority=10)
pipeline = dict(
    type="Pipeline",
    operations={
        "mllm_qi_filter": mllm_qi_filter,
    },
)
