"""Example 3: MLLM answer rewriting

Use case: Use a large language model to regenerate answers, e.g., producing
more detailed or accurate responses. Suitable for data augmentation or
answer quality improvement.

Prerequisites:
    Deploy an OpenAI-compatible inference service.

    Tip: If you have multiple LLM backends, use the bundled LLMRouter
    (tools/LLMRouter/) as a reverse proxy for load balancing and auto
    failover, then point ``api_base`` to the router address.

Usage:
    python run.py -c configs/examples/mllm_answer_rewrite.py
"""

_base_ = [
    "@/_base_/models/local_api_model.py",
    "@/_base_/dataset.py",
]

work_dir = "./work_dirs/examples/mllm_answer_rewrite"

logger = dict(type="Logger", log_file="logs/mllm_answer_rewrite.log")

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
    output_dir="./output/examples/mllm_answer_rewrite",
    save_yaml_name="rewritten",
)

# ---- Model ----
# Plain text mode: model outputs new answer directly (no JSON parsing)
model = dict(
    model="Qwen3-VL-30B-A3B-Instruct",
    api_base="http://127.0.0.1",
    port=8000,
    thread_num=512,
    return_dict=False,   # Plain text output
    max_tokens=4096,
)

# ---- MLLM Rewriter ----
# Plain text mode: send original question + image to model, get new answer directly
rewriter_request = dict(
    type="RequestBuilder",
    prompt=None,              # No template, send original question as-is
    key_templates=None,       # Plain text mode, no JSON parsing
    with_image=True,
    with_question=True,
    with_answer=False,
)

mllm_rewriter = dict(
    cfg=dict(
        type="SubPipeline",
        operators=[
            dict(
                type="MLLMRewriter",
                request_builder=rewriter_request,
                rewrite_type="answer",  # Rewrite answer (original saved to ori_answer)
            ),
        ],
    ),
    priority=0,
)

pipeline = dict(
    type="Pipeline",
    operations={
        "mllm_rewriter": mllm_rewriter,
    },
)
