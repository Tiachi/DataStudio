"""HoneyPipe: Complete data processing pipeline
(rule filter + MLLM filter + MLLM rewrite + consistency check + normalization)

This is a fully self-contained example with all configuration expanded inline,
serving as a reference for understanding every available option.

Use case: End-to-end data processing pipeline covering all stages from raw
data to high-quality training data.

Execution order (by priority, lowest first):
    priority=0   Rule filters: quickly remove obviously abnormal data
    priority=10  MLLM filter: semantic-level quality assessment
    priority=20  MLLM rewriter: regenerate answers with a large model
    priority=30  Consistency check: compare old vs new answers, filter inconsistent ones
    priority=40  Normalization: clean tags and standardize formats

Prerequisites:
    Deploy an OpenAI-compatible inference service.

Usage:
    python run.py -c configs/examples/honeypipe.py
"""

_base_ = [
    "@/_base_/models/local_api_model.py",
    "@/_base_/dataset.py",
]

work_dir = "./work_dirs/examples/honeypipe"

logger = dict(type="Logger", log_file="logs/honeypipe.log")

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
    output_dir="./output/examples/honeypipe",
    save_yaml_name="honeypipe_processed",
)

# ---- Model ----
model = dict(
    model="Qwen3-VL-30B-A3B-Instruct",
    api_base="http://127.0.0.1",
    port=8000,
    thread_num=1024,
    return_dict=True,
    max_tokens=4096,
)

# ---- Stage 1: Rule filters (priority=0) ----
rule_filters = dict(
    cfg=dict(
        type="SubPipeline",
        operators=[
            dict(type="ConvLengthFilter", min_length=1, max_length=20),
            dict(type="ImageSizeFilter", min_size=14),
            dict(type="ImageAspectRatioFilter", max_aspect_ratio=200),
            dict(type="ImageExtFilter"),
            dict(
                type="LengthAnomalyFilter",
                min_length=2,
                max_length=4096,
                check_question=True,
                check_answer=True,
                use_tokenizer=True,
            ),
            dict(
                type="TextRepeatFilter",
                check_question=True,
                check_answer=True,
            ),
        ],
    ),
    priority=0,
)

# ---- Stage 2: MLLM filter (priority=10) ----
mllm_filter_request = dict(
    type="RequestBuilder",
    prompt="prompts/filter/question_image_consist_v2.txt",
    key_templates={"result": "q{idx}", "reason": "q{idx}_reason"},
    with_image=True,
    with_question=True,
    with_answer=False,
)

mllm_filter = dict(
    cfg=dict(
        type="SubPipeline",
        operators=[
            dict(
                type="MLLMFilter",
                request_builder=mllm_filter_request,
            ),
        ],
    ),
    priority=10,
)

# ---- Stage 3: MLLM rewriter (priority=20) ----
# Rewriter uses plain text mode (return_dict=False), model outputs new answer directly.
# Different from the global model (return_dict=True used by filters), so we specify
# a per-operator model override here.
mllm_rewriter_model = dict(
    type="OpenAIAPI",
    model="Qwen3-VL-30B-A3B-Instruct",
    api_base="http://127.0.0.1",
    port=8000,
    thread_num=1024,
    return_dict=False,   # Plain text output for rewriting
    max_tokens=4096,
)

mllm_rewriter_request = dict(
    type="RequestBuilder",
    prompt=None,
    key_templates=None,
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
                model=mllm_rewriter_model,
                request_builder=mllm_rewriter_request,
                rewrite_type="answer",
            ),
        ],
    ),
    priority=20,
)

# ---- Stage 4: Consistency check (priority=30) ----
consist_request = dict(
    type="RequestBuilder",
    prompt="prompts/filter/compare_answer_v2.txt",
    key_templates={"result": "q{idx}", "reason": "q{idx}_reason"},
    with_image=True,
    with_answer=True,
    with_original=True,
)

consist_filter = dict(
    cfg=dict(
        type="SubPipeline",
        operators=[
            dict(
                type="MLLMFilter",
                request_builder=consist_request,
            ),
        ],
    ),
    priority=30,
)

# ---- Stage 5: Format normalization (priority=40) ----
norm_rewriters = dict(
    cfg=dict(
        type="SubPipeline",
        operators=[
            dict(type="RemoveThinkRewriter"),    # Remove <think>...</think> tags
            dict(type="NormImageTagRewriter"),    # Standardize <image> tag placement
        ],
    ),
    priority=40,
)

# ---- Assemble pipeline ----
pipeline = dict(
    type="Pipeline",
    operations={
        "rule_filters": rule_filters,
        "mllm_filter": mllm_filter,
        "mllm_rewriter": mllm_rewriter,
        "consist_filter": consist_filter,
        "norm_rewriters": norm_rewriters,
    },
)
