"""Example: GUI grounding rewriting with MAI-UI-8B model.

Use case: Use MAI-UI-8B to locate UI elements on screenshots based on
grounding instructions. The model takes a screenshot + instruction and
returns coordinates of the target element.

Prerequisites:
    Deploy an OpenAI-compatible inference service with the MAI-UI-8B model.

Usage:
    python run.py -c configs/examples/gui_grounding/mai-ui-8b_gui-grounding.py
"""

_base_ = ["@/_base_/models/local_api_model.py", "@/_base_/dataset.py"]

work_dir = "./work_dirs/examples/gui_grounding"

logger = dict(type="Logger", log_file="logs/mai-ui-8b_gui-grounding.log")

# ---- Dataset ----
# Replace with your actual grounding dataset yaml path
dataset_yaml = "/path/to/your/grounding_dataset.yaml"

dataloader = dict(
    dataset=dataset_yaml,
    batch_size=100000,
    use_image=True,
    cache_dir="./cache/images_lmdb_gui_grounding",
    resize_image_size=4096,
)

datasaver = dict(
    dataset=dataset_yaml,
    output_dir="./output/examples/gui_grounding",
    save_yaml_name="mai-ui-8b_gui-grounding",
)

# ---- Model ----
# Plain text mode: grounding response is raw text (coordinates), not JSON
model = dict(
    model="MAI-UI-8B",
    return_dict=False,
    thread_num=8192,
)

# ---- GUI Grounding Rewriter ----
grounding_request_builder = dict(
    type="RequestBuilder",
    prompt=None,  # No user-level prompt; question is the grounding instruction
    system_prompt="prompts/grounding/mai-ui_system_prompt.txt",
    key_templates=None,  # Raw text response, no JSON parsing
    with_image=True,
    with_answer=False,
)

rewriter_grounding = dict(
    cfg=dict(
        type="SubPipeline",
        operators=[
            dict(
                type="MLLMRewriter",
                request_builder=grounding_request_builder,
                rewrite_type="answer",
            )
        ],
    ),
    priority=0,
)

# ---- Assemble pipeline ----
pipeline = dict(
    type="Pipeline",
    operations={
        "rewriter_grounding": rewriter_grounding,
    },
)
