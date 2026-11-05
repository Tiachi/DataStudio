"""DeepSeek official small-pilot template. Fill private paths and budgets first.

All four roles share one key and therefore one quota pool. The public model ID
is deepseek-flash, not the model's display name. Never put the actual key here.
"""

from pathlib import Path
import runpy

image_cot = runpy.run_path(str(Path(__file__).with_name("base.py")))["image_cot"]
for role, model in image_cot["models"].items():
    model["connection_ref"] = "deepseek_validation"
    model["effective_options"] = {
        "thinking": {"type": "enabled" if role == "long" else "disabled"}
    }
    if role == "long":
        model["effective_options"]["reasoning_effort"] = "high"

# Define sources explicitly, for example:
# image_cot['sources'] = [{
#     'schema': 'standard_qa_v1', 'name': 'my_source', 'revision': '<revision>',
#     'path': '<absolute annotation JSONL>', 'media_root': '<absolute media root>',
#     'max_image_pixels': <integer>,
#     'task_policy': {'task_type': 'numeric', 'reference_grade': 'strong'},
# }]
# Set private_work_root, all None budgets/timeouts, and teacher_account quotas.
# Do not infer quota or production speed from this intentionally incomplete file.
