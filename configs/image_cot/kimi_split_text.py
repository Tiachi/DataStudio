"""V3 template: mandatory four-way rating, independent answer and CoT audits.

Load via runpy from a private wrapper; provide actual budgets, sources and model
services. This file contains neither credentials nor an executable paid budget.
"""

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
image_cot = runpy.run_path(str(ROOT / "configs/image_cot/kimi_cascade.py"))["image_cot"]
image_cot["schema_version"] = "image_cot.config/3"
image_cot["policy"].update(review_mode="split_text", rating_enabled=True)
del image_cot["policy"]["fixed_effort_cap"]
base_review = image_cot["models"].pop("text_verify")
for role in ("answer_verify", "reasoning_verify"):
    image_cot["models"][role] = {
        **base_review,
        "request_revision": "split-text-v3",
        "effective_options": {},
        "prompt": str(ROOT / "prompts/image_cot/v3" / f"{role}.txt"),
    }
image_cot["models"]["effort_rating"].update(
    prompt=str(ROOT / "prompts/image_cot/v3/effort_rating.txt"),
    request_revision="quality-rating-v3",
)
# Keep the rating endpoint on a vision-capable model. The two audit endpoints may
# be different text reasoning models; configure their connection_ref/quota_pool.
