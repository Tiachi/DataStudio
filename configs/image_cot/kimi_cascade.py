"""V2 template, intentionally no real endpoint credentials, budget or data paths.

Load this template with runpy from a private wrapper and supply source contracts,
quotas and limits. IP/port/name services use the same connection format as base.py.
"""

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
image_cot = runpy.run_path(str(ROOT / "configs/image_cot/base.py"))["image_cot"]
image_cot.update(
    schema_version="image_cot.config/2",
    policy={
        "tiers": ["low", "high", "max"],
        "effort_cap": "max",
        "fixed_effort_cap": "high",
        "qa_check_enabled": False,
        "rating_enabled": True,
        "review_mode": "text_only",
        "regenerate_all": True,
        "retry_unresolved": False,
        "partial_release": False,
    },
)
image_cot["connections"]["kimi_teacher"] = {
    "provider_profile": "kimi_official",
    "endpoint": "https://api.moonshot.cn",
    "api_prefix": "/v1",
    "model": "kimi-k3.1-pro",
    "auth": {"mode": "env_bearer", "key_env": "KIMI_API_KEY"},
}
image_cot["models"] = {
    role: {
        "connection_ref": "kimi_teacher"
        if role.startswith("generate_")
        else "lan_teacher",
        "request_revision": "native-cascade-v2",
        "prompt": None
        if role.startswith("generate_")
        else str(ROOT / "prompts/image_cot/v2" / f"{role}.txt"),
        "max_output_tokens": None,
        "read_timeout_s": None,
        "batch_deadline_s": None,
        "effective_options": {"reasoning_effort": role.removeprefix("generate_")}
        if role.startswith("generate_")
        else {},
        "transport_retry": 1,
        "quota_pool": "teacher_account"
        if role.startswith("generate_")
        else "review_account",
    }
    for role in (
        "effort_rating",
        "generate_low",
        "generate_high",
        "generate_max",
        "text_verify",
    )
}
# Optional roles: question_check and joint_verify. Add their model entries when
# enabling QA checks or visual_joint/text_then_visual (see v2 operations manual).
