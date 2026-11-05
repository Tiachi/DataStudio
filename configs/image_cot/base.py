"""Explicit, intentionally incomplete image-CoT configuration; no API calls."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
image_cot = {
    "schema_version": "image_cot.config/1",
    "mode": "pilot",
    "scope": {"single_image": True, "single_turn": True},
    "private_work_root": None,
    "input_manifest": None,
    "sources": [],
    "reuse_from": [],
    "review_resolutions": None,
    "limits": {
        "source_records_per_part": 256,
        "source_bytes_per_part": 16 * 1024 * 1024,
        "requests_per_microbatch": 16,
        "max_inflight": 8,
        "max_response_bytes": 2 * 1024 * 1024,
        "max_request_bytes": None,
        "max_decoded_pixels_per_batch": None,
        "max_requests_total": None,
        "max_output_tokens_total": None,
        "max_wall_time_s": None,
    },
    "connections": {
        "lan_teacher": {
            "provider_profile": "openai_compatible",
            "ip": "192.0.2.10",
            "port": 8000,
            "api_prefix": "/v1",
            "model": "replace-with-served-model-name",
            "auth": {"mode": "none"},
        },
        "deepseek_validation": {
            "provider_profile": "deepseek_official",
            "endpoint": "https://api.deepseek.com",
            "api_prefix": "",
            "model": "deepseek-flash",
            "auth": {"mode": "env_bearer", "key_env": "DEEPSEEK_API_KEY"},
        },
    },
    "models": {
        role: {
            "connection_ref": "lan_teacher",
            "request_revision": "1",
            "prompt": str(ROOT / "prompts" / "image_cot" / f"{role}.txt"),
            "max_output_tokens": None,
            "read_timeout_s": None,
            "batch_deadline_s": None,
            "effective_options": {},
            "transport_retry": 1,
            "quota_pool": "teacher_account",
        }
        for role in ("question_check", "short", "long", "joint_verify")
    },
    "quota_pools": {},
    "policy": {
        "max_short_slots": 1,
        "max_long_slots": 1,
        "long_source_allowlist": [],
        "prefer_short_if_both_pass": True,
        "retry_unresolved": False,
        "partial_release": False,
    },
    "export": {
        "dataset_name": None,
        "training_media_root": None,
        "original_field_mapping": {},
        "student_profile": None,
        "quality_gate": None,
        "training_probe": None,
        "readme_template": None,
        "release_readme": None,
        "data_format_path": None,
        "delivery_validation": None,
        "completion_date": None,
        "maintainer": None,
        "output_jsonl": None,
        "output_readme": None,
        "output_provenance": None,
        "output_manifest": None,
    },
}
