"""Version-two policy: supported tiers, immutable caps and explicit audit scope."""

from .contracts import require

V2_SCHEMA = "image_cot.config/2"
V3_SCHEMA = "image_cot.config/3"
EFFORTS = ("low", "high", "max")
RATINGS = ("none", *EFFORTS)


def is_v3(config):
    return config.get("schema_version") == V3_SCHEMA


def is_v2(config):
    return config.get("schema_version") in {V2_SCHEMA, V3_SCHEMA}


def stage_roles(config):
    if not is_v2(config):
        return {
            "question_check": "question_check",
            "short": "short",
            "short_verify": "joint_verify",
            "long": "long",
            "long_verify": "joint_verify",
        }
    policy = config["policy"]
    stages = {}
    if policy["qa_check_enabled"] or any(
        s.get("qa_check_enabled") for s in config.get("sources", [])
    ):
        stages["question_check"] = "question_check"
    if policy["rating_enabled"] or any(
        s.get("rating_enabled") for s in config.get("sources", [])
    ):
        stages["effort_rating"] = "effort_rating"
    for tier in policy["tiers"]:
        stages[f"generate_{tier}"] = f"generate_{tier}"
        if is_v3(config):
            stages[f"answer_{tier}_verify"] = "answer_verify"
            stages[f"reasoning_{tier}_verify"] = "reasoning_verify"
        if policy["review_mode"] in {"text_only", "text_then_visual"}:
            stages[f"text_{tier}_verify"] = "text_verify"
        if policy["review_mode"] in {"visual_joint", "text_then_visual"}:
            stages[f"visual_{tier}_verify"] = "joint_verify"
    return stages


def validate_policy(config):
    p = config.get("policy", {})
    allowed = {
        "tiers",
        "effort_cap",
        "fixed_effort_cap",
        "qa_check_enabled",
        "rating_enabled",
        "review_mode",
        "regenerate_all",
        "retry_unresolved",
        "partial_release",
    }
    require(set(p).issubset(allowed), "unknown_v2_policy")
    require(p.get("regenerate_all") is True, "v2_requires_regenerate_all")
    require(
        not config.get("review_resolutions"),
        "v2_rejudge_receipts_instead_of_manual_override",
    )
    for flag in ("qa_check_enabled", "rating_enabled"):
        require(type(p.get(flag)) is bool, f"explicit_boolean_required:{flag}")
    require(
        type(p.get("tiers")) is list
        and p["tiers"]
        and all(type(t) is str and t in EFFORTS for t in p["tiers"])
        and p["tiers"] == [t for t in EFFORTS if t in p["tiers"]]
        and p["tiers"][0] == "low",
        "invalid_supported_tiers",
    )
    require(p.get("effort_cap") in p["tiers"], "explicit_global_effort_cap_required")
    if p.get("fixed_effort_cap") is not None:
        require(p["fixed_effort_cap"] in p["tiers"], "invalid_fixed_cap")
    require(
        p.get("review_mode")
        in (
            {"split_text"}
            if is_v3(config)
            else {"text_only", "visual_joint", "text_then_visual"}
        ),
        "invalid_review_mode",
    )
    if is_v3(config):
        require(p["rating_enabled"] is True, "v3_rating_is_mandatory")
    for source in config.get("sources", []):
        require(
            set(source).issubset(
                {
                    "schema",
                    "name",
                    "revision",
                    "path",
                    "media_root",
                    "max_image_pixels",
                    "task_policy",
                    "category_field",
                    "category_overrides",
                    "scope_attested_non_math_code",
                    "qa_check_enabled",
                    "rating_enabled",
                    "effort_cap",
                    "fixed_effort_cap",
                    "allowed_categories",
                    "excluded_categories",
                    "reference_field",
                }
            ),
            "unknown_v2_source_option",
        )
        require(
            source.get("scope_attested_non_math_code") is True,
            "source_non_math_code_scope_attestation_required",
        )
        if is_v3(config):
            require(
                source.get("rating_enabled", True) is True,
                "v3_source_rating_is_mandatory",
            )
        for flag in ("qa_check_enabled", "rating_enabled"):
            require(
                flag not in source or type(source[flag]) is bool,
                "invalid_source_switch",
            )
        require(
            source.get("effort_cap", p["effort_cap"]) in p["tiers"],
            "invalid_source_cap",
        )
        if not source.get("rating_enabled", p["rating_enabled"]):
            require(
                source.get("fixed_effort_cap", p.get("fixed_effort_cap")) in p["tiers"],
                "rating_off_requires_explicit_fixed_cap",
            )
    if not p["rating_enabled"]:
        require(
            p.get("fixed_effort_cap") in p["tiers"],
            "rating_off_requires_explicit_fixed_cap",
        )
    required = set(stage_roles(config).values())
    require(set(config.get("models", {})) == required, "v2_model_role_coverage")
    for tier in p["tiers"]:
        options = config["models"][f"generate_{tier}"].get("effective_options", {})
        require(options.get("reasoning_effort") == tier, "tier_request_effort_mismatch")
        require(
            options.get("thinking") != {"type": "disabled"},
            "native_tiers_require_thinking",
        )
    return required


def cap_for(source, policy, rating=None):
    caps = [policy["effort_cap"], source.policy["effort_cap"]]
    if source.policy["rating_enabled"]:
        require(rating in policy["tiers"], "rating_cap_not_supported")
        caps.append(rating)
    else:
        caps.append(source.policy["fixed_effort_cap"])
    return min(caps, key=EFFORTS.index)


def source_policy(source, config):
    p = config["policy"]
    return {
        key: source.get(key, p.get(key))
        for key in (
            "qa_check_enabled",
            "rating_enabled",
            "effort_cap",
            "fixed_effort_cap",
        )
    }
