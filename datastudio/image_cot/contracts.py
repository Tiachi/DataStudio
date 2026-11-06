"""Strict input/configuration contracts and stable identities for image CoT.

No model requests occur in this module. Configuration files are trusted Python,
just like upstream DataStudio configurations; never load untrusted Python files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
from typing import Any
from urllib.parse import urlsplit, urlunsplit

SCHEMA = "image_cot.config/1"
ROLES = ("question_check", "short", "long", "joint_verify")
OUTCOMES = {"pass", "fail", "uncertain", "error", "not_dispatched"}
TERMINALS = {
    "accepted_cot",
    "bypass_verified",
    "input_rejected",
    "candidate_rejected",
    "needs_review",
    "technical_pending",
}


class CoTError(Exception):
    """Expected fail-closed error safe to display without raw payloads."""


class ConfigurationError(CoTError):
    pass


class InvariantError(CoTError):
    pass


class StorageError(CoTError):
    pass


class BudgetExhausted(CoTError):
    pass


def require(condition, reason, error=ConfigurationError):
    if not condition:
        raise error(reason)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _constant(_value):
    raise ConfigurationError("nonfinite_json_constant")


def strict_json(text):
    try:
        return json.loads(text, object_pairs_hook=_unique, parse_constant=_constant)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ConfigurationError("invalid_json") from exc


def canonical(value) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ConfigurationError("noncanonical_value") from exc


def identity(*parts) -> str:
    return sha256(canonical(parts)).hexdigest()


def file_hash(path) -> str:
    result = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def within(path, root) -> bool:
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def media_path(value, root) -> Path:
    require(type(value) is str and value and "\\" not in value, "invalid_media_path")
    relative = PurePosixPath(value)
    require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and not re.match(r"^[a-zA-Z]:", value),
        "media_path_escape",
    )
    path = (Path(root) / value).resolve()
    require(within(path, root) and path.is_file(), "missing_or_escaped_media")
    return path


def positive_int(value, name):
    require(type(value) is int and value > 0, f"positive_integer_required:{name}")
    return value


def normalize_connection(connection):
    """Resolve IP:port or a root URL exactly once; api_base is a full POST URL."""
    result = dict(connection)
    profile = result.get("provider_profile", "openai_compatible")
    require(
        profile in {"openai_compatible", "deepseek_official", "kimi_official"},
        "unknown_provider_profile",
    )
    endpoint = result.get("endpoint")
    if not endpoint and result.get("ip"):
        host = result["ip"]
        require(type(host) is str and not any(c in host for c in "/?#@"), "invalid_ip")
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = positive_int(result.get("port"), "port")
        require(port <= 65535, "invalid_port")
        endpoint = f"http://{host}:{port}"
    require(type(endpoint) is str, "endpoint_required")
    try:
        parsed = urlsplit(endpoint)
        parsed.port
    except ValueError as exc:
        raise ConfigurationError("invalid_endpoint") from exc
    require(
        parsed.scheme in {"http", "https"}
        and parsed.hostname
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment,
        "invalid_endpoint",
    )
    require(
        not parsed.path.rstrip("/").endswith("/chat/completions"),
        "endpoint_must_be_server_root",
    )
    prefix = result.get("api_prefix", "" if profile == "deepseek_official" else "/v1")
    require(
        type(prefix) is str
        and (not prefix or prefix.startswith("/"))
        and ".." not in prefix.split("/")
        and not any(c in prefix for c in "?#\\"),
        "invalid_api_prefix",
    )
    root = parsed.path.rstrip("/")
    require(not (prefix and root.endswith(prefix.rstrip("/"))), "duplicate_api_prefix")
    path = root + prefix.rstrip("/") + "/chat/completions"
    result["api_base"] = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    result["endpoint"] = endpoint.rstrip("/")
    result["provider_profile"] = profile
    require(
        type(result.get("model")) is str and result["model"].strip(), "model_required"
    )
    auth = result.get("auth", {"mode": "none"})
    require(
        type(auth) is dict and auth.get("mode") in {"none", "env_bearer"},
        "invalid_auth",
    )
    if auth["mode"] == "env_bearer":
        require(
            type(auth.get("key_env")) is str
            and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", auth["key_env"]),
            "invalid_key_env",
        )
    result["auth"] = auth
    require(
        not any(k in result for k in ("key", "api_key", "token", "headers")),
        "secrets_not_allowed_in_config",
    )
    if profile == "deepseek_official":
        require(
            parsed.scheme == "https"
            and parsed.hostname == "api.deepseek.com"
            and parsed.port in (None, 443),
            "deepseek_official_host_required",
        )
        require(auth["mode"] == "env_bearer", "deepseek_auth_required")
    if profile == "kimi_official":
        require(
            parsed.scheme == "https"
            and parsed.hostname in {"api.moonshot.ai", "api.moonshot.cn"}
            and parsed.port in (None, 443),
            "kimi_official_host_required",
        )
        require(auth["mode"] == "env_bearer", "kimi_auth_required")
    return result


def load_config(path):
    path = Path(path).resolve()
    config = runpy.run_path(str(path)).get("image_cot")
    require(type(config) is dict, "image_cot_config_required")
    canonical(config)
    return config


def validate_config(config, *, runtime=False, backend="http"):
    """Pure validation/resolution; missing real budgets fail before any dispatch."""
    config = strict_json(canonical(config))
    from .policy import V2_SCHEMA, V3_SCHEMA, validate_policy

    version_two = config.get("schema_version") in {V2_SCHEMA, V3_SCHEMA}
    require(
        config.get("schema_version") in {SCHEMA, V2_SCHEMA, V3_SCHEMA}, "config_schema"
    )
    expected_roles = validate_policy(config) if version_two else set(ROLES)
    require(config.get("mode") in {"pilot", "production", "mock"}, "invalid_mode")
    require(backend in {"http", "mock"}, "invalid_backend")
    require(
        not (backend == "http" and config["mode"] == "mock"),
        "mock_mode_cannot_call_http",
    )
    scope = config.get("scope", {})
    require(scope == {"single_image": True, "single_turn": True}, "unsupported_scope")
    policy = config.get("policy", {})
    for name in ("retry_unresolved", "partial_release"):
        require(policy.get(name, False) is False, f"unsupported_policy:{name}")
    for name in ("max_short_slots", "max_long_slots"):
        require(policy.get(name, 1) == 1, "one_candidate_slot_per_stage")
    require(
        policy.get("prefer_short_if_both_pass", True) is True,
        "unsupported_selection_policy",
    )
    limits = config.get("limits", {})
    require(
        set(limits).issubset(
            {
                "source_records_per_part",
                "source_bytes_per_part",
                "requests_per_microbatch",
                "max_inflight",
                "max_response_bytes",
                "max_request_bytes",
                "max_decoded_pixels_per_batch",
                "max_requests_total",
                "max_output_tokens_total",
                "max_wall_time_s",
            }
        ),
        "unsupported_budget_field_no_currency_budget_implemented",
    )
    for name in (
        "source_records_per_part",
        "source_bytes_per_part",
        "requests_per_microbatch",
        "max_inflight",
        "max_response_bytes",
    ):
        positive_int(limits.get(name), name)
    require(
        limits["requests_per_microbatch"] <= limits["source_records_per_part"],
        "batch_larger_than_part",
    )
    if runtime:
        for name in (
            "max_decoded_pixels_per_batch",
            "max_requests_total",
            "max_output_tokens_total",
            "max_wall_time_s",
            "max_request_bytes",
        ):
            positive_int(limits.get(name), name)
        require(config.get("private_work_root"), "private_work_root_required")
    connections = {
        key: normalize_connection(value)
        for key, value in config.get("connections", {}).items()
    }
    config["connections"] = connections
    models = config.get("models", {})
    require(set(models) == expected_roles, "all_four_model_roles_required")
    for role, model in models.items():
        require(
            type(model.get("request_revision", "1")) is str
            and bool(model.get("request_revision", "1")),
            "request_revision_required",
        )
        ref = model.get("connection_ref")
        if backend == "http":
            require(ref in connections, f"connection_required:{role}")
        require(model.get("transport_retry", 1) == 1, "hidden_retries_forbidden")
        options = model.get("effective_options", {})
        require(type(options) is dict, "options_object_required")
        allowed = {
            "temperature",
            "top_p",
            "seed",
            "thinking",
            "reasoning_effort",
            "chat_template_kwargs",
            "response_format",
        }
        require(set(options).issubset(allowed), "unsupported_request_option")
        if (
            ref in connections
            and connections[ref]["provider_profile"] == "kimi_official"
        ):
            require(
                options.get("reasoning_effort", "max") in {"low", "high", "max"},
                "kimi_unsupported_effort",
            )
            require(
                options.get("thinking") != {"type": "disabled"}, "kimi_always_thinking"
            )
        if (
            ref in connections
            and connections[ref]["provider_profile"] == "deepseek_official"
        ):
            require(
                set(options).issubset(
                    {"thinking", "reasoning_effort", "response_format"}
                ),
                "deepseek_unsupported_option",
            )
            if "thinking" in options:
                require(
                    options["thinking"] in ({"type": "enabled"}, {"type": "disabled"}),
                    "invalid_thinking",
                )
            if "reasoning_effort" in options:
                require(
                    options["reasoning_effort"] in {"low", "high", "max"},
                    "invalid_effort",
                )
        if runtime:
            for name in ("max_output_tokens", "read_timeout_s", "batch_deadline_s"):
                positive_int(model.get(name), f"{role}.{name}")
            require(
                (
                    version_two
                    and role.startswith("generate_")
                    and model.get("prompt") is None
                )
                or (
                    type(model.get("prompt")) is str and Path(model["prompt"]).is_file()
                ),
                f"missing_prompt:{role}",
            )
            if backend == "http":
                auth = connections[ref]["auth"]
                require(
                    auth["mode"] == "none"
                    or bool(os.environ.get(auth.get("key_env", ""))),
                    "api_key_env_missing",
                )
    if runtime and backend == "http":
        pools = config.get("quota_pools", {})
        for role, model in models.items():
            pool = pools.get(model.get("quota_pool"))
            require(type(pool) is dict, f"quota_pool_required:{role}")
            for name in ("rpm", "input_tpm", "output_tpm", "input_tokens_per_request"):
                positive_int(pool.get(name), f"quota.{name}")
            require(
                pool["input_tokens_per_request"] <= pool["input_tpm"]
                and model["max_output_tokens"] <= pool["output_tpm"],
                f"single_request_exceeds_quota:{role}",
            )
        # Sharing a credential/endpoint cannot multiply its effective quota.
        owners = {}
        for role, model in models.items():
            conn = connections[model["connection_ref"]]
            auth = conn["auth"]
            credential = (
                sha256(os.environ.get(auth.get("key_env", ""), "").encode()).hexdigest()
                if auth["mode"] == "env_bearer"
                else "none"
            )
            owner = (conn["endpoint"], auth["mode"], credential)
            require(
                owner not in owners or owners[owner] == model["quota_pool"],
                "shared_account_requires_one_quota_pool",
            )
            owners[owner] = model["quota_pool"]
    return config


def validate_input(original, source_spec, source_hash, row):
    """Semantic input failures are retained; malformed JSON is a prepare error."""
    policy = dict(source_spec["task_policy"])
    category_field = source_spec.get("category_field", "category")
    category = original.get(category_field) if type(original) is dict else None
    policy.update(source_spec.get("category_overrides", {}).get(str(category), {}))
    require(
        policy.get("task_type")
        in {"numeric", "mcq", "open", "pure_ocr", "box_grounding"},
        "unregistered_task_type",
    )
    require(
        policy.get("reference_grade") in {"strong", "weak", "unknown"},
        "reference_grade_required",
    )
    for name in ("max_candidate_chars", "min_image_size"):
        if name in policy:
            positive_int(policy[name], name)
    if "max_image_ratio" in policy:
        require(
            type(policy["max_image_ratio"]) in (int, float)
            and policy["max_image_ratio"] >= 1,
            "invalid_image_ratio",
        )
    if policy["task_type"] == "numeric":
        from decimal import Decimal, InvalidOperation

        for name in ("absolute_tolerance", "relative_tolerance"):
            try:
                value = Decimal(str(policy.get(name, "0")))
            except InvalidOperation as exc:
                raise ConfigurationError("invalid_numeric_tolerance") from exc
            require(value.is_finite() and value >= 0, "invalid_numeric_tolerance")
    transform = policy.get("question_transform", "identity")
    require(
        transform in {"identity", "remove_exact_suffix"},
        "unregistered_question_transform",
    )
    if transform == "remove_exact_suffix":
        require(
            type(policy.get("remove_suffix")) is str and bool(policy["remove_suffix"]),
            "remove_suffix_required",
        )
    q, answer, media, checksum, outcome, reason = "", "", "", "", "pass", ""
    try:
        require(type(original) is dict, "input_object_required")
        require(
            not any(k in original for k in ("data", "messages", "tools")),
            "unsupported_input_schema",
        )
        conversations = original.get("conversations")
        require(
            type(conversations) is list
            and len(conversations) == 2
            and all(type(c) is dict for c in conversations),
            "single_turn_required",
        )
        require(
            [c.get("from") for c in conversations] == ["human", "gpt"]
            and all(
                type(c.get("value")) is str and c["value"].strip()
                for c in conversations
            ),
            "invalid_qa_roles_or_text",
        )
        q, answer = conversations[0]["value"], conversations[1]["value"]
        require(
            not original.get("videos") and not original.get("audios"),
            "unsupported_media_modality",
        )
        images = original.get("images")
        require(type(images) is list and len(images) == 1, "single_image_required")
        require(
            "modal_mask" not in original
            or type(original["modal_mask"]) is int
            and original["modal_mask"] == 1,
            "input_modal_mask_mismatch",
        )
        media = images[0]
        require(
            q.count("<image>") == 1 and "<audio>" not in q and "<video>" not in q,
            "image_placeholder_mismatch",
        )
        path = media_path(media, source_spec["media_root"])
        from PIL import Image

        with Image.open(path) as image:
            require(
                image.width * image.height <= source_spec["max_image_pixels"],
                "source_image_pixel_limit",
            )
            require(getattr(image, "n_frames", 1) == 1, "animated_images_unsupported")
            from datastudio.operators.core.data_item import DataItem
            from datastudio.operators.filters.image_size import ImageSizeFilter

            result = ImageSizeFilter(
                min_size=policy.get("min_image_size", 14),
                max_ratio=policy.get("max_image_ratio", 200),
                require_image=True,
            ).process(DataItem({"image_pil": image}))
            require(not result.has_filter, "upstream_image_size_or_ratio_filter")
            image.verify()
        checksum = file_hash(path)
    except (ConfigurationError, OSError, ValueError) as exc:
        outcome = "fail"
        reason = (
            str(exc) if isinstance(exc, ConfigurationError) else "image_decode_failed"
        )
    original_question = q
    if transform == "remove_exact_suffix" and q.endswith(policy["remove_suffix"]):
        q = q[: -len(policy["remove_suffix"])].rstrip()
    qid = identity(
        source_spec["name"],
        source_spec["revision"],
        source_hash,
        original.get("id", row) if type(original) is dict else row,
        row,
        q,
        media,
        checksum,
    )
    return SourceRecord(
        qid,
        source_spec["name"],
        source_spec["revision"],
        source_hash,
        row,
        original,
        original_question,
        q,
        answer,
        media,
        str(Path(source_spec["media_root"]).resolve()),
        checksum,
        policy,
        outcome,
        reason,
    )


@dataclass(frozen=True)
class SourceRecord:
    question_id: str
    source: str
    source_revision: str
    source_file_hash: str
    row: int
    original: dict
    question_original: str
    question: str
    reference: str
    media: str
    media_root: str
    media_sha256: str
    policy: dict
    input_outcome: str = "pass"
    input_reason: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class ModelTask:
    task_id: str
    attempt_id: str
    operation: str
    question_id: str
    payload: list
    candidate_id: str | None = None
    input_digest: str = ""
    recipe_signature: str = ""

    def intent(self):
        return {
            key: getattr(self, key)
            for key in (
                "task_id",
                "attempt_id",
                "operation",
                "question_id",
                "candidate_id",
                "input_digest",
                "recipe_signature",
            )
        }


@dataclass
class GenerationReceipt:
    task_id: str
    attempt_id: str
    status: str
    raw: dict | None = None
    http_status: int | None = None
    provider_request_id: str | None = None
    error: str | None = None
    elapsed_s: float = 0.0
    request_digest: str | None = None
    reuse_origin: dict | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    question_id: str
    stage: str
    receipt_ref: dict
    text_hash: str
    renderer: str = "honey_compact_native/1"


@dataclass(frozen=True)
class ReviewResolution:
    candidate_id: str
    candidate_text_hash: str
    judge_signature: str
    reviewer: str
    timestamp: str
    evidence: str
    outcome: str


@dataclass
class RuntimeContext:
    sources: dict[str, SourceRecord]
    operation: str
    recipe_signature: str
    dispatch: Any
    candidate_ids: dict[str, str] = field(default_factory=dict)
    outcomes: dict[str, dict] = field(default_factory=dict)
    settings: dict = field(default_factory=dict)

    def record_outcome(self, question_id, outcome):
        require(
            question_id in self.sources and question_id not in self.outcomes,
            "duplicate_or_unknown_outcome",
            InvariantError,
        )
        require(
            outcome.get("status") in OUTCOMES,
            "invalid_operation_outcome",
            InvariantError,
        )
        self.outcomes[question_id] = outcome
