"""Verified selection, private previews, and gated, no-clobber training release."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
import os
import re
import shutil
import uuid

from PIL import Image

from .contracts import (
    GenerationReceipt,
    SourceRecord,
    canonical,
    file_hash,
    identity,
    media_path,
    require,
    within,
)
from .operations import (
    parse_judgment,
    render_candidate,
    task_input_digest,
    validate_review_resolution,
    verify_final,
)
from .storage import (
    RunStore,
    atomic_bytes,
    atomic_json,
    read_json,
    read_jsonl,
    sync_directory,
)

BANNED_FIELDS = {
    "estimated_tokens",
    "token_length",
    "text_tokens",
    "audio_tokens",
    "video_tokens",
    "answer_tokens",
    "audio_tokens_est",
    "text_tokens_est",
    "_cot",
}
RESERVED = {"id", "conversations", "images", "audios", "videos", "modal_mask"}


class TrainingProbe(Protocol):
    """Trainer adapters implement this outside the data generation hot path.

    A returned report must contain actual encode/loss/media and forward/backward
    checks. A mock implementation can never authorize a production release.
    """

    def encode(self, training_record, student_spec): ...
    def check(self, encoded_examples, trainer_spec) -> dict: ...


def verified_ref(store, ref):
    part = (store.root / ref["part"]).resolve()
    require(
        within(part, store.root / "parts")
        and file_hash(part / "commit.json") == ref["commit_sha256"],
        "candidate_commit_identity",
    )
    outcome = store.read_outcome(part, ref["task_id"])
    require(outcome is not None, "candidate_outcome_reference")
    receipt = store.recover_part(part)["receipts"].get(ref["task_id"])
    require(receipt is not None, "candidate_receipt_missing")
    return outcome, GenerationReceipt(**receipt), read_json(part / "plan.json")


def verified_selection(run_root):
    root = Path(run_root).resolve()
    spec, report = read_json(root / "run.json"), read_json(root / "report.json")
    require(
        report["job_id"] == spec["job_id"]
        and report["selection_digest"] == identity(report["selection_parts"]),
        "selection_identity",
    )
    manifest_path = Path(spec["input_manifest"])
    require(
        file_hash(manifest_path) == spec["input_manifest_sha256"],
        "selection_input_changed",
    )
    manifest = read_json(manifest_path)
    for selection_part, input_part in zip(report["selection_parts"], manifest["parts"]):
        path = (root / selection_part["file"]).resolve()
        require(
            within(path, root / "selection")
            and file_hash(path) == selection_part["sha256"],
            "selection_part_changed",
        )
        source_path = manifest_path.parent / input_part["file"]
        require(
            file_hash(source_path) == input_part["sha256"], "export_source_part_changed"
        )
        sources = {
            row["question_id"]: SourceRecord(**row)
            for row in read_jsonl(
                source_path, spec["config"]["limits"]["source_bytes_per_part"]
            )
        }
        selections = list(read_jsonl(path))
        require(
            len(selections) == selection_part["count"] == len(sources)
            and {s["question_id"] for s in selections} == set(sources),
            "selection_source_coverage",
        )
        for selection in selections:
            yield selection, sources[selection["question_id"]]
    require(
        len(report["selection_parts"]) == len(manifest["parts"]), "selection_part_count"
    )


def selected_text(selection, source, store):
    if selection.get("schema_version") == "image_cot.selection/2":
        return selected_text_v2(selection, source, store)
    generation, receipt, _plan = verified_ref(store, selection["receipt_ref"])
    require(
        generation["status"] == "pass"
        and generation["candidate_id"] == selection["candidate_id"]
        and generation["question_id"] == source.question_id,
        "selected_generation_mismatch",
    )
    text = render_candidate(receipt, selection["stage"])
    require(
        identity(text) == selection["text_hash"] == generation["text_hash"],
        "selected_text_changed",
    )
    judgment, judge_receipt, judge_plan = verified_ref(
        store, selection["verification_ref"]
    )
    require(
        judgment["candidate_id"] == selection["candidate_id"]
        and judgment["question_id"] == source.question_id
        and judgment["recipe_signature"] == selection["judge_signature"],
        "selected_judgment_mismatch",
    )
    expected_digest = task_input_digest(
        source, selection["candidate_id"], text, "joint_verify"
    )
    judge_task = next(
        t for t in judge_plan["tasks"] if t["task_id"] == judgment["task_id"]
    )
    require(
        judge_task["input_digest"] == expected_digest,
        "judgment_candidate_text_mismatch",
    )
    judge_status = parse_judgment(judge_receipt, source.policy)["status"]
    resolution = selection.get("review_resolution")
    if resolution:
        require(
            judge_status in {"pass", "fail", "uncertain"},
            "cannot_review_technical_error_as_content",
        )
        judge_status = validate_review_resolution(resolution, generation, judgment)
    require(
        judge_status == "pass" and verify_final(text, source)["status"] == "pass",
        "revalidated_candidate_not_pass",
    )
    return text


def selected_text_v2(selection, source, store):
    """Recompute the cap and every audit from durable receipts, not selection flags."""
    from .operations import (
        parse_rating,
        parse_question_check,
        parse_text_judgment,
        check_candidate,
    )
    from .policy import EFFORTS, cap_for

    config = read_json(store.root / "run.json")["config"]
    policy = config["policy"]
    rated = None
    if source.policy["qa_check_enabled"]:
        outcome, raw, _ = verified_ref(store, selection["question_check_ref"])
        require(
            outcome["question_id"] == source.question_id
            and parse_question_check(raw)["status"] == "pass",
            "v2_question_revalidation",
        )
    if source.policy["rating_enabled"]:
        outcome, raw, _ = verified_ref(store, selection["rating_ref"])
        rating = parse_rating(raw, policy["tiers"])
        require(
            outcome["question_id"] == source.question_id and rating["status"] == "pass",
            "v2_rating_revalidation",
        )
        rated = rating["effort_cap"]
    cap = cap_for(source, policy, rated)
    require(
        selection["effort_cap"] == cap
        and selection["effort"] in policy["tiers"]
        and EFFORTS.index(selection["effort"]) <= EFFORTS.index(cap)
        and selection["stage"] == f"generate_{selection['effort']}",
        "export_effort_cap_violation",
    )
    generation, receipt, plan = verified_ref(store, selection["receipt_ref"])
    text = render_candidate(receipt, selection["stage"])
    require(
        plan["stage"] == selection["stage"]
        and generation["status"] == "pass"
        and generation["question_id"] == source.question_id
        and generation["candidate_id"] == selection["candidate_id"]
        and generation["recipe_signature"] == selection["recipe_signature"]
        and identity(text) == selection["text_hash"] == generation["text_hash"]
        and check_candidate(text, source.policy)["status"] == "pass"
        and verify_final(text, source)["status"] == "pass",
        "v2_generation_revalidation",
    )
    modes = {
        "text_only": ["text_verify"],
        "visual_joint": ["joint_verify"],
        "text_then_visual": ["text_verify", "joint_verify"],
    }
    expected = modes[policy["review_mode"]]
    require(
        selection["review_mode"] == policy["review_mode"]
        and [a["role"] for a in selection["audit_refs"]] == expected,
        "v2_review_role_coverage",
    )
    require(
        selection["visual_audit"]
        == ("checked" if "joint_verify" in expected else "not_checked"),
        "v2_visual_scope_mismatch",
    )
    for audit in selection["audit_refs"]:
        judgment, raw, plan = verified_ref(store, audit["receipt_ref"])
        role = audit["role"]
        parser = parse_text_judgment if role == "text_verify" else parse_judgment
        task = next(t for t in plan["tasks"] if t["task_id"] == judgment["task_id"])
        require(
            judgment["status"] == "pass"
            and judgment["question_id"] == source.question_id
            and judgment["candidate_id"] == selection["candidate_id"]
            and judgment["recipe_signature"] == audit["recipe_signature"]
            and task["operation"] == role
            and task["input_digest"]
            == task_input_digest(source, selection["candidate_id"], text, role)
            and parser(raw, source.policy)["status"] == "pass",
            "v2_judgment_revalidation",
        )
    return text


def training_record(selection, source, text, policy):
    dataset = policy.get("dataset_name")
    require(
        type(dataset) is str and re.fullmatch(r"[A-Za-z0-9_.-]+", dataset),
        "dataset_name_required",
    )
    root = policy.get("training_media_root")
    require(root, "training_media_root_required")
    path = media_path(source.media, root)
    require(file_hash(path) == source.media_sha256, "training_media_identity_mismatch")
    with Image.open(path) as image:
        image.verify()
    row = {
        "id": dataset + "_" + selection["candidate_id"],
        "conversations": [
            {"from": "human", "value": source.question},
            {"from": "gpt", "value": text},
        ],
        "images": [source.media],
        "modal_mask": 1,
    }
    mapping = policy.get("original_field_mapping", {})
    for key, value in source.original.items():
        target = mapping.get(key, key)
        require(
            type(target) is str
            and target
            and target not in row
            and target not in RESERVED
            and target not in BANNED_FIELDS,
            "original_field_mapping_conflict",
        )
        row[target] = value
    require(
        row["conversations"][0]["value"].count("<image>") == 1,
        "export_image_placeholder",
    )
    return row


def verify_release_gates(spec, report, policy):
    require(
        spec["backend"] == "http" and spec["config"]["mode"] != "mock",
        "mock_cannot_publish",
    )
    require(
        spec["implementation"]["clean"]
        and spec["implementation"]["commit"] != "unknown",
        "release_requires_clean_code",
    )
    require(
        report["status"] == "completed"
        and not report["counts"].get("needs_review")
        and not report["counts"].get("technical_pending"),
        "release_has_unresolved_records",
    )
    quality = read_json(policy["quality_gate"]) if policy.get("quality_gate") else None
    probe = (
        read_json(policy["training_probe"]) if policy.get("training_probe") else None
    )
    for gate, schema in (
        (quality, "image_cot.quality_gate/1"),
        (probe, "image_cot.training_probe/1"),
    ):
        require(
            type(gate) is dict
            and gate.get("schema_version") == schema
            and gate.get("status") == "pass"
            and gate.get("selection_digest") == report["selection_digest"]
            and gate.get("job_id") == spec["job_id"]
            and gate.get("real_execution") is True,
            "release_gate_identity_or_status",
        )
    audit = quality.get("human_audit", {})
    n = audit.get("independent_groups")
    require(
        type(n) is int
        and n >= 300
        and audit.get("severe_errors") == 0
        and type(audit.get("severe_errors")) is int
        and n <= report["counts"].get("accepted_cot", 0)
        and audit.get("one_sided_confidence") == 0.95
        and 1 - 0.05 ** (1 / n) <= 0.01
        and audit.get("reviewer")
        and audit.get("sampling_plan")
        and audit.get("representative_strata")
        and audit.get("groups_independent") is True,
        "human_quality_gate_insufficient",
    )
    require(
        type(probe.get("student")) is dict
        and probe["student"].get("profile") == policy.get("student_profile")
        and type(policy.get("student_profile")) is str
        and bool(policy["student_profile"])
        and all(
            probe["student"].get(k)
            for k in (
                "model",
                "model_revision",
                "tokenizer_revision",
                "trainer",
                "trainer_revision",
            )
        ),
        "student_probe_version_required",
    )
    checks = probe.get("checks", {})
    for name in (
        "image_order",
        "bos_eos",
        "reasoning_loss",
        "answer_loss",
        "no_final_truncation",
        "no_duplicate_think",
        "actual_forward_backward",
    ):
        require(checks.get(name) is True, f"student_probe_failed:{name}")
    require(probe.get("sample_digest"), "student_probe_sample_identity")
    return quality, probe


def export_training(run_root, *, preview=True, export_policy=None):
    root = Path(run_root).resolve()
    spec, report = read_json(root / "run.json"), read_json(root / "report.json")
    policy = (
        export_policy if export_policy is not None else spec["config"].get("export", {})
    )
    artifacts = {}
    if not preview:
        verify_release_gates(spec, report, policy)
        for key in (
            "readme_template",
            "release_readme",
            "data_format_path",
            "delivery_validation",
            "quality_gate",
            "training_probe",
        ):
            require(
                policy.get(key) and Path(policy[key]).is_file(),
                f"release_artifact_required:{key}",
            )
            artifacts[key] = file_hash(policy[key])
    export_sig = identity(report["selection_digest"], policy, preview, artifacts)
    target = root / "export_staging" / export_sig
    require(within(target, root), "export_output_path_escape")
    with RunStore(
        root, max_line_bytes=spec["config"]["limits"]["max_response_bytes"] * 8 + 65536
    ) as store:
        if (target / "manifest.json").exists():
            saved = read_json(target / "manifest.json")
            require(
                all(
                    file_hash(target / name) == checksum
                    for name, checksum in saved["files"].items()
                ),
                "export_integrity",
            )
            return target
        target.mkdir(parents=True, exist_ok=True)
        nonce = uuid.uuid4().hex
        dataset_tmp, sidecar_tmp = (
            target / f".{nonce}.jsonl.pending",
            target / f".{nonce}.provenance.pending",
        )
        count, bypass_count, field_types, source_names = 0, 0, {}, set()
        with dataset_tmp.open("xb") as dataset, sidecar_tmp.open("xb") as sidecar:
            current_source_part = None
            for selection, source in verified_selection(root):
                if selection["source_part"] != current_source_part:
                    store.clear_cache()
                    current_source_part = selection["source_part"]
                source_names.add(source.source)
                if selection["terminal"] == "bypass_verified":
                    bypass_count += 1
                    continue  # Separate source preservation; never counted as CoT.
                if selection["terminal"] != "accepted_cot":
                    continue
                text = selected_text(selection, source, store)
                row = training_record(selection, source, text, policy)
                dataset.write(canonical(row) + b"\n")
                sidecar.write(
                    canonical(
                        {
                            "id": row["id"],
                            "selection": selection,
                            "source_file_hash": source.source_file_hash,
                            "source_row": source.row,
                        }
                    )
                    + b"\n"
                )
                for name, value in source.original.items():
                    field_types.setdefault(name, set()).add(type(value).__name__)
                count += 1
            for handle in (dataset, sidecar):
                handle.flush()
                os.fsync(handle.fileno())
        require(
            count == report["counts"].get("accepted_cot", 0),
            "export_accepted_count_mismatch",
        )
        for temporary, name in (
            (dataset_tmp, "data.jsonl"),
            (sidecar_tmp, "provenance.jsonl"),
        ):
            destination = target / name
            if destination.exists():
                require(
                    file_hash(temporary) == file_hash(destination),
                    "export_file_conflict",
                )
                temporary.unlink()  # Our exact derived temporary file, never original media.
            else:
                os.replace(temporary, destination)
        fields = "\n".join(
            f"| {name} | {', '.join(sorted(types))} | 原始字段，按映射保留 |"
            for name, types in sorted(field_types.items())
        )
        sections = {
            "DATASET_NAME": policy["dataset_name"],
            "COMPLETION_DATE": policy.get("completion_date") or "未发布",
            "MAINTAINER": policy.get("maintainer") or "待指定",
            "INTRODUCTION": f"合格图片CoT：{count}条；modal_mask=1。源：{', '.join(sorted(source_names))}。旁路{bypass_count}条未混入CoT。",
            "ORIGINAL_FIELDS": "| 字段 | 类型 | 说明 |\n|---|---|---|\n" + fields,
            "PROCESSING": (
                "统一重新生成；评级为硬上限；从low起首个合格即停；审核模式="
                + spec["config"]["policy"]["review_mode"]
                + "。text_only未逐条核验视觉事实；原始媒体未修改。原始字段映射："
                if spec["config"]["schema_version"] == "image_cot.config/2"
                else "短链生成、严格终答/视觉/过程审核、选择性长链；原始媒体未修改。原始字段映射："
            )
            + canonical(policy.get("original_field_mapping", {})).decode(),
            "PATHS": "训练JSON中的images相对于配置的training_media_root；逐条校验媒体摘要。工作目录路径不代表服务器已发布。",
        }
        if preview:
            readme = "# {{DATASET_NAME}}（PREVIEW，未发布）\n\n完成日期：{{COMPLETION_DATE}}；维护人：{{MAINTAINER}}\n\n"
            readme += "\n\n".join(
                f"## {title}\n\n{{{{{key}}}}}"
                for title, key in (
                    ("一、数据集介绍", "INTRODUCTION"),
                    ("二、原始字段说明", "ORIGINAL_FIELDS"),
                    ("三、处理说明", "PROCESSING"),
                    ("四、路径对应", "PATHS"),
                )
            )
            for key, value in sections.items():
                readme = readme.replace("{{" + key + "}}", value)
        else:
            # Do not reinterpret the user's authoritative template as our own.
            # A completed README and exact-format validation are external inputs.
            require(
                policy.get("release_readme")
                and policy.get("data_format_path")
                and policy.get("delivery_validation"),
                "completed_readme_and_format_validation_required",
            )
            Path(policy["data_format_path"]).read_text(
                encoding="utf-8"
            )  # Current full contract, not a cached summary.
            readme = Path(policy["release_readme"]).read_bytes().decode("utf-8")
            require(
                policy.get("maintainer")
                and policy.get("completion_date")
                and all(
                    str(policy[key]) in readme
                    for key in ("dataset_name", "maintainer", "completion_date")
                )
                and readme.lstrip().startswith("#"),
                "completed_readme_structure_or_identity",
            )
            validation = read_json(policy["delivery_validation"])
            require(
                validation.get("schema_version") == "image_cot.delivery_validation/1"
                and validation.get("status") == "pass"
                and validation.get("real_execution") is True
                and validation.get("selection_digest") == report["selection_digest"]
                and validation.get("data_sha256") == file_hash(target / "data.jsonl")
                and validation.get("readme_sha256")
                == file_hash(policy["release_readme"])
                and validation.get("readme_template_sha256")
                == file_hash(policy["readme_template"])
                and validation.get("format_spec_sha256")
                == file_hash(policy["data_format_path"]),
                "delivery_validation_identity",
            )
            for key in (
                "actual_json_checked",
                "media_paths_checked",
                "authoritative_readme_template_checked",
            ):
                require(
                    validation.get("checks", {}).get(key) is True,
                    "delivery_validation_incomplete",
                )
            require(
                validation.get("training_media_root")
                == str(Path(policy["training_media_root"]).resolve())
                and validation.get("destinations")
                == {
                    key: policy.get(key)
                    for key in (
                        "output_jsonl",
                        "output_readme",
                        "output_provenance",
                        "output_manifest",
                    )
                },
                "delivery_path_correspondence",
            )
        atomic_bytes(target / "README.md", readme.encode("utf-8"))
        files = {
            name: file_hash(target / name)
            for name in ("data.jsonl", "provenance.jsonl", "README.md")
        }
        require(
            all(
                file_hash(policy[key]) == checksum
                for key, checksum in artifacts.items()
            ),
            "release_artifact_changed_during_export",
        )
        manifest = {
            "schema_version": "image_cot.export/1",
            "release_id": export_sig,
            "selection_digest": report["selection_digest"],
            "preview": preview,
            "published": False,
            "backend": spec["backend"],
            "count": count,
            "bypass_count_separate": bypass_count,
            "files": files,
            "export_policy_digest": identity(policy),
            "release_artifact_sha256": artifacts,
            "delivery_template_compliance": (
                "not_certified" if preview else "external_validation_bound"
            ),
        }
        atomic_json(target / "manifest.json", manifest)
    return target


def publish_release(run_root, *, export_policy=None):
    """Publish only generated annotations/docs, never move or rewrite media.

    Hard-link publication from a same-directory temporary is atomic no-clobber.
    Unsupported target filesystems fail closed. The final manifest is last;
    partially published matching files can be verified and reused on rerun.
    """
    target = export_training(run_root, preview=False, export_policy=export_policy)
    root = Path(run_root).resolve()
    spec, report = read_json(root / "run.json"), read_json(root / "report.json")
    policy = export_policy if export_policy is not None else spec["config"]["export"]
    verify_release_gates(spec, report, policy)
    manifest = read_json(target / "manifest.json")
    require(
        all(
            file_hash(policy[key]) == checksum
            for key, checksum in manifest["release_artifact_sha256"].items()
        ),
        "release_artifact_changed_before_publish",
    )
    names = {
        "data.jsonl": "output_jsonl",
        "README.md": "output_readme",
        "provenance.jsonl": "output_provenance",
    }
    destinations = {}
    for key in (*names.values(), "output_manifest"):
        require(
            type(policy.get(key)) is str and Path(policy[key]).is_absolute(),
            "absolute_release_destinations_required",
        )
        path = Path(policy[key]).resolve()
        from .runner import REPOSITORY

        require(
            not within(path, REPOSITORY)
            and not within(path, policy["training_media_root"]),
            "release_overlaps_code_or_media",
        )
        require(not within(root, path), "release_destination_is_run_ancestor")
        for source in spec["config"].get("sources", []):
            require(
                path != Path(source["path"]).resolve()
                and not within(path, source["media_root"]),
                "release_overlaps_source",
            )
        destinations[key] = path
    require(len(set(destinations.values())) == 4, "release_destinations_overlap")
    published = {
        **manifest,
        "published": True,
        "destinations": {key: str(value) for key, value in destinations.items()},
        "evidence_sha256": {
            key: file_hash(policy[key])
            for key in ("quality_gate", "training_probe", "delivery_validation")
        },
    }
    with RunStore(root):
        for name, key in names.items():
            require(
                file_hash(target / name) == manifest["files"][name],
                "staged_release_changed",
            )
            _publish_file(target / name, destinations[key], manifest["files"][name])
        # Logical release commit: every annotation/doc must match before this file.
        atomic_json(target / "published_manifest.json", published)
        _publish_file(
            target / "published_manifest.json",
            destinations["output_manifest"],
            file_hash(target / "published_manifest.json"),
        )
    return destinations["output_manifest"]


def _publish_file(source, destination, expected_hash):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(
            destination.is_file() and file_hash(destination) == expected_hash,
            "release_target_conflict",
        )
        return
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.pending")
    with Path(source).open("rb") as incoming, temporary.open("xb") as output:
        shutil.copyfileobj(incoming, output, 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    require(file_hash(temporary) == expected_hash, "release_copy_integrity")
    try:
        os.link(temporary, destination)
    except FileExistsError:
        require(file_hash(destination) == expected_hash, "release_target_conflict")
    finally:
        temporary.unlink()  # Exact owned generated temporary; never source data.
    sync_directory(destination.parent)
