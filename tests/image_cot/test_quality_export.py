from dataclasses import replace
from pathlib import Path

import pytest

from datastudio.image_cot.contracts import ConfigurationError, canonical, validate_input
from datastudio.image_cot.export import (
    verify_release_gates,
    export_training,
    _publish_file,
)
from datastudio.image_cot.operations import (
    validate_bypass,
    validate_review_resolution,
    task_input_digest,
)
from datastudio.image_cot.runner import ModelHandle, apply_reviews, run
from datastudio.image_cot.audited_api import MockAPI
from datastudio.image_cot.storage import read_json, read_jsonl


def test_box_bypass_requires_registered_format_and_valid_coordinates(cot_config):
    spec = cot_config["sources"][0]
    record = validate_input(next(read_jsonl(spec["path"])), spec, "hash", 1)
    source = replace(
        record,
        reference="[0.1, 0.2, 0.8, 0.9]",
        policy={"task_type": "box_grounding", "box_format": "xyxy_normalized"},
    )
    assert validate_bypass(source)["status"] == "pass"
    assert (
        validate_bypass(replace(source, reference="[0.8, 0.2, 0.1, 0.9]"))["status"]
        == "uncertain"
    )
    assert (
        validate_bypass(replace(source, policy={"task_type": "box_grounding"}))[
            "status"
        ]
        == "uncertain"
    )


def test_gold_policy_changes_judge_identity_but_not_generation(cot_config):
    spec = cot_config["sources"][0]
    source = validate_input(next(read_jsonl(spec["path"])), spec, "hash", 1)
    revised = replace(source, policy={**source.policy, "relative_tolerance": "0.001"})
    assert task_input_digest(source, "candidate", "text", "short") == task_input_digest(
        revised, "candidate", "text", "short"
    )
    assert task_input_digest(
        source, "candidate", "text", "joint_verify"
    ) != task_input_digest(revised, "candidate", "text", "joint_verify")


def test_review_resolution_cannot_override_wrong_text_or_technical_error():
    candidate = {"candidate_id": "c", "text_hash": "hash"}
    judge = {"recipe_signature": "judge", "status": "uncertain"}
    resolution = {
        "schema_version": "image_cot.review_resolution/1",
        "candidate_id": "c",
        "candidate_text_hash": "hash",
        "judge_signature": "judge",
        "reviewer": "synthetic-human",
        "timestamp": "2026-11-05T12:00:00+08:00",
        "evidence": "synthetic audit fixture",
        "outcome": "pass",
    }
    assert validate_review_resolution(resolution, candidate, judge) == "pass"
    with pytest.raises(ConfigurationError):
        validate_review_resolution(
            {**resolution, "candidate_text_hash": "different"}, candidate, judge
        )
    with pytest.raises(ConfigurationError, match="technical"):
        apply_reviews(
            {"q": {**judge, "status": "error"}}, {"q": candidate}, {"c": resolution}
        )


def test_reordered_batch_receipts_are_matched_by_identity(cot_config, monkeypatch):
    path = Path(cot_config["sources"][0]["path"])
    row = next(read_jsonl(path))
    path.write_bytes(
        canonical(row) + b"\n" + canonical({**row, "id": "second"}) + b"\n"
    )
    monkeypatch.setattr(
        ModelHandle,
        "generate_audited",
        lambda self, tasks, **kwargs: list(reversed(MockAPI().generate_audited(tasks))),
    )
    root = run(cot_config, backend="mock")
    assert read_json(root / "report.json")["counts"] == {"accepted_cot": 2}
    output = export_training(root)
    rows = list(read_jsonl(output / "data.jsonl"))
    assert len({r["id"] for r in rows}) == 2


def test_duplicate_receipts_are_quarantined_and_not_passed(cot_config, monkeypatch):
    def duplicate(self, tasks, **kwargs):
        receipt = MockAPI().generate_audited(tasks)[0]
        return [receipt, receipt]

    monkeypatch.setattr(ModelHandle, "generate_audited", duplicate)
    with pytest.raises(Exception, match="coverage"):
        run(cot_config, backend="mock")
    roots = list((Path(cot_config["private_work_root"]) / "runs").iterdir())
    parts = list((roots[0] / "parts/input-000000/question_check").iterdir())
    assert list(parts[0].glob("quarantine-*.jsonl"))
    assert not (parts[0] / "commit.json").exists()


def test_release_gates_bind_selection_real_loss_and_human_audit(tmp_path):
    # Specification fixtures only; this is not a real human/model quality result.
    spec = {
        "backend": "http",
        "config": {"mode": "pilot"},
        "implementation": {"clean": True, "commit": "hash"},
        "job_id": "job",
    }
    report = {
        "status": "completed",
        "counts": {"accepted_cot": 300},
        "selection_digest": "selected",
    }
    common = {
        "status": "pass",
        "selection_digest": "selected",
        "job_id": "job",
        "real_execution": True,
    }
    quality = {
        **common,
        "schema_version": "image_cot.quality_gate/1",
        "human_audit": {
            "independent_groups": 300,
            "severe_errors": 0,
            "one_sided_confidence": 0.95,
            "reviewer": "synthetic",
            "sampling_plan": "synthetic",
            "representative_strata": ["synthetic"],
            "groups_independent": True,
        },
    }
    probe = {
        **common,
        "schema_version": "image_cot.training_probe/1",
        "sample_digest": "samples",
        "student": dict.fromkeys(
            (
                "model",
                "model_revision",
                "tokenizer_revision",
                "trainer",
                "trainer_revision",
            ),
            "synthetic",
        ),
        "checks": dict.fromkeys(
            (
                "image_order",
                "bos_eos",
                "reasoning_loss",
                "answer_loss",
                "no_final_truncation",
                "no_duplicate_think",
                "actual_forward_backward",
            ),
            True,
        ),
    }
    probe["student"]["profile"] = "test_student"
    qpath, ppath = tmp_path / "quality.json", tmp_path / "probe.json"
    policy = {
        "quality_gate": str(qpath),
        "training_probe": str(ppath),
        "student_profile": "test_student",
    }
    qpath.write_bytes(canonical(quality))
    ppath.write_bytes(canonical(probe))
    assert verify_release_gates(spec, report, policy)
    for bad_probe in (
        {**probe, "selection_digest": "other"},
        {**probe, "real_execution": False},
        {**probe, "checks": {**probe["checks"], "reasoning_loss": False}},
    ):
        ppath.write_bytes(canonical(bad_probe))
        with pytest.raises(ConfigurationError):
            verify_release_gates(spec, report, policy)


def test_publication_is_no_clobber_and_safely_reusable(tmp_path):
    from datastudio.image_cot.contracts import file_hash

    source = tmp_path / "generated.txt"
    target = tmp_path / "release" / "generated.txt"
    source.write_bytes(b"reviewed synthetic output\n")
    checksum = file_hash(source)
    _publish_file(source, target, checksum)
    _publish_file(source, target, checksum)
    assert target.read_bytes() == source.read_bytes()
    target.write_bytes(b"existing unrelated work\n")
    with pytest.raises(ConfigurationError, match="conflict"):
        _publish_file(source, target, checksum)
    assert target.read_bytes() == b"existing unrelated work\n"


def test_formal_file_protocol_binds_completed_readme_and_validation(
    cot_config, tmp_path, monkeypatch
):
    """Synthetic publication IO only; real model/human gates tested separately.

    The gate override exists only inside this test. It does not constitute a real
    quality, student-training, or template-compliance result.
    """
    import datastudio.image_cot.export as module
    from datastudio.image_cot.contracts import file_hash

    root = run(cot_config, backend="mock")
    preview = export_training(root)
    report = read_json(root / "report.json")
    policy = dict(cot_config["export"])
    readme = tmp_path / "completed.md"
    readme.write_bytes(
        "# synthetic_cot\r\n\r\n完成日期：2026-11-05；维护人：合成测试\r\n\r\n## 数据集介绍\r\nSynthetic fixture only.\r\n".encode()
    )
    template = tmp_path / "template.md"
    template.write_bytes(b"# Synthetic template fixture\n")
    format_spec = tmp_path / "format_spec.md"
    format_spec.write_bytes(
        b"Synthetic format contract fixture, not the user template.\n"
    )
    validation = tmp_path / "delivery_validation.json"
    quality, probe = tmp_path / "quality_stub.json", tmp_path / "probe_stub.json"
    quality.write_bytes(b'{"test_fixture_only":true}')
    probe.write_bytes(b'{"test_fixture_only":true}')
    policy.update(
        release_readme=str(readme),
        readme_template=str(template),
        data_format_path=str(format_spec),
        delivery_validation=str(validation),
        quality_gate=str(quality),
        training_probe=str(probe),
        maintainer="合成测试",
        completion_date="2026-11-05",
    )
    for key, name in (
        ("output_jsonl", "data.jsonl"),
        ("output_readme", "README.md"),
        ("output_provenance", "provenance.jsonl"),
        ("output_manifest", "release.json"),
    ):
        policy[key] = str(tmp_path / "published_fixture" / name)
    validation.write_bytes(
        canonical(
            {
                "schema_version": "image_cot.delivery_validation/1",
                "status": "pass",
                "real_execution": True,
                "selection_digest": report["selection_digest"],
                "data_sha256": file_hash(preview / "data.jsonl"),
                "readme_sha256": file_hash(readme),
                "readme_template_sha256": file_hash(template),
                "format_spec_sha256": file_hash(format_spec),
                "training_media_root": str(
                    Path(policy["training_media_root"]).resolve()
                ),
                "destinations": {
                    key: policy[key]
                    for key in (
                        "output_jsonl",
                        "output_readme",
                        "output_provenance",
                        "output_manifest",
                    )
                },
                "checks": {
                    "actual_json_checked": True,
                    "media_paths_checked": True,
                    "authoritative_readme_template_checked": True,
                },
            }
        )
    )
    monkeypatch.setattr(module, "verify_release_gates", lambda *args: ({}, {}))
    staged = export_training(root, preview=False, export_policy=policy)
    assert (
        staged / "README.md"
    ).read_bytes() == readme.read_bytes()  # preserve CRLF/template bytes
    manifest = module.publish_release(root, export_policy=policy)
    assert read_json(manifest)["published"] is True
    assert Path(policy["output_readme"]).read_bytes() == readme.read_bytes()
    assert module.publish_release(root, export_policy=policy) == manifest
    original_output = Path(policy["output_jsonl"]).read_bytes()
    readme.write_bytes(readme.read_bytes() + b"unvalidated change\n")
    with pytest.raises(ConfigurationError, match="delivery_validation_identity"):
        export_training(root, preview=False, export_policy=policy)
    assert Path(policy["output_jsonl"]).read_bytes() == original_output
