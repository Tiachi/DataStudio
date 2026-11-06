"""Four-way rating, strict tool exclusion and separate saved text audit stages."""

import json
from pathlib import Path
import runpy

import pytest

from datastudio.image_cot.audited_api import MockAPI
from datastudio.image_cot.contracts import (
    ConfigurationError,
    canonical,
    validate_config,
)
from datastudio.image_cot.export import (
    export_training,
    selected_text,
    verified_selection,
)
from datastudio.image_cot.rating_calibration import rating_calibration
from datastudio.image_cot.runner import Budget, run, resume, stage_batch_limits
from datastudio.image_cot.storage import RunStore, read_json, read_jsonl
from datastudio.image_cot.tool_guard import contains_tool

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def split_config(cot_config):
    template = runpy.run_path(str(ROOT / "configs/image_cot/kimi_split_text.py"))[
        "image_cot"
    ]
    cot_config.update(
        schema_version=template["schema_version"],
        policy=template["policy"],
        models=template["models"],
    )
    cot_config["sources"][0]["scope_attested_non_math_code"] = True
    for model in cot_config["models"].values():
        model.update(
            connection_ref="lan_teacher",
            max_output_tokens=1024,
            read_timeout_s=5,
            batch_deadline_s=10,
            quota_pool="teacher_account",
        )
    return cot_config


def intercept(monkeypatch, transform):
    original = MockAPI.generate_audited
    calls = []

    def generate(self, tasks, *, receipt_sink=None):
        receipts = original(self, tasks)
        for task, receipt in zip(tasks, receipts):
            calls.append(task)
            transform(task, receipt)
            if receipt_sink:
                receipt_sink(receipt)
        return receipts

    monkeypatch.setattr(MockAPI, "generate_audited", generate)
    return calls


def value(receipt):
    return json.loads(receipt.raw["choices"][0]["message"]["content"])


def set_value(receipt, data):
    receipt.raw["choices"][0]["message"]["content"] = canonical(data).decode()


def rate(receipt, cap="high", quality="answerable", kind="question_quality"):
    data = value(receipt)
    data.update(
        effort_cap=cap if quality == "answerable" else None, task_status=quality
    )
    if quality != "answerable":
        for check in data["checks"]:
            if check["kind"] == kind:
                check.update(
                    finding="unsupported" if quality == "invalid" else "uncertain",
                    reason="synthetic quality issue",
                )
    set_value(receipt, data)


def audit(receipt, status="fail", kind="candidate"):
    data = value(receipt)
    data.update(status=status, failure_kind=kind)
    data["critical_errors"] = ["synthetic definite issue"] if status == "fail" else []
    data["uncertainties"] = (
        ["synthetic unresolved issue"] if status == "uncertain" else []
    )
    data["checks"][0].update(finding="unsupported" if status == "fail" else "uncertain")
    set_value(receipt, data)


def only_selection(root):
    return next(verified_selection(root))


@pytest.mark.parametrize("cap", ["none", "low", "high", "max"])
def test_four_grades_and_first_pass(split_config, monkeypatch, cap):
    def change(task, receipt):
        if task.operation == "effort_rating":
            rate(receipt, cap)

    calls = intercept(monkeypatch, change)
    root = run(split_config, backend="mock")
    chosen, source = only_selection(root)
    expected = (
        ["effort_rating"]
        if cap == "none"
        else ["effort_rating", "generate_low", "answer_verify", "reasoning_verify"]
    )
    assert [t.operation for t in calls] == expected
    assert chosen["terminal"] == (
        "not_required_cot" if cap == "none" else "accepted_cot"
    )
    report = read_json(root / "report.json")
    assert report["rating_distribution"][cap] == 1
    assert report["stage_statistics"]["effort_rating"]["planned"] == 1
    assert report["stage_statistics"]["generate_max"]["planned"] == 0
    if cap != "none":
        with RunStore(root) as store:
            assert "<think>" in selected_text(chosen, source, store)
    count = len(calls)
    resume(root)
    assert len(calls) == count and read_json(root / "report.json") == report
    if cap == "none":
        preview = export_training(root, preview=True)
        assert list(read_jsonl(preview / "data.jsonl")) == []


@pytest.mark.parametrize(
    "quality,kind,terminal",
    [
        ("invalid", "question_quality", "input_rejected"),
        ("invalid", "reference_quality", "needs_review"),
        ("uncertain", "question_quality", "needs_review"),
        ("uncertain", "reference_quality", "needs_review"),
    ],
)
def test_quality_is_not_effort(split_config, monkeypatch, quality, kind, terminal):
    def change(task, receipt):
        if task.operation == "effort_rating":
            rate(receipt, quality=quality, kind=kind)

    calls = intercept(monkeypatch, change)
    root = run(split_config, backend="mock")
    assert only_selection(root)[0]["terminal"] == terminal
    assert [t.operation for t in calls] == ["effort_rating"]
    report = read_json(root / "report.json")
    assert report["rating_decisions"][quality] == 1
    assert (
        report["rating_distribution"]["none"]
        == report["rating_distribution"]["not_rated"]
        == 0
    )


@pytest.mark.parametrize(
    "bad", ["medium", "invalid_with_max", "missing_checks", "contradictory"]
)
def test_invalid_rating_never_becomes_max(split_config, monkeypatch, bad):
    def change(task, receipt):
        if task.operation == "effort_rating":
            data = value(receipt)
            if bad == "medium":
                data["effort_cap"] = "medium"
            if bad == "invalid_with_max":
                data.update(task_status="invalid", effort_cap="max")
            if bad == "missing_checks":
                data["checks"] = []
            if bad == "contradictory":
                data["checks"][0]["finding"] = "uncertain"
            set_value(receipt, data)

    calls = intercept(monkeypatch, change)
    assert (
        only_selection(run(split_config, backend="mock"))[0]["terminal"]
        == "technical_pending"
    )
    assert [t.operation for t in calls] == ["effort_rating"]


def test_answer_and_reasoning_have_separate_payloads(split_config, monkeypatch):
    calls = intercept(monkeypatch, lambda t, r: None)
    root = run(split_config, backend="mock")
    answer = next(t for t in calls if t.operation == "answer_verify")
    cot = next(t for t in calls if t.operation == "reasoning_verify")
    a = canonical(answer.payload).decode()
    c = canonical(cot.payload).decode()
    assert "candidate_reasoning" not in a and "Read the two groups" not in a
    assert "candidate_reasoning" in c and "Read the two groups" in c
    answer_data = json.loads(
        next(p["value"] for p in answer.payload if p["type"] == "text").split(
            "UNTRUSTED_DATA_JSON:\n", 1
        )[1]
    )
    assert answer_data["task_policy"]["options"] == {"A": "one", "B": "two"}
    assert all(p["type"] != "image" for t in (answer, cot) for p in t.payload)
    assert answer.task_id != cot.task_id
    chosen, source = only_selection(root)
    assert [r["role"] for r in chosen["audit_refs"]] == [
        "answer_verify",
        "reasoning_verify",
    ]
    with RunStore(root) as store:
        with pytest.raises(ConfigurationError, match="both_text_audits_required"):
            selected_text(
                {**chosen, "audit_refs": chosen["audit_refs"][:1]}, source, store
            )
    for role in ("answer_verify", "reasoning_verify"):
        text = (
            Path(split_config["models"][role]["prompt"])
            .read_text(encoding="utf-8")
            .casefold()
        )
        assert not any(
            w in text for w in ("visual", "image", "picture", "视觉", "图片")
        )


@pytest.mark.parametrize("stage", ["answer_verify", "reasoning_verify"])
def test_definite_candidate_failure_upgrades_only_within_cap(
    split_config, monkeypatch, stage
):
    current = [None]

    def change(task, receipt):
        if task.operation.startswith("generate_"):
            current[0] = task.operation
        if task.operation == stage:
            audit(receipt)

    calls = intercept(monkeypatch, change)
    root = run(split_config, backend="mock")
    chosen, _ = only_selection(root)
    assert (
        chosen["terminal"] == "candidate_rejected"
        and chosen["reason"] == "effort_cap_exhausted"
    )
    assert [t.operation for t in calls if t.operation.startswith("generate_")] == [
        "generate_low",
        "generate_high",
    ]
    if stage == "answer_verify":
        assert not any(t.operation == "reasoning_verify" for t in calls)
    assert (
        read_json(root / "report.json")["stage_statistics"]["answer_low_verify"][
            "planned"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "stage,kind,status,terminal",
    [
        ("answer_verify", "reference", "fail", "needs_review"),
        ("answer_verify", "input", "fail", "input_rejected"),
        ("answer_verify", "unknown", "uncertain", "needs_review"),
        ("reasoning_verify", "reference", "fail", "needs_review"),
        ("reasoning_verify", "candidate", "uncertain", "needs_review"),
    ],
)
def test_source_or_uncertainty_does_not_upgrade(
    split_config, monkeypatch, stage, kind, status, terminal
):
    def change(task, receipt):
        if task.operation == stage:
            audit(receipt, status, kind)

    calls = intercept(monkeypatch, change)
    chosen, _ = only_selection(run(split_config, backend="mock"))
    assert chosen["terminal"] == terminal
    assert not any(t.operation in {"generate_high", "generate_max"} for t in calls)


def test_local_gold_mismatch_still_reaches_independent_answer_judge(
    split_config, monkeypatch
):
    def change(task, receipt):
        if task.operation == "generate_low":
            receipt.raw["choices"][0]["message"]["content"] = "Answer: 1"
        if task.operation == "answer_verify":
            audit(receipt, kind="reference")

    calls = intercept(monkeypatch, change)
    chosen, _ = only_selection(run(split_config, backend="mock"))
    assert chosen["terminal"] == "needs_review"
    assert [t.operation for t in calls] == [
        "effort_rating",
        "generate_low",
        "answer_verify",
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"tools": []},
        {"tool_calls": None},
        {"function_call": {"name": "lookup"}},
        {"from": "tool"},
        {"nested": {"role": "function"}},
        {"value": "<tool_call>lookup</tool_call>"},
        {"value": "[TOOL_CALL]lookup"},
        {"type": "tool_result"},
    ],
)
def test_any_source_tool_is_rejected_before_reference_rewrite(
    split_config, monkeypatch, payload
):
    spec = split_config["sources"][0]
    spec["reference_field"] = "gold"
    row = next(read_jsonl(spec["path"]))
    row["gold"] = "2"
    row["conversations"][1].update(payload)
    Path(spec["path"]).write_bytes(canonical(row) + b"\n")
    calls = intercept(monkeypatch, lambda t, r: None)
    assert (
        only_selection(run(split_config, backend="mock"))[0]["terminal"]
        == "input_rejected"
    )
    assert not calls


@pytest.mark.parametrize(
    "stage", ["effort_rating", "generate_low", "answer_verify", "reasoning_verify"]
)
def test_tools_in_any_provider_stage_are_saved_and_excluded(
    split_config, monkeypatch, stage
):
    def change(task, receipt):
        if task.operation == stage:
            receipt.raw["choices"][0]["message"]["tool_calls"] = [
                {"function": {"name": "lookup"}}
            ]

    calls = intercept(monkeypatch, change)
    root = run(split_config, backend="mock")
    assert only_selection(root)[0]["terminal"] == "input_rejected"
    assert not any(t.operation in {"generate_high", "generate_max"} for t in calls)
    assert any(
        b"lookup" in p.read_bytes() for p in root.glob("parts/*/*/*/journal-*.jsonl")
    )


def test_null_provider_tool_stub_is_not_a_tool_operation(split_config, monkeypatch):
    def change(task, receipt):
        receipt.raw["choices"][0]["message"]["tool_calls"] = None

    intercept(monkeypatch, change)
    assert (
        only_selection(run(split_config, backend="mock"))[0]["terminal"]
        == "accepted_cot"
    )
    assert not contains_tool({"question": "Which hand tool is a hammer?"})


@pytest.mark.parametrize("role", ["answer_verify", "reasoning_verify"])
def test_malformed_audit_does_not_lose_paid_candidate(split_config, monkeypatch, role):
    def change(task, receipt):
        if task.operation == role:
            receipt.raw["choices"][0]["message"]["content"] = "malformed audit JSON"

    calls = intercept(monkeypatch, change)
    root = run(split_config, backend="mock")
    assert only_selection(root)[0]["terminal"] == "technical_pending"
    assert any(
        b"reasoning_content" in p.read_bytes()
        for p in root.glob("parts/*/generate_low/*/journal-*.jsonl")
    )
    assert not any(t.operation == "generate_high" for t in calls)


def test_no_cot_task_type_is_rated_without_forced_generation(split_config, monkeypatch):
    split_config["sources"][0]["task_policy"]["task_type"] = "pure_ocr"

    def change(task, receipt):
        if task.operation == "effort_rating":
            rate(receipt, "none")

    calls = intercept(monkeypatch, change)
    assert (
        only_selection(run(split_config, backend="mock"))[0]["terminal"]
        == "not_required_cot"
    )
    assert [t.operation for t in calls] == ["effort_rating"]


@pytest.mark.parametrize("source_field", [False, True])
def test_rating_is_mandatory(split_config, source_field):
    (split_config["sources"][0] if source_field else split_config["policy"])[
        "rating_enabled"
    ] = False
    with pytest.raises(ConfigurationError, match="rating_is_mandatory"):
        validate_config(split_config, runtime=True, backend="mock")


def test_honey_policy_overrides_survive_projection(split_config, monkeypatch):
    spec = split_config["sources"][0]
    row = next(read_jsonl(spec["path"]))
    spec.update(
        schema="honey_data_v2",
        category_field="source_subset",
        category_overrides={"weak": {"reference_grade": "weak"}},
    )
    row = {
        "images": row["images"],
        "data": [
            {
                "n_turns": 1,
                "messages": row["conversations"],
                "original_answers": ["2"],
                "source_subset": "weak",
            }
        ],
    }
    Path(spec["path"]).write_bytes(canonical(row) + b"\n")
    calls = intercept(monkeypatch, lambda t, r: None)
    chosen, source = only_selection(run(split_config, backend="mock"))
    assert (
        source.policy["reference_grade"] == "weak"
        and chosen["terminal"] == "needs_review"
    )
    assert not calls


def test_per_role_quota_splits_microbatches_before_paid_work(split_config, monkeypatch):
    split_config["limits"]["max_wall_time_s"] = 600
    split_config["quota_pools"] = {
        "teacher_account": {
            "rpm": 100,
            "input_tpm": 100000,
            "output_tpm": 1500,
            "input_tokens_per_request": 100,
        }
    }
    split_config["models"]["effort_rating"]["max_output_tokens"] = 128
    for role in ("generate_low", "generate_high", "generate_max"):
        split_config["models"][role]["max_output_tokens"] = 512
    assert (
        stage_batch_limits(split_config, "answer_verify")["requests_per_microbatch"]
        == 1
    )
    assert (
        stage_batch_limits(split_config, "generate_low")["requests_per_microbatch"] == 2
    )
    validate_config(split_config, runtime=True, backend="http")
    spec = split_config["sources"][0]
    row = next(read_jsonl(spec["path"]))
    Path(spec["path"]).write_bytes(
        canonical(row) + b"\n" + canonical({**row, "id": "q2"}) + b"\n"
    )
    now = [1000.0]

    class FastBudget(Budget):
        def __init__(self, config, store, manifest):
            super().__init__(
                config,
                store,
                manifest,
                now=lambda: now[0],
                sleep=lambda t: now.__setitem__(0, now[0] + t),
            )

    monkeypatch.setattr("datastudio.image_cot.runner.Budget", FastBudget)
    report = read_json(run(split_config, backend="mock") / "report.json")
    assert report["counts"] == {"accepted_cot": 2}
    split_config["quota_pools"]["teacher_account"]["output_tpm"] = 500
    with pytest.raises(ConfigurationError, match="single_request_exceeds_quota"):
        validate_config(split_config, runtime=True, backend="http")


def test_rating_calibration_and_stage_statistics(split_config, tmp_path):
    root = run(split_config, backend="mock")
    chosen, _ = only_selection(root)
    labels = tmp_path / "gold_labels.jsonl"
    labels.write_bytes(
        canonical(
            {
                "question_id": chosen["question_id"],
                "task_status": "answerable",
                "effort_cap": "low",
                "reviewer": "synthetic-human-fixture",
            }
        )
        + b"\n"
    )
    report = read_json(rating_calibration(root, labels))
    assert (
        report["counts"]["overrated"] == 1
        and report["counts"]["unnecessary_high_max"] == 1
    )
    stats = read_json(root / "report.json")["stage_statistics"]
    for stage in (
        "effort_rating",
        "generate_low",
        "answer_low_verify",
        "reasoning_low_verify",
    ):
        assert (
            stats[stage]["outcomes"]["pass"] == 1
            and stats[stage]["completion_tokens_observed"] == 20
        )
        saved = list((root / "statistics" / stage).glob("*.json"))
        assert len(saved) == 1 and read_json(saved[0])["planned"] == 1


def test_budget_stop_retains_completed_generation_and_answer(split_config):
    split_config["limits"]["max_requests_total"] = 3
    root = run(split_config, backend="mock")
    chosen, _ = only_selection(root)
    assert chosen["terminal"] == "technical_pending"
    stats = read_json(root / "report.json")["stage_statistics"]
    assert stats["generate_low"]["requests_with_receipt"] == 1
    assert stats["answer_low_verify"]["requests_with_receipt"] == 1
    assert stats["reasoning_low_verify"]["outcomes"]["not_dispatched"] == 1


def test_rating_calibration_counts_unspent_rating_without_receipt(
    split_config, tmp_path
):
    split_config["limits"]["max_output_tokens_total"] = 1
    root = run(split_config, backend="mock")
    chosen, _ = only_selection(root)
    assert chosen["rating_status"] == "not_dispatched"
    labels = tmp_path / "labels.jsonl"
    labels.write_bytes(
        canonical(
            {
                "question_id": chosen["question_id"],
                "task_status": "answerable",
                "effort_cap": "low",
                "reviewer": "fixture",
            }
        )
        + b"\n"
    )
    report = read_json(rating_calibration(root, labels))
    assert report["counts"]["rating_not_dispatched"] == 1


def test_legacy_visual_evidence_switch_cannot_bypass_v2_contract():
    from datastudio.image_cot.contracts import ModelTask
    from datastudio.image_cot.operations import parse_judgment

    receipt = MockAPI().generate_audited(
        [ModelTask("t", "a", "joint_verify", "q", [])]
    )[0]
    data = value(receipt)
    data["checks"] = [c for c in data["checks"] if c["kind"] != "visual"]
    set_value(receipt, data)
    assert (
        parse_judgment(
            receipt, {"evidence_contract": "v2", "visual_evidence_required": False}
        )["status"]
        == "uncertain"
    )


def test_actual_loopback_uses_two_distinct_text_requests(split_config):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from datastudio.image_cot.contracts import ModelTask

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = json.dumps(payload["messages"])
            role = (
                "effort_rating"
                if "image_cot.effort_rating/2" in text
                else "answer_verify"
                if "cot.answer_judgment/1" in text
                else "reasoning_verify"
                if "cot.reasoning_judgment/1" in text
                else "generate_low"
            )
            calls.append((role, payload))
            receipt = MockAPI().generate_audited(
                [ModelTask("t", "a", role, "q", [{"type": "system", "value": text}])]
            )[0]
            body = canonical(receipt.raw)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        split_config["connections"]["lan_teacher"].update(
            ip="127.0.0.1", port=server.server_address[1]
        )
        split_config["quota_pools"] = {
            "teacher_account": {
                "rpm": 1000,
                "input_tpm": 1000000,
                "output_tpm": 1000000,
                "input_tokens_per_request": 5000,
            }
        }
        root = run(split_config, backend="http")
        assert only_selection(root)[0]["terminal"] == "accepted_cot"
        assert [role for role, _ in calls] == [
            "effort_rating",
            "generate_low",
            "answer_verify",
            "reasoning_verify",
        ]
        for role, payload in calls:
            assert ("image_url" in json.dumps(payload)) == (
                role in {"effort_rating", "generate_low"}
            )
        assert "candidate_reasoning" not in json.dumps(calls[2][1])
        assert "candidate_reasoning" in json.dumps(calls[3][1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
