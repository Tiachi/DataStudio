"""Policy, evidence, adapters, export and recovery for the native effort cascade."""

from copy import deepcopy
from pathlib import Path

import pytest

from datastudio.image_cot.audited_api import MockAPI
from datastudio.image_cot.contracts import (
    ConfigurationError,
    ModelTask,
    canonical,
    validate_config,
)
from datastudio.image_cot.export import (
    export_training,
    selected_text,
    verified_selection,
)
from datastudio.image_cot.operations import (
    extract_final,
    parse_text_judgment,
    parse_judgment,
    parse_rating,
)
from datastudio.image_cot.profile import profile_sources
from datastudio.image_cot.runner import (
    Budget,
    ModelHandle,
    prepare,
    resume,
    run,
    verify_prepared,
)
from datastudio.image_cot.storage import RunStore, read_json, read_jsonl

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cascade(cot_config):
    config = cot_config
    config["schema_version"] = "image_cot.config/2"
    config["policy"] = {
        "tiers": ["low", "high", "max"],
        "effort_cap": "max",
        "fixed_effort_cap": "high",
        "qa_check_enabled": False,
        "rating_enabled": True,
        "review_mode": "text_only",
        "regenerate_all": True,
    }
    config["sources"][0]["scope_attested_non_math_code"] = True
    base = deepcopy(config["models"]["long"])
    config["models"] = {}
    for role in (
        "effort_rating",
        "generate_low",
        "generate_high",
        "generate_max",
        "text_verify",
    ):
        model = deepcopy(base)
        model["prompt"] = (
            None
            if role.startswith("generate_")
            else str(ROOT / "prompts/image_cot/v2" / f"{role}.txt")
        )
        model["effective_options"] = (
            {"reasoning_effort": role[9:]} if role.startswith("generate_") else {}
        )
        config["models"][role] = model
    return config


def scripted(monkeypatch, transform):
    original = MockAPI.generate_audited
    calls = []

    def generate(self, tasks, *, receipt_sink=None):
        result = original(self, tasks)
        for task, receipt in zip(tasks, result):
            calls.append(task)
            transform(task, receipt)
            if receipt_sink:
                receipt_sink(receipt)
        return result

    monkeypatch.setattr(MockAPI, "generate_audited", generate)
    return calls


def selection(root):
    return next(read_jsonl(root / "selection/input-000000.jsonl"))


@pytest.mark.parametrize(
    "cap,wrong,expected,terminal",
    [
        ("low", ["low"], ["low"], "candidate_rejected"),
        ("high", ["low", "high"], ["low", "high"], "candidate_rejected"),
        ("max", ["low", "high"], ["low", "high", "max"], "accepted_cot"),
        ("max", [], ["low"], "accepted_cot"),
        ("high", ["low"], ["low", "high"], "accepted_cot"),
    ],
)
def test_rated_cap_and_first_pass(cascade, monkeypatch, cap, wrong, expected, terminal):
    def transform(task, receipt):
        if task.operation == "effort_rating":
            value = read_value(receipt)
            value["effort_cap"] = cap
            set_value(receipt, value)
        elif task.operation.startswith("generate_") and task.operation[9:] in wrong:
            receipt.raw["choices"][0]["message"]["content"] = "Answer: 1"

    calls = scripted(monkeypatch, transform)
    root = run(cascade, backend="mock")
    chosen = selection(root)
    assert [
        t.operation[9:] for t in calls if t.operation.startswith("generate_")
    ] == expected
    assert chosen["terminal"] == terminal and chosen["effort_cap"] == cap
    if terminal == "candidate_rejected":
        assert chosen["reason"] == "effort_cap_exhausted"
    assert all("question_check" != t.operation for t in calls)
    before = len(calls)
    resume(root)
    assert len(calls) == before
    assert read_json(root / "report.json")["requests_reserved"] == before
    with RunStore(root) as store:
        spec = read_json(root / "run.json")
        budget = Budget(spec["config"], store, verify_prepared(spec["input_manifest"]))
        assert budget.requests == before


def payload_text(task):
    return canonical([p for p in task.payload if p["type"] != "image"]).decode()


def read_value(receipt):
    import json

    return json.loads(receipt.raw["choices"][0]["message"]["content"])


def set_value(receipt, value):
    receipt.raw["choices"][0]["message"]["content"] = canonical(value).decode()


@pytest.mark.parametrize(
    "failure,terminal",
    [
        ("uncertain", "needs_review"),
        ("malformed", "technical_pending"),
        ("truncated", "technical_pending"),
        ("unresolved", "technical_pending"),
    ],
)
def test_pending_never_escalates(cascade, monkeypatch, failure, terminal):
    def transform(task, receipt):
        if task.operation == "generate_low" and failure == "truncated":
            receipt.raw["choices"][0]["finish_reason"] = "length"
        if task.operation == "generate_low" and failure == "unresolved":
            receipt.status, receipt.raw, receipt.error = (
                "unresolved",
                None,
                "unknown_execution",
            )
        if task.operation == "text_verify":
            if failure == "malformed":
                receipt.raw["choices"][0]["message"]["content"] = "not json"
            if failure == "uncertain":
                value = read_value(receipt)
                value["reasoning_status"] = "uncertain"
                set_value(receipt, value)

    calls = scripted(monkeypatch, transform)
    root = run(cascade, backend="mock")
    assert selection(root)["terminal"] == terminal
    assert not any(t.operation in {"generate_high", "generate_max"} for t in calls)


@pytest.mark.parametrize("where", ["global", "source", "fixed"])
def test_caps_only_tighten(cascade, monkeypatch, where):
    if where == "global":
        cascade["policy"]["effort_cap"] = "low"
    elif where == "source":
        cascade["sources"][0]["effort_cap"] = "low"
    else:
        cascade["policy"].update(rating_enabled=False, fixed_effort_cap="low")
        del cascade["models"]["effort_rating"]

    def transform(task, receipt):
        if task.operation.startswith("generate_"):
            receipt.raw["choices"][0]["message"]["content"] = "Answer: 1"

    calls = scripted(monkeypatch, transform)
    assert selection(run(cascade, backend="mock"))["reason"] == "effort_cap_exhausted"
    assert [t.operation for t in calls if t.operation.startswith("generate_")] == [
        "generate_low"
    ]


def test_text_scope_and_export_revalidates_cap(cascade):
    root = run(cascade, backend="mock")
    chosen, source = next(verified_selection(root))
    assert chosen["visual_audit"] == "not_checked"
    with RunStore(root) as store:
        assert "<think>" in selected_text(chosen, source, store)
        tampered = {**chosen, "effort": "max", "stage": "generate_max"}
        with pytest.raises(ConfigurationError, match="cap_violation"):
            selected_text(tampered, source, store)
        with pytest.raises(ConfigurationError, match="scope_mismatch"):
            selected_text({**chosen, "visual_audit": "checked"}, source, store)
    exported = export_training(root, preview=True)
    rows = list(read_jsonl(exported / "data.jsonl"))
    assert len(rows) == 1 and "<think>" in rows[0]["conversations"][1]["value"]


def test_native_teacher_has_no_reference_or_old_chain(cascade, monkeypatch):
    calls = scripted(monkeypatch, lambda t, r: None)
    run(cascade, backend="mock")
    task = next(t for t in calls if t.operation == "generate_low")
    text = payload_text(task)
    assert "How many marked groups?" in text
    assert "source_reference" not in text and "Answer: 2" not in text
    assert not any(m["type"] in {"system", "assistant"} for m in task.payload)
    judge = next(t for t in calls if t.operation == "text_verify")
    assert "source_reference" in payload_text(judge)
    assert "image_url" not in payload_text(judge)


@pytest.mark.parametrize("mode", ["visual_joint", "text_then_visual"])
def test_visual_modes_require_actual_visual_audit(cascade, mode):
    cascade["policy"]["review_mode"] = mode
    visual = deepcopy(cascade["models"]["text_verify"])
    visual["prompt"] = str(ROOT / "prompts/image_cot/v2/joint_verify.txt")
    cascade["models"]["joint_verify"] = visual
    if mode == "visual_joint":
        del cascade["models"]["text_verify"]
    root = run(cascade, backend="mock")
    chosen, source = next(verified_selection(root))
    assert chosen["visual_audit"] == "checked"
    with RunStore(root) as store:
        assert selected_text(chosen, source, store)


def test_honey_regenerates_independent_dialogs_and_preserves_source(
    cascade, monkeypatch
):
    spec = cascade["sources"][0]
    raw = next(read_jsonl(spec["path"]))
    old = "<think>OLD_SOURCE_REASONING_DO_NOT_REUSE</think>Answer: 2"
    dialog = {
        "id": "d0",
        "n_turns": 1,
        "messages": [raw["conversations"][0], {"from": "gpt", "value": old}],
        "original_answers": ["2"],
        "cot_level": "long",
        "category": "reasoning",
        "provenance": {"reasoning_rewrite": "old-teacher"},
    }
    honey = {
        "id": "group0",
        "images": raw["images"],
        "data": [dialog, {**dialog, "id": "d1"}],
    }
    Path(spec["path"]).write_bytes(canonical(honey) + b"\n")
    spec["schema"] = "honey_data_v2"
    calls = scripted(monkeypatch, lambda t, r: None)
    root = run(cascade, backend="mock")
    pairs = list(verified_selection(root))
    assert len(pairs) == 2 and len({p[0]["candidate_id"] for p in pairs}) == 2
    assert all(p[1].original == honey for p in pairs)
    assert all("OLD_SOURCE_REASONING" not in payload_text(t) for t in calls)
    with RunStore(root) as store:
        assert all(
            "OLD_SOURCE_REASONING" not in selected_text(s, r, store) for s, r in pairs
        )


@pytest.mark.parametrize(
    "kind", ["missing_reference", "tool", "multi_turn", "bad_think", "math", "code"]
)
def test_honey_unsafe_sources_never_reach_teacher(cascade, monkeypatch, kind):
    spec = cascade["sources"][0]
    raw = next(read_jsonl(spec["path"]))
    dialog = {
        "n_turns": 1,
        "messages": raw["conversations"],
        "original_answers": ["2"],
        "category": "reasoning",
    }
    if kind == "missing_reference":
        dialog["original_answers"] = []
    if kind == "tool":
        dialog["tools"] = [{"name": "search"}]
    if kind == "multi_turn":
        dialog["n_turns"] = 2
    if kind == "bad_think":
        dialog["messages"][1]["value"] = "<think>unterminated"
    if kind in {"math", "code"}:
        dialog["category"] = kind
    Path(spec["path"]).write_bytes(
        canonical({"images": raw["images"], "data": [dialog]}) + b"\n"
    )
    spec["schema"] = "honey_data_v2"
    calls = scripted(monkeypatch, lambda t, r: None)
    chosen = selection(run(cascade, backend="mock"))
    assert chosen["terminal"] == (
        "needs_review" if kind == "missing_reference" else "input_rejected"
    )
    assert not calls


def test_invalid_rating_never_defaults_to_max(cascade, monkeypatch):
    def transform(task, receipt):
        if task.operation == "effort_rating":
            value = read_value(receipt)
            value["effort_cap"] = "medium"
            set_value(receipt, value)

    calls = scripted(monkeypatch, transform)
    assert selection(run(cascade, backend="mock"))["terminal"] == "technical_pending"
    assert [t.operation for t in calls] == ["effort_rating"]


@pytest.mark.parametrize(
    "mutation", ["medium", "regenerate", "scope", "fixed", "thinking"]
)
def test_invalid_policy_fails_before_dispatch(cascade, mutation):
    if mutation == "medium":
        cascade["policy"]["tiers"].insert(1, "medium")
    if mutation == "regenerate":
        cascade["policy"]["regenerate_all"] = False
    if mutation == "scope":
        del cascade["sources"][0]["scope_attested_non_math_code"]
    if mutation == "fixed":
        cascade["policy"].update(rating_enabled=False, fixed_effort_cap=None)
    if mutation == "thinking":
        cascade["models"]["generate_low"]["effective_options"]["thinking"] = {
            "type": "disabled"
        }
    with pytest.raises(ConfigurationError):
        validate_config(cascade, runtime=True, backend="mock")


@pytest.mark.parametrize("where", ["global", "source"])
def test_source_policy_changes_cannot_reuse_stale_preparation(cascade, where):
    frozen = prepare(cascade)
    cascade["input_manifest"] = str(frozen)
    if where == "global":
        cascade["policy"]["effort_cap"] = "low"
    else:
        cascade["sources"][0]["effort_cap"] = "low"
    with pytest.raises(ConfigurationError, match="reprepare_required"):
        run(cascade, backend="mock")


def test_rating_recipe_binds_supported_caps(cascade):
    from datastudio.image_cot.runner import stage_recipe

    first = stage_recipe(cascade, "effort_rating", "mock")
    cascade["policy"]["tiers"] = ["low", "high"]
    assert stage_recipe(cascade, "effort_rating", "mock") != first


def test_judge_failure_escalates_but_unchanged_teacher_is_reused_on_rejudge(
    cascade, monkeypatch
):
    generations = []

    def transform(task, receipt):
        if task.operation.startswith("generate_"):
            generations.append(task.operation)
        elif task.operation == "text_verify" and generations[-1] == "generate_low":
            value = read_value(receipt)
            value.update(
                reasoning_status="fail",
                critical_errors=["synthetic faulty justification"],
            )
            set_value(receipt, value)

    scripted(monkeypatch, transform)
    first = run(cascade, backend="mock")
    assert selection(first)["effort"] == "high"
    cascade["reuse_from"] = [str(first)]
    cascade["models"]["text_verify"]["request_revision"] = "new-auditor-v2"
    calls = scripted(monkeypatch, lambda t, r: None)
    second = run(cascade, backend="mock")
    assert selection(second)["effort"] == "low"
    assert [t.operation for t in calls] == ["text_verify"]


def test_profile_reads_all_annotations_not_media(cascade):
    report = read_json(profile_sources(cascade))
    assert report["sources"][0]["rows"] == 1
    assert report["sources"][0]["labels"][".category"] == {"reasoning": 1}
    assert report["scope"] == "all_annotation_rows_no_media_io"


def test_judge_cannot_fabricate_visual_evidence_or_omit_process():
    task = ModelTask("t", "a", "text_verify", "q", [])
    receipt = MockAPI().generate_audited([task])[0]
    value = read_value(receipt)
    value["visual_status"] = "pass"
    set_value(receipt, value)
    assert parse_text_judgment(receipt, {})["status"] == "error"
    receipt = MockAPI().generate_audited(
        [ModelTask("t", "a", "joint_verify", "q", [])]
    )[0]
    value = read_value(receipt)
    value["checks"] = value["checks"][:1]
    set_value(receipt, value)
    assert parse_judgment(receipt, {"evidence_contract": "v2"})["status"] == "uncertain"
    assert parse_rating(receipt, ["low", "high", "max"])["status"] == "error"


@pytest.mark.parametrize(
    "text,fmt,expected",
    [
        (r"x \boxed{A}", "boxed", "A"),
        (r"x \box{2}", "boxed", "2"),
        (r"x \boxed{\frac{1}{2}}", "boxed", r"\frac{1}{2}"),
        (r"\boxed{1} \boxed{2}", "boxed", None),
        ("<think>x</think>two", "plain", "two"),
        ("Answer: A", "answer_line", "A"),
    ],
)
def test_registered_answer_formats(text, fmt, expected):
    assert extract_final(text, fmt) == expected


def test_completed_response_is_durable_even_if_batch_raises(cascade, monkeypatch):
    original = ModelHandle.generate_audited
    crashed = False

    def fail_after_sink(self, tasks, **kwargs):
        nonlocal crashed
        receipts = original(self, tasks, **kwargs)
        if self.role == "generate_low" and not crashed:
            crashed = True
            raise RuntimeError("crash_after_individual_receipt")
        return receipts

    monkeypatch.setattr(ModelHandle, "generate_audited", fail_after_sink)
    with pytest.raises(RuntimeError, match="individual_receipt"):
        run(cascade, backend="mock")
    root = next((Path(cascade["private_work_root"]) / "runs").iterdir())
    with RunStore(root) as store:
        part = next((root / "parts/input-000000/generate_low").iterdir())
        recovery = store.recover_part(part)
        assert len(recovery["receipts"]) == 1
        raw = next(iter(recovery["receipts"].values()))["raw"]
        assert raw["choices"][0]["message"]["reasoning_content"]
    calls = scripted(monkeypatch, lambda t, r: None)
    resume(root)
    assert not any(t.operation == "generate_low" for t in calls)
    assert selection(root)["terminal"] == "accepted_cot"


def test_source_switches_are_explicit_and_no_skipped_qa_receipt(cascade, monkeypatch):
    model = deepcopy(cascade["models"]["text_verify"])
    model["prompt"] = str(ROOT / "prompts/image_cot/v2/question_check.txt")
    cascade["models"]["question_check"] = model
    cascade["sources"][0].update(
        qa_check_enabled=True, rating_enabled=False, fixed_effort_cap="low"
    )
    calls = scripted(monkeypatch, lambda t, r: None)
    root = run(cascade, backend="mock")
    chosen = selection(root)
    assert chosen["qa_check"] == "checked" and chosen["question_check_ref"]
    assert "rating_ref" not in chosen and chosen["effort_cap"] == "low"
    assert [t.operation for t in calls] == [
        "question_check",
        "generate_low",
        "text_verify",
    ]
    with RunStore(root) as store:
        s, r = next(verified_selection(root))
        assert selected_text(s, r, store)


def test_unusable_reference_does_not_spend_teacher_requests(cascade, monkeypatch):
    spec = cascade["sources"][0]
    row = next(read_jsonl(spec["path"]))
    row["conversations"][1]["value"] = "two, perhaps"
    Path(spec["path"]).write_bytes(canonical(row) + b"\n")
    calls = scripted(monkeypatch, lambda t, r: None)
    assert (
        selection(run(cascade, backend="mock"))["reason"]
        == "reference_handler_not_ready"
    )
    assert not calls


def test_reference_format_is_explicit_and_original_is_preserved(cascade):
    spec = cascade["sources"][0]
    row = next(read_jsonl(spec["path"]))
    row["conversations"][1]["value"] = r"source answer: \boxed{2}"
    Path(spec["path"]).write_bytes(canonical(row) + b"\n")
    spec["task_policy"]["reference_format"] = "boxed"
    root = run(cascade, backend="mock")
    chosen, source = next(verified_selection(root))
    assert chosen["terminal"] == "accepted_cot" and source.reference == "2"
    assert source.original == row


def test_budget_stop_and_scope_exclusion_do_not_upgrade(cascade, monkeypatch):
    cascade["limits"]["max_requests_total"] = 1
    calls = scripted(monkeypatch, lambda t, r: None)
    root = run(cascade, backend="mock")
    assert selection(root)["terminal"] == "technical_pending"
    assert [t.operation for t in calls] == ["effort_rating"]
    cascade["limits"]["max_requests_total"] = 100

    def transform(task, receipt):
        if task.operation == "text_verify":
            value = read_value(receipt)
            value.update(
                instruction_status="fail", critical_errors=["scope_excluded: code task"]
            )
            set_value(receipt, value)

    calls = scripted(monkeypatch, transform)
    assert (
        selection(run(cascade, backend="mock"))["reason"] == "scope_excluded_by_audit"
    )
    assert not any(t.operation == "generate_high" for t in calls)
