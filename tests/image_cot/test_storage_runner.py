from pathlib import Path

import pytest

from datastudio.image_cot.contracts import InvariantError, StorageError, canonical
from datastudio.image_cot.export import export_training
from datastudio.image_cot.runner import prepare, resume, run
from datastudio.image_cot.storage import RunStore, read_json, read_jsonl


def plan():
    return {
        "input_part": "input-0",
        "stage": "short",
        "part_id": "part-0",
        "tasks": [{"task_id": "t", "attempt_id": "a"}],
    }


def test_intent_without_receipt_is_unresolved_and_no_replay(tmp_path):
    with RunStore(tmp_path / "run") as store:
        part = store.open_part(plan())
        store.append_intents(part, [{"task_id": "t", "attempt_id": "a"}])
        assert store.recover_part(part)["unresolved"] == ["t"]
        with pytest.raises(InvariantError):
            store.append_intents(part, [{"task_id": "t", "attempt_id": "a"}])


@pytest.mark.parametrize(
    "point",
    [
        "before_intents",
        "after_intents",
        "before_receipts",
        "after_receipts",
        "before_results",
        "after_results",
        "before_commit",
        "after_commit",
    ],
)
def test_crash_boundaries_preserve_authority(tmp_path, point):
    class SimulatedFailure(OSError):
        pass

    def fail(actual):
        if actual == point:
            raise SimulatedFailure("injected disk-full/crash")

    root = tmp_path / "run"
    with pytest.raises(SimulatedFailure):
        with RunStore(root, fault=fail) as store:
            part = store.open_part(plan())
            store.append_intents(part, [{"task_id": "t", "attempt_id": "a"}])
            store.append_receipts(
                part, [{"task_id": "t", "attempt_id": "a", "status": "received"}]
            )
            store.commit_part(
                part, [{"task_id": "t", "attempt_id": "a", "status": "pass"}]
            )
    with RunStore(root) as store:
        part = store.open_part(plan())
        recovery = store.recover_part(part)
        if point == "after_commit":
            assert store.verify_commit(part)["count"] == 1
        else:
            assert not (part / "commit.json").exists()
        if point in {
            "after_receipts",
            "before_results",
            "after_results",
            "before_commit",
            "after_commit",
        }:
            assert "t" in recovery["receipts"]


def test_lock_hash_corruption_and_torn_tail(tmp_path):
    root = tmp_path / "run"
    with RunStore(root) as store:
        with pytest.raises(StorageError):
            RunStore(root).acquire_owner()
        part = store.open_part(plan())
        store.append_intents(part, [{"task_id": "t", "attempt_id": "a"}])
        journal = next(part.glob("journal-*.jsonl"))
        with journal.open("ab") as handle:
            handle.write(b'{"incomplete":')
        assert store.recover_part(part)["torn"]
        store.append_receipts(
            part, [{"task_id": "t", "attempt_id": "a", "status": "received"}]
        )
        assert len(list(part.glob("journal-*.jsonl"))) == 2
        store.commit_part(part, [{"task_id": "t", "attempt_id": "a", "status": "pass"}])
        manifest = store.verify_commit(part)
        with (part / manifest["results"]).open("ab") as handle:
            handle.write(b"corruption")
        with pytest.raises(StorageError):
            store.verify_commit(part)


def test_mock_end_to_end_resume_export_and_bypass(cot_config):
    source = cot_config["sources"][0]
    row = next(read_jsonl(source["path"]))
    ocr = {**row, "id": "ocr-1", "category": "ocr"}
    unsupported = {"data": [], "images": ["images/sample.png"]}
    Path(source["path"]).write_bytes(
        b"".join(canonical(r) + b"\n" for r in (row, ocr, unsupported))
    )
    root = run(cot_config, backend="mock")
    report = read_json(root / "report.json")
    assert report["counts"] == {
        "accepted_cot": 1,
        "bypass_verified": 1,
        "input_rejected": 1,
    }
    assert (
        report["requests_reserved"] == 4
    )  # two question checks + one generation + one joint audit
    assert resume(root) == root
    assert read_json(root / "report.json") == report
    target = export_training(root, preview=True)
    output = list(read_jsonl(target / "data.jsonl"))
    assert len(output) == 1 and output[0]["modal_mask"] == 1
    assert output[0]["original_id"] == row["id"]
    assert output[0]["original_conversations"] == row["conversations"]
    assert output[0]["conversations"][1]["value"].startswith("<think>")
    assert read_json(target / "manifest.json")["preview"] is True
    with pytest.raises(Exception, match="mock_cannot_publish"):
        export_training(root, preview=False)


def test_complex_source_long_once_and_short_preferred(cot_config):
    cot_config["policy"]["long_source_allowlist"] = ["synthetic"]
    root = run(cot_config, backend="mock")
    selection = next(read_jsonl(root / "selection/input-000000.jsonl"))
    assert selection["stage"] == "short"
    assert read_json(root / "report.json")["requests_reserved"] == 5


def test_input_changed_or_malformed_never_silently_resumes(cot_config):
    manifest = prepare(cot_config)
    value = read_json(manifest)
    with (manifest.parent / value["parts"][0]["file"]).open("ab") as handle:
        handle.write(b"bad")
    with pytest.raises(Exception, match="prepared_part_changed"):
        prepare(cot_config)


def test_pass_without_durable_receipt_cannot_be_committed(tmp_path):
    with RunStore(tmp_path / "run") as store:
        part = store.open_part(plan())
        with pytest.raises(InvariantError, match="durable_receipt"):
            store.commit_part(
                part, [{"task_id": "t", "attempt_id": "a", "status": "pass"}]
            )
        assert not (part / "commit.json").exists()


def test_journal_append_does_not_reread_or_rewrite_prefix_per_flush(
    tmp_path, monkeypatch
):
    original_open = Path.open
    reads = []

    def traced_open(path, mode="r", *args, **kwargs):
        if path.name.startswith("journal-") and mode == "rb":
            reads.append(path.stat().st_size)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", traced_open)
    with RunStore(tmp_path / "run") as store:
        specification = plan()
        specification["tasks"] = [
            {"task_id": str(i), "attempt_id": "a" + str(i)} for i in range(20)
        ]
        part = store.open_part(specification)
        for task in specification["tasks"]:
            store.append_intents(part, [task])
            store.append_receipts(part, [{**task, "status": "received"}])
        store.commit_part(
            part, [{**task, "status": "pass"} for task in specification["tasks"]]
        )
        size = next(part.glob("journal-*.jsonl")).stat().st_size
        assert sum(reads) <= 3 * size  # hashing at commit, not sum(1..flush_count)


def test_budget_stop_preserves_accepted_progress_without_long(cot_config):
    cot_config["limits"]["max_requests_total"] = 1
    root = run(cot_config, backend="mock")
    assert read_json(root / "report.json")["counts"] == {"technical_pending": 1}
    assert read_json(root / "report.json")["requests_reserved"] == 1
    assert not (root / "parts/input-000000/long").exists()


def test_duplicate_sources_do_not_duplicate_training_exposure(cot_config):
    cot_config["sources"].append(dict(cot_config["sources"][0]))
    with pytest.raises(Exception, match="duplicate_source_identity"):
        prepare(cot_config)
