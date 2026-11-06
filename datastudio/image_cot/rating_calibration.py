"""Compare saved ratings with independently supplied labels; no model requests."""

from collections import Counter
from pathlib import Path

from .contracts import file_hash, identity, require
from .export import verified_selection, verified_ref
from .policy import RATINGS
from .split_audit import parse_rating_v3
from .storage import RunStore, atomic_json, read_json, read_jsonl


def rating_calibration(run_root, labels_path):
    root, labels_path = Path(run_root).resolve(), Path(labels_path).resolve()
    spec = read_json(root / "run.json")
    require(
        spec["config"]["schema_version"] == "image_cot.config/3", "v3_ratings_required"
    )
    labels = {}
    checksum = file_hash(labels_path)
    for row in read_jsonl(labels_path, 1024 * 1024):
        require(
            type(row) is dict
            and len(labels) < 10000
            and type(row.get("question_id")) is str
            and row["question_id"] not in labels,
            "unique_bounded_calibration_labels_required",
        )
        require(
            row.get("task_status") in {"answerable", "invalid", "uncertain"}
            and type(row.get("reviewer")) is str
            and row["reviewer"].strip(),
            "human_rating_label_required",
        )
        require(
            row.get("effort_cap") in RATINGS
            if row["task_status"] == "answerable"
            else row.get("effort_cap") is None,
            "invalid_gold_rating",
        )
        labels[row["question_id"]] = row
    require(
        bool(labels) and file_hash(labels_path) == checksum,
        "empty_or_changed_calibration_labels",
    )
    counts, grades, quality = Counter(), Counter(), Counter()
    found = set()
    with RunStore(root) as store:
        for selection, source in verified_selection(root):
            qid = source.question_id
            if qid not in labels:
                continue
            found.add(qid)
            if not selection.get("rating_ref"):
                counts["not_rated_due_to_earlier_exit"] += 1
                continue
            if selection.get("rating_status") == "not_dispatched":
                counts["rating_not_dispatched"] += 1
                continue
            outcome, receipt, _ = verified_ref(store, selection["rating_ref"])
            require(outcome["question_id"] == qid, "rating_label_binding")
            parsed = parse_rating_v3(receipt, spec["config"]["policy"]["tiers"])
            if parsed["status"] == "error":
                counts["technical_rating_error"] += 1
                continue
            expected, actual = labels[qid], parsed["rating"]
            counts["compared"] += 1
            quality[expected["task_status"] + "->" + actual["task_status"]] += 1
            counts["quality_match"] += expected["task_status"] == actual["task_status"]
            if (
                expected["task_status"] != "answerable"
                and actual["task_status"] == "answerable"
            ):
                counts["unusable_task_rated_answerable"] += 1
            if expected["task_status"] == actual["task_status"] == "answerable":
                gold, predicted = expected["effort_cap"], actual["effort_cap"]
                grades[gold + "->" + predicted] += 1
                counts["effort_compared"] += 1
                counts["effort_match"] += gold == predicted
                counts["unnecessary_cot"] += gold == "none" and predicted != "none"
                counts["missed_cot"] += gold != "none" and predicted == "none"
                counts["overrated"] += RATINGS.index(predicted) > RATINGS.index(gold)
                counts["underrated"] += RATINGS.index(predicted) < RATINGS.index(gold)
                counts["unnecessary_high_max"] += predicted in {
                    "high",
                    "max",
                } and RATINGS.index(predicted) > RATINGS.index(gold)
    require(found == set(labels), "calibration_labels_not_in_run")
    report = {
        "schema_version": "image_cot.rating_calibration/1",
        "job_id": spec["job_id"],
        "labels_sha256": checksum,
        "label_count": len(labels),
        "counts": dict(counts),
        "grade_confusion": dict(grades),
        "quality_confusion": dict(quality),
        "claim": "agreement_with_supplied_labels_only_not_guaranteed_rating_accuracy",
    }
    target = root / "reports" / ("rating-calibration-" + identity(report) + ".json")
    atomic_json(target, report)
    return target
