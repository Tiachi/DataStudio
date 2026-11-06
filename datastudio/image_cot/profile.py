"""Streaming annotation inventory; no media enumeration, decoding or model calls."""

from collections import Counter, defaultdict
from pathlib import Path
import re

from .contracts import canonical, file_hash, identity, require, validate_config
from .storage import atomic_json, read_jsonl


def profile_sources(config):
    from .runner import validate_work_root

    config = validate_config(config, backend="mock")
    root = validate_work_root(config)
    reports = []
    for spec in config["sources"]:
        fields = defaultdict(Counter)
        labels = defaultdict(Counter)
        formats = Counter()
        samples = []
        count = 0
        path = Path(spec["path"])
        checksum = file_hash(path)

        def visit(value, key="", depth=0):
            require(depth <= 32 and len(fields) < 8192, "profile_field_or_depth_limit")
            fields[key][type(value).__name__] += 1
            if type(value) is dict:
                for k, v in value.items():
                    visit(v, key + "." + k, depth + 1)
            elif type(value) is list:
                for entry in value:
                    visit(entry, key + "[]", depth + 1)
            elif type(value) is str:
                if key.rsplit(".", 1)[-1].removesuffix("[]") in {
                    "from",
                    "role",
                    "category",
                    "cot_level",
                    "source",
                    "subset",
                    "source_subset",
                    "categories",
                    "sources",
                    "source_subsets",
                }:
                    require(
                        len(labels[key]) < 4096 or value in labels[key],
                        "profile_label_cardinality_limit",
                    )
                    labels[key][value] += 1
                for tag, pattern in {
                    "think_open": r"<think>",
                    "think_close": r"</think>",
                    "boxed": r"\\boxed\{",
                    "box": r"\\box\{",
                    "answer_line": r"(?im)^\s*(?:Answer|答案)[:：]",
                    "tool_tag": r"<tool(?:_call|_response)?>",
                }.items():
                    if re.search(pattern, value):
                        formats[tag] += 1

        for record in read_jsonl(path, config["limits"]["source_bytes_per_part"]):
            count += 1
            visit(record)
            if len(samples) < 8:
                # Representative snippets are private; full raw rows remain immutable.
                samples.append(
                    {"row": count, "preview": canonical(record).decode()[:4096]}
                )
        require(file_hash(path) == checksum, "source_changed_during_profile")
        reports.append(
            {
                "name": spec["name"],
                "revision": spec["revision"],
                "file_sha256": checksum,
                "rows": count,
                "fields": {k: dict(v) for k, v in fields.items()},
                "labels": {k: dict(v) for k, v in labels.items()},
                "formats": dict(formats),
                "representative_snippets": samples,
            }
        )
    report = {
        "schema_version": "image_cot.profile/1",
        "scope": "all_annotation_rows_no_media_io",
        "sources": reports,
        "adapter_review_required": True,
    }
    target = root / "profiles" / (identity(report) + ".json")
    atomic_json(target, report)
    return target
