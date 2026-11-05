"""Single-writer, append-only request evidence and immutable part commits.

The OS lock releases on process exit. A lock filename is not a lease to steal.
Only verified commit manifests authorize consumption; legacy checkpoints do not.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import os
from pathlib import Path
import re
import uuid

from .contracts import (
    InvariantError,
    StorageError,
    canonical,
    file_hash,
    identity,
    require,
    strict_json,
    within,
)


def sync_directory(path):
    if os.name == "nt":
        return False  # Windows requires separate target-filesystem qualification.
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def atomic_bytes(path, content):
    """No-clobber publication under the caller's exclusive owner lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(
            path.stat().st_size == len(content)
            and file_hash(path) == sha256(content).hexdigest(),
            "immutable_file_conflict",
            StorageError,
        )
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    # All writers must hold the same owner lock. Recheck rather than overwrite.
    require(not path.exists(), "publication_race", StorageError)
    os.replace(temporary, path)
    sync_directory(path.parent)


def atomic_json(path, value):
    atomic_bytes(path, canonical(value) + b"\n")


def read_json(path):
    return strict_json(Path(path).read_bytes())


def read_jsonl(path, max_line_bytes=4 * 1024 * 1024):
    with Path(path).open("rb") as handle:
        while True:
            line = handle.readline(max_line_bytes + 1)
            if not line:
                return
            require(
                len(line) <= max_line_bytes and line.endswith(b"\n"),
                "invalid_jsonl_line",
                StorageError,
            )
            yield strict_json(line)


class RunStore:
    def __init__(self, root, *, max_line_bytes=4 * 1024 * 1024, fault=None):
        self.root = Path(root).resolve()
        self.max_line_bytes = max_line_bytes
        self._lock = None
        self.fault = fault or (lambda _point: None)
        self._event_cache = {}
        self._verified_cache = {}
        self._recovery_cache = {}
        self._outcome_cache = {}

    def clear_cache(self):
        self._event_cache.clear()
        self._verified_cache.clear()
        self._recovery_cache.clear()
        self._outcome_cache.clear()

    def acquire_owner(self):
        require(self._lock is None, "owner_already_acquired", StorageError)
        self.root.mkdir(parents=True, exist_ok=True)
        handle = (self.root / "owner.lock").open("a+b")
        try:
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise StorageError("run_has_another_owner") from exc
        self._lock = handle
        return self

    def close(self):
        if self._lock is not None:
            handle, self._lock = self._lock, None
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def __enter__(self):
        return self.acquire_owner()

    def __exit__(self, *_args):
        self.close()

    def _owner(self):
        require(self._lock is not None, "write_requires_owner", StorageError)

    def _owned_part(self, part):
        self._owner()
        require(
            within(part, self.root / "parts"), "cannot_write_another_run", StorageError
        )

    def create_run(self, spec):
        self._owner()
        atomic_json(self.root / "run.json", spec)

    def open_part(self, plan):
        self._owner()
        for key in ("input_part", "stage", "part_id"):
            require(
                type(plan[key]) is str and re.fullmatch(r"[a-zA-Z0-9_-]+", plan[key]),
                "invalid_part_component",
            )
        path = (
            self.root / "parts" / plan["input_part"] / plan["stage"] / plan["part_id"]
        )
        require(within(path, self.root), "part_escape", StorageError)
        tasks = plan["tasks"]
        require(
            len({t["task_id"] for t in tasks}) == len(tasks),
            "duplicate_plan_task",
            InvariantError,
        )
        atomic_json(path / "plan.json", plan)
        return path

    def _events(self, part):
        """Validate complete lines; a torn last line never implies no execution."""
        events, torn = [], []
        previous = None
        expected_seq = 0
        segments = sorted(Path(part).glob("journal-*.jsonl"))
        signature = tuple(
            (p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in segments
        )
        saved = self._event_cache.get(str(part))
        if saved and saved[0] == signature:
            return saved[1], saved[2]
        for segment in segments:
            with segment.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline(self.max_line_bytes + 1)
                    if not line:
                        break
                    require(
                        len(line) <= self.max_line_bytes,
                        "journal_line_too_large",
                        StorageError,
                    )
                    if not line.endswith(b"\n"):
                        torn.append({"segment": segment.name, "offset": offset})
                        break
                    event = strict_json(line)
                    saved_hash = event.pop("event_hash", None)
                    require(
                        saved_hash == identity(event),
                        "journal_hash_mismatch",
                        StorageError,
                    )
                    require(
                        event["seq"] == expected_seq and event["previous"] == previous,
                        "journal_chain_mismatch",
                        StorageError,
                    )
                    event["event_hash"] = saved_hash
                    previous, expected_seq = saved_hash, expected_seq + 1
                    events.append(event)
        self._event_cache[str(part)] = (signature, events, torn)
        return events, torn

    def _append(self, part, event_type, records):
        self._owned_part(part)
        require(
            not (part / "commit.json").exists(),
            "committed_part_is_immutable",
            StorageError,
        )
        events, torn = self._events(part)
        segments = sorted(part.glob("journal-*.jsonl"))
        rotate = bool(torn and segments and torn[-1]["segment"] == segments[-1].name)
        index = len(segments) if rotate else max(0, len(segments) - 1)
        segment = part / f"journal-{index:06d}.jsonl"
        seq = len(events)
        previous = events[-1]["event_hash"] if events else None
        added = []
        with segment.open("ab") as handle:
            for record in records:
                event = {
                    "schema_version": "image_cot.event/1",
                    "seq": seq,
                    "previous": previous,
                    "event_type": event_type,
                    "record": record,
                }
                event["event_hash"] = identity(event)
                encoded = canonical(event) + b"\n"
                require(
                    len(encoded) <= self.max_line_bytes,
                    "journal_record_too_large",
                    StorageError,
                )
                handle.write(encoded)
                added.append(event)
                seq, previous = seq + 1, event["event_hash"]
            handle.flush()
            os.fsync(handle.fileno())
        sync_directory(part)
        segments = sorted(part.glob("journal-*.jsonl"))
        signature = tuple(
            (p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in segments
        )
        self._event_cache[str(part)] = (signature, events + added, torn)

    def append_intents(self, part, tasks):
        self.fault("before_intents")
        recovery = self.recover_part(part)
        require(
            len({t["task_id"] for t in tasks}) == len(tasks),
            "duplicate_intent_batch",
            InvariantError,
        )
        for task in tasks:
            require(
                task["task_id"] not in recovery["intents"],
                "attempt_already_dispatched",
                InvariantError,
            )
        self._append(part, "intent", tasks)
        self.fault("after_intents")

    def append_receipts(self, part, receipts):
        recovery = self.recover_part(part)
        require(
            len({r["task_id"] for r in receipts}) == len(receipts),
            "duplicate_receipt_batch",
            InvariantError,
        )
        for receipt in receipts:
            task_id = receipt["task_id"]
            require(
                task_id in recovery["intents"] and task_id not in recovery["receipts"],
                "unknown_or_duplicate_receipt",
                InvariantError,
            )
            require(
                receipt["attempt_id"] == recovery["intents"][task_id]["attempt_id"],
                "receipt_attempt_mismatch",
                InvariantError,
            )
        self.fault("before_receipts")
        self._append(part, "receipt", receipts)
        self.fault("after_receipts")

    def quarantine_receipts(self, part, receipts):
        self._owned_part(part)
        content = b"".join(canonical(r) + b"\n" for r in receipts)
        if content:
            atomic_bytes(part / f"quarantine-{identity(receipts)}.jsonl", content)

    def recover_part(self, part):
        part = Path(part)
        if (part / "commit.json").exists():
            self.verify_commit(part)
        events, torn = self._events(part)
        signature = (
            len(events),
            events[-1]["event_hash"] if events else None,
            canonical(torn),
        )
        cached = self._recovery_cache.get(str(part))
        if cached and cached[0] == signature:
            return cached[1]
        intents, receipts = {}, {}
        for event in events:
            record = event["record"]
            target = intents if event["event_type"] == "intent" else receipts
            require(
                event["event_type"] in {"intent", "receipt"},
                "unknown_journal_event",
                StorageError,
            )
            key = record["task_id"]
            require(key not in target, "duplicate_journal_record", StorageError)
            target[key] = record
        for task_id, receipt in receipts.items():
            require(
                task_id in intents
                and receipt["attempt_id"] == intents[task_id]["attempt_id"],
                "orphan_receipt",
                StorageError,
            )
        result = {
            "intents": intents,
            "receipts": receipts,
            "torn": torn,
            "unresolved": sorted(set(intents) - set(receipts)),
        }
        self._recovery_cache[str(part)] = (signature, result)
        return result

    def commit_part(self, part, outcomes):
        self._owned_part(part)
        plan = read_json(part / "plan.json")
        expected = {task["task_id"] for task in plan["tasks"]}
        actual = [outcome["task_id"] for outcome in outcomes]
        require(
            len(actual) == len(set(actual)) and set(actual) == expected,
            "incomplete_part_outcomes",
            InvariantError,
        )
        require(
            all(
                o.get("status")
                in {"pass", "fail", "uncertain", "error", "not_dispatched"}
                for o in outcomes
            ),
            "invalid_part_outcome",
            InvariantError,
        )
        recovery = self.recover_part(part)
        planned = {t["task_id"]: t for t in plan["tasks"]}
        for outcome in outcomes:
            task_id = outcome["task_id"]
            require(
                outcome.get("attempt_id") == planned[task_id]["attempt_id"],
                "outcome_attempt_mismatch",
                InvariantError,
            )
            if outcome["status"] in {"pass", "fail", "uncertain"}:
                require(
                    task_id in recovery["receipts"],
                    "outcome_without_durable_receipt",
                    InvariantError,
                )
        content = b"".join(
            canonical(row) + b"\n"
            for row in sorted(outcomes, key=lambda x: x["task_id"])
        )
        results_name = f"results-{sha256(content).hexdigest()}.jsonl"
        self.fault("before_results")
        atomic_bytes(part / results_name, content)
        self.fault("after_results")
        files = [
            part / "plan.json",
            part / results_name,
            *sorted(part.glob("journal-*.jsonl")),
            *sorted(part.glob("quarantine-*.jsonl")),
        ]
        manifest = {
            "schema_version": "image_cot.commit/1",
            "plan_digest": identity(plan),
            "parents": plan.get("parents", []),
            "results": results_name,
            "count": len(outcomes),
            "outcomes": dict(Counter(o["status"] for o in outcomes)),
            "files": {path.name: file_hash(path) for path in files},
        }
        self.fault("before_commit")
        atomic_json(part / "commit.json", manifest)
        self.fault("after_commit")
        self.verify_commit(part)
        return manifest

    def verify_commit(self, part, _ancestors=()):
        part = Path(part).resolve()
        require(
            part.parents[2].name == "parts",
            "invalid_committed_part_layout",
            StorageError,
        )
        owner_root = part.parents[3]
        require(
            str(part.resolve()) not in _ancestors, "commit_parent_cycle", StorageError
        )
        ancestors = (*_ancestors, str(part.resolve()))
        manifest = read_json(part / "commit.json")
        for parent in manifest.get("parents", []):
            path = (owner_root / parent["part"]).resolve()
            require(
                within(path, owner_root / "parts")
                and file_hash(path / "commit.json") == parent["commit_sha256"],
                "parent_commit_integrity",
                StorageError,
            )
            self.verify_commit(path, ancestors)
        paths = [part / name for name in manifest["files"]]
        require(
            all(Path(name).name == name for name in manifest["files"]),
            "commit_path_escape",
            StorageError,
        )
        signature = tuple(
            (p.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in [part / "commit.json", *paths]
            if p.is_file()
        )
        saved = self._verified_cache.get(str(part))
        if saved and saved[0] == signature and saved[1] == manifest:
            return manifest
        require(
            manifest.get("schema_version") == "image_cot.commit/1",
            "commit_schema",
            StorageError,
        )
        for name, checksum in manifest["files"].items():
            require(Path(name).name == name, "commit_path_escape", StorageError)
            path = part / name
            require(
                path.is_file() and file_hash(path) == checksum,
                "commit_integrity_failure",
                StorageError,
            )
        plan = read_json(part / "plan.json")
        require(
            identity(plan) == manifest["plan_digest"],
            "commit_plan_mismatch",
            StorageError,
        )
        require(
            manifest.get("parents", []) == plan.get("parents", []),
            "commit_parent_mismatch",
            StorageError,
        )
        rows = list(read_jsonl(part / manifest["results"], self.max_line_bytes))
        require(
            len(rows) == manifest["count"]
            and len({r["task_id"] for r in rows}) == len(rows)
            and {r["task_id"] for r in rows} == {t["task_id"] for t in plan["tasks"]},
            "committed_outcome_coverage",
            StorageError,
        )
        self._verified_cache[str(part)] = (signature, manifest)
        self._outcome_cache[str(part)] = {r["task_id"]: r for r in rows}
        return manifest

    def read_outcome(self, part, task_id):
        part = Path(part).resolve()
        self.verify_commit(part)
        return self._outcome_cache[str(part)].get(task_id)

    def iter_committed(self, input_part, stage):
        root = self.root / "parts" / input_part / stage
        for part in sorted(root.iterdir()) if root.exists() else []:
            if part.is_dir() and (part / "commit.json").exists():
                manifest = self.verify_commit(part)
                yield part, list(
                    read_jsonl(part / manifest["results"], self.max_line_bytes)
                )

    def find_reusable(self, task_id, reuse_parts):
        """Only explicit bounded part roots, never a recursive global cache scan."""
        for part in reuse_parts:
            if self.read_outcome(part, task_id) is not None:
                receipt = self.recover_part(part)["receipts"].get(task_id)
                if receipt:
                    return {
                        **receipt,
                        "reuse_origin": {
                            "part": str(part),
                            "commit_sha256": file_hash(Path(part) / "commit.json"),
                        },
                    }
        return None
