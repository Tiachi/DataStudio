"""Bounded stage runner. Deliberately independent of legacy save/checkpoint order."""

from __future__ import annotations

from collections import Counter
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from PIL import Image

from datastudio.pipelines import Pipeline

from .audited_api import AuditedOpenAIAPI, MockAPI
from .contracts import (
    BudgetExhausted,
    ConfigurationError,
    GenerationReceipt,
    InvariantError,
    RuntimeContext,
    SourceRecord,
    canonical,
    file_hash,
    identity,
    media_path,
    require,
    strict_json,
    validate_config,
    validate_input,
    within,
)
from .operations import (
    decide_route,
    make_model_task,
    render_candidate,
    validate_bypass,
    validate_review_resolution,
    verify_final,
)
from .storage import RunStore, atomic_bytes, atomic_json, read_json, read_jsonl
from .policy import is_v2, stage_roles

LOG = logging.getLogger("image_cot")
REPOSITORY = Path(__file__).resolve().parents[2]


def implementation_identity():
    files = [
        *sorted(Path(__file__).parent.glob("*.py")),
        REPOSITORY / "run_image_cot.py",
    ]
    checksum = identity({str(p.relative_to(REPOSITORY)): file_hash(p) for p in files})
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=10,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return {
        "commit": head.stdout.strip() if head.returncode == 0 else "unknown",
        "code_digest": checksum,
        "clean": dirty.returncode == 0 and not dirty.stdout.strip(),
    }


def validate_work_root(config):
    root = Path(config["private_work_root"]).resolve()
    require(
        not within(root, REPOSITORY) and not within(REPOSITORY, root),
        "work_root_overlaps_code",
    )
    for source in config.get("sources", []):
        for path in (source["media_root"], source["path"]):
            require(
                not within(path, root) and not within(root, path),
                "work_root_overlaps_input",
            )
    return root


def prepare(config, *, limit=None):
    """Read explicit allowlisted annotations/media; never enumerate media trees."""
    config = validate_config(config, backend="mock")
    require(config.get("private_work_root"), "private_work_root_required")
    root = validate_work_root(config)
    sources = config.get("sources")
    require(type(sources) is list and sources, "sources_required")
    require(limit is None or type(limit) is int and limit > 0, "invalid_prepare_limit")
    descriptions = []
    for source in sources:
        require(
            source.get("schema")
            in (
                {"standard_qa_v1", "honey_data_v2"}
                if is_v2(config)
                else {"standard_qa_v1"}
            )
            and source.get("name")
            and source.get("revision")
            and type(source.get("task_policy")) is dict,
            "explicit_source_contract_required",
        )
        require(
            type(source.get("max_image_pixels")) is int
            and source["max_image_pixels"] > 0,
            "max_image_pixels_required",
        )
        path = Path(source["path"]).resolve()
        require(path.is_file(), "source_file_missing")
        descriptions.append({"spec": source, "file_hash": file_hash(path)})
    require(
        len(
            {
                (d["spec"]["name"], d["spec"]["revision"], d["file_hash"])
                for d in descriptions
            }
        )
        == len(descriptions),
        "duplicate_source_identity_would_duplicate_questions",
    )
    recipe = {
        "sources": descriptions,
        "limit": limit,
        "schema": "image_cot.prepared/1",
        "part_records": config["limits"]["source_records_per_part"],
        "part_bytes": config["limits"]["source_bytes_per_part"],
        "prepare_code": implementation_identity()["code_digest"],
    }
    if is_v2(config):
        from .policy import source_policy

        recipe["v2_defaults"] = source_policy({}, config)
    input_digest = identity(recipe)
    target = root / "prepared" / input_digest
    require(within(target, root), "prepared_output_path_escape")
    with RunStore(target):
        if (target / "input_manifest.json").exists():
            verify_prepared(target / "input_manifest.json")
            return target / "input_manifest.json"
        parts, rows, size, total, invalid = [], [], 0, 0, 0

        def flush():
            nonlocal rows, size
            if not rows:
                return
            name = f"sources-{len(parts):06d}.jsonl"
            content = b"".join(canonical(row) + b"\n" for row in rows)
            atomic_bytes(target / name, content)
            parts.append(
                {"file": name, "sha256": file_hash(target / name), "count": len(rows)}
            )
            rows, size = [], 0

        for description in descriptions:
            source, checksum = description["spec"], description["file_hash"]
            path = Path(source["path"])
            with path.open("rb") as handle:
                row_number = 0
                while limit is None or total < limit:
                    line = handle.readline(
                        config["limits"]["source_bytes_per_part"] + 1
                    )
                    if not line:
                        break
                    row_number += 1
                    require(
                        len(line) <= config["limits"]["source_bytes_per_part"],
                        "source_line_too_large",
                    )
                    require(bool(line.strip()), f"blank_source_line:{row_number}")
                    try:
                        original = strict_json(line)
                    except ConfigurationError as exc:
                        raise ConfigurationError(
                            f"malformed_source_json_line:{row_number}"
                        ) from exc
                    if is_v2(config):
                        from .adapters import adapt_records

                        adapted = adapt_records(
                            original, source, checksum, row_number, config
                        )
                    else:
                        adapted = [
                            validate_input(original, source, checksum, row_number)
                        ]
                    for record in adapted:
                        if limit is not None and total >= limit:
                            break
                        encoded_size = len(canonical(record.to_dict())) + 1
                        require(
                            encoded_size <= config["limits"]["source_bytes_per_part"],
                            "prepared_record_too_large",
                        )
                        if rows and (
                            len(rows) >= recipe["part_records"]
                            or size + encoded_size > recipe["part_bytes"]
                        ):
                            flush()
                        rows.append(record.to_dict())
                        size += encoded_size
                        total += 1
                        invalid += record.input_outcome != "pass"
            require(file_hash(path) == checksum, "source_changed_during_prepare")
            flush()  # Source roots/policies never get mixed by a loader.
            if limit is not None and total >= limit:
                break
        require(total > 0, "empty_input")
        manifest = {
            "schema_version": "image_cot.prepared/1",
            "input_digest": input_digest,
            "recipe": recipe,
            "parts": parts,
            "total": total,
            "input_rejected": invalid,
        }
        atomic_json(target / "input_manifest.json", manifest)
    return target / "input_manifest.json"


def verify_prepared(path):
    path = Path(path).resolve()
    manifest = read_json(path)
    require(
        manifest.get("schema_version") == "image_cot.prepared/1"
        and identity(manifest["recipe"]) == manifest["input_digest"],
        "prepared_manifest_identity",
    )
    count = 0
    for part in manifest["parts"]:
        require(Path(part["file"]).name == part["file"], "prepared_path_escape")
        source = path.parent / part["file"]
        require(file_hash(source) == part["sha256"], "prepared_part_changed")
        count += part["count"]
    require(count == manifest["total"], "prepared_count_mismatch")
    return manifest


class Budget:
    """Conservative reservations reconstructed from intents, including unknown usage."""

    def __init__(self, config, store, manifest, *, sleep=time.sleep, now=time.time):
        self.config, self.store, self.manifest = config, store, manifest
        self.sleep, self.now = sleep, now
        self.started = now()
        self.requests, self.outputs = 0, 0
        self.events = []
        self.timings = Counter()
        self.cache_hits = 0
        # Enumerate only known input parts/fixed stage names, never media or all runs.
        for index in range(len(manifest["parts"])):
            for stage in stage_roles(config):
                directory = store.root / "parts" / f"input-{index:06d}" / stage
                for part in sorted(directory.iterdir()) if directory.exists() else []:
                    if (part / "plan.json").is_file():
                        recovery = store.recover_part(part)
                        for task_id, intent in recovery["intents"].items():
                            if intent.get("reused"):
                                continue
                            self.requests += 1
                            receipt = recovery["receipts"].get(task_id)
                            actual = self.actual_output(receipt)
                            self.outputs += (
                                actual
                                if actual is not None
                                else intent["reserved_output_tokens"]
                            )
                            if now() - intent["sent_at"] < 60:
                                self.events.append(intent)
                        store.clear_cache()

    @staticmethod
    def actual_output(receipt):
        if not receipt or type(receipt.get("raw")) is not dict:
            return None
        usage = receipt["raw"].get("usage")
        value = usage.get("completion_tokens") if type(usage) is dict else None
        return value if type(value) is int and value >= 0 else None

    def reserve(self, tasks, role):
        limits = self.config["limits"]
        model = self.config["models"][role]
        outputs = len(tasks) * model["max_output_tokens"]
        if (
            self.requests + len(tasks) > limits["max_requests_total"]
            or self.outputs + outputs > limits["max_output_tokens_total"]
        ):
            raise BudgetExhausted("request_or_token_budget_exhausted")
        pool_name = model.get("quota_pool", "mock")
        pool = self.config.get("quota_pools", {}).get(pool_name)
        inputs = (pool or {}).get("input_tokens_per_request", 0)
        if pool:
            require(
                len(tasks) <= pool["rpm"]
                and len(tasks) * inputs <= pool["input_tpm"]
                and outputs <= pool["output_tpm"],
                "microbatch_exceeds_quota_window",
            )
            while True:
                recent = [
                    e
                    for e in self.events
                    if e["quota_pool"] == pool_name and self.now() - e["sent_at"] < 60
                ]
                valid = (
                    len(recent) + len(tasks) <= pool["rpm"]
                    and sum(e["reserved_input_tokens"] for e in recent)
                    + len(tasks) * inputs
                    <= pool["input_tpm"]
                    and sum(e["reserved_output_tokens"] for e in recent) + outputs
                    <= pool["output_tpm"]
                )
                if valid:
                    break
                self.check_time()
                self.sleep(
                    min(
                        1.0,
                        max(
                            0.01, 60 - (self.now() - min(e["sent_at"] for e in recent))
                        ),
                    )
                )
        self.check_time()
        sent_at = self.now()
        intents = [
            {
                **task.intent(),
                "sent_at": sent_at,
                "quota_pool": pool_name,
                "reserved_input_tokens": inputs,
                "reserved_output_tokens": model["max_output_tokens"],
            }
            for task in tasks
        ]
        self.requests += len(tasks)
        self.outputs += outputs
        self.events.extend(intents)
        # Only keep quota-window events in memory, not the whole dataset ledger.
        self.events = [e for e in self.events if sent_at - e["sent_at"] < 60]
        return intents

    def check_time(self):
        if self.now() - self.started >= self.config["limits"]["max_wall_time_s"]:
            raise BudgetExhausted("wall_time_budget_exhausted")

    def remaining_wall(self):
        return max(
            0.001,
            self.config["limits"]["max_wall_time_s"] - (self.now() - self.started),
        )

    def settle(self, intents, receipts):
        reservations = {r["task_id"]: r["reserved_output_tokens"] for r in intents}
        input_reservations = {r["task_id"]: r["reserved_input_tokens"] for r in intents}
        for receipt in receipts:
            actual = self.actual_output(receipt.to_dict())
            if actual is not None:
                require(
                    actual <= reservations[receipt.task_id],
                    "provider_exceeded_output_reservation",
                    InvariantError,
                )
                self.outputs -= reservations[receipt.task_id] - actual
            usage = (receipt.raw or {}).get("usage", {})
            actual_input = usage.get("prompt_tokens") if type(usage) is dict else None
            if type(actual_input) is int and input_reservations[receipt.task_id]:
                require(
                    0 <= actual_input <= input_reservations[receipt.task_id],
                    "input_token_estimate_not_conservative_stop_and_recalibrate",
                    InvariantError,
                )


def stage_recipe(config, role, backend, code_digest=None):
    model = config["models"][role]
    connection = config["connections"].get(
        model.get("connection_ref"), {"model": "mock"}
    )
    connection_signature = {
        key: connection.get(key)
        for key in ("provider_profile", "api_base", "model", "declared_revision")
    }
    if role == "effort_rating":
        connection_signature["supported_caps"] = config["policy"]["tiers"]
    return identity(
        "image_cot.stage/1",
        role,
        backend,
        connection_signature,
        {
            k: v
            for k, v in model.items()
            if k
            not in {
                "prompt",
                "quota_pool",
                "read_timeout_s",
                "batch_deadline_s",
                "connection_ref",
            }
        },
        file_hash(model["prompt"]) if model.get("prompt") else "native_question_only/1",
        {"image_encoding": "PNG", "resize": False},
        code_digest or implementation_identity()["code_digest"],
    )


def load_images(records, pixel_limit):
    datas, pixels = [], 0
    try:
        for source in records:
            path = media_path(source.media, source.media_root)
            require(file_hash(path) == source.media_sha256, "source_media_changed")
            with Image.open(path) as image:
                pixels += image.width * image.height
                require(pixels <= pixel_limit, "decoded_batch_pixel_limit")
                image.load()
                copy = image.convert("RGB")
            datas.append(
                {
                    "conversations": [
                        {"from": "human", "value": source.question},
                        {"from": "gpt", "value": source.reference},
                    ],
                    "image_pil": copy,
                    "_cot": {"question_id": source.question_id},
                }
            )
        return datas
    except BaseException:
        release_images(datas)
        raise


def release_images(datas):
    for data in datas:
        image = data.pop("image_pil", None)
        if image is not None:
            image.close()


class StageDispatcher:
    def __init__(self, store, part, model, budget, role, reuse_parts=()):
        self.store, self.part, self.model = store, part, model
        self.budget, self.role, self.reuse_parts = budget, role, reuse_parts

    def __call__(self, tasks):
        recovery = self.store.recover_part(self.part)
        result, fresh = {}, []
        for task in tasks:
            raw = recovery["receipts"].get(task.task_id)
            if raw:
                result[task.task_id] = GenerationReceipt(**raw)
            elif task.task_id in recovery["intents"]:
                result[task.task_id] = GenerationReceipt(
                    task.task_id,
                    task.attempt_id,
                    "unresolved",
                    error="unresolved_no_auto_replay",
                )
            else:
                reusable = self.store.find_reusable(task.task_id, self.reuse_parts)
                if reusable:
                    # Copy bounded evidence once into the new run; preserve origin.
                    intent = {
                        **task.intent(),
                        "sent_at": 0,
                        "quota_pool": "reused",
                        "reserved_input_tokens": 0,
                        "reserved_output_tokens": 0,
                        "reused": True,
                    }
                    self.store.append_intents(self.part, [intent])
                    self.store.append_receipts(self.part, [reusable])
                    result[task.task_id] = GenerationReceipt(**reusable)
                else:
                    fresh.append(task)
        if fresh:
            try:
                intents = self.budget.reserve(fresh, self.role)
            except BudgetExhausted:
                for task in fresh:
                    result[task.task_id] = GenerationReceipt(
                        task.task_id,
                        task.attempt_id,
                        "error",
                        error="budget_not_dispatched",
                    )
                return [result[t.task_id] for t in tasks]
            started = time.monotonic()
            self.store.append_intents(self.part, intents)
            self.budget.timings["intent_fsync_s"] += time.monotonic() - started
            started = time.monotonic()
            expected = {t.task_id: t for t in fresh}
            persisted = set()

            def persist_completion(receipt):
                task = expected.get(receipt.task_id)
                require(
                    task is not None
                    and receipt.attempt_id == task.attempt_id
                    and receipt.task_id not in persisted,
                    "completion_sink_identity",
                    InvariantError,
                )
                self.store.append_receipts(self.part, [receipt.to_dict()])
                persisted.add(receipt.task_id)

            self.model.receipt_sink = persist_completion
            try:
                receipts = self.model.generate_audited(
                    fresh, deadline_s=self.budget.remaining_wall()
                )
            finally:
                self.model.receipt_sink = None
            self.budget.timings[f"{self.role}.model_batch_s"] += (
                time.monotonic() - started
            )
            expected = {t.task_id: t for t in fresh}
            received_ids = [r.task_id for r in receipts]
            identifiable = [
                r.to_dict()
                for r in receipts
                if received_ids.count(r.task_id) == 1
                and r.task_id in expected
                and r.attempt_id == expected[r.task_id].attempt_id
            ]
            remaining = [r for r in identifiable if r["task_id"] not in persisted]
            if remaining:
                started = time.monotonic()
                self.store.append_receipts(self.part, remaining)
                self.budget.timings["receipt_fsync_s"] += time.monotonic() - started
            unidentifiable = [
                r.to_dict()
                for r in receipts
                if received_ids.count(r.task_id) != 1
                or r.task_id not in expected
                or r.attempt_id != expected[r.task_id].attempt_id
            ]
            self.store.quarantine_receipts(self.part, unidentifiable)
            require(
                len(received_ids) == len(set(received_ids))
                and set(received_ids) == set(expected)
                and len(identifiable) == len(fresh),
                "model_receipt_coverage_mismatch",
                InvariantError,
            )
            self.budget.settle(intents, receipts)
            result.update({r.task_id: r for r in receipts})
        return [result[t.task_id] for t in tasks]


def iter_microbatches(records, limits, *, with_image=True):
    batch, pixels = [], 0
    for record in records:
        count = 0
        if with_image:
            with Image.open(media_path(record.media, record.media_root)) as image:
                count = image.width * image.height
        require(
            count <= limits["max_decoded_pixels_per_batch"],
            "single_image_exceeds_batch_pixel_budget",
        )
        if batch and (
            len(batch) >= limits["requests_per_microbatch"]
            or pixels + count > limits["max_decoded_pixels_per_batch"]
        ):
            yield batch
            batch, pixels = [], 0
        batch.append(record)
        pixels += count
    if batch:
        yield batch


class ModelHandle:
    """Construct a provider only on fresh dispatch, not while reading old receipts."""

    def __init__(self, config, role, backend):
        self.config, self.role, self.backend = config, role, backend
        self.instance = None
        ref = config["models"][role].get("connection_ref")
        self.model = (
            config["connections"].get(ref, {}).get("model", "synthetic-image-cot-mock")
        )

    def generate_audited(self, tasks, *, deadline_s=None):
        if self.instance is None:
            model = self.config["models"][self.role]
            self.instance = (
                MockAPI()
                if self.backend == "mock"
                else AuditedOpenAIAPI(
                    self.config["connections"][model["connection_ref"]],
                    model,
                    self.config["limits"],
                )
            )
        return (
            self.instance.generate_audited(
                tasks, receipt_sink=getattr(self, "receipt_sink", None)
            )
            if self.backend == "mock"
            else self.instance.generate_audited(
                tasks,
                deadline_s=deadline_s,
                receipt_sink=getattr(self, "receipt_sink", None),
            )
        )


def _reuse_parts(config, input_part, stage):
    result = []
    for root in config.get("reuse_from", []):
        directory = Path(root).resolve() / "parts" / input_part / stage
        require((Path(root).resolve() / "run.json").is_file(), "invalid_reuse_run")
        for part in sorted(directory.iterdir()) if directory.exists() else []:
            if (part / "commit.json").is_file():
                result.append(part)
    return result


def execute_stage(
    records,
    stage,
    config,
    backend,
    store,
    budget,
    input_part,
    handles,
    *,
    candidates=None,
    parents=(),
):
    if not records:
        return {}
    candidates = candidates or {}
    role = stage_roles(config)[stage]
    operation = role
    recipe = budget.stage_recipes[role]
    candidate_review = role in {"joint_verify", "text_verify"}
    is_verifier = role in {
        "question_check",
        "effort_rating",
        "joint_verify",
        "text_verify",
    }
    candidate_ids = {
        source.question_id: (
            candidates[source.question_id]["candidate_id"]
            if candidate_review
            else identity(source.question_id, recipe, role, 0)
        )
        for source in records
    }
    context = RuntimeContext(
        {s.question_id: s for s in records},
        operation,
        recipe,
        None,
        candidate_ids=candidate_ids,
        settings=config.get("policy", {}),
    )
    tasks = []
    for source in records:
        answer = (
            candidates[source.question_id]["text"]
            if candidate_review
            else source.reference
        )
        tasks.append(
            make_model_task(
                context, source.question_id, [], answer if is_verifier else None
            ).intent()
        )
    plan = {
        "schema_version": "image_cot.plan/1",
        "input_part": input_part,
        "stage": stage,
        "part_id": identity(recipe, [t["task_id"] for t in tasks]),
        "recipe_signature": recipe,
        "tasks": tasks,
        "parents": list(parents),
    }
    for parent in parents:
        path = (store.root / parent["part"]).resolve()
        require(
            within(path, store.root / "parts")
            and file_hash(path / "commit.json") == parent["commit_sha256"],
            "stage_parent_identity",
        )
        store.verify_commit(path)
    part = store.open_part(plan)
    if (part / "commit.json").exists():
        budget.cache_hits += 1
        manifest = store.verify_commit(part)
        outcomes = list(read_jsonl(part / manifest["results"], store.max_line_bytes))
        return _stage_outputs(outcomes, store, part, role)
    handle = handles.setdefault(role, ModelHandle(config, role, backend))
    context.dispatch = StageDispatcher(
        store, part, handle, budget, role, _reuse_parts(config, input_part, stage)
    )
    builder = {
        "type": (
            "VerificationRequestBuilder"
            if candidate_review
            else (
                "RatingRequestBuilder" if role == "effort_rating" else "RequestBuilder"
            )
        ),
        "prompt": None if is_verifier else config["models"][role]["prompt"],
        "system_prompt": config["models"][role]["prompt"] if is_verifier else None,
        "key_templates": None,
        "with_image": role != "text_verify",
        "with_question": True,
        "with_answer": candidate_review,
        "with_original": candidate_review,
    }
    operator = {
        "type": "StrictCoTFilter" if is_verifier else "CoTRewriter",
        "request_builder": builder,
        "batch_qa": False,
    }
    pipeline = Pipeline(
        {"stage": {"cfg": {"type": "SubPipeline", "operators": [operator]}}},
        LOG,
        model=handle,
    )
    for subpipeline, _name in pipeline.sub_pipelines:
        for op in subpipeline.operators:
            op.bind_context(context)
    for batch in iter_microbatches(
        records, config["limits"], with_image=role != "text_verify"
    ):
        started = time.monotonic()
        datas = (
            [
                {
                    "conversations": [
                        {"from": "human", "value": s.question},
                        {"from": "gpt", "value": s.reference},
                    ],
                    "_cot": {"question_id": s.question_id},
                }
                for s in batch
            ]
            if role == "text_verify"
            else load_images(batch, config["limits"]["max_decoded_pixels_per_batch"])
        )
        budget.timings["decode_s"] += time.monotonic() - started
        try:
            for data in datas:
                qid = data["_cot"]["question_id"]
                if candidate_review:
                    data["conversations"][1]["value"] = candidates[qid]["text"]
                    data["ori_answer"] = {"0": context.sources[qid].reference}
            started = time.monotonic()
            output = pipeline(datas)
            budget.timings[f"{stage}.pipeline_s"] += time.monotonic() - started
            expected = {s.question_id for s in batch}
            require(
                len(output) == len(batch)
                and {d["_cot"]["question_id"] for d in output} == expected
                and expected.issubset(context.outcomes),
                "pipeline_outcome_coverage",
                InvariantError,
            )
        finally:
            release_images(datas)
    outcomes = list(context.outcomes.values())
    require(
        len(outcomes) == len(records), "stage_missing_explicit_outcomes", InvariantError
    )
    started = time.monotonic()
    store.commit_part(part, outcomes)
    budget.timings["part_commit_s"] += time.monotonic() - started
    return _stage_outputs(outcomes, store, part, role)


def parent_refs(*groups):
    result = {}
    for group in groups:
        for row in group.values():
            ref = row.get("receipt_ref")
            if ref:
                # Parent commit identity, not a separate per-request completion bit.
                result[(ref["part"], ref["commit_sha256"])] = {
                    "part": ref["part"],
                    "commit_sha256": ref["commit_sha256"],
                }
    return sorted(result.values(), key=lambda r: r["part"])


def _stage_outputs(outcomes, store, part, role):
    result = {}
    recovery = store.recover_part(part)
    commit_digest = file_hash(part / "commit.json")
    for outcome in outcomes:
        row = dict(outcome)
        row["receipt_ref"] = {
            "part": str(part.relative_to(store.root)),
            "task_id": row["task_id"],
            "commit_sha256": commit_digest,
        }
        if (role in {"short", "long"} or role.startswith("generate_")) and row.get(
            "text_hash"
        ):
            receipt = recovery["receipts"].get(row["task_id"])
            require(receipt is not None, "candidate_missing_receipt", InvariantError)
            text = render_candidate(GenerationReceipt(**receipt), role)
            require(
                identity(text) == row["text_hash"],
                "candidate_render_hash_mismatch",
                InvariantError,
            )
            row["text"] = text
        result[row["question_id"]] = row
    store.clear_cache()
    return result


def run(config, *, backend="http", limit=None):
    invocation_started = time.monotonic()
    config = validate_config(config, runtime=True, backend=backend)
    resolutions = {}
    if config.get("review_resolutions"):
        path = Path(config["review_resolutions"]).resolve()
        config["review_resolutions_sha256"] = file_hash(path)
        for row in read_jsonl(path, 1024 * 1024):
            require(
                len(resolutions) < 4096
                and type(row) is dict
                and row.get("candidate_id") not in resolutions,
                "review_resolution_limit_or_duplicate",
            )
            resolutions[row.get("candidate_id")] = row
    root = validate_work_root(config)
    version = implementation_identity()
    if config["mode"] == "production":
        require(
            version["clean"] and version["commit"] != "unknown",
            "production_requires_clean_commit",
        )
        require(backend == "http", "production_cannot_use_mock")
    input_path = config.get("input_manifest")
    require(
        not (input_path and limit is not None),
        "limit_requires_fresh_prepared_selection",
    )
    input_path = (
        Path(input_path).resolve() if input_path else prepare(config, limit=limit)
    )
    manifest = verify_prepared(input_path)
    if is_v2(config):
        from .policy import source_policy

        require(
            manifest["recipe"].get("v2_defaults") == source_policy({}, config)
            and [d["spec"] for d in manifest["recipe"]["sources"]] == config["sources"],
            "prepared_source_policy_changed_reprepare_required",
        )
    config["input_manifest"] = str(input_path)
    job_id = identity(manifest["input_digest"], config, version, backend)
    run_root = root / "runs" / job_id
    require(within(run_root, root), "run_output_path_escape")
    run_spec = {
        "schema_version": "image_cot.run/1",
        "job_id": job_id,
        "config": config,
        "implementation": version,
        "backend": backend,
        "input_manifest": str(input_path),
        "input_manifest_sha256": file_hash(input_path),
    }
    with RunStore(
        run_root, max_line_bytes=config["limits"]["max_response_bytes"] * 8 + 65536
    ) as store:
        store.create_run(run_spec)
        budget = Budget(config, store, manifest)
        previous_requests = budget.requests
        budget.stage_recipes = {
            role: stage_recipe(config, role, backend, version["code_digest"])
            for role in config["models"]
        }
        handles = {}
        counts = Counter()
        effort_outcomes, reasons, source_outcomes = Counter(), Counter(), Counter()
        selection_parts = []
        for index, prepared_part in enumerate(manifest["parts"]):
            input_part = f"input-{index:06d}"
            records = [
                SourceRecord(**row)
                for row in read_jsonl(
                    input_path.parent / prepared_part["file"],
                    config["limits"]["source_bytes_per_part"],
                )
            ]
            require(
                len(records) == prepared_part["count"]
                and len({r.question_id for r in records}) == len(records),
                "prepared_record_coverage",
                InvariantError,
            )
            if is_v2(config):
                from .runner_v2 import process_part

                selections = process_part(
                    records,
                    prepared_part,
                    config,
                    backend,
                    store,
                    budget,
                    input_part,
                    handles,
                )
            else:
                reference_ready = {
                    r.question_id: (
                        validate_bypass(r)
                        if r.policy["task_type"] in {"pure_ocr", "box_grounding"}
                        else verify_final("Answer: " + r.reference, r)
                    )["status"]
                    == "pass"
                    for r in records
                    if r.input_outcome == "pass"
                }
                eligible = [
                    r
                    for r in records
                    if r.input_outcome == "pass"
                    and r.policy["reference_grade"] == "strong"
                    and reference_ready[r.question_id]
                ]
                question = execute_stage(
                    eligible,
                    "question_check",
                    config,
                    backend,
                    store,
                    budget,
                    input_part,
                    handles,
                )
                cot_records = [
                    r
                    for r in eligible
                    if question[r.question_id]["status"] == "pass"
                    and r.policy["task_type"] not in {"pure_ocr", "box_grounding"}
                ]
                short = execute_stage(
                    cot_records,
                    "short",
                    config,
                    backend,
                    store,
                    budget,
                    input_part,
                    handles,
                    parents=parent_refs(question),
                )
                short_pass = [
                    r for r in cot_records if short[r.question_id]["status"] == "pass"
                ]
                short_verify = execute_stage(
                    short_pass,
                    "short_verify",
                    config,
                    backend,
                    store,
                    budget,
                    input_part,
                    handles,
                    candidates=short,
                    parents=parent_refs(short),
                )
                apply_reviews(short_verify, short, resolutions)

                def short_status(record):
                    qid = record.question_id
                    return (
                        short_verify[qid]["status"]
                        if qid in short_verify
                        else short[qid]["status"]
                    )

                long_records = [
                    r
                    for r in cot_records
                    if decide_route(
                        short=short_status(r),
                        complex_source=r.source
                        in config.get("policy", {}).get("long_source_allowlist", []),
                    )
                    == "dispatch_long"
                ]
                long = execute_stage(
                    long_records,
                    "long",
                    config,
                    backend,
                    store,
                    budget,
                    input_part,
                    handles,
                    parents=parent_refs(question, short, short_verify),
                )
                long_pass = [
                    r for r in long_records if long[r.question_id]["status"] == "pass"
                ]
                long_verify = execute_stage(
                    long_pass,
                    "long_verify",
                    config,
                    backend,
                    store,
                    budget,
                    input_part,
                    handles,
                    candidates=long,
                    parents=parent_refs(long),
                )
                apply_reviews(long_verify, long, resolutions)
                selections = []
                for source in records:
                    qid = source.question_id
                    if source.input_outcome != "pass":
                        terminal = "input_rejected"
                    elif (
                        source.policy["reference_grade"] != "strong"
                        or not reference_ready[qid]
                    ):
                        terminal = "needs_review"
                    else:
                        long_status = long_verify.get(qid, long.get(qid, {})).get(
                            "status", "absent"
                        )
                        terminal = decide_route(
                            question_check=question[qid]["status"],
                            bypass=source.policy["task_type"]
                            in {"pure_ocr", "box_grounding"},
                            short=short_status(source) if qid in short else "absent",
                            long=long_status,
                        )
                        if (
                            terminal == "bypass_verified"
                            and validate_bypass(source)["status"] != "pass"
                        ):
                            terminal = "needs_review"
                    require(
                        terminal
                        in {
                            "accepted_cot",
                            "bypass_verified",
                            "input_rejected",
                            "candidate_rejected",
                            "needs_review",
                            "technical_pending",
                        },
                        "nonterminal_selection",
                        InvariantError,
                    )
                    chosen = None
                    if terminal == "accepted_cot":
                        chosen = (
                            short[qid] if short_status(source) == "pass" else long[qid]
                        )
                    selection = {
                        "question_id": qid,
                        "terminal": terminal,
                        "source_row": source.row,
                        "source_part": prepared_part["file"],
                        "input_reason": source.input_reason,
                    }
                    if chosen:
                        selection.update(
                            {
                                key: chosen[key]
                                for key in (
                                    "candidate_id",
                                    "text_hash",
                                    "receipt_ref",
                                    "recipe_signature",
                                )
                            }
                        )
                        selection["stage"] = (
                            "short" if chosen is short.get(qid) else "long"
                        )
                        verification = (
                            short_verify[qid]
                            if selection["stage"] == "short"
                            else long_verify[qid]
                        )
                        selection["verification_ref"] = verification["receipt_ref"]
                        selection["judge_signature"] = verification["recipe_signature"]
                        if verification.get("review_resolution"):
                            selection["review_resolution"] = verification[
                                "review_resolution"
                            ]
                    selections.append(selection)
            counts.update(s["terminal"] for s in selections)
            if is_v2(config):
                names = {r.question_id: r.source for r in records}
                for selected in selections:
                    reasons[selected["reason"]] += 1
                    source_outcomes[
                        names[selected["question_id"]] + ":" + selected["terminal"]
                    ] += 1
                    for attempt in selected["attempts"]:
                        effort_outcomes[
                            attempt["effort"] + ":" + attempt["status"]
                        ] += 1
            selection_path = store.root / "selection" / f"{input_part}.jsonl"
            atomic_bytes(
                selection_path, b"".join(canonical(row) + b"\n" for row in selections)
            )
            selection_parts.append(
                {
                    "file": str(selection_path.relative_to(store.root)),
                    "sha256": file_hash(selection_path),
                    "count": len(selections),
                }
            )
        require(
            sum(counts.values()) == manifest["total"],
            "terminal_input_count_mismatch",
            InvariantError,
        )
        report = {
            "schema_version": "image_cot.report/1",
            "job_id": job_id,
            "backend": backend,
            "status": (
                "completed_with_pending"
                if counts["technical_pending"] or counts["needs_review"]
                else "completed"
            ),
            "counts": dict(counts),
            "input_count": manifest["total"],
            "requests_reserved": budget.requests,
            "output_tokens_accounted": budget.outputs,
            "selection_parts": selection_parts,
            "usage_accounting": "completion_tokens_when_available_otherwise_conservative_output_reservation",
            "monetary_cost": "unknown",
            "selection_digest": identity(selection_parts),
            "quality_verified": False,
            "throughput_verified": False,
            "student_training_verified": False,
        }
        if is_v2(config):
            report.update(
                effort_outcomes=dict(effort_outcomes),
                terminal_reasons=dict(reasons),
                source_outcomes=dict(source_outcomes),
                review_mode=config["policy"]["review_mode"],
            )
        atomic_json(store.root / "report.json", report)
        observation = {
            "schema_version": "image_cot.execution_observation/1",
            "job_id": job_id,
            "backend": backend,
            "scope": "this_invocation_not_a_capacity_benchmark",
            "elapsed_s": time.monotonic() - invocation_started,
            "new_requests_reserved": budget.requests - previous_requests,
            "committed_stage_cache_hits": budget.cache_hits,
            "phase_timings_s": dict(budget.timings),
            "peak_rss_bytes": peak_rss_bytes(),
            "directory_fsync_supported_by_platform": os.name != "nt",
        }
        atomic_json(
            store.root / "reports" / f"execution-{uuid.uuid4().hex}.json", observation
        )
    return run_root


def peak_rss_bytes():
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if sys.platform == "darwin" else value * 1024
    except ImportError:
        return None


def apply_reviews(judgments, candidates, resolutions):
    for qid, judgment in judgments.items():
        candidate = candidates[qid]
        resolution = resolutions.get(candidate["candidate_id"])
        if resolution:
            require(
                judgment["status"] in {"pass", "fail", "uncertain"},
                "cannot_review_technical_error_as_content",
            )
            judgment["status"] = validate_review_resolution(
                resolution, candidate, judgment
            )
            judgment["review_resolution"] = resolution


def resume(run_root):
    spec = read_json(Path(run_root) / "run.json")
    require(
        file_hash(spec["input_manifest"]) == spec["input_manifest_sha256"],
        "resume_input_manifest_changed",
    )
    require(
        implementation_identity() == spec["implementation"],
        "resume_requires_original_code_identity",
    )
    return run(spec["config"], backend=spec["backend"])
