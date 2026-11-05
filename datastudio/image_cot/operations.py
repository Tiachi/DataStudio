"""Opt-in CoT operators and strict, side-effect-free semantic contracts."""

from __future__ import annotations

from decimal import Decimal, DecimalException
from collections import Counter
import math
from datetime import datetime
import re

from datastudio.operators.core.result import Result
from datastudio.operators.core.data_item import DataItem
from datastudio.operators.filters.text_repeat import TextRepeatFilter
from datastudio.operators.mllm.filter import MLLMFilter
from datastudio.operators.mllm.request import RequestBuilder
from datastudio.operators.mllm.rewriter import MLLMRewriter
from datastudio.utils.registry import OPERATORS, REQUESTS

from .contracts import (
    ConfigurationError,
    InvariantError,
    ModelTask,
    canonical,
    identity,
    require,
    strict_json,
)


def completed_message(receipt):
    require(
        receipt.status == "received" and type(receipt.raw) is dict,
        receipt.error or "request_not_received",
    )
    choices = receipt.raw.get("choices")
    require(
        type(choices) is list and len(choices) == 1 and type(choices[0]) is dict,
        "exactly_one_choice_required",
    )
    require(choices[0].get("finish_reason") == "stop", "incomplete_or_unknown_finish")
    message = choices[0].get("message")
    require(
        type(message) is dict
        and not message.get("tool_calls")
        and not message.get("function_call"),
        "tool_or_missing_message",
    )
    require(
        type(message.get("content")) is str and message["content"].strip(),
        "empty_content",
    )
    require(
        message.get("reasoning_content") is None
        or type(message["reasoning_content"]) is str,
        "invalid_reasoning_type",
    )
    return message


def render_candidate(receipt, stage):
    message = completed_message(receipt)
    content, reasoning = message["content"], message.get("reasoning_content") or ""
    inline = "<think>" in content or "</think>" in content
    require(not (reasoning.strip() and inline), "ambiguous_reasoning_encoding")
    if inline:
        require(
            content.count("<think>") == content.count("</think>") == 1
            and re.fullmatch(r"\s*<think>.*?</think>\s*\S.*", content, re.S),
            "invalid_think_boundaries",
        )
        if stage == "long":
            require(
                bool(re.search(r"<think>\s*\S.*?</think>", content, re.S)),
                "missing_long_reasoning",
            )
        return content
    if reasoning.strip():
        require(
            "<think>" not in reasoning and "</think>" not in reasoning,
            "nested_native_reasoning",
        )
        return "<think>\n" + reasoning + "\n</think>\n" + content
    require(stage == "short", "missing_long_reasoning")
    return "<think>\n\n</think>\n" + content


def check_candidate(text, policy):
    if any(
        token in text
        for token in ("<image>", "<video>", "<audio>", "[INSERT", "[PLACEHOLDER")
    ):
        return {"status": "fail", "reason": "placeholder_leak"}
    lines = Counter(line.strip() for line in text.splitlines() if line.strip())
    if any(len(line) >= 20 and count >= 5 for line, count in lines.items()):
        return {"status": "fail", "reason": "repetition_collapse"}
    # Upstream's tail-pattern routine has quadratic worst-case slicing. Bound its
    # window for long CoTs instead of scanning multi-megabyte answers quadratically.
    tail = DataItem(
        {
            "conversations": [
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": text[-8192:]},
            ]
        }
    )
    repeated, _reason = TextRepeatFilter(
        check_question=False, check_answer=True, check_ori_answer=False
    ).check(tail, 0)
    if repeated:
        return {"status": "fail", "reason": "upstream_tail_repetition"}
    maximum = policy.get("max_candidate_chars")
    if maximum is not None and len(text) > maximum:
        return {"status": "fail", "reason": "candidate_length_limit"}
    return {"status": "pass", "reason": "local_structure_pass"}


def extract_final(text):
    outside = text.split("</think>", 1)[-1]
    matches = re.findall(
        r"(?im)^\s*(?:Answer|Final Answer|最终答案|答案)\s*[:：]\s*(.+?)\s*$", outside
    )
    if len(matches) != 1:
        return None
    return matches[0].strip()


def _number_unit(value):
    match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\s*/\s*(\d+))?\s*([^\d]*)\s*",
        value,
    )
    if not match:
        return None
    try:
        number = Decimal(match[1])
        if match[2]:
            number /= Decimal(match[2])
        if not number.is_finite():
            return None
        return number, match[3].strip()
    except (DecimalException, ZeroDivisionError):
        return None


def verify_final(text, source):
    policy = source.policy
    if policy.get("reference_grade") != "strong":
        return {"status": "uncertain", "reason": "reference_not_strong"}
    task_type = policy["task_type"]
    if task_type == "open":
        return {"status": "pass", "reason": "open_conclusion_requires_joint_judge"}
    final = extract_final(text)
    if final is None:
        return {"status": "uncertain", "reason": "ambiguous_final_extraction"}
    if task_type == "mcq":
        options = source.original.get(
            policy.get("options_field", "options"), policy.get("options")
        )
        if (
            type(options) is not dict
            or not options
            or not all(type(k) is str and type(v) is str for k, v in options.items())
        ):
            return {"status": "uncertain", "reason": "explicit_option_mapping_required"}

        def label(value):
            value = value.strip()
            if value in options:
                return value
            matches = [
                key for key, option in options.items() if value == option.strip()
            ]
            return matches[0] if len(matches) == 1 else None

        actual, expected = label(final), label(source.reference)
        if actual is None or expected is None:
            return {"status": "uncertain", "reason": "unmapped_option"}
        return {
            "status": "pass" if actual == expected else "fail",
            "reason": "exact_option_check",
        }
    if task_type == "numeric":
        actual, expected = _number_unit(final), _number_unit(source.reference)
        if actual is None or expected is None:
            return {"status": "uncertain", "reason": "numeric_extraction_uncertain"}
        if actual[1] != expected[1]:
            return {"status": "fail", "reason": "unit_mismatch"}
        absolute = Decimal(str(policy.get("absolute_tolerance", "0")))
        relative = Decimal(str(policy.get("relative_tolerance", "0")))
        require(
            absolute.is_finite()
            and relative.is_finite()
            and absolute >= 0
            and relative >= 0,
            "invalid_numeric_tolerance",
        )
        try:
            ok = abs(actual[0] - expected[0]) <= max(
                absolute, relative * abs(expected[0])
            )
        except DecimalException:
            return {"status": "uncertain", "reason": "numeric_range_requires_review"}
        return {
            "status": "pass" if ok else "fail",
            "reason": "declared_numeric_tolerance",
        }
    return {"status": "uncertain", "reason": "unregistered_task_handler"}


def validate_bypass(source):
    if source.policy["task_type"] == "pure_ocr":
        return {
            "status": "pass" if source.reference.strip() else "fail",
            "reason": "preserved_trusted_ocr_annotation",
        }
    try:
        require(
            source.policy.get("box_format") in {"xyxy_normalized", "xyxy_absolute"},
            "box_format_required",
        )
        boxes = strict_json(source.reference)
        require(type(boxes) is list and bool(boxes), "box_list_required")
        if len(boxes) == 4 and all(type(v) in (int, float) for v in boxes):
            boxes = [boxes]
        x_limit, y_limit = 1, 1
        if source.policy["box_format"] == "xyxy_absolute":
            from PIL import Image
            from .contracts import media_path

            with Image.open(media_path(source.media, source.media_root)) as image:
                x_limit, y_limit = image.size
        require(
            all(
                type(box) is list
                and len(box) == 4
                and all(type(v) in (int, float) and math.isfinite(v) for v in box)
                and 0 <= box[0] < box[2] <= x_limit
                and 0 <= box[1] < box[3] <= y_limit
                for box in boxes
            ),
            "invalid_box_coordinates",
        )
        return {"status": "pass", "reason": "preserved_trusted_box_annotation"}
    except (ConfigurationError, TypeError, OSError):
        return {"status": "uncertain", "reason": "box_annotation_requires_review"}


def parse_judgment(receipt, policy):
    try:
        value = strict_json(completed_message(receipt)["content"])
        require(
            type(value) is dict
            and value.get("schema_version") == "image_cot.judgment/1",
            "judgment_schema",
        )
        keys = (
            "answer_status",
            "visual_status",
            "reasoning_status",
            "instruction_status",
        )
        statuses = [value.get(key) for key in keys]
        require(
            all(
                type(s) is str and s in {"pass", "fail", "uncertain"} for s in statuses
            ),
            "judgment_status",
        )
        for key in ("critical_errors", "uncertainties"):
            require(
                type(value.get(key)) is list
                and all(type(v) is str and v.strip() for v in value[key]),
                "judgment_lists",
            )
        require(
            type(value.get("reason")) is str and value["reason"].strip(),
            "judgment_reason",
        )
        require(type(value.get("checks")) is list, "judgment_checks")
        for check in value["checks"]:
            require(
                type(check) is dict
                and all(
                    type(check.get(k)) is str and check[k].strip()
                    for k in ("kind", "claim", "evidence_source", "finding", "reason")
                ),
                "judgment_check_fields",
            )
            require(
                check["finding"] in {"supported", "unsupported", "uncertain"},
                "judgment_finding",
            )
        require(
            not (value["critical_errors"] and all(s == "pass" for s in statuses)),
            "contradictory_judgment",
        )
        # Evidence cannot contradict all-pass flags and still be accepted.
        require(
            not (
                all(s == "pass" for s in statuses)
                and any(c["finding"] == "unsupported" for c in value["checks"])
            ),
            "contradictory_evidence",
        )
        if "fail" in statuses or value["critical_errors"]:
            status = "fail"
        elif (
            "uncertain" in statuses
            or value["uncertainties"]
            or any(c["finding"] == "uncertain" for c in value["checks"])
        ):
            status = "uncertain"
        elif policy.get("visual_evidence_required", True) and not any(
            c["kind"] == "visual"
            and c["evidence_source"] == "image:0"
            and c["finding"] == "supported"
            for c in value["checks"]
        ):
            status = "uncertain"
        else:
            status = "pass"
        return {"status": status, "reason": value["reason"], "judgment": value}
    except (ConfigurationError, KeyError, TypeError):
        return {"status": "error", "reason": "judge_schema_error"}


def parse_question_check(receipt):
    try:
        value = strict_json(completed_message(receipt)["content"])
        require(
            type(value) is dict
            and value.get("schema_version") == "image_cot.question_check/1"
            and type(value.get("status")) is str
            and value["status"] in {"pass", "fail", "uncertain"}
            and type(value.get("reason")) is str
            and value["reason"].strip(),
            "question_check_schema",
        )
        return {"status": value["status"], "reason": value["reason"]}
    except (ConfigurationError, TypeError):
        return {"status": "error", "reason": "question_check_schema_error"}


def decide_route(
    *,
    input_valid=True,
    question_check="pass",
    bypass=False,
    short="absent",
    long="absent",
    complex_source=False,
):
    if not input_valid or question_check == "fail":
        return "input_rejected"
    if any(s in {"error", "not_dispatched"} for s in (question_check, short, long)):
        return "technical_pending"
    if "uncertain" in (question_check, short, long):
        return "needs_review"
    if bypass:
        return "bypass_verified"
    if long == "pass":
        return "accepted_cot"
    if long == "fail":
        return "accepted_cot" if short == "pass" else "candidate_rejected"
    if complex_source or short == "fail":
        return "dispatch_long"
    return "accepted_cot" if short == "pass" else "dispatch_short"


@REQUESTS.register_module()
class VerificationRequestBuilder(RequestBuilder):
    def bind_context(self, context):
        self.context = context

    def format_content(self, item, qa_idx):
        require(hasattr(self, "context"), "builder_context_missing", InvariantError)
        source = self.context.sources[item.data["_cot"]["question_id"]]
        fields = {
            "question_original": source.question_original,
            "question_used": source.question,
            "registered_transform": source.policy.get("question_transform", "identity"),
            "candidate": item.get_qa(qa_idx).answer,
            "source_reference": source.reference,
            "reference_grade": source.policy["reference_grade"],
            "task_policy": source.policy,
        }
        return (
            self.prompt_text
            + "\nUNTRUSTED_DATA_JSON:\n"
            + canonical(fields).decode("utf-8")
        )


def execute_audited_batch(operator, items, context):
    require(context is not None, "operator_context_missing", InvariantError)
    tasks = []
    for item in items:
        require(
            item.qa_count == 1 and item.image is not None,
            "single_decoded_image_qa_required",
            InvariantError,
        )
        qid = item.data["_cot"]["question_id"]
        tasks.append(
            make_model_task(
                context,
                qid,
                operator.request_builder.build_request(item, 0)["payload"],
                item.get_qa(0).answer if operator.is_verifier else None,
            )
        )
    receipts = context.dispatch(tasks)
    require(
        len(receipts) == len(tasks)
        and {r.task_id for r in receipts} == {t.task_id for t in tasks},
        "response_coverage_mismatch",
        InvariantError,
    )
    by_id = {r.task_id: r for r in receipts}
    results = []
    for task, item in zip(tasks, items):
        receipt = by_id[task.task_id]
        source = context.sources[task.question_id]
        result = Result(item_idx=item.idx)
        if receipt.error == "budget_not_dispatched":
            outcome = {"status": "not_dispatched", "reason": "budget_not_dispatched"}
        elif operator.is_verifier:
            outcome = (
                parse_question_check(receipt)
                if context.operation == "question_check"
                else parse_judgment(receipt, source.policy)
            )
        else:
            try:
                text = render_candidate(receipt, context.operation)
                outcome = check_candidate(text, source.policy)
                if outcome["status"] == "pass":
                    outcome = verify_final(text, source)
                outcome.update(
                    text_hash=identity(text), renderer="honey_compact_native/1"
                )
                # Rewrite before filtering preserves the candidate for failed-chain review.
                operator._add_decision(result, 0, {"result": text})
            except ConfigurationError as exc:
                outcome = {"status": "error", "reason": str(exc)}
        outcome.update(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            question_id=task.question_id,
            candidate_id=task.candidate_id,
            recipe_signature=context.recipe_signature,
        )
        context.record_outcome(task.question_id, outcome)
        item.data["_cot"]["outcome"] = {
            k: outcome[k] for k in ("status", "task_id", "question_id")
        }
        if operator.is_verifier or outcome["status"] != "pass":
            result.add_filter(
                -1, rejected=outcome["status"] != "pass", reason=outcome["reason"]
            )
        results.append(result)
    return results


def make_model_task(context, question_id, payload, answer=None):
    source = context.sources[question_id]
    candidate = context.candidate_ids.get(question_id)
    input_digest = task_input_digest(source, candidate, answer, context.operation)
    task_id = identity(context.operation, input_digest, context.recipe_signature)
    return ModelTask(
        task_id,
        identity(task_id, 0),
        context.operation,
        question_id,
        payload,
        candidate,
        input_digest,
        context.recipe_signature,
    )


def task_input_digest(source, candidate, answer, operation):
    inputs = [source.question_id, source.media_sha256, candidate, answer]
    if operation == "joint_verify":
        inputs.append(
            {
                "source_reference": source.reference,
                "task_policy": source.policy,
                "question_original": source.question_original,
            }
        )
    return identity(*inputs)


def validate_review_resolution(value, candidate, judgment):
    require(
        type(value) is dict
        and value.get("schema_version") == "image_cot.review_resolution/1",
        "review_schema",
    )
    for key in ("reviewer", "timestamp", "evidence"):
        require(
            type(value.get(key)) is str and value[key].strip(),
            "review_evidence_required",
        )
    try:
        timestamp = datetime.fromisoformat(value["timestamp"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfigurationError("review_timestamp_required") from exc
    require(timestamp.tzinfo is not None, "review_timestamp_timezone_required")
    require(
        value.get("outcome") in {"pass", "fail"}
        and value.get("candidate_id") == candidate["candidate_id"]
        and value.get("candidate_text_hash") == candidate["text_hash"]
        and value.get("judge_signature") == judgment["recipe_signature"],
        "review_identity_mismatch",
    )
    return value["outcome"]


class _ContextMixin:
    context = None

    def bind_context(self, context):
        self.context = context
        if isinstance(self.request_builder, VerificationRequestBuilder):
            self.request_builder.bind_context(context)

    def process_batch(self, items):
        return execute_audited_batch(self, items, self.context)


@OPERATORS.register_module()
class CoTRewriter(_ContextMixin, MLLMRewriter):
    is_verifier = False


@OPERATORS.register_module()
class StrictCoTFilter(_ContextMixin, MLLMFilter):
    is_verifier = True

    def _add_decision(self, result, qa_idx, parsed):
        # Deliberately do not inherit upstream's bool/default-keep parser.
        status = parsed.get("status")
        result.add_filter(
            -1,
            rejected=status != "pass",
            reason=parsed.get("reason", "missing_strict_outcome"),
        )
