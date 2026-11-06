"""V3 rating and two independent text-only audit contracts."""

from .contracts import ConfigurationError, canonical, require, strict_json
from .operations import completed_message
from datastudio.operators.mllm.request import RequestBuilder
from datastudio.utils.registry import REQUESTS

TEXT_AUDITS = {"answer_verify", "reasoning_verify"}


def candidate_parts(text):
    require(
        text.count("<think>") == text.count("</think>") == 1
        and text.lstrip().startswith("<think>"),
        "candidate_reasoning_boundaries",
    )
    reasoning, answer = text.split("<think>", 1)[1].split("</think>", 1)
    require(bool(reasoning.strip()) and bool(answer.strip()), "candidate_parts_empty")
    return reasoning.strip(), answer.strip()


def _checks(value, required, allowed_sources=None):
    checks = value.get("checks")
    require(type(checks) is list and bool(checks), "audit_checks_required")
    for check in checks:
        require(
            type(check) is dict
            and all(
                type(check.get(k)) is str and check[k].strip()
                for k in ("kind", "claim", "evidence_source", "finding", "reason")
            ),
            "audit_check_fields",
        )
        require(
            check["finding"] in {"supported", "unsupported", "uncertain"},
            "audit_check_finding",
        )
        if allowed_sources is not None:
            require(
                check["evidence_source"] in allowed_sources
                and check["kind"] in required,
                "text_audit_evidence_contract",
            )
    require(
        all(any(c["kind"] == kind for c in checks) for kind in required),
        "audit_evidence_coverage",
    )
    return checks


def parse_rating_v3(receipt, tiers):
    try:
        value = strict_json(completed_message(receipt)["content"])
        require(
            type(value) is dict
            and value.get("schema_version") == "image_cot.effort_rating/2",
            "rating_schema",
        )
        require(
            value.get("task_status") in {"answerable", "invalid", "uncertain"}
            and type(value.get("reason")) is str
            and value["reason"].strip(),
            "rating_decision",
        )
        checks = _checks(
            value, {"question_quality", "reference_quality", "reasoning_need", "effort"}
        )
        quality = [
            c for c in checks if c["kind"] in {"question_quality", "reference_quality"}
        ]
        if value["task_status"] == "answerable":
            require(
                value.get("effort_cap") in {"none", *tiers}
                and all(c["finding"] == "supported" for c in checks),
                "rating_contradiction",
            )
            status, kind = "pass", "none"
        else:
            require(value.get("effort_cap") is None, "invalid_task_has_no_effort_cap")
            if value["task_status"] == "invalid":
                require(
                    any(c["finding"] == "unsupported" for c in quality),
                    "invalid_rating_needs_evidence",
                )
                status = "fail"
            else:
                require(
                    any(c["finding"] == "uncertain" for c in checks),
                    "uncertain_rating_needs_evidence",
                )
                status = "uncertain"
            kind = (
                "reference"
                if any(
                    c["kind"] == "reference_quality" and c["finding"] != "supported"
                    for c in quality
                )
                else "input"
            )
        return {
            "status": status,
            "reason": value["reason"],
            "failure_kind": kind,
            "effort_cap": value.get("effort_cap"),
            "rating": value,
        }
    except (ConfigurationError, TypeError, KeyError):
        return {"status": "error", "reason": "rating_schema_error"}


def parse_text_audit(receipt, role):
    try:
        schema = (
            "cot.answer_judgment/1"
            if role == "answer_verify"
            else "cot.reasoning_judgment/1"
        )
        value = strict_json(completed_message(receipt)["content"])
        require(
            type(value) is dict
            and set(value)
            == {
                "schema_version",
                "status",
                "failure_kind",
                "critical_errors",
                "uncertainties",
                "checks",
                "reason",
            }
            and value["schema_version"] == schema,
            "text_audit_schema",
        )
        require(
            value["status"] in {"pass", "fail", "uncertain"}
            and value["failure_kind"]
            in {"none", "candidate", "reference", "input", "unknown"}
            and type(value["reason"]) is str
            and value["reason"].strip(),
            "text_audit_status",
        )
        for name in ("critical_errors", "uncertainties"):
            require(
                type(value[name]) is list
                and all(type(v) is str and v.strip() for v in value[name]),
                "text_audit_lists",
            )
        required = (
            {"answer", "instruction"}
            if role == "answer_verify"
            else {"reasoning", "consistency"}
        )
        checks = _checks(
            value,
            required,
            {
                "question",
                "reference",
                "candidate_answer",
                "candidate_reasoning",
                "task_policy",
            },
        )
        if value["status"] == "pass":
            require(
                value["failure_kind"] == "none"
                and not value["critical_errors"]
                and not value["uncertainties"]
                and all(c["finding"] == "supported" for c in checks),
                "text_audit_contradiction",
            )
        elif value["status"] == "fail":
            require(
                value["failure_kind"] in {"candidate", "reference", "input"}
                and bool(value["critical_errors"])
                and any(c["finding"] == "unsupported" for c in checks),
                "text_failure_needs_evidence",
            )
        else:
            require(
                value["failure_kind"] != "none"
                and bool(value["uncertainties"])
                and any(c["finding"] == "uncertain" for c in checks),
                "text_uncertainty_needs_evidence",
            )
        return {
            "status": value["status"],
            "failure_kind": value["failure_kind"],
            "reason": value["reason"],
            "judgment": value,
        }
    except (ConfigurationError, TypeError, KeyError):
        return {"status": "error", "reason": "text_audit_schema_error"}


@REQUESTS.register_module()
class TextAuditRequestBuilder(RequestBuilder):
    def bind_context(self, context):
        self.context = context

    def format_content(self, item, qa_idx):
        source = self.context.sources[item.data["_cot"]["question_id"]]
        reasoning, answer = candidate_parts(item.get_qa(qa_idx).answer)
        fields = {
            "question_original": source.question_original,
            "question_used": source.question,
            "candidate_answer": answer,
            "source_reference": source.reference,
            "reference_grade": source.policy["reference_grade"],
            "task_policy": {
                k: v
                for k, v in source.policy.items()
                if k
                in {
                    "task_type",
                    "reference_grade",
                    "final_format",
                    "options",
                    "absolute_tolerance",
                    "relative_tolerance",
                    "question_transform",
                    "remove_suffix",
                }
            },
        }
        options = source.original.get(
            source.policy.get("options_field", "options"), source.policy.get("options")
        )
        if options is not None:
            fields["task_policy"]["options"] = options
        if self.context.operation == "reasoning_verify":
            fields["candidate_reasoning"] = reasoning
        return "UNTRUSTED_DATA_JSON:\n" + canonical(fields).decode()


@REQUESTS.register_module()
class QualityRatingRequestBuilder(RequestBuilder):
    def bind_context(self, context):
        self.context = context

    def format_content(self, item, qa_idx):
        source = self.context.sources[item.data["_cot"]["question_id"]]
        return canonical(
            {
                "question": source.question,
                "source_reference": source.reference,
                "reference_grade": source.policy["reference_grade"],
                "supported_caps": ["none", *self.context.settings["tiers"]],
            }
        ).decode()
