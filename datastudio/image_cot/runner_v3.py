"""Four ratings; quality failures stop; answer and reasoning use separate calls."""

from .contracts import require, InvariantError
from .policy import EFFORTS, cap_for


def stop_for_outcome(state, outcome, *, input_check=False):
    status, kind = outcome["status"], outcome.get("failure_kind")
    if kind == "tool":
        state.update(terminal="input_rejected", reason="tool_protocol_excluded")
    elif status in {"error", "not_dispatched"}:
        state.update(terminal="technical_pending", reason=outcome["reason"])
    elif status == "uncertain" or kind == "reference":
        state.update(terminal="needs_review", reason=outcome["reason"])
    elif status == "fail" and (input_check or kind == "input"):
        state.update(terminal="input_rejected", reason=outcome["reason"])


def process_part(
    records, prepared_part, config, backend, store, budget, input_part, handles
):
    from .runner import execute_stage, parent_refs

    policy, states = config["policy"], {}
    for source in records:
        state = {
            "schema_version": "image_cot.selection/3",
            "question_id": source.question_id,
            "terminal": None,
            "source_row": source.row,
            "source_part": prepared_part["file"],
            "input_reason": source.input_reason,
            "attempts": [],
            "visual_audit": "not_checked",
            "rating_status": "not_eligible",
        }
        if source.input_outcome != "pass":
            state.update(terminal="input_rejected", reason=source.input_reason)
        elif source.policy["reference_grade"] != "strong":
            state.update(terminal="needs_review", reason="reference_not_strong")
        states[source.question_id] = state

    def active():
        return [r for r in records if states[r.question_id]["terminal"] is None]

    def stage(subset, name, **kwargs):
        return execute_stage(
            subset, name, config, backend, store, budget, input_part, handles, **kwargs
        )

    question = stage(
        [r for r in active() if r.policy["qa_check_enabled"]], "question_check"
    )
    for source in active():
        state, qid = states[source.question_id], source.question_id
        state["qa_check"] = "checked" if qid in question else "skipped_by_source_policy"
        if qid in question:
            state["question_check_ref"] = question[qid]["receipt_ref"]
            stop_for_outcome(state, question[qid], input_check=True)
    rating = stage(active(), "effort_rating", parents=parent_refs(question))
    for source in active():
        state, qid = states[source.question_id], source.question_id
        outcome = rating[qid]
        state.update(
            rating_ref=outcome["receipt_ref"],
            rating_status=outcome["status"],
            rating_reason=outcome["reason"],
            rating_task_status=outcome.get("rating", {}).get(
                "task_status", outcome["status"]
            ),
        )
        stop_for_outcome(state, outcome, input_check=True)
        if state["terminal"]:
            continue
        state["rated_cap"] = outcome["effort_cap"]
        if state["rated_cap"] == "none":
            state.update(
                terminal="not_required_cot",
                reason="rating_no_cot_required",
                effort_cap="none",
            )
            continue
        if source.policy["task_type"] in {"pure_ocr", "box_grounding"}:
            state.update(
                terminal="needs_review",
                reason="rating_conflicts_with_non_reasoning_task",
            )
            continue
        state["effort_cap"] = cap_for(source, policy, state["rated_cap"])

    for tier in policy["tiers"]:
        subset = [
            r
            for r in active()
            if EFFORTS.index(tier) <= EFFORTS.index(states[r.question_id]["effort_cap"])
        ]
        generated = stage(
            subset, f"generate_{tier}", parents=parent_refs(question, rating)
        )
        candidate_ready = [
            r for r in subset if generated[r.question_id]["status"] == "pass"
        ]
        answers = stage(
            candidate_ready,
            f"answer_{tier}_verify",
            candidates=generated,
            parents=parent_refs(generated),
        )
        answer_ready = [
            r for r in candidate_ready if answers[r.question_id]["status"] == "pass"
        ]
        reasoning = stage(
            answer_ready,
            f"reasoning_{tier}_verify",
            candidates=generated,
            parents=parent_refs(generated, answers),
        )
        for source in subset:
            qid = source.question_id
            state, candidate = states[qid], generated[qid]
            attempt = {
                "effort": tier,
                "candidate_id": candidate["candidate_id"],
                "generation_ref": candidate["receipt_ref"],
                "generation_status": candidate["status"],
                "generation_reason": candidate["reason"],
                "reviews": [],
            }
            outcome = candidate
            for role, group in (
                ("answer_verify", answers),
                ("reasoning_verify", reasoning),
            ):
                if qid in group:
                    outcome = group[qid]
                    attempt["reviews"].append(
                        {
                            "role": role,
                            "receipt_ref": outcome["receipt_ref"],
                            "recipe_signature": outcome["recipe_signature"],
                            "status": outcome["status"],
                            "reason": outcome["reason"],
                            "failure_kind": outcome.get("failure_kind"),
                        }
                    )
            attempt.update(status=outcome["status"], reason=outcome["reason"])
            state["attempts"].append(attempt)
            if outcome["status"] == "pass":
                require(
                    qid in reasoning and len(attempt["reviews"]) == 2,
                    "both_text_audits_required",
                    InvariantError,
                )
                state.update(
                    {
                        k: candidate[k]
                        for k in (
                            "candidate_id",
                            "text_hash",
                            "receipt_ref",
                            "recipe_signature",
                        )
                    }
                )
                state.update(
                    terminal="accepted_cot",
                    reason="answer_and_reasoning_pass",
                    stage=f"generate_{tier}",
                    effort=tier,
                    audit_refs=attempt["reviews"],
                    review_mode="split_text",
                )
            else:
                stop_for_outcome(state, outcome)
                if state["terminal"] is None and tier == state["effort_cap"]:
                    state.update(
                        terminal="candidate_rejected", reason="effort_cap_exhausted"
                    )
    require(
        all(s["terminal"] for s in states.values()),
        "v3_nonterminal_selection",
        InvariantError,
    )
    return list(states.values())
