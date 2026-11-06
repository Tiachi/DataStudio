"""Bounded v2 per-part orchestration; transport, journal and budgets stay shared."""

from .contracts import require, InvariantError
from .policy import EFFORTS, cap_for


def process_part(
    records, prepared_part, config, backend, store, budget, input_part, handles
):
    from .runner import execute_stage, parent_refs
    from .operations import verify_final

    policy = config["policy"]
    states = {}
    for source in records:
        state = {
            "question_id": source.question_id,
            "terminal": None,
            "source_row": source.row,
            "source_part": prepared_part["file"],
            "input_reason": source.input_reason,
            "attempts": [],
            "visual_audit": "not_checked",
            "schema_version": "image_cot.selection/2",
        }
        if source.input_outcome != "pass":
            state.update(terminal="input_rejected", reason=source.input_reason)
        elif source.policy["reference_grade"] != "strong":
            state.update(terminal="needs_review", reason="reference_not_strong")
        elif source.policy["task_type"] in {"numeric", "mcq"}:
            fmt = source.policy.get("final_format", "answer_line")
            reference_text = {
                "answer_line": "Answer: " + source.reference,
                "boxed": "\\boxed{" + source.reference + "}",
                "plain": source.reference,
            }.get(fmt)
            require(reference_text is not None, "unregistered_final_format")
            if verify_final(reference_text, source)["status"] != "pass":
                state.update(
                    terminal="needs_review", reason="reference_handler_not_ready"
                )
        states[source.question_id] = state

    def active():
        return [r for r in records if states[r.question_id]["terminal"] is None]

    def stage(subset, name, **kwargs):
        return execute_stage(
            subset, name, config, backend, store, budget, input_part, handles, **kwargs
        )

    def stop(state, outcome, *, input_check=False):
        status = outcome["status"]
        if outcome.get("failure_kind") == "tool":
            state.update(terminal="input_rejected", reason="tool_protocol_excluded")
        elif any(
            e.startswith("scope_excluded")
            for e in outcome.get("judgment", {}).get("critical_errors", [])
        ):
            state.update(terminal="input_rejected", reason="scope_excluded_by_audit")
        elif status == "uncertain":
            state.update(terminal="needs_review", reason=outcome["reason"])
        elif status in {"error", "not_dispatched"}:
            state.update(terminal="technical_pending", reason=outcome["reason"])
        elif input_check and status == "fail":
            state.update(terminal="input_rejected", reason=outcome["reason"])

    question = stage(
        [r for r in active() if r.policy["qa_check_enabled"]], "question_check"
    )
    for source in active():
        state, qid = states[source.question_id], source.question_id
        state["qa_check"] = "checked" if qid in question else "skipped_by_source_policy"
        if qid in question:
            state["question_check_ref"] = question[qid]["receipt_ref"]
            stop(state, question[qid], input_check=True)
    rating = stage(
        [r for r in active() if r.policy["rating_enabled"]],
        "effort_rating",
        parents=parent_refs(question),
    )
    for source in active():
        state, qid = states[source.question_id], source.question_id
        if qid in rating:
            state["rating_ref"] = rating[qid]["receipt_ref"]
            stop(state, rating[qid])
            if state["terminal"]:
                continue
            state["rated_cap"] = rating[qid]["effort_cap"]
        state["effort_cap"] = cap_for(source, policy, state.get("rated_cap"))

    for tier in policy["tiers"]:
        subset = [
            r
            for r in active()
            if EFFORTS.index(tier) <= EFFORTS.index(states[r.question_id]["effort_cap"])
        ]
        generated = stage(
            subset, f"generate_{tier}", parents=parent_refs(question, rating)
        )
        reviews = []
        previous = generated
        passed = [r for r in subset if generated[r.question_id]["status"] == "pass"]
        modes = (
            {
                "text_only": ["text"],
                "visual_joint": ["visual"],
                "text_then_visual": ["text", "visual"],
            }
        )[policy["review_mode"]]
        for mode in modes:
            judged = stage(
                passed,
                f"{mode}_{tier}_verify",
                candidates=generated,
                parents=parent_refs(previous, generated),
            )
            reviews.append((mode, judged))
            previous = judged
            passed = [r for r in passed if judged[r.question_id]["status"] == "pass"]
        for source in subset:
            qid = source.question_id
            state, candidate = states[qid], generated[qid]
            attempt = {
                "effort": tier,
                "generation_ref": candidate["receipt_ref"],
                "candidate_id": candidate["candidate_id"],
                "reviews": [],
            }
            outcome = candidate
            for mode, judged in reviews:
                if qid in judged:
                    outcome = judged[qid]
                    attempt["reviews"].append(
                        {
                            "role": "text_verify" if mode == "text" else "joint_verify",
                            "receipt_ref": outcome["receipt_ref"],
                            "recipe_signature": outcome["recipe_signature"],
                        }
                    )
            attempt.update(status=outcome["status"], reason=outcome["reason"])
            state["attempts"].append(attempt)
            if outcome["status"] == "pass":
                state.update(
                    {
                        key: candidate[key]
                        for key in (
                            "candidate_id",
                            "text_hash",
                            "receipt_ref",
                            "recipe_signature",
                        )
                    }
                )
                state.update(
                    terminal="accepted_cot",
                    reason="first_pass_within_cap",
                    stage=f"generate_{tier}",
                    effort=tier,
                    audit_refs=attempt["reviews"],
                    review_mode=policy["review_mode"],
                    visual_audit="checked" if "visual" in modes else "not_checked",
                )
            else:
                stop(state, outcome)
                if state["terminal"] is None and tier == state["effort_cap"]:
                    state.update(
                        terminal="candidate_rejected", reason="effort_cap_exhausted"
                    )
    require(
        all(s["terminal"] for s in states.values()),
        "v2_nonterminal_selection",
        InvariantError,
    )
    return list(states.values())
