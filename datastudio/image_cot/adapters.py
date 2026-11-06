"""Explicit JSONL adapters. Never guess the meaning of Honey's from/value fields."""

from dataclasses import replace

from .contracts import ConfigurationError, identity, require, validate_input
from .policy import source_policy, is_v3
from .tool_guard import contains_tool


def validate_source_chain(value):
    require(type(value) is str and bool(value.strip()), "source_answer_missing")
    if "<think>" in value or "</think>" in value:
        require(
            value.count("<think>") == value.count("</think>") == 1
            and value.index("<think>") < value.index("</think>"),
            "source_think_malformed",
        )
    require(
        not any(t in value for t in ("<tool_call>", "<tool_response>", "[TOOL_CALL]")),
        "source_tool_dependency",
    )


def adapt_records(original, spec, checksum, row, config):
    require(type(original) is dict, "input_object_required")
    if spec["schema"] == "honey_data_v2":
        dialogs = original.get("data")
        require(type(dialogs) is list and bool(dialogs), "honey_data_list_required")
    else:
        dialogs = [original]
    source_has_tools = contains_tool(original)
    for index, dialog in enumerate(dialogs):
        error = None
        reference_missing = False
        try:
            require(type(dialog) is dict, "honey_dialog_object_required")
            require(not source_has_tools, "source_tool_dependency")
            require(
                not dialog.get("tools") and not original.get("tools"),
                "source_tool_dependency",
            )
            if spec["schema"] == "honey_data_v2":
                require(
                    type(dialog.get("n_turns")) is int and dialog["n_turns"] == 1,
                    "single_turn_required",
                )
                messages = dialog.get("messages")
                require(
                    type(messages) is list and len(messages) == 2,
                    "single_turn_required",
                )
                require(all(type(m) is dict for m in messages), "invalid_message")
                require(
                    not any(
                        any(m.get(k) for k in ("tools", "tool_calls", "function_call"))
                        for m in messages
                    ),
                    "source_tool_dependency",
                )
                validate_source_chain(messages[1].get("value"))
                answers = dialog.get("original_answers")
                reference_missing = not (
                    type(answers) is list
                    and len(answers) == 1
                    and type(answers[0]) is str
                    and answers[0].strip()
                )
                reference = "" if reference_missing else answers[0]
                projected = {
                    "id": identity(original.get("id"), dialog.get("id"), index),
                    "conversations": [
                        messages[0],
                        {
                            "from": messages[1].get("from"),
                            "value": reference or "[missing reference]",
                        },
                    ],
                    "images": original.get("images"),
                    "category": dialog.get("category", original.get("category")),
                }
                category_field = spec.get("category_field", "category")
                if "category_field" in spec:
                    require(
                        category_field in dialog or category_field in original,
                        "source_category_field_missing",
                    )
                projected[category_field] = dialog.get(
                    category_field, original.get(category_field)
                )
                # HF Image objects/bytes must be exported to explicit media paths first.
                require(
                    type(projected["images"]) is list
                    and all(type(p) is str for p in projected["images"]),
                    "honey_requires_materialized_media_paths",
                )
            else:
                projected = dict(original)
                conversations = projected.get("conversations", [])
                if (
                    type(conversations) is list
                    and len(conversations) == 2
                    and type(conversations[1]) is dict
                ):
                    require(
                        all(type(m) is dict for m in conversations)
                        and [m.get("from") for m in conversations] == ["human", "gpt"],
                        "invalid_qa_roles_or_text",
                    )
                    validate_source_chain(conversations[1].get("value"))
                    reference_field = spec.get("reference_field")
                    if reference_field:
                        reference = original.get(reference_field)
                        require(
                            type(reference) is str and bool(reference.strip()),
                            "reference_field_missing",
                        )
                        projected["conversations"] = [
                            conversations[0],
                            {"from": "gpt", "value": reference},
                        ]
                    elif "<think>" in conversations[1].get("value", ""):
                        # Old reasoning cannot silently become a trusted answer key.
                        reference_missing = True
            messages = projected.get("conversations", [])
            require(
                not any(
                    type(m) is dict
                    and any(m.get(k) for k in ("tools", "tool_calls", "function_call"))
                    for m in messages
                ),
                "source_tool_dependency",
            )
            require(
                not dialog.get("tool_calls") and not dialog.get("function_call"),
                "source_tool_dependency",
            )
            record = validate_input(projected, spec, checksum, row)
        except ConfigurationError as exc:
            error = str(exc)
            record = validate_input({}, spec, checksum, row)
        policy = {**record.policy, **source_policy(spec, config)}
        policy["evidence_contract"] = "v2"
        reference = record.reference
        reference_format = policy.get("reference_format", "plain")
        require(
            reference_format in {"plain", "answer_line", "boxed"},
            "unregistered_reference_format",
        )
        if reference_format != "plain" and record.input_outcome == "pass":
            from .operations import extract_final

            extracted = extract_final(reference, reference_format)
            if extracted is None:
                reference_missing = True
            else:
                reference = extracted
        category = (
            dialog.get(spec.get("category_field", "category"), original.get("category"))
            if type(dialog) is dict
            else None
        )
        excluded = set(spec.get("excluded_categories", ["math", "code"]))
        allowed = spec.get("allowed_categories")
        if str(category).casefold() in {str(c).casefold() for c in excluded} or (
            allowed is not None and category not in allowed
        ):
            error = "source_category_out_of_scope"
        if not is_v3(config) and record.policy["task_type"] in {
            "pure_ocr",
            "box_grounding",
        }:
            error = "non_cot_task_out_of_scope"
        if reference_missing:
            policy["reference_grade"] = "unknown"
        policy["source_dialog_index"] = index
        if spec["schema"] == "honey_data_v2":
            policy["source_group_id"] = original.get("id")
            policy["source_dialog_id"] = (
                dialog.get("id") if type(dialog) is dict else None
            )
        yield replace(
            record,
            original=original,
            reference=reference,
            policy=policy,
            question_id=identity(record.question_id, index),
            input_outcome="fail" if error else record.input_outcome,
            input_reason=error or record.input_reason,
        )
