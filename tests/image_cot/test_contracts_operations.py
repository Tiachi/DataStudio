from dataclasses import replace
import json

import pytest

from datastudio.image_cot.audited_api import MockAPI
from datastudio.image_cot.contracts import (
    ConfigurationError,
    GenerationReceipt,
    ModelTask,
    canonical,
    normalize_connection,
    strict_json,
    validate_config,
    validate_input,
)
from datastudio.image_cot.operations import (
    decide_route,
    parse_judgment,
    render_candidate,
    verify_final,
)


def receipt(content="Answer: 2", reasoning=None, finish="stop"):
    return GenerationReceipt(
        "task",
        "attempt",
        "received",
        raw={
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {"content": content, "reasoning_content": reasoning},
                }
            ]
        },
    )


@pytest.mark.parametrize(
    "text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', "```json\n{}\n```"]
)
def test_strict_json_fails_closed(text):
    with pytest.raises(ConfigurationError):
        strict_json(text)


def test_lan_and_deepseek_connection_profiles(cot_config):
    lan = normalize_connection(cot_config["connections"]["lan_teacher"])
    assert lan["api_base"] == "http://192.0.2.10:8000/v1/chat/completions"
    ds = normalize_connection(cot_config["connections"]["deepseek_validation"])
    assert ds["api_base"] == "https://api.deepseek.com/chat/completions"
    assert ds["model"] == "deepseek-flash"
    with pytest.raises(ConfigurationError):
        normalize_connection(
            {"endpoint": "http://127.0.0.1:8000/v1", "model": "served"}
        )
    with pytest.raises(ConfigurationError):
        normalize_connection(
            {"endpoint": "https://user:secret@example.org", "model": "served"}
        )


def test_unknown_budget_blocks_real_run(cot_config):
    cot_config["limits"]["max_output_tokens_total"] = None
    with pytest.raises(ConfigurationError):
        validate_config(cot_config, runtime=True)


def test_currency_budget_is_not_silently_accepted(cot_config):
    cot_config["limits"]["max_cost_total"] = 1
    with pytest.raises(ConfigurationError, match="currency"):
        validate_config(cot_config, runtime=True)


def test_credential_alias_changes_do_not_invalidate_generation(cot_config):
    from datastudio.image_cot.runner import stage_recipe

    config = validate_config(cot_config, backend="mock")
    first = stage_recipe(config, "short", "http", "fixed-code")
    config["connections"]["renamed"] = config["connections"].pop("lan_teacher")
    config["connections"]["renamed"]["auth"] = {
        "mode": "env_bearer",
        "key_env": "NEW_ENV_NAME",
    }
    config["models"]["short"]["connection_ref"] = "renamed"
    assert stage_recipe(config, "short", "http", "fixed-code") == first


def test_reasoning_preserved_and_ambiguous_or_incomplete_rejected():
    assert "native reasoning" in render_candidate(
        receipt(reasoning="native reasoning"), "long"
    )
    assert render_candidate(
        receipt("brief explanation\nAnswer: 2"), "short"
    ).startswith("<think>\n\n</think>")
    for sample in (
        receipt("<think>inline</think>Answer: 2", "native"),
        receipt(finish="length"),
        receipt("<think>unclosed"),
        receipt("<think></think>Answer: 2"),
    ):
        with pytest.raises(ConfigurationError):
            render_candidate(sample, "long")


def test_judge_counterexamples():
    base = MockAPI().generate_audited([ModelTask("t", "a", "joint_verify", "q", [])])[0]
    good = json.loads(base.raw["choices"][0]["message"]["content"])
    assert parse_judgment(base, {})["status"] == "pass"
    for patch, expected in (
        ({"answer_status": True}, "error"),
        ({"answer_status": "yes"}, "error"),
        ({"reasoning_status": "fail"}, "fail"),
        ({"visual_status": "uncertain"}, "uncertain"),
        ({"critical_errors": ["wrong visual value"]}, "error"),
        ({"checks": []}, "uncertain"),
    ):
        value = {**good, **patch}
        assert (
            parse_judgment(receipt(canonical(value).decode()), {})["status"] == expected
        )
    bad = dict(good)
    del bad["visual_status"]
    assert parse_judgment(receipt(canonical(bad).decode()), {})["status"] == "error"


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"short": "error", "complex_source": True}, "technical_pending"),
        ({"short": "not_dispatched"}, "technical_pending"),
        ({"short": "uncertain"}, "needs_review"),
        ({"short": "fail"}, "dispatch_long"),
        ({"short": "pass", "complex_source": True}, "dispatch_long"),
        ({"short": "fail", "long": "pass"}, "accepted_cot"),
        ({"short": "fail", "long": "fail"}, "candidate_rejected"),
        ({"bypass": True}, "bypass_verified"),
    ],
)
def test_routing(kwargs, expected):
    assert decide_route(**kwargs) == expected


def test_input_schema_and_originals(cot_config):
    source = cot_config["sources"][0]
    row = json.loads(open(source["path"], encoding="utf-8").read())
    record = validate_input(row, source, "source-hash", 1)
    assert record.input_outcome == "pass"
    assert record.original == row
    assert (
        verify_final("<think></think>\nExplanation\nAnswer: 2", record)["status"]
        == "pass"
    )
    assert verify_final("Answer: 12", record)["status"] == "fail"
    assert verify_final("Answer: 2 cm", record)["reason"] == "unit_mismatch"
    bad = {"messages": [{"from": "human", "value": "q"}]}
    assert validate_input(bad, source, "source-hash", 2).input_outcome == "fail"
    mcq = replace(
        record, reference="B", policy={"task_type": "mcq", "reference_grade": "strong"}
    )
    assert verify_final("Answer: A", mcq)["status"] == "fail"
    assert verify_final("Answer: B", mcq)["status"] == "pass"
