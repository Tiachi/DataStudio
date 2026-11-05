"""Real aiohttp/parent-engine calls against loopback only; no vendor credentials."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest

from datastudio.image_cot.audited_api import AuditedOpenAIAPI, MockAPI
from datastudio.image_cot.contracts import InvariantError, ModelTask, canonical
from datastudio.image_cot.runner import ModelHandle, run
from datastudio.image_cot.storage import RunStore, read_json, read_jsonl


@contextmanager
def fake_teacher(*, wrong_short=False, http_status=200, judge_patch=None):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(
                {
                    "payload": payload,
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                }
            )
            text = "\n".join(
                p.get("text", "") for m in payload["messages"] for p in m["content"]
            )
            if "UNTRUSTED_DATA_JSON:" in text:
                role = "joint_verify"
            elif "Check whether the user question" in text:
                role = "question_check"
            elif "Solve the supplied image question carefully" in text:
                role = "long"
            else:
                role = "short"
            response = (
                MockAPI().generate_audited([ModelTask("t", "a", role, "q", [])])[0].raw
            )
            response["model"] = "served-model"
            response["system_fingerprint"] = "synthetic-server-v1"
            if role == "short" and wrong_short:
                response["choices"][0]["message"][
                    "content"
                ] = "I read one group.\nAnswer: 1"
            if role == "joint_verify" and judge_patch:
                judgment = json.loads(response["choices"][0]["message"]["content"])
                judgment.update(judge_patch)
                response["choices"][0]["message"]["content"] = canonical(
                    judgment
                ).decode()
            encoded = canonical(
                response
                if http_status == 200
                else {"error": {"message": "synthetic error"}}
            )
            self.send_response(http_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("x-request-id", f"local-{len(calls)}")
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def use_http(config, port):
    config["connections"]["lan_teacher"] = {
        "provider_profile": "openai_compatible",
        "ip": "127.0.0.1",
        "port": port,
        "model": "served-model",
        "auth": {"mode": "none"},
    }
    config["quota_pools"] = {
        "teacher_account": {
            "rpm": 1000,
            "input_tpm": 1000000,
            "output_tpm": 1000000,
            "input_tokens_per_request": 5000,
        }
    }
    return config


def test_actual_http_short_failure_long_success_and_no_hidden_retries(
    cot_config, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak-to-unauthed-local-service")
    with fake_teacher(wrong_short=True) as (port, calls):
        config = use_http(cot_config, port)
        root = run(config)
        selection = next(read_jsonl(root / "selection/input-000000.jsonl"))
        assert selection["terminal"] == "accepted_cot" and selection["stage"] == "long"
        assert len(calls) == 4  # no short verification after an exact-answer failure
        assert all(
            c["path"] == "/v1/chat/completions" and c["auth"] is None for c in calls
        )
        for call in calls:
            payload = call["payload"]
            assert payload["model"] == "served-model"
            assert not any(
                k in payload
                for k in ("top_k", "min_p", "temperature", "presence_penalty")
            )
            assert any(
                p.get("image_url", {})
                .get("url", "")
                .startswith("data:image/png;base64,")
                for m in payload["messages"]
                for p in m["content"]
            )
        assert read_json(root / "report.json")["quality_verified"] is False


def test_gold_only_reaches_judge_and_original_question_is_visible(cot_config):
    source = cot_config["sources"][0]
    row = next(read_jsonl(source["path"]))
    row["conversations"][0]["value"] += " Answer only the number."
    row["conversations"][1]["value"] = "REFERENCE_CANARY_9acb"
    Path(source["path"]).write_bytes(canonical(row) + b"\n")
    source["task_policy"].update(
        task_type="open",
        question_transform="remove_exact_suffix",
        remove_suffix=" Answer only the number.",
    )
    with fake_teacher() as (port, calls):
        run(use_http(cot_config, port))
        assert len(calls) == 3
        assert calls[0]["payload"]["messages"][0]["role"] == "system"
        assert calls[2]["payload"]["messages"][0]["role"] == "system"
        assert "REFERENCE_CANARY_9acb" not in canonical(calls[0]["payload"]).decode()
        assert "REFERENCE_CANARY_9acb" not in canonical(calls[1]["payload"]).decode()
        verify = canonical(calls[2]["payload"]).decode()
        assert (
            "REFERENCE_CANARY_9acb" in verify
            and "question_original" in verify
            and "Answer only the number." in verify
        )


@pytest.mark.parametrize("status", [400, 429, 503])
def test_technical_errors_never_upgrade_to_long_or_auto_replay(cot_config, status):
    cot_config["policy"]["long_source_allowlist"] = ["synthetic"]
    with fake_teacher(http_status=status) as (port, calls):
        config = use_http(cot_config, port)
        root = run(config)
        assert read_json(root / "report.json")["counts"] == {"technical_pending": 1}
        assert run(config) == root
        assert len(calls) == 1


def test_durable_generation_receipt_survives_commit_crash(cot_config, monkeypatch):
    original = RunStore.commit_part
    failed = []

    def fail_once(self, part, outcomes):
        if part.parent.name == "short" and not failed:
            failed.append(True)
            raise OSError("injected after durable receipt, before commit")
        return original(self, part, outcomes)

    with fake_teacher() as (port, calls):
        config = use_http(cot_config, port)
        monkeypatch.setattr(RunStore, "commit_part", fail_once)
        with pytest.raises(OSError):
            run(config)
        assert len(calls) == 2
        root = run(config)
        assert len(calls) == 3  # generation parsed locally, only judge dispatched
        assert read_json(root / "report.json")["counts"] == {"accepted_cot": 1}


def test_missing_responses_are_preserved_as_unresolved(cot_config, monkeypatch):
    with fake_teacher() as (port, calls):
        config = use_http(cot_config, port)
        original = ModelHandle.generate_audited
        monkeypatch.setattr(
            ModelHandle, "generate_audited", lambda self, tasks, **kwargs: []
        )
        with pytest.raises(InvariantError, match="coverage"):
            run(config)
        monkeypatch.setattr(ModelHandle, "generate_audited", original)
        root = run(config)
        assert not calls
        assert read_json(root / "report.json")["counts"] == {"technical_pending": 1}


def test_changing_judge_reuses_generation_without_new_teacher_calls(
    cot_config, tmp_path
):
    with fake_teacher() as (port, calls):
        config = use_http(cot_config, port)
        first = run(config)
        assert len(calls) == 3
        prompt = tmp_path / "new_judge.txt"
        prompt.write_text(
            Path(config["models"]["joint_verify"]["prompt"]).read_text()
            + "\nAudit revision 2.\n"
        )
        config["models"]["joint_verify"]["prompt"] = str(prompt)
        config["reuse_from"] = [str(first)]
        second = run(config)
        assert first != second and len(calls) == 4
        assert read_json(second / "report.json")["requests_reserved"] == 1
        assert run(config) == second and len(calls) == 4


def test_deepseek_exact_payload_and_key_scoping(cot_config, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-unit-test-key")
    config = cot_config["models"]["long"]
    config["effective_options"] = {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }
    api = AuditedOpenAIAPI(
        cot_config["connections"]["deepseek_validation"], config, cot_config["limits"]
    )
    payload = api._build_payload([{"role": "user", "content": "fixture"}])
    assert payload["thinking"] == {"type": "enabled"} and "extra_body" not in payload
    assert "temperature" not in payload and payload["model"] == "deepseek-flash"
    assert (
        api._build_openai_headers()["Authorization"] == "Bearer synthetic-unit-test-key"
    )
    assert api.retry == 1
    system = api._build_payload(
        [{"role": "system", "content": [{"type": "text", "text": "audit policy"}]}]
    )
    assert system["messages"][0]["content"] == "audit policy"


def test_correct_final_with_failed_visual_or_process_judgment_never_becomes_sft(
    cot_config,
):
    with fake_teacher(
        judge_patch={
            "reasoning_status": "fail",
            "critical_errors": ["wrong decisive visual relation"],
        }
    ) as (port, calls):
        root = run(use_http(cot_config, port))
        assert len(calls) == 5  # short and long both audited once and rejected
        assert read_json(root / "report.json")["counts"] == {"candidate_rejected": 1}
        selection = next(read_jsonl(root / "selection/input-000000.jsonl"))
        assert "candidate_id" not in selection
