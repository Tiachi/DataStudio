"""Loopback protocol tests for native effort and immediate per-response durability."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from datastudio.image_cot.audited_api import AuditedOpenAIAPI, MockAPI
from datastudio.image_cot.contracts import ModelTask, canonical
from datastudio.image_cot.storage import RunStore


@pytest.mark.parametrize("sink_failure", [False, True])
def test_http_saves_first_completion_before_slow_request_finishes(
    cot_config, tmp_path, sink_failure
):
    persisted = threading.Event()
    seen = []
    waits = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(payload)
            text = str(payload["messages"])
            if "slow" in text:
                waits.append(persisted.wait(timeout=3))
            raw = (
                MockAPI()
                .generate_audited([ModelTask("t", "a", "generate_low", "q", [])])[0]
                .raw
            )
            body = canonical(raw)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("x-request-id", "loopback-only")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = {
            "provider_profile": "openai_compatible",
            "ip": "127.0.0.1",
            "port": server.server_address[1],
            "model": "local-synthetic",
            "auth": {"mode": "none"},
        }
        model = {
            **cot_config["models"]["long"],
            "effective_options": {"reasoning_effort": "low"},
        }
        api = AuditedOpenAIAPI(connection, model, cot_config["limits"])
        tasks = [
            ModelTask(
                name,
                "a-" + name,
                "generate_low",
                name,
                [{"type": "text", "value": name}],
            )
            for name in ("fast", "slow")
        ]
        with RunStore(tmp_path / "private_run") as store:
            plan = {
                "input_part": "input-0",
                "stage": "generate_low",
                "part_id": "p0",
                "tasks": [t.intent() for t in tasks],
            }
            part = store.open_part(plan)
            store.append_intents(part, [t.intent() for t in tasks])

            def sink(receipt):
                if sink_failure:
                    persisted.set()
                    raise OSError("synthetic_disk_failure")
                store.append_receipts(part, [receipt.to_dict()])
                # This callback occurs while the second HTTP request is waiting.
                assert store.recover_part(part)["receipts"][receipt.task_id]["raw"]
                persisted.set()

            if sink_failure:
                with pytest.raises(OSError, match="synthetic_disk_failure"):
                    api.generate_audited(tasks, receipt_sink=sink)
            else:
                receipts = api.generate_audited(tasks, receipt_sink=sink)
                assert len(receipts) == 2 and waits == [True]
                assert len(store.recover_part(part)["receipts"]) == 2
                assert all(r.provider_request_id == "loopback-only" for r in receipts)
        assert all(
            p["reasoning_effort"] == "low" and p["stream"] is False for p in seen
        )
    finally:
        persisted.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
