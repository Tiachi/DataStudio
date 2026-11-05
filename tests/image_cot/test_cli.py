"""Subprocess smoke tests of the installed-dependency/source-checkout entry point."""

import json
from pathlib import Path
import subprocess
import sys

from datastudio.image_cot.storage import read_json, read_jsonl

ROOT = Path(__file__).resolve().parents[2]


def invoke(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "run_image_cot.py"), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_check_run_and_export_override_do_not_require_api(cot_config, tmp_path):
    config_path = tmp_path / "private_config.py"
    config_path.write_text("image_cot = " + repr(cot_config), encoding="utf-8")
    checked = invoke("check", "-c", str(config_path), "--backend", "mock")
    assert checked.returncode == 0, checked.stderr
    ran = invoke("run", "-c", str(config_path), "--backend", "mock", "--limit", "1")
    assert ran.returncode == 0, ran.stderr
    lines = ran.stdout.strip().splitlines()
    run_root = Path(lines[0])
    assert json.loads(lines[-1])["quality_verified"] is False
    policy = {**cot_config["export"], "dataset_name": "export_only_rename"}
    export_path = tmp_path / "export_config.py"
    export_path.write_text("image_cot = " + repr({"export": policy}), encoding="utf-8")
    exported = invoke(
        "export", "--run", str(run_root), "-c", str(export_path), "--preview"
    )
    assert exported.returncode == 0, exported.stderr
    target = Path(exported.stdout.strip())
    assert next(read_jsonl(target / "data.jsonl"))["id"].startswith(
        "export_only_rename_"
    )
    assert read_json(run_root / "report.json")["requests_reserved"] == 3


def test_cli_reports_pending_budget_as_non_success(cot_config, tmp_path):
    cot_config["limits"]["max_requests_total"] = 1
    path = tmp_path / "limited.py"
    path.write_text("image_cot = " + repr(cot_config), encoding="utf-8")
    result = invoke("run", "-c", str(path), "--backend", "mock")
    assert result.returncode == 4, result.stderr
    assert (
        json.loads(result.stdout.strip().splitlines()[-1])["status"]
        == "completed_with_pending"
    )
