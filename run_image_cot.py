"""Opt-in image-CoT CLI; never imports or invokes the legacy Runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from datastudio.image_cot.contracts import (
    CoTError,
    canonical,
    load_config,
    validate_config,
)
from datastudio.image_cot.export import export_training, publish_release
from datastudio.image_cot.runner import prepare, resume, run
from datastudio.image_cot.storage import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare", "run"):
        command = sub.add_parser(name)
        command.add_argument("-c", "--config", required=True)
        command.add_argument("--backend", choices=("http", "mock"), default="http")
        if name != "check":
            command.add_argument("--limit", type=int)
    for name in ("resume", "report", "export"):
        command = sub.add_parser(name)
        command.add_argument("--run", required=True)
        if name == "export":
            command.add_argument(
                "-c",
                "--config",
                help="optional export policy; does not regenerate model outputs",
            )
            group = command.add_mutually_exclusive_group(required=True)
            group.add_argument("--preview", action="store_true")
            group.add_argument("--stage-release", action="store_true")
            group.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            validate_config(
                load_config(args.config), runtime=True, backend=args.backend
            )
            print(
                "PASS: configuration preflight only; no endpoint call or quality test"
            )
            return 0
        if args.command == "prepare":
            result = prepare(load_config(args.config), limit=args.limit)
        elif args.command == "run":
            result = run(
                load_config(args.config), backend=args.backend, limit=args.limit
            )
        elif args.command == "resume":
            result = resume(args.run)
        elif args.command == "report":
            print(canonical(read_json(Path(args.run) / "report.json")).decode("utf-8"))
            return 0
        else:
            policy = load_config(args.config)["export"] if args.config else None
            result = (
                publish_release(args.run, export_policy=policy)
                if args.publish
                else export_training(
                    args.run, preview=args.preview, export_policy=policy
                )
            )
        print(result)
        if args.command in {"run", "resume"}:
            report = read_json(Path(result) / "report.json")
            print(
                canonical(
                    {
                        "phase": "construction",
                        "status": report["status"],
                        "counts": report["counts"],
                        "quality_verified": False,
                        "throughput_verified": False,
                    }
                ).decode()
            )
            return 4 if report["status"] == "completed_with_pending" else 0
        return 0
    except KeyboardInterrupt:
        print(
            "STOP: interrupted; preserve run journals and resume without replaying unresolved requests",
            file=sys.stderr,
        )
        return 130
    except CoTError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 3 if args.command == "export" else 2
    except (OSError, KeyError, TypeError, ValueError):
        print(
            "FAIL: invalid configuration or IO; private run evidence preserved",
            file=sys.stderr,
        )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
