"""Bounded offline documentation QA, with an optional synthetic CLI smoke run.

Default: named Markdown files only; checks local links/anchors, JSON/Python
examples and fence balance. --smoke-root must name a new directory outside the
checkout. It extracts only the two reviewed DOC-CHECK examples, generates a tiny
image and calls the CLI with mode=mock/backend=mock. No external API or secrets.
The smoke root is preserved and never recursively deleted or overwritten.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    ROOT / "README.md",
    ROOT / "docs/image_cot_operations_manual.md",
    ROOT / "docs/image_cot_implementation.md",
    ROOT / "docs/image_cot_validation_2026-11-05.md",
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def json_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, f"Duplicate JSON example key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON example constant: {value}")


def parse_json(text):
    return json.loads(
        text, object_pairs_hook=json_object, parse_constant=reject_constant
    )


def anchors(text):
    result = set(re.findall(r'<a\s+(?:id|name)="([^"]+)"', text))
    counts = {}
    outside = re.sub(r"^```[^\n]*\n.*?^```[ \t]*$", "", text, flags=re.M | re.S)
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", outside, flags=re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.casefold()).replace(" ", "-")
        number = counts.get(slug, 0)
        result.add(slug + (f"-{number}" if number else ""))
        counts[slug] = number + 1
    return result


def check_docs():
    report = []
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        need(
            len(re.findall(r"^```", text, re.M)) % 2 == 0,
            f"Unbalanced fences: {path.name}",
        )
        blocks = re.findall(r"^```([^\n]*)\n(.*?)^```[ \t]*$", text, re.M | re.S)
        parsed = {"json": 0, "python": 0}
        for language, code in blocks:
            language = language.strip()
            if language == "json":
                parse_json(code)
                parsed["json"] += 1
            elif language == "python":
                ast.parse(code)
                parsed["python"] += 1
        links = 0
        for target in re.findall(r"(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)", text):
            parsed_url = urlsplit(target)
            if parsed_url.scheme or parsed_url.netloc:
                continue
            destination = (
                (path.parent / unquote(parsed_url.path)).resolve()
                if parsed_url.path
                else path
            )
            need(destination.is_file(), f"Missing local link in {path.name}: {target}")
            if parsed_url.fragment:
                need(
                    unquote(parsed_url.fragment)
                    in anchors(destination.read_text(encoding="utf-8")),
                    f"Missing anchor in {path.name}: {target}",
                )
            links += 1
        report.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "local_links": links,
                "parsed_examples": parsed,
            }
        )
    return report


def named_example(name, language):
    text = DOCS[1].read_text(encoding="utf-8")
    matches = re.findall(
        r"<!-- DOC-CHECK: "
        + re.escape(name)
        + r" -->\s*```"
        + language
        + r"\n(.*?)^```",
        text,
        re.M | re.S,
    )
    need(len(matches) == 1, f"Exactly one {name} example required")
    return matches[0]


def smoke(root):
    need(root.is_absolute(), "--smoke-root must be absolute")
    root = root.resolve()
    need(not root.exists(), "Refusing to reuse an existing smoke directory")
    need(
        not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root),
        "Smoke root must not overlap the checkout",
    )
    root.mkdir(parents=True)
    inputs, work = root / "inputs", root / "work"
    (inputs / "images").mkdir(parents=True)
    from PIL import Image, ImageDraw

    with Image.new("RGB", (128, 96), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((15, 20, 45, 60), fill="blue")
        draw.rectangle((80, 20, 110, 60), fill="blue")
        image.save(inputs / "images/example.png")
    annotation = parse_json(named_example("input-example", "json"))
    (inputs / "data.jsonl").write_text(
        json.dumps(annotation, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    config_text = named_example("tiny-config", "python")
    for original, replacement in (
        ("/path/to/DataStudio", ROOT),
        ("/path/to/cot-input", inputs),
        ("/path/to/cot-work", work),
    ):
        pattern = f'Path("{original}")'
        need(config_text.count(pattern) == 1, f"Config path marker changed: {original}")
        config_text = config_text.replace(pattern, f"Path({str(replacement)!r})")
    config_path = root / "cot_demo.py"
    config_path.write_text(config_text, encoding="utf-8")
    env = os.environ.copy()
    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "TEACHER_API_KEY"):
        env.pop(name, None)
    env["PYTHONIOENCODING"] = "utf-8"
    steps = []

    def cli(*args, expected=0):
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_image_cot.py"), *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=45,
        )
        need(
            result.returncode == expected,
            f"CLI {args[0]} returned {result.returncode}: {result.stderr[-2000:]}",
        )
        steps.append({"command": args[0], "returncode": result.returncode})
        return result

    cli("check", "-c", str(config_path), "--backend", "mock")
    prepared = cli(
        "prepare", "-c", str(config_path), "--limit", "1", "--backend", "mock"
    )
    need(Path(prepared.stdout.strip()).is_file(), "prepare did not return its manifest")
    built = cli("run", "-c", str(config_path), "--backend", "mock", "--limit", "1")
    run_root = Path(built.stdout.strip().splitlines()[0])
    report = json.loads(cli("report", "--run", str(run_root)).stdout)
    need(
        report["counts"] == {"accepted_cot": 1} and report["requests_reserved"] == 3,
        "Unexpected mock flow",
    )
    need(
        report["backend"] == "mock" and report["quality_verified"] is False,
        "Mock must not claim model quality",
    )
    cli("resume", "--run", str(run_root))
    exported = cli("export", "--run", str(run_root), "--preview")
    preview = Path(exported.stdout.strip())
    manifest = json.loads((preview / "manifest.json").read_text(encoding="utf-8"))
    need(
        manifest["preview"] is True and manifest["published"] is False,
        "Preview identity",
    )
    rows = [
        json.loads(line)
        for line in (preview / "data.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    need(
        len(rows) == 1
        and rows[0]["modal_mask"] == 1
        and rows[0]["original_id"] == annotation["id"],
        "Preview fields",
    )
    refused = cli("export", "--run", str(run_root), "--stage-release", expected=3)
    need(
        "mock_cannot_publish" in refused.stderr,
        "Mock release was not refused for the right reason",
    )
    return {
        "root": str(root),
        "run": str(run_root),
        "preview": str(preview),
        "steps": steps,
        "backend": "mock",
        "external_api_calls": 0,
        "scope": "documented synthetic control flow only; no quality/training/template certification",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path)
    args = parser.parse_args()
    try:
        report = {"status": "PASS", "documents": check_docs()}
        if args.smoke_root:
            report["smoke"] = smoke(args.smoke_root)
            with (args.smoke_root / "documentation_validation.json").open(
                "x", encoding="utf-8"
            ) as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, SyntaxError, subprocess.TimeoutExpired) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
