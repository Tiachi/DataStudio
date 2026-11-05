"""Only tiny synthetic fixtures; no user datasets or external API calls."""

from pathlib import Path
import runpy

from PIL import Image
import pytest

from datastudio.image_cot.contracts import canonical

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cot_config(tmp_path):
    inputs = tmp_path / "inputs"
    media = inputs / "images"
    media.mkdir(parents=True)
    Image.new("RGB", (32, 24), "white").save(media / "sample.png")
    row = {
        "id": "synthetic-0",
        "conversations": [
            {"from": "human", "value": "<image>\nHow many marked groups?"},
            {"from": "gpt", "value": "2"},
        ],
        "images": ["images/sample.png"],
        "category": "reasoning",
        "options": {"A": "one", "B": "two"},
    }
    annotation = inputs / "source.jsonl"
    annotation.write_bytes(canonical(row) + b"\n")
    config = runpy.run_path(str(ROOT / "configs/image_cot/base.py"))["image_cot"]
    config["private_work_root"] = str(tmp_path / "private_work")
    config["sources"] = [
        {
            "schema": "standard_qa_v1",
            "name": "synthetic",
            "revision": "test-v1",
            "path": str(annotation),
            "media_root": str(inputs),
            "max_image_pixels": 4096,
            "task_policy": {"task_type": "numeric", "reference_grade": "strong"},
            "category_overrides": {"ocr": {"task_type": "pure_ocr"}},
        }
    ]
    config["limits"].update(
        max_requests_total=100,
        max_output_tokens_total=100000,
        max_wall_time_s=60,
        max_request_bytes=1024 * 1024,
        max_decoded_pixels_per_batch=4096,
        requests_per_microbatch=2,
        source_records_per_part=4,
    )
    for model in config["models"].values():
        model.update(max_output_tokens=1024, read_timeout_s=5, batch_deadline_s=10)
    config["export"].update(
        dataset_name="synthetic_cot",
        training_media_root=str(inputs),
        original_field_mapping={
            "id": "original_id",
            "conversations": "original_conversations",
            "images": "original_images",
        },
    )
    return config
