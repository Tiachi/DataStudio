# Examples Guide

DataStudio includes 151 demo data samples (multimodal QA with images) and 5 ready-to-run example configs. You can try the examples immediately after installation — no additional data download required.

> Demo data location: `configs/examples/demo_data/`
> Example configs: `configs/examples/*.py`

---

## Demo Data

The demo dataset contains 151 multimodal QA samples covering diverse sources (Caption, VQA, OCR, Math, Chart, etc.). Each sample includes:

- **Multi-turn conversations** in `messages` format (OpenAI-style)
- **Image references** (265 images included in `demo_data/images/`)
- **`<think>` tags** in answers (useful for testing normalization)
- **`ori_answer` fields** in 126 samples (original answers before rewriting)

Data format:
```json
{
  "messages": [
    {"role": "user", "content": "<image>\nWhat color is the phone?"},
    {"role": "assistant", "content": "<think>\n\n</think>\nThe phone is red."}
  ],
  "images": ["images/xxx.jpg"],
  "id": "sample_001",
  "source": "st_vqa"
}
```

---

## Example Overview

| # | Config | MLLM Required | Description |
|---|--------|:---:|-------------|
| 1 | `rule_filter_only.py` | No | Rule filtering (length, repetition, image validation) |
| 2 | `mllm_quality_filter.py` | Yes | Rule filters + MLLM question-image consistency check |
| 3 | `mllm_answer_rewrite.py` | Yes | MLLM answer rewriting (plain text mode) |
| 4 | `honeypipe.py` | Yes | 5-stage end-to-end pipeline (HoneyPipe) |
| 5 | `text_normalization.py` | No | Remove think tags, normalize image tags and prompts |

**Recommended order**: Start with Example 1 or 5 (no MLLM needed), then try 2–4 if you have a deployed inference service.

---

## Examples Without MLLM (Run Directly)

### Example 1: Rule-Based Filtering

Filters out samples with abnormal conversation lengths, text length anomalies, or repetitive content. Text-only, no image loading. Inherits rule filters from `_base_/filters/filter_rule_base_for_answer.py`.

```bash
python run.py -c configs/examples/rule_filter_only.py
```

**Inherited operators** (from `_base_`):
- `ImageSizeFilter`, `ImageAspectRatioFilter`, `ImageExtFilter` (no-ops when `use_image=False`)
- `ConvLengthFilter`: Keep conversations with 1–300 turns
- `LengthAnomalyFilter`: Reject questions < 2 or > 4096 tokens; reject answers < 1 or > 4096 tokens
- `TextRepeatFilter`: Detect repetitive patterns in questions and answers

**Output:** `./output/examples/rule_filter_only/`

---

### Example 5: Text Normalization

Cleans up data formatting without filtering. The demo data contains `<think>...</think>` tags in all 151 answers, making it a good test case.

```bash
python run.py -c configs/examples/text_normalization.py
```

**What it does:**
- `RemoveThinkRewriter`: Remove `<think>...</think>` tags and their content
- `NormImageTagRewriter`: Move `<image>` tags to the beginning of questions
- `NormPromptRewriter`: Standardize prompt format
- `RemoveReasonRewriter`: Remove reasoning prefix content

**Output:** `./output/examples/text_normalization/`

---

## Examples With MLLM (Require Inference Service)

These examples require an OpenAI-compatible inference service. Deploy one first:

```bash
# vLLM
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --port 8000 --tensor-parallel-size 4

# or SGLang
python -m sglang.launch_server \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --port 8000 --tp 4
```

Then update the `model` section in the config if your setup differs (model name, port, etc.).

---

### Example 2: MLLM Quality Filtering

Rule-based filters run first (inherited from `_base_`), then an MLLM checks question-image consistency.

```bash
python run.py -c configs/examples/mllm_quality_filter.py
```

**Pipeline:**
1. `priority=0` — Rule filters (from `_base_/filters/filter_rule_base_for_question.py`)
2. `priority=10` — MLLM question-image consistency filter

**Key config:**
```python
qi_consist_request = dict(
    type="RequestBuilder",
    prompt="prompts/filter/question_image_consist_v2.txt",
    key_templates={"result": "q{idx}", "reason": "q{idx}_reason"},
    with_image=True,
    with_question=True,
)
```

**Output:** `./output/examples/mllm_quality_filter/`

---

### Example 3: MLLM Answer Rewriting

Sends the original question + image to the MLLM, which generates a new answer. Uses plain text mode (no JSON parsing).

```bash
python run.py -c configs/examples/mllm_answer_rewrite.py
```

**Key config:**
```python
model = dict(
    return_dict=False,   # Plain text output — model response is the new answer
)

rewriter_request = dict(
    type="RequestBuilder",
    prompt=None,          # No template, send original question directly
    key_templates=None,   # Plain text mode
    with_image=True,
    with_question=True,
)
```

After rewriting, the original answer is saved in the `ori_answer` field.

**Output:** `./output/examples/mllm_answer_rewrite/`

---

### Example 4: HoneyPipe Full Pipeline (5 Stages)

End-to-end processing from raw data to clean training data. All configuration is expanded inline, serving as a reference for understanding every available option.

```bash
python run.py -c configs/examples/honeypipe.py
```

**Execution order:**

| Priority | Stage | Description |
|:--------:|-------|-------------|
| 0 | Rule Filters | ConvLength, ImageSize, AspectRatio, ImageExt, LengthAnomaly, TextRepeat |
| 10 | MLLM Filter | Question-image consistency check |
| 20 | MLLM Rewriter | Regenerate answers (plain text mode) |
| 30 | Consistency Check | Compare old vs new answers |
| 40 | Normalization | Remove think tags, normalize image tags |

**Output:** `./output/examples/honeypipe/`

---

## Customizing Examples

To adapt any example for your own data:

1. **Replace the dataset**: Change `dataset_yaml` to point to your own YAML file
2. **Set `data_root`**: Set to the base directory for resolving relative image paths in your data
3. **Adjust model config**: Update `model`, `api_base`, `port` to match your inference service
4. **Tune parameters**: Adjust `batch_size`, `thread_num`, filter thresholds, etc.

```python
# Example: use your own data
dataset_yaml = "/path/to/your/dataset.yaml"

dataloader = dict(
    dataset=dataset_yaml,
    data_root="/path/to/your/data/root",   # Base path for image resolution
    batch_size=5000,
    use_image=True,
)
```
