# Quick Start

This guide helps you get started with DataStudio for multimodal data processing from scratch.

## 1. Installation

### Requirements

- Python >= 3.10
- Sufficient disk space (LMDB image cache requires additional storage)

### Steps

```bash
# Clone the repository
git clone https://github.com/Open-Bee/DataStudio.git
cd DataStudio

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install
pip install -e .

# Verify
python -c "import datastudio; print(datastudio.__version__)"
```

### Optional Dependencies

```bash
# Weights & Biases monitoring
pip install wandb
wandb login
```

### Try It Out

After installation, run the built-in examples to verify your setup (no additional data needed):

```bash
# Rule filtering example (CPU only, no MLLM required)
python run.py -c configs/examples/rule_filter_only.py

# Text normalization example (remove think tags, etc.)
python run.py -c configs/examples/text_normalization.py
```

See the **[Examples Guide](examples.md)** for all 5 examples covering different scenarios.

---

## 2. Data Format

DataStudio uses standard multimodal conversation format, supporting JSON and JSONL files.

### Basic Format

```json
{
  "id": "sample_001",
  "image": "/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is in this image?"},
    {"from": "gpt", "value": "The image shows a beautiful sunset."}
  ]
}
```

Field descriptions:

| Field | Required | Description |
|-------|----------|-------------|
| `id` | No | Unique sample identifier |
| `image` | No | Image path; string for single image, list for multiple images |
| `conversations` | Yes | Conversation list; `from` is `"human"` or `"gpt"`, `value` is text content |

### Multi-Image Format

```json
{
  "id": "multi_img_001",
  "image": ["/path/to/img1.jpg", "/path/to/img2.jpg"],
  "conversations": [
    {"from": "human", "value": "<image>\n<image>\nCompare these two images."},
    {"from": "gpt", "value": "The first one is daytime, the second is nighttime."}
  ]
}
```

### Schema Compatibility

DataStudio supports both `conversations` format and `messages` format (OpenAI-style), automatically converting to the internal standard format during loading.

---

## 3. Dataset YAML Configuration

Datasets are described via YAML files. A single YAML can contain multiple data sources:

```yaml
# dataset.yaml
datasets:
  - json_path: /data/dataset_a.jsonl
    source: dataset_a

  - json_path: /data/dataset_b.json
    source: dataset_b
```

Field descriptions:

| Field | Description |
|-------|-------------|
| `json_path` / `file_path` / `jsonl_path` | Data file path (all three are equivalent) |
| `source` | Data source name, used for organizing output directories |

---

## 4. Core Concepts

### Operator Types

DataStudio provides two types of operators, each with rule-based and MLLM-powered versions:

| Type | Purpose | Core Method |
|------|---------|-------------|
| **Filter** | Decide whether to keep a sample | `check(item, qa_idx) -> (rejected, reason)` |
| **Rewriter** | Modify content without removing samples | `rewrite(item, qa_idx) -> new_answer` |
| **MLLMFilter** | MLLM-powered quality filtering | Via `RequestBuilder` |
| **MLLMRewriter** | MLLM-powered content rewriting | Via `RequestBuilder` |

### Pipeline Structure

A `Pipeline` consists of multiple `SubPipeline` stages, executed in order of `priority` (lower values run first). Within each `SubPipeline`, operators run sequentially — samples rejected by an earlier operator are skipped by later ones.

```
Pipeline
├── SubPipeline (priority=0)  ← runs first
│   ├── ConvLengthFilter
│   └── ImageSizeFilter
├── SubPipeline (priority=1)
│   └── MLLMFilter
└── SubPipeline (priority=2)  ← runs last
    └── MLLMRewriter
```

### Config Inheritance

Config files support inheritance via `_base_`. The `@/` prefix resolves from the `configs/` directory:

```python
_base_ = [
    "@/_base_/models/local_api_model.py",
    "@/_base_/dataset.py",
    "@/_base_/filters/filter_rule_base_for_question.py",
]
```

- Base configs are merged in order
- Child config fields override base values
- `dict` fields are merged recursively

Pre-defined base configs:

| File | Description |
|------|-------------|
| `@/_base_/dataset.py` | Default DataLoader and DataSaver parameters |
| `@/_base_/models/local_api_model.py` | Default local API model parameters |
| `@/_base_/filters/filter_rule_base_for_question.py` | Common question filter rules (priority=0) |
| `@/_base_/filters/filter_rule_base_for_answer.py` | Common answer filter rules (priority=50) |
| `@/_base_/rewriters/rewriter_rule_base.py` | Rewriter base template (priority=1) |

---

## 5. Built-in Operators

### Filters

| Operator | Description | Key Parameters |
|----------|-------------|----------------|
| `ConvLengthFilter` | Filter by conversation turn count | `min_length`, `max_length` |
| `ImageSizeFilter` | Filter by image dimensions | `min_size`, `max_ratio` |
| `ImageAspectRatioFilter` | Filter by aspect ratio | `max_aspect_ratio` |
| `ImageExtFilter` | Filter by image format | — |
| `LengthAnomalyFilter` | Detect text length anomalies | `min_length`, `max_length`, `use_tokenizer` |
| `ResponseTagFilter` | Filter by response tags | — |
| `TextRepeatFilter` | Detect text repetition | `check_question`, `check_answer` |
| `MLLMFilter` | MLLM quality filtering | `request_builder` |

### Rewriters

| Operator | Description |
|----------|-------------|
| `RemoveThinkRewriter` | Remove `<think>...</think>` tags |
| `NormThinkRewriter` | Normalize think tag format |
| `AddNoThinkRewriter` | Add `/no_think` prefix |
| `NormImageTagRewriter` | Standardize `<image>` tag placement |
| `NormPromptRewriter` | Standardize prompt format |
| `NormMultiTurnPromptRewriter` | Standardize multi-turn prompts |
| `RemoveAnswerRewriter` | Remove answer content |
| `RemoveReasonRewriter` | Remove reasoning prefix |
| `SplitRewriter` | Split multi-turn conversations into single-turn |
| `MLLMRewriter` | MLLM content rewriting |
| `SelectiveMLLMRewriter` | Conditional MLLM rewriting |

---

## 6. Config File Details

### Minimal Config Example

The following example shows a pipeline using only rule-based operators:

```python
# configs/my_first_pipeline.py
_base_ = ["@/_base_/dataset.py"]

work_dir = "./work_dirs/my_first_run"
logger = dict(type="Logger", log_file="logs/run.log")

dataset_yaml = "/path/to/dataset.yaml"

dataloader = dict(
    dataset=dataset_yaml,
    batch_size=1000,
    use_image=False,             # Text-only processing, no image loading
)

datasaver = dict(
    dataset=dataset_yaml,
    output_dir="./output",
    save_yaml_name="my_output",  # Output YAML file name prefix
)

pipeline = dict(
    type="Pipeline",
    operations={
        "rule_filters": dict(
            cfg=dict(type="SubPipeline", operators=[
                dict(type="ConvLengthFilter", min_length=1, max_length=20),
                dict(type="LengthAnomalyFilter", min_length=2, max_length=4096,
                     check_question=True, check_answer=True, use_tokenizer=True),
                dict(type="TextRepeatFilter", check_question=True, check_answer=True),
            ]),
            priority=0,
        ),
        "rewriters": dict(
            cfg=dict(type="SubPipeline", operators=[
                dict(type="RemoveThinkRewriter"),
                dict(type="NormImageTagRewriter"),
            ]),
            priority=1,
        ),
    }
)
```

### Config with MLLM Operators

When using MLLM operators, configure the `model` field and deploy an OpenAI API-compatible inference service:

```python
# configs/my_mllm_pipeline.py
_base_ = [
    "@/_base_/models/local_api_model.py",
    "@/_base_/dataset.py",
    "@/_base_/filters/filter_rule_base_for_question.py",
]

work_dir = "./work_dirs/mllm_run"
logger = dict(type="Logger", log_file="logs/mllm_run.log")

dataset_yaml = "/path/to/dataset.yaml"

dataloader = dict(
    dataset=dataset_yaml,
    batch_size=5000,
    use_image=True,
    cache_dir="~/cache/images_lmdb",    # LMDB image cache directory
    resize_image_size=2048,             # Max image edge length
)

datasaver = dict(
    dataset=dataset_yaml,
    output_dir="./output",
    save_yaml_name="mllm_processed",
)

model = dict(
    model="Qwen3-VL-30B-A3B-Instruct",
    api_base="http://127.0.0.1",
    port=8000,
    thread_num=512,                     # Concurrent request count
    return_dict=True,                   # JSON output mode
)

# Request builder for MLLM filter
filter_request = dict(
    type="RequestBuilder",
    prompt="prompts/filter/question_image_consist_v2.txt",    # Prompt file
    key_templates={"result": "q{idx}", "reason": "q{idx}_reason"},  # Response parsing template
    with_image=True,
    with_question=True,
)

# Add MLLM filter after rule filters (inherited from _base_)
mllm_filter = dict(
    cfg=dict(type="SubPipeline", operators=[
        dict(type="MLLMFilter", request_builder=filter_request),
    ]),
    priority=10,
)

pipeline = dict(
    type="Pipeline",
    operations={
        "filter_rule_base_for_question": filter_rule_base_for_question,  # from _base_
        "mllm_filter": mllm_filter,
    }
)
```

---

## 7. Running the Pipeline

### Basic Run

```bash
python run.py -c configs/my_pipeline.py
```

### Cache Images Only

For the first run with image data, you can optionally pre-cache images to LMDB for faster subsequent processing:

```bash
python run.py -c configs/my_pipeline.py --cache-images
```

> **Note:** Pre-caching is not mandatory. If `cache_dir` is configured, images will be cached on-the-fly during the pipeline run. The `--cache-images` mode is useful when you want to populate the cache before the actual processing starts.

### Checkpoint Resume

DataStudio automatically saves progress via checkpoints. If a run is interrupted, simply re-run the same command to resume from the last position:

```bash
# Re-run after interruption — automatically resumes
python run.py -c configs/my_pipeline.py
```

### Output Structure

After processing, the output directory structure is:

```
output/
├── dataset_a/                    # Organized by source
│   └── data.jsonl                # Kept data
├── dataset_b/
│   ├── data.jsonl
│   └── rejected/                 # Filtered-out data
│       └── data.jsonl
├── mllm_processed.yaml           # YAML index for kept data
└── mllm_processed_rejected.yaml  # YAML index for filtered data
```

The output YAML can be used directly as input configuration for downstream tasks.

---

## 8. Model Backend

DataStudio uses `OpenAIAPI` to call any service compatible with the OpenAI Chat Completions API:

```python
model = dict(
    type="OpenAIAPI",
    model="Qwen3-VL-30B-A3B-Instruct",   # Model name
    api_base="http://127.0.0.1",        # API address
    port=8000,                          # Port
    key="sk-123456",                    # API key
    thread_num=512,                     # Max concurrency
    return_dict=True,                   # True: JSON output; False: plain text output
    max_tokens=16384,                   # Max output tokens
    temperature=0.6,
    retry=10,                           # Retry count on failure
    timeout=(30, 1800),                 # (connect timeout, read timeout) in seconds
)
```

For CPU-intensive scenarios (encoding many high-resolution images), use `MPOpenAIAPI` for multi-process parallel processing:

```python
model = dict(
    type="MPOpenAIAPI",
    model="Qwen3-VL-30B-A3B-Instruct",
    thread_num=512,
    num_workers=4,          # Number of worker processes
)
```

`MPOpenAIAPI` creates multiple worker processes via fork, using Copy-on-Write to share pre-encoded message data with near-single-process memory overhead.

---

## 9. MLLM Operators

### RequestBuilder

`RequestBuilder` controls how data is assembled into model requests and how responses are parsed:

```python
request_builder = dict(
    type="RequestBuilder",
    prompt="prompts/filter/question_image_consist_v2.txt",   # Prompt: file path or inline text
    system_prompt=None,                    # System prompt (optional)
    key_templates={                        # Response field mapping (JSON mode)
        "result": "q{idx}",
        "reason": "q{idx}_reason",
    },
    with_image=True,                       # Include image in request
    with_question=True,                    # Include question text
    with_answer=False,                     # Include answer text
    with_original=False,                   # Include original answer (before rewriting)
)
```

**Prompt files** (`.txt`) can use `{question}`, `{answer}`, `{ori_answer}` placeholders, which are automatically substituted at runtime.

**`key_templates`**:

- When set to a `dict`, the model must return JSON (requires `return_dict=True`); `{idx}` is replaced with the QA pair index
- When set to `None`, the model returns plain text, used directly as the rewritten output

### MLLMFilter

The MLLM filter expects the model to return JSON in the format:

```json
{"q0": true, "q0_reason": "Answer is inconsistent with the image"}
```

`true` means reject, `false` means keep.

```python
dict(
    type="MLLMFilter",
    request_builder=dict(
        type="RequestBuilder",
        prompt="prompts/filter/question_image_consist_v2.txt",
        key_templates={"result": "q{idx}", "reason": "q{idx}_reason"},
        with_image=True,
        with_question=True,
    ),
)
```

### MLLMRewriter

The MLLM rewriter can work in JSON mode or plain text mode:

```python
# JSON mode: suitable for batch processing with structured output
dict(
    type="MLLMRewriter",
    request_builder=dict(
        type="RequestBuilder",
        prompt="Your rewrite prompt here or path to .txt file",
        key_templates={"answer": "q{idx}_answer"},
        with_image=True,
    ),
    rewrite_type="answer",      # "answer" or "question"
)

# Plain text mode: model output used directly as new answer
dict(
    type="MLLMRewriter",
    request_builder=dict(
        type="RequestBuilder",
        prompt=None,                # No template, send original question directly
        key_templates=None,         # Plain text response
        with_image=True,
    ),
    rewrite_type="answer",
)
```

### SelectiveMLLMRewriter

Only rewrites QA pairs that meet certain conditions, reducing unnecessary API calls:

```python
dict(
    type="SelectiveMLLMRewriter",
    request_builder=rewriter_request,
    should_rewrite_fn="lambda q, a: len(a) < 100",   # Only rewrite short answers
)
```

---

## 10. Custom Operators

### Custom Filter

Inherit the `Filter` class and implement the `check` method:

```python
from datastudio.operators import Filter, DataItem
from datastudio.utils.registry import OPERATORS
from typing import Tuple

@OPERATORS.register_module()
class AnswerLengthFilter(Filter):
    """Filter by answer length."""

    def __init__(self, min_length: int = 10, max_length: int = 10000, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
        answer = item.get_answer(qa_idx)
        length = len(answer)

        if length < self.min_length:
            return True, f"Answer too short: {length} < {self.min_length}"
        if length > self.max_length:
            return True, f"Answer too long: {length} > {self.max_length}"
        return False, ""
```

Return values of the `check` method:
- `(True, reason)` — reject this QA pair
- `(False, "")` — keep this QA pair

### Custom Rewriter

Inherit the `Rewriter` class and implement the `rewrite` method:

```python
from datastudio.operators import Rewriter, DataItem
from datastudio.utils.registry import OPERATORS
from typing import Optional

@OPERATORS.register_module()
class CleanWhitespaceRewriter(Rewriter):
    """Clean extra whitespace in answers."""

    def rewrite(self, item: DataItem, qa_idx: int) -> Optional[str]:
        answer = item.get_answer(qa_idx)
        cleaned = " ".join(answer.split())
        return cleaned if cleaned != answer else None
```

Return values of the `rewrite` method:
- Return a new answer string — replaces the original answer (original is automatically saved to `ori_answer`)
- Return `None` — no modification

Then use in config: `dict(type="AnswerLengthFilter", min_length=10)`

---

## 11. Performance Tuning

### LMDB Image Cache

When processing data with images, enable LMDB caching to avoid repeated image reading and decoding:

```python
dataloader = dict(
    cache_dir="~/cache/images_lmdb",    # Cache directory
    use_lmdb_cache=True,                # Enabled by default
    resize_image_size=2048,             # Max edge length when caching
)
```

Pre-cache images on first run (optional — images are also cached on-the-fly during pipeline execution):

```bash
python run.py -c my_config.py --cache-images
```

### Concurrency Tuning

`thread_num` controls the maximum number of concurrent requests to the inference service:

```python
model = dict(
    thread_num=512,     # Adjust based on inference service capacity
)
```

Reference values:
- Single A100: 256–1024
- Multi-GPU / large cluster: 2048–8192

### Batch Size

`batch_size` controls the number of entries loaded per batch. Larger batches improve throughput for MLLM operators:

```python
dataloader = dict(
    batch_size=5000,        # Recommended range: 1000–50000
)
```

---

## 12. Full Example

Here is a complete config with rule filtering + MLLM filtering + MLLM rewriting + consistency check + normalization. This mirrors the actual `configs/examples/honeypipe.py`.

> For more ready-to-run examples, see the **[Examples Guide](examples.md)** with 5 configs and built-in demo data.

```python
# configs/honeypipe.py
_base_ = [
    "@/_base_/models/local_api_model.py",
    "@/_base_/dataset.py",
]

work_dir = "./work_dirs/honeypipe"
logger = dict(type="Logger", log_file="logs/honeypipe.log")

dataset_yaml = "/path/to/dataset.yaml"

dataloader = dict(
    dataset=dataset_yaml,
    batch_size=5000,
    use_image=True,
    cache_dir="./cache/images_lmdb",
    resize_image_size=2048,
)

datasaver = dict(
    dataset=dataset_yaml,
    output_dir="./output/full",
    save_yaml_name="full_processed",
)

model = dict(
    model="Qwen3-VL-30B-A3B-Instruct",
    api_base="http://127.0.0.1",
    port=8000,
    thread_num=1024,
    return_dict=True,
    max_tokens=4096,
)

# --- Stage 1: Rule Filtering (priority=0, runs first) ---
rule_filters = dict(
    cfg=dict(type="SubPipeline", operators=[
        dict(type="ConvLengthFilter", min_length=1, max_length=20),
        dict(type="ImageSizeFilter", min_size=14),
        dict(type="ImageAspectRatioFilter", max_aspect_ratio=200),
        dict(type="ImageExtFilter"),
        dict(type="LengthAnomalyFilter", min_length=2, max_length=4096,
             check_question=True, check_answer=True, use_tokenizer=True),
        dict(type="TextRepeatFilter", check_question=True, check_answer=True),
    ]),
    priority=0,
)

# --- Stage 2: MLLM Filtering (priority=10) ---
mllm_filter = dict(
    cfg=dict(type="SubPipeline", operators=[
        dict(type="MLLMFilter", request_builder=dict(
            type="RequestBuilder",
            prompt="prompts/filter/question_image_consist_v2.txt",
            key_templates={"result": "q{idx}", "reason": "q{idx}_reason"},
            with_image=True,
            with_question=True,
        )),
    ]),
    priority=10,
)

# --- Stage 3: MLLM Rewriting (priority=20) ---
# Rewriter uses plain text mode (return_dict=False), so we override the model per-operator
mllm_rewriter_model = dict(
    type="OpenAIAPI",
    model="Qwen3-VL-30B-A3B-Instruct",
    api_base="http://127.0.0.1",
    port=8000,
    thread_num=1024,
    return_dict=False,   # Plain text output for rewriting
    max_tokens=4096,
)

mllm_rewriter = dict(
    cfg=dict(type="SubPipeline", operators=[
        dict(type="MLLMRewriter",
             model=mllm_rewriter_model,
             request_builder=dict(
                 type="RequestBuilder",
                 prompt=None,
                 key_templates=None,
                 with_image=True,
                 with_question=True,
             ),
             rewrite_type="answer",
        ),
    ]),
    priority=20,
)

# --- Stage 4: Consistency Check (priority=30) ---
# Compare old vs new answers after rewriting
consist_filter = dict(
    cfg=dict(type="SubPipeline", operators=[
        dict(type="MLLMFilter", request_builder=dict(
            type="RequestBuilder",
            prompt="prompts/filter/compare_answer_v2.txt",
            key_templates={"result": "q{idx}", "reason": "q{idx}_reason"},
            with_image=True,
            with_answer=True,
            with_original=True,
        )),
    ]),
    priority=30,
)

# --- Stage 5: Normalization (priority=40, runs last) ---
norm_rewriters = dict(
    cfg=dict(type="SubPipeline", operators=[
        dict(type="RemoveThinkRewriter"),
        dict(type="NormImageTagRewriter"),
    ]),
    priority=40,
)

pipeline = dict(
    type="Pipeline",
    operations={
        "rule_filters": rule_filters,
        "mllm_filter": mllm_filter,
        "mllm_rewriter": mllm_rewriter,
        "consist_filter": consist_filter,
        "norm_rewriters": norm_rewriters,
    }
)
```

Run:

```bash
python run.py -c configs/honeypipe.py
```

---

## 13. Deploying Inference Services

DataStudio is compatible with any OpenAI Chat Completions API-compatible service. We recommend [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang):

```bash
# vLLM example
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --port 8000 \
    --tensor-parallel-size 4

# SGLang example
python -m sglang.launch_server \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --port 8000 \
    --tp 4
```

---

## 14. Auxiliary Tools

DataStudio ships with two tools under the `tools/` directory:

### LLMRouter

An OpenAI API-compatible reverse proxy router for managing multiple LLM backends. Features include:

- **Load Balancing**: Weighted random, least-connections (P2C), least-waiting (P2C + Prometheus)
- **Rate Limiting**: Sliding-window RPM control per backend
- **Auto Failover**: Async health checks with automatic unhealthy backend removal
- **Hot Reload**: Watch YAML config changes and update backends incrementally


```bash
cd tools/LLMRouter
go build -o llmrouter ./cmd
./llmrouter -config config.yaml
```

### DataVis

A multimodal data visualization and analysis platform (React + FastAPI):

- Browse and search processed data with pagination
- Filter by source, language, image presence, conversation turns, rejection status
- Statistical analysis: token distribution, turn count distribution, image count, message length
- Supports JSONL / JSON / YAML data files

```bash
cd tools/DataVis
bash start.sh
```
