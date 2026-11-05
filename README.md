<p align="center">
  <img src="https://img.shields.io/badge/🐝-DataStudio-FFD700?style=for-the-badge&labelColor=000000" alt="DataStudio Logo" height="60"/>
</p>

<p align="center">
  <strong>Config-driven multimodal data processing pipeline for MLLMs</strong>
</p>

---

<p align="center">
  <a href="https://arxiv.org/abs/2510.13795"><img src="https://img.shields.io/badge/ICLR%202026-arXiv:2510.13795-B31B1B?logo=arxiv&logoColor=white" alt="Paper"/></a>
  <a href="https://open-bee.github.io/"><img src="https://img.shields.io/badge/Project-Open--Bee-4285F4?logo=googlechrome&logoColor=white" alt="Project Page"/></a>
  <a href="https://huggingface.co/Open-Bee"><img src="https://img.shields.io/badge/🤗-Models%20%26%20Datasets-FFD21E" alt="HuggingFace"/></a>
  <br/>
  <a href="https://github.com/Open-Bee/DataStudio"><img src="https://img.shields.io/github/stars/Open-Bee/DataStudio?style=social" alt="GitHub Stars"/></a>
  <a href="https://github.com/Open-Bee/DataStudio/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"/></a>
</p>

<p align="center">
  <a href="docs/guide/readme_zh.md">简体中文</a> | <a href="https://open-bee.github.io/DataStudio/">Documentation</a>
</p>

---

## Introduction

**DataStudio** is a data processing toolkit for Multimodal Large Language Models (MLLMs). It supports both rule-based and MLLM-powered operators for filtering and rewriting data. DataStudio drives the HoneyPipe data cleaning pipeline that produced [Honey-Data-15M](https://huggingface.co/datasets/Open-Bee/Honey-Data-15M) — a high-quality dataset of 15 million QA pairs.

### Key Features

- **Config-Driven**: Define complete processing pipelines via Python config files with config inheritance (`_base_`)
- **Dual-Engine Operators**: 16 built-in rule operators (7 filters + 9 rewriters), plus MLLM-powered operators for intelligent filtering and rewriting
- **High Performance**: Multi-process async concurrent API requests (8192+ via `MPOpenAIAPI`), LMDB sharded image caching, fork + COW memory sharing
- **Checkpoint Resume**: Automatic progress saving with checkpoint-based recovery after interruption

### Auxiliary Tools

- **[LLMRouter](https://github.com/Open-Bee/LLMRouter/)**: A high-performance reverse proxy router built in Go, OpenAI API-compatible, supporting multiple LLM backends with weighted load balancing, sliding-window RPM throttling, async health checks with auto failover, and config hot-reload. See its [README](tools/LLMRouter/README.md) for details.
- **[DataVis](https://github.com/Open-Bee/DataVis/)**: A multimodal data visualization and analysis platform (React + FastAPI) for browsing, filtering, and analyzing processed datasets.

---

## Installation

```bash
git clone https://github.com/Open-Bee/DataStudio.git
cd DataStudio
pip install -e .
```

Requirements: Python >= 3.10.

---

## Quick Start

### 1. Prepare a Dataset YAML

```yaml
# dataset.yaml
datasets:
  - json_path: /path/to/data.jsonl
    source: my_dataset
```

### 2. Write a Config File

```python
# configs/my_pipeline.py
_base_ = ["@/_base_/dataset.py"]

work_dir = "./work_dirs/my_experiment"
logger = dict(type="Logger", log_file="logs/process.log")
dataset_yaml = "/path/to/dataset.yaml"

dataloader = dict(dataset=dataset_yaml, batch_size=1000, use_image=False)
datasaver = dict(dataset=dataset_yaml, output_dir="./output", save_yaml_name="processed")

pipeline = dict(
    type="Pipeline",
    operations={
        "rule_filters": dict(
            cfg=dict(type="SubPipeline", operators=[
                dict(type="ConvLengthFilter", min_length=1, max_length=20),
                dict(type="ImageSizeFilter", min_size=14),
            ]),
            priority=0,
        ),
        "rewriters": dict(
            cfg=dict(type="SubPipeline", operators=[
                dict(type="RemoveThinkRewriter"),
            ]),
            priority=1,
        ),
    }
)
```

### 3. Run

```bash
python run.py -c configs/my_pipeline.py
```

For a complete tutorial covering data formats, MLLM operators, custom operators, performance tuning, and more, see the **[Quick Start Guide](docs/guide/quick_start.md)**.

---

## Image CoT SFT Construction

The opt-in `run_image_cot.py` entry point adds audited short/long image-CoT
generation, strict verification, crash recovery, and gated exports. It supports
self-hosted OpenAI-compatible teachers and a DeepSeek API pilot profile, without
changing the default runner. See the [implementation and usage guide](docs/image_cot_implementation.md)
for configuration, local tests, and the separate runtime quality/training gates.

## Project Structure

```
DataStudio/
├── run.py                  # Main entry point (supports -c/--config, --cache-images)
├── datastudio/
│   ├── operators/          # Operators
│   │   ├── core/           # Base: Operator, Filter, Rewriter, DataItem, OperatorResult
│   │   ├── filters/        # 7 rule-based filters
│   │   ├── rewriters/      # 9 rule-based rewriters
│   │   └── mllm/           # MLLM operators: MLLMFilter, MLLMRewriter, SelectiveMLLMRewriter
│   ├── models/             # Model backends: OpenAIAPI, MPOpenAIAPI
│   ├── pipelines/          # Pipeline, SubPipeline
│   ├── datasets/           # Data loading/saving, format conversion, YAML generation
│   └── utils/              # Registry, config, logging, LMDB cache, checkpoint
├── configs/                # Configuration files
│   ├── _base_/             # Base configs (models, datasets, filters, rewriters)
│   └── examples/           # Ready-to-run example configurations with demo data
├── prompts/                # MLLM prompt templates
├── tools/                  # Auxiliary tools
│   ├── LLMRouter/          # OpenAI API-compatible reverse proxy with load balancing
│   └── DataVis/            # Multimodal data visualization and analysis platform
├── docs/                   # Documentation (Sphinx + Markdown guides)
└── tests/                  # Unit tests
```

---

## About the Bee Project

DataStudio is the data cleaning engine of the [Bee](https://open-bee.github.io/) project. Through the HoneyPipe pipeline, it produced the Honey-Data-15M dataset.

| Component | Description | Link |
|-----------|-------------|------|
| **Honey-Data-15M** | 15M high-quality QA pairs (with CoT) | [HuggingFace](https://huggingface.co/datasets/Open-Bee/Honey-Data-15M) |
| **Bee-8B** | Fully open-source 8B MLLM | [HuggingFace](https://huggingface.co/Open-Bee/Bee-8B) |
| **DataStudio** | Data cleaning pipeline | [GitHub](https://github.com/Open-Bee/DataStudio) |

---

## Citation

```bibtex
@article{zhang2025bee,
  title={Bee: A High-Quality Corpus and Full-Stack Suite to Unlock Advanced Fully Open MLLMs},
  author={Zhang, Yi and Ni, Bolin and Chen, Xin-Sheng and Zhang, Heng-Rui and Rao, Yongming and Peng, Houwen and Lu, Qinglin and Hu, Han and Guo, Meng-Hao and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2510.13795},
  year={2025}
}
```

## Contributing

If you encounter any bugs or have feature requests, feel free to open an [Issue](https://github.com/Open-Bee/DataStudio/issues) or submit a [Pull Request](https://github.com/Open-Bee/DataStudio/pulls). Contributions are welcome!

## License

[Apache License 2.0](LICENSE)
