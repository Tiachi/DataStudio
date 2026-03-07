<p align="center">
  <img src="https://img.shields.io/badge/🐝-DataStudio-FFD700?style=for-the-badge&labelColor=000000" alt="DataStudio Logo" height="60"/>
</p>

<p align="center">
  <strong>配置驱动的多模态数据处理流水线</strong>
</p>

---

<p align="center">
  <a href="https://arxiv.org/abs/2510.13795"><img src="https://img.shields.io/badge/ICLR%202026-arXiv:2510.13795-B31B1B?logo=arxiv&logoColor=white" alt="Paper"/></a>
  <a href="https://open-bee.github.io/"><img src="https://img.shields.io/badge/项目-Open--Bee-4285F4?logo=googlechrome&logoColor=white" alt="Project Page"/></a>
  <a href="https://huggingface.co/Open-Bee"><img src="https://img.shields.io/badge/🤗-模型%20%26%20数据集-FFD21E" alt="HuggingFace"/></a>
  <br/>
  <a href="https://github.com/Open-Bee/DataStudio"><img src="https://img.shields.io/github/stars/Open-Bee/DataStudio?style=social" alt="GitHub Stars"/></a>
  <a href="https://github.com/Open-Bee/DataStudio/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"/></a>
</p>

<p align="center">
  <a href="../../README.md">English</a> | <a href="https://open-bee.github.io/DataStudio/">文档</a>
</p>

---

## 简介

**DataStudio** 是一个面向多模态大语言模型（MLLM）的数据处理工具包，支持规则和 MLLM 两种算子对数据进行过滤与重写。它驱动了 HoneyPipe 数据清洗流水线，生产了 [Honey-Data-15M](https://huggingface.co/datasets/Open-Bee/Honey-Data-15M) —— 一个包含 1500 万 QA 对的高质量数据集。

### 核心特性

- **配置驱动**：通过 Python 配置文件定义完整的处理流水线，支持配置继承（`_base_`）
- **双引擎算子**：内置 16 个规则算子（7 个过滤器 + 9 个重写器），同时支持调用 MLLM 进行智能过滤和重写
- **高性能**：多进程异步并发 API 请求（通过 `MPOpenAIAPI` 支持 8192+）、LMDB 分片图像缓存、fork + COW 内存共享
- **断点续传**：基于 checkpoint 的进度保存，中断后自动恢复

### 辅助工具

- **[LLMRouter](https://github.com/Open-Bee/LLMRouter/)**：基于 Go 开发的高性能反向代理路由器，兼容 OpenAI API，支持多个 LLM 后端的加权负载均衡、滑动窗口 RPM 限流、异步健康检查与自动故障转移、配置热加载等。详见其 [README](../../tools/LLMRouter/README.md)。
- **[DataVis](https://github.com/Open-Bee/DataVis/)**：多模态数据可视化与分析平台（React + FastAPI），支持分页浏览、多维度过滤和统计分析。

---

## 安装

```bash
git clone https://github.com/Open-Bee/DataStudio.git
cd DataStudio
pip install -e .
```

环境要求：Python >= 3.10。

---

## 快速上手

### 1. 准备数据集 YAML

```yaml
# dataset.yaml
datasets:
  - json_path: /path/to/data.jsonl
    source: my_dataset
```

### 2. 编写配置文件

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

### 3. 运行

```bash
python run.py -c configs/my_pipeline.py
```

更完整的教程（数据格式、MLLM 算子、自定义算子、性能调优等）请参考 **[快速开始文档](quick_start_zh.md)**。

---

## 项目结构

```
DataStudio/
├── run.py                  # 主入口（支持 -c/--config, --cache-images）
├── datastudio/
│   ├── operators/          # 算子
│   │   ├── core/           # 基类：Operator, Filter, Rewriter, DataItem, OperatorResult
│   │   ├── filters/        # 7 个规则过滤器
│   │   ├── rewriters/      # 9 个规则重写器
│   │   └── mllm/           # MLLM 算子：MLLMFilter, MLLMRewriter, SelectiveMLLMRewriter
│   ├── models/             # 模型后端：OpenAIAPI, MPOpenAIAPI
│   ├── pipelines/          # Pipeline, SubPipeline
│   ├── datasets/           # 数据加载/保存、格式转换、YAML 生成
│   └── utils/              # 注册表、配置、日志、LMDB 缓存、checkpoint
├── configs/                # 配置文件
│   ├── _base_/             # 基础配置（模型、数据集、过滤器、重写器）
│   └── examples/           # 可直接运行的示例配置及演示数据
├── prompts/                # MLLM 提示词模板
├── tools/                  # 辅助工具
│   ├── LLMRouter/          # OpenAI API 兼容的反向代理路由器（负载均衡、故障转移）
│   └── DataVis/            # 多模态数据可视化与分析平台
├── docs/                   # 文档（Sphinx + Markdown 指南）
└── tests/                  # 单元测试
```

---

## 关于 Bee 项目

DataStudio 是 [Bee](https://open-bee.github.io/) 项目的数据清洗引擎，通过 HoneyPipe 流水线生产了 Honey-Data-15M 数据集。

| 组件 | 说明 | 链接 |
|------|------|------|
| **Honey-Data-15M** | 1500 万高质量 QA 对（含 CoT） | [HuggingFace](https://huggingface.co/datasets/Open-Bee/Honey-Data-15M) |
| **Bee-8B** | 全开源 8B MLLM | [HuggingFace](https://huggingface.co/Open-Bee/Bee-8B) |
| **DataStudio** | 数据清洗流水线 | [GitHub](https://github.com/Open-Bee/DataStudio) |

---

## 引用

```bibtex
@article{zhang2025bee,
  title={Bee: A High-Quality Corpus and Full-Stack Suite to Unlock Advanced Fully Open MLLMs},
  author={Zhang, Yi and Ni, Bolin and Chen, Xin-Sheng and Zhang, Heng-Rui and Rao, Yongming and Peng, Houwen and Lu, Qinglin and Hu, Han and Guo, Meng-Hao and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2510.13795},
  year={2025}
}
```

## 贡献

如果在使用过程中发现 Bug 或有功能建议，欢迎提 [Issue](https://github.com/Open-Bee/DataStudio/issues) 或 [Pull Request](https://github.com/Open-Bee/DataStudio/pulls)，期待你的贡献！

## 许可证

[Apache License 2.0](../../LICENSE)
