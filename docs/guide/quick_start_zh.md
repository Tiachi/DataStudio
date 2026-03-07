# 快速开始

本文档帮助你从零开始使用 DataStudio 处理多模态数据。

## 1. 安装

### 环境要求

- Python >= 3.10
- 足够的磁盘空间（LMDB 图像缓存需要额外存储）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/Open-Bee/DataStudio.git
cd DataStudio

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate

# 安装
pip install -e .

# 验证
python -c "import datastudio; print(datastudio.__version__)"
```

### 可选依赖

```bash
# Weights & Biases 监控
pip install wandb
wandb login
```

### 快速体验

安装完成后，可以直接运行内置示例验证环境（无需额外数据）：

```bash
# 规则过滤示例（纯 CPU，无需 MLLM）
python run.py -c configs/examples/rule_filter_only.py

# 文本规范化示例（移除 think 标签等）
python run.py -c configs/examples/text_normalization.py
```

更多示例请参考 **[示例指南](examples_zh.md)**，包含 5 个覆盖不同场景的完整配置。

---

## 2. 数据格式

DataStudio 使用标准的多模态对话格式，支持 JSON 和 JSONL 文件。

### 基本格式

```json
{
  "id": "sample_001",
  "image": "/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\n这张图片里有什么？"},
    {"from": "gpt", "value": "图片展示了一个美丽的日落。"}
  ]
}
```

字段说明：

| 字段 | 必需 | 说明 |
|------|------|------|
| `id` | 否 | 样本唯一标识 |
| `image` | 否 | 图像路径，单图为字符串，多图为列表 |
| `conversations` | 是 | 对话列表，`from` 为 `"human"` 或 `"gpt"`，`value` 为文本内容 |

### 多图格式

```json
{
  "id": "multi_img_001",
  "image": ["/path/to/img1.jpg", "/path/to/img2.jpg"],
  "conversations": [
    {"from": "human", "value": "<image>\n<image>\n比较这两张图片。"},
    {"from": "gpt", "value": "第一张是白天，第二张是夜晚。"}
  ]
}
```

### Schema 兼容

DataStudio 同时支持 `conversations` 格式和 `messages` 格式（OpenAI 风格），加载时自动转换为内部标准格式。

---

## 3. 数据集 YAML 配置

数据集通过 YAML 文件描述，一个 YAML 可包含多个数据源：

```yaml
# dataset.yaml
datasets:
  - json_path: /data/dataset_a.jsonl
    source: dataset_a

  - json_path: /data/dataset_b.json
    source: dataset_b
```

字段说明：

| 字段 | 说明 |
|------|------|
| `json_path` / `file_path` / `jsonl_path` | 数据文件路径（三者等价） |
| `source` | 数据源名称，用于输出时分目录保存 |

---

## 4. 核心概念

### 算子类型

DataStudio 提供两类算子，每类都有规则版本和 MLLM 版本：

| 类型 | 作用 | 核心方法 |
|------|------|----------|
| **Filter** | 判断是否保留样本 | `check(item, qa_idx) -> (rejected, reason)` |
| **Rewriter** | 修改内容（不删除样本） | `rewrite(item, qa_idx) -> new_answer` |
| **MLLMFilter** | 调用 MLLM 进行质量过滤 | 通过 `RequestBuilder` 构建请求 |
| **MLLMRewriter** | 调用 MLLM 进行内容重写 | 通过 `RequestBuilder` 构建请求 |

### 流水线结构

`Pipeline` 由多个 `SubPipeline` 组成，按 `priority`（数值越小越先执行）依次执行。每个 `SubPipeline` 内的算子顺序执行，被前序算子拒绝的样本不会传入后续算子。

```
Pipeline
├── SubPipeline (priority=0)  ← 先执行
│   ├── ConvLengthFilter
│   └── ImageSizeFilter
├── SubPipeline (priority=1)
│   └── MLLMFilter
└── SubPipeline (priority=2)  ← 后执行
    └── MLLMRewriter
```

### 配置继承

配置文件通过 `_base_` 字段实现继承，`@/` 前缀表示从 `configs/` 目录解析：

```python
_base_ = [
    "@/_base_/models/local_api_model.py",
    "@/_base_/dataset.py",
    "@/_base_/filters/filter_rule_base_for_question.py",
]
```

- `_base_` 列表中的配置文件按顺序合并
- 子配置中的同名字段会覆盖基础配置
- `dict` 类型的字段会递归合并

预置的基础配置：

| 文件 | 说明 |
|------|------|
| `@/_base_/dataset.py` | DataLoader 和 DataSaver 默认参数 |
| `@/_base_/models/local_api_model.py` | 本地 API 模型默认参数 |
| `@/_base_/filters/filter_rule_base_for_question.py` | 常用问题过滤规则组合（priority=0） |
| `@/_base_/filters/filter_rule_base_for_answer.py` | 常用答案过滤规则组合（priority=50） |
| `@/_base_/rewriters/rewriter_rule_base.py` | 重写器基础模板（priority=1） |

---

## 5. 内置算子

### 过滤器

| 算子 | 说明 | 关键参数 |
|------|------|----------|
| `ConvLengthFilter` | 按对话轮数过滤 | `min_length`, `max_length` |
| `ImageSizeFilter` | 按图像尺寸过滤 | `min_size`, `max_ratio` |
| `ImageAspectRatioFilter` | 按宽高比过滤 | `max_aspect_ratio` |
| `ImageExtFilter` | 按图像格式过滤 | — |
| `LengthAnomalyFilter` | 文本长度异常检测 | `min_length`, `max_length`, `use_tokenizer` |
| `ResponseTagFilter` | 按响应标签过滤 | — |
| `TextRepeatFilter` | 文本重复检测 | `check_question`, `check_answer` |
| `MLLMFilter` | MLLM 质量过滤 | `request_builder` |

### 重写器

| 算子 | 说明 |
|------|------|
| `RemoveThinkRewriter` | 移除 `<think>...</think>` 标签 |
| `NormThinkRewriter` | 标准化 think 标签格式 |
| `AddNoThinkRewriter` | 添加 `/no_think` 前缀 |
| `NormImageTagRewriter` | 标准化 `<image>` 标签位置 |
| `NormPromptRewriter` | 标准化提示词格式 |
| `NormMultiTurnPromptRewriter` | 标准化多轮对话提示词 |
| `RemoveAnswerRewriter` | 移除答案内容 |
| `RemoveReasonRewriter` | 移除推理前缀 |
| `SplitRewriter` | 多轮对话拆分为单轮 |
| `MLLMRewriter` | MLLM 内容重写 |
| `SelectiveMLLMRewriter` | 条件触发的 MLLM 重写 |

---

## 6. 配置文件详解

### 最小配置示例

以下示例展示了一个仅使用规则算子的流水线：

```python
# configs/my_first_pipeline.py
_base_ = ["@/_base_/dataset.py"]

work_dir = "./work_dirs/my_first_run"
logger = dict(type="Logger", log_file="logs/run.log")

dataset_yaml = "/path/to/dataset.yaml"

dataloader = dict(
    dataset=dataset_yaml,
    batch_size=1000,
    use_image=False,             # 纯文本处理，不加载图像
)

datasaver = dict(
    dataset=dataset_yaml,
    output_dir="./output",
    save_yaml_name="my_output",  # 输出 YAML 文件名前缀
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

### 包含 MLLM 算子的配置

使用 MLLM 算子时，需要配置 `model` 字段并部署一个 OpenAI API 兼容的推理服务：

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
    cache_dir="~/cache/images_lmdb",    # LMDB 图像缓存目录
    resize_image_size=2048,             # 图像最大边长
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
    thread_num=512,                     # 并发请求数
    return_dict=True,                   # JSON 输出模式
)

# MLLM 过滤器的请求构建器
filter_request = dict(
    type="RequestBuilder",
    prompt="prompts/filter/question_image_consist_v2.txt",    # 提示词文件
    key_templates={"result": "q{idx}", "reason": "q{idx}_reason"},  # 响应解析模板
    with_image=True,
    with_question=True,
)

# 在已有的规则过滤器（继承自 _base_）之后添加 MLLM 过滤器
mllm_filter = dict(
    cfg=dict(type="SubPipeline", operators=[
        dict(type="MLLMFilter", request_builder=filter_request),
    ]),
    priority=10,
)

pipeline = dict(
    type="Pipeline",
    operations={
        "filter_rule_base_for_question": filter_rule_base_for_question,  # 来自 _base_
        "mllm_filter": mllm_filter,
    }
)
```

---

## 7. 运行流水线

### 基本运行

```bash
python run.py -c configs/my_pipeline.py
```

### 仅缓存图像

首次处理包含图像的数据时，可以选择先运行缓存模式将图像写入 LMDB，后续处理直接读取缓存：

```bash
python run.py -c configs/my_pipeline.py --cache-images
```

> **注意：** 预缓存不是必须的。只要配置了 `cache_dir`，图像会在 pipeline 运行过程中自动边跑边缓存。`--cache-images` 模式适用于希望在正式处理前先完成缓存的场景。

### 断点续传

DataStudio 基于 checkpoint 自动保存处理进度。如果运行中断，直接重新运行相同命令即可从上次位置恢复：

```bash
# 中断后重新运行，自动恢复
python run.py -c configs/my_pipeline.py
```

### 输出结构

处理完成后，输出目录结构如下：

```
output/
├── dataset_a/                    # 按 source 分目录
│   └── data.jsonl                # 保留的数据
├── dataset_b/
│   ├── data.jsonl
│   └── rejected/                 # 被过滤的数据
│       └── data.jsonl
├── mllm_processed.yaml           # 保留数据的 YAML 索引
└── mllm_processed_rejected.yaml  # 过滤数据的 YAML 索引
```

输出 YAML 可直接作为下游任务的输入配置。

---

## 8. 模型后端

DataStudio 通过 `OpenAIAPI` 调用任何兼容 OpenAI Chat Completions API 的服务：

```python
model = dict(
    type="OpenAIAPI",
    model="Qwen3-VL-30B-A3B-Instruct",   # 模型名称
    api_base="http://127.0.0.1",        # API 地址
    port=8000,                          # 端口
    key="sk-123456",                    # API 密钥
    thread_num=512,                     # 最大并发数
    return_dict=True,                   # True: JSON 输出; False: 纯文本输出
    max_tokens=16384,                   # 最大输出 token 数
    temperature=0.6,
    retry=10,                           # 失败重试次数
    timeout=(30, 1800),                 # (连接超时, 读取超时) 秒
)
```

对于 CPU 密集型场景（大量高分辨率图像编码），可使用 `MPOpenAIAPI` 通过多进程并行处理：

```python
model = dict(
    type="MPOpenAIAPI",
    model="Qwen3-VL-30B-A3B-Instruct",
    thread_num=512,
    num_workers=4,          # 进程数
)
```

`MPOpenAIAPI` 通过 fork 创建多个工作进程，利用 Copy-on-Write 共享预编码的消息数据，内存开销接近单进程。

---

## 9. MLLM 算子详解

### RequestBuilder

`RequestBuilder` 控制如何将数据组装为模型请求、如何解析模型响应：

```python
request_builder = dict(
    type="RequestBuilder",
    prompt="prompts/filter/question_image_consist_v2.txt",   # 提示词：文件路径或直接文本
    system_prompt=None,                    # 系统提示词（可选）
    key_templates={                        # 响应字段映射（JSON 模式）
        "result": "q{idx}",
        "reason": "q{idx}_reason",
    },
    with_image=True,                       # 是否包含图像
    with_question=True,                    # 是否包含问题文本
    with_answer=False,                     # 是否包含答案文本
    with_original=False,                   # 是否包含原始答案（重写前）
)
```

**提示词文件**（`.txt`）中可以使用 `{question}`、`{answer}`、`{ori_answer}` 占位符，运行时自动替换。

**`key_templates`**：

- 设置为 `dict` 时，模型需要返回 JSON（需配合 `return_dict=True`），`{idx}` 会被替换为 QA 对索引
- 设置为 `None` 时，模型返回纯文本，直接作为重写结果

### MLLMFilter

MLLM 过滤器期望模型返回 JSON，格式为：

```json
{"q0": true, "q0_reason": "答案与图像不一致"}
```

`true` 表示拒绝，`false` 表示保留。

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

MLLM 重写器可以工作在 JSON 模式或纯文本模式：

```python
# JSON 模式：适合批量处理，配合结构化输出
dict(
    type="MLLMRewriter",
    request_builder=dict(
        type="RequestBuilder",
        prompt="你的重写提示词，或 .txt 文件路径",
        key_templates={"answer": "q{idx}_answer"},
        with_image=True,
    ),
    rewrite_type="answer",      # "answer" 或 "question"
)

# 纯文本模式：模型输出直接作为新答案
dict(
    type="MLLMRewriter",
    request_builder=dict(
        type="RequestBuilder",
        prompt=None,                # 不使用模板，直接发送原始问题
        key_templates=None,         # 纯文本响应
        with_image=True,
    ),
    rewrite_type="answer",
)
```

### SelectiveMLLMRewriter

仅对满足条件的 QA 对执行重写，减少不必要的 API 调用：

```python
dict(
    type="SelectiveMLLMRewriter",
    request_builder=rewriter_request,
    should_rewrite_fn="lambda q, a: len(a) < 100",   # 仅重写短答案
)
```

---

## 10. 自定义算子

### 自定义过滤器

继承 `Filter` 类，实现 `check` 方法：

```python
from datastudio.operators import Filter, DataItem
from datastudio.utils.registry import OPERATORS
from typing import Tuple

@OPERATORS.register_module()
class AnswerLengthFilter(Filter):
    """按答案长度过滤。"""

    def __init__(self, min_length: int = 10, max_length: int = 10000, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
        answer = item.get_answer(qa_idx)
        length = len(answer)

        if length < self.min_length:
            return True, f"答案过短: {length} < {self.min_length}"
        if length > self.max_length:
            return True, f"答案过长: {length} > {self.max_length}"
        return False, ""
```

`check` 方法的返回值：
- `(True, reason)` — 拒绝该 QA 对
- `(False, "")` — 保留该 QA 对

### 自定义重写器

继承 `Rewriter` 类，实现 `rewrite` 方法：

```python
from datastudio.operators import Rewriter, DataItem
from datastudio.utils.registry import OPERATORS
from typing import Optional

@OPERATORS.register_module()
class CleanWhitespaceRewriter(Rewriter):
    """清理答案中的多余空白字符。"""

    def rewrite(self, item: DataItem, qa_idx: int) -> Optional[str]:
        answer = item.get_answer(qa_idx)
        cleaned = " ".join(answer.split())
        return cleaned if cleaned != answer else None
```

`rewrite` 方法的返回值：
- 返回新答案字符串 — 替换原答案（原答案自动保存到 `ori_answer`）
- 返回 `None` — 不修改

然后在配置文件中使用：`dict(type="AnswerLengthFilter", min_length=10)`

---

## 11. 性能调优

### LMDB 图像缓存

处理带图像的数据时，启用 LMDB 缓存可以避免重复读取和解码图像：

```python
dataloader = dict(
    cache_dir="~/cache/images_lmdb",    # 缓存目录
    use_lmdb_cache=True,                # 默认开启
    resize_image_size=2048,             # 缓存时的最大边长
)
```

首次运行可先缓存图像（非必须，pipeline 运行时也会自动边跑边缓存）：

```bash
python run.py -c my_config.py --cache-images
```

### 并发调优

`thread_num` 控制同时发往推理服务的最大请求数：

```python
model = dict(
    thread_num=512,     # 根据推理服务的承载能力调整
)
```

参考值：
- 单卡 A100：256-1024
- 多卡 / 大集群：2048-8192

### 批量大小

`batch_size` 控制每批读取的数据量。较大的批次可以提高 MLLM 算子的吞吐效率：

```python
dataloader = dict(
    batch_size=5000,        # 推荐范围：1000-50000
)
```

---

## 12. 完整示例

以下是一个包含规则过滤 + MLLM 过滤 + MLLM 重写 + 一致性检查 + 格式规范化的完整配置，与实际的 `configs/examples/honeypipe.py` 保持一致。

> 更多可直接运行的示例请参考 **[示例指南](examples_zh.md)**，包含 5 个配置和内置演示数据。

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

# --- 阶段 1：规则过滤（priority=0，最先执行）---
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

# --- 阶段 2：MLLM 过滤（priority=10）---
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

# --- 阶段 3：MLLM 重写（priority=20）---
# 重写器使用纯文本模式（return_dict=False），需要独立覆盖 model
mllm_rewriter_model = dict(
    type="OpenAIAPI",
    model="Qwen3-VL-30B-A3B-Instruct",
    api_base="http://127.0.0.1",
    port=8000,
    thread_num=1024,
    return_dict=False,   # 纯文本输出，用于重写
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

# --- 阶段 4：一致性检查（priority=30）---
# 重写后比较新旧答案，过滤不一致的样本
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

# --- 阶段 5：格式规范化（priority=40，最后执行）---
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

运行：

```bash
python run.py -c configs/honeypipe.py
```

---

## 13. 部署推理服务

DataStudio 兼容任何 OpenAI Chat Completions API 格式的服务。推荐使用 [vLLM](https://github.com/vllm-project/vllm) 或 [SGLang](https://github.com/sgl-project/sglang) 部署：

```bash
# vLLM 示例
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --port 8000 \
    --tensor-parallel-size 4

# SGLang 示例
python -m sglang.launch_server \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --port 8000 \
    --tp 4
```

---

## 14. 辅助工具

DataStudio 在 `tools/` 目录下提供了两个辅助工具：

### LLMRouter

OpenAI API 兼容的反向代理路由器，用于管理多个 LLM 后端。核心特性：

- **负载均衡**：支持加权随机、最少连接数（P2C）、最少等待（P2C + Prometheus）
- **RPM 限流**：滑动窗口计数，按后端独立控制
- **自动故障转移**：异步健康检查，连续失败自动标记不健康
- **配置热加载**：监听 YAML 配置变更，增量更新后端


```bash
cd tools/LLMRouter
go build -o llmrouter ./cmd
./llmrouter -config config.yaml
```

### DataVis

多模态数据可视化与分析平台（React + FastAPI）：

- 分页浏览和搜索处理后的数据
- 按来源、语言、是否含图片、对话轮次、是否被拒绝等多维度过滤
- 统计分析：token 分布、对话轮次分布、图片数量分布、消息长度分布
- 支持 JSONL / JSON / YAML 格式数据文件

```bash
cd tools/DataVis
bash start.sh
```
