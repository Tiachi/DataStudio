# 示例指南

DataStudio 内置了 151 条多模态演示数据（含图片）和 5 个开箱即用的示例配置。安装完成后即可直接运行，无需额外下载数据。

> 演示数据位置：`configs/examples/demo_data/`
> 示例配置：`configs/examples/*.py`

---

## 演示数据

演示数据集包含 151 条多模态 QA 样本，覆盖多种数据源（Caption、VQA、OCR、Math、Chart 等）。每条样本包含：

- **对话格式**：`messages` 格式（OpenAI 风格）
- **图像引用**：共 265 张图片，位于 `demo_data/images/`
- **`<think>` 标签**：所有答案均包含 think 标签（适合测试文本规范化）
- **`ori_answer` 字段**：126 条样本包含重写前的原始答案

数据格式：
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

## 示例概览

| # | 配置文件 | 需要 MLLM | 说明 |
|---|---------|:---------:|------|
| 1 | `rule_filter_only.py` | 否 | 规则过滤（长度、重复检测、图像校验） |
| 2 | `mllm_quality_filter.py` | 是 | 规则过滤 + MLLM 问题-图像一致性检查 |
| 3 | `mllm_answer_rewrite.py` | 是 | MLLM 答案重写（纯文本模式） |
| 4 | `honeypipe.py` | 是 | 5 阶段端到端流水线（HoneyPipe） |
| 5 | `text_normalization.py` | 否 | 移除 think 标签、规范化 image 标签和提示词 |

**建议顺序**：先运行示例 1 或 5（无需 MLLM），再尝试示例 2–4（需要部署推理服务）。

---

## 无需 MLLM 的示例（直接运行）

### 示例 1：规则过滤

过滤对话轮数异常、文本长度异常、文本重复的样本。纯文本处理，不加载图像。

```bash
python run.py -c configs/examples/rule_filter_only.py
```

**继承的算子**（来自 `_base_`）：
- `ImageSizeFilter`、`ImageAspectRatioFilter`、`ImageExtFilter`（`use_image=False` 时不生效）
- `ConvLengthFilter`：保留 1–300 轮对话
- `LengthAnomalyFilter`：拒绝问题 < 2 或 > 4096 token 的样本；拒绝答案 < 1 或 > 4096 token 的样本
- `TextRepeatFilter`：检测问题和答案中的重复模式

**输出目录：** `./output/examples/rule_filter_only/`

---

### 示例 5：文本规范化

仅做格式清理，不过滤数据。演示数据的 151 条答案均包含 `<think>...</think>` 标签，适合测试此功能。

```bash
python run.py -c configs/examples/text_normalization.py
```

**包含的算子：**
- `RemoveThinkRewriter`：移除 `<think>...</think>` 标签及其内容
- `NormImageTagRewriter`：将 `<image>` 标签移至问题开头
- `NormPromptRewriter`：标准化提示词格式
- `RemoveReasonRewriter`：移除推理前缀内容

**输出目录：** `./output/examples/text_normalization/`

---

## 需要 MLLM 的示例（需部署推理服务）

以下示例需要一个 OpenAI 兼容的推理服务。请先部署：

```bash
# vLLM
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --port 8000 --tensor-parallel-size 4

# 或 SGLang
python -m sglang.launch_server \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --port 8000 --tp 4
```

如果你的配置不同（模型名、端口等），请相应修改配置文件中的 `model` 部分。

---

### 示例 2：MLLM 质量过滤

先执行规则过滤（继承自 `_base_`），再使用 MLLM 检查问题与图像的一致性。

```bash
python run.py -c configs/examples/mllm_quality_filter.py
```

**流水线：**
1. `priority=0` — 规则过滤（来自 `_base_/filters/filter_rule_base_for_question.py`）
2. `priority=10` — MLLM 问题-图像一致性过滤

**关键配置：**
```python
qi_consist_request = dict(
    type="RequestBuilder",
    prompt="prompts/filter/question_image_consist_v2.txt",
    key_templates={"result": "q{idx}", "reason": "q{idx}_reason"},
    with_image=True,
    with_question=True,
)
```

**输出目录：** `./output/examples/mllm_quality_filter/`

---

### 示例 3：MLLM 答案重写

将原始问题和图像发送给 MLLM，由模型生成新答案。使用纯文本模式（无 JSON 解析）。

```bash
python run.py -c configs/examples/mllm_answer_rewrite.py
```

**关键配置：**
```python
model = dict(
    return_dict=False,   # 纯文本输出 — 模型响应即为新答案
)

rewriter_request = dict(
    type="RequestBuilder",
    prompt=None,          # 不使用模板，直接发送原始问题
    key_templates=None,   # 纯文本模式
    with_image=True,
    with_question=True,
)
```

重写后，原始答案会保存在 `ori_answer` 字段中。

**输出目录：** `./output/examples/mllm_answer_rewrite/`

---

### 示例 4：HoneyPipe 完整流水线（5 阶段）

从原始数据到高质量训练数据的端到端处理，综合所有技术。所有配置均内联展开，方便查看每个可用选项。

```bash
python run.py -c configs/examples/honeypipe.py
```

**执行顺序：**

| 优先级 | 阶段 | 说明 |
|:------:|------|------|
| 0 | 规则过滤 | ConvLength、ImageSize、AspectRatio、ImageExt、LengthAnomaly、TextRepeat |
| 10 | MLLM 过滤 | 问题-图像一致性检查 |
| 20 | MLLM 重写 | 重新生成答案（纯文本模式） |
| 30 | 一致性检查 | 比较新旧答案 |
| 40 | 格式规范化 | 移除 think 标签、规范化 image 标签 |

**输出目录：** `./output/examples/honeypipe/`

---

## 自定义示例

将示例适配到你自己的数据：

1. **替换数据集**：将 `dataset_yaml` 指向你的 YAML 文件
2. **设置 `data_root`**：设为你的数据中图像相对路径的基目录
3. **调整模型配置**：修改 `model`、`api_base`、`port` 以匹配你的推理服务
4. **调优参数**：根据需要调整 `batch_size`、`thread_num`、过滤阈值等

```python
# 示例：使用自己的数据
dataset_yaml = "/path/to/your/dataset.yaml"

dataloader = dict(
    dataset=dataset_yaml,
    data_root="/path/to/your/data/root",   # 图像路径解析的基目录
    batch_size=5000,
    use_image=True,
)
```
