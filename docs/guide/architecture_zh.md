# DataStudio 架构快速理解指南

## 一、项目定位

**DataStudio** 是一个多模态数据处理工具包，专门用于：
- 清洗和过滤训练数据（Filter）
- 重写和标准化数据格式（Rewriter）
- 利用大模型（MLLM）进行智能数据处理

---

## 二、核心数据流与类调用关系

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 run.py                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                              Runner                                      │   │
│  │                                                                          │   │
│  │   __init__():                                                            │   │
│  │     1. cfg = get_cfg()                    # 加载配置                      │   │
│  │     2. self.logger = build_from_cfg(...)  # 构建日志                      │   │
│  │     3. self.model = build_from_cfg(...)   # 构建模型 (OpenAIAPI)          │   │
│  │     4. self.dataloaders = build_dataloaders(...)  # 构建数据加载器         │   │
│  │     5. self.datasaver = build_from_cfg(...)       # 构建数据保存器         │   │
│  │     6. self.pipeline = build_from_cfg(...)        # 构建处理管线          │   │
│  │                                                                          │   │
│  │   run():                                                                 │   │
│  │     for dataloader in self.dataloaders:                                  │   │
│  │         for batch in dataloader:          # 迭代获取批次数据              │   │
│  │             datas = self.pipeline(batch)  # 管线处理                      │   │
│  │             self.datasaver(datas)         # 累积结果                      │   │
│  │         self.datasaver.save()             # 保存到磁盘                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据加载流程 (StandardDataLoader)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          StandardDataLoader                                     │
│                                                                                 │
│   __init__(data_root, dataset, batch_size, ...):                               │
│     │                                                                           │
│     ├─▶ FormatRegistry.load(data_file)      # 根据扩展名自动选择格式器           │
│     │       │                                                                   │
│     │       ├─▶ JsonFormat.load()           # .json 文件                        │
│     │       └─▶ JsonlFormat.load()          # .jsonl 文件                       │
│     │                                                                           │
│     ├─▶ ShardedLMDBManager.get_instance()   # 获取 LMDB 单例                    │
│     │                                                                           │
│     └─▶ write_images_to_sharded_lmdb()      # 预处理图像到缓存                   │
│             │                                                                   │
│             └─▶ 多线程读取图像 → resize → 写入 LMDB                              │
│                                                                                 │
│   __iter__() / __next__():                                                     │
│     │                                                                           │
│     └─▶ 返回 List[Dict]                     # 每个 Dict 是一条原始数据           │
│             │                                                                   │
│             └─▶ Dict 结构:                                                      │
│                   {                                                             │
│                     "id": "xxx",                                                │
│                     "image": "path/to/img.jpg",                                 │
│                     "image_pil": <PIL.Image>,    # 预加载的图像对象              │
│                     "conversations": [...]                                      │
│                   }                                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 管线处理流程 (Pipeline → SubPipeline → Operator)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               Pipeline                                          │
│                                                                                 │
│   __call__(datas: List[Dict]) -> List[Dict]:                                   │
│     │                                                                           │
│     │  # 按 priority 排序后依次执行 SubPipeline                                  │
│     │                                                                           │
│     for sub_pipeline, name in self.sub_pipelines:  # 按优先级顺序               │
│         │                                                                       │
│         ├─▶ result = sub_pipeline(datas)                                       │
│         │       │                                                               │
│         │       │  ┌─────────────────────────────────────────────────────────┐ │
│         │       │  │              SubPipeline.__call__()                     │ │
│         │       │  │                                                         │ │
│         │       │  │  # Step 1: 包装原始数据                                  │ │
│         │       │  │  items = [DataItem(d, i) for i, d in enumerate(datas)]  │ │
│         │       │  │                                                         │ │
│         │       │  │  # Step 2: 依次执行每个 Operator                         │ │
│         │       │  │  for op in self.operators:                              │ │
│         │       │  │      │                                                  │ │
│         │       │  │      ├─▶ results = op.process_batch(items)              │ │
│         │       │  │      │       │                                          │ │
│         │       │  │      │       │  ┌───────────────────────────────────┐   │ │
│         │       │  │      │       │  │  Operator.process_batch()         │   │ │
│         │       │  │      │       │  │                                   │   │ │
│         │       │  │      │       │  │  # 默认实现：逐个处理              │   │ │
│         │       │  │      │       │  │  return [self.process(item)       │   │ │
│         │       │  │      │       │  │          for item in items]       │   │ │
│         │       │  │      │       │  │                                   │   │ │
│         │       │  │      │       │  │  # MLLM 算子重写此方法实现批处理   │   │ │
│         │       │  │      │       │  └───────────────────────────────────┘   │ │
│         │       │  │      │       │                                          │ │
│         │       │  │      │       └─▶ List[Result]                           │ │
│         │       │  │      │                                                  │ │
│         │       │  │      │  # Step 3: 应用结果到数据                         │ │
│         │       │  │      │                                                  │ │
│         │       │  │      └─▶ for item, result in zip(items, results):       │ │
│         │       │  │              │                                          │ │
│         │       │  │              └─▶ kept, rejected = result.apply_to(item) │ │
│         │       │  │                      │                                  │ │
│         │       │  │                      │  ┌─────────────────────────────┐ │ │
│         │       │  │                      │  │  Result.apply_to()          │ │ │
│         │       │  │                      │  │                             │ │ │
│         │       │  │                      │  │  1. _apply_rewrites()       │ │ │
│         │       │  │                      │  │     → item.set_answer()     │ │ │
│         │       │  │                      │  │                             │ │ │
│         │       │  │                      │  │  2. _apply_filters()        │ │ │
│         │       │  │                      │  │     → item.mark_rejected()  │ │ │
│         │       │  │                      │  │     → item.split()          │ │ │
│         │       │  │                      │  │                             │ │ │
│         │       │  │                      │  │  return (kept, rejected)    │ │ │
│         │       │  │                      │  └─────────────────────────────┘ │ │
│         │       │  │                      │                                  │ │
│         │       │  │                      └─▶ (DataItem | None, DataItem | None)│
│         │       │  │                                                         │ │
│         │       │  │  # Step 4: 分离保留和拒绝的数据                          │ │
│         │       │  │  items = [i for i in result if not i.is_rejected]       │ │
│         │       │  │  rejected.extend([i for i in result if i.is_rejected])  │ │
│         │       │  │                                                         │ │
│         │       │  │  return [item.data for item in items + rejected]        │ │
│         │       │  └─────────────────────────────────────────────────────────┘ │
│         │       │                                                               │
│         │       └─▶ List[Dict]  # 包含 kept 和 rejected                        │
│         │                                                                       │
│         ├─▶ datas = [d for d in result if not d.get("rejected")]  # 继续处理   │
│         └─▶ all_rejected.extend([d for d in result if d.get("rejected")])      │
│                                                                                 │
│     return datas + all_rejected                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Operator 执行细节

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Operator 类型与执行流程                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Filter (规则过滤器)                              │   │
│  │                                                                          │   │
│  │   process(item: DataItem) -> Result:                                    │   │
│  │     result = Result(item_idx=item.idx)                                  │   │
│  │     for qa in item.qa_pairs:                                            │   │
│  │         rejected, reason = self.check(item, qa.idx)  # 子类实现          │   │
│  │         if rejected:                                                    │   │
│  │             result.add_filter(qa.idx, rejected=True, reason=reason)     │   │
│  │     return result                                                       │   │
│  │                                                                          │   │
│  │   # 子类只需实现 check():                                                │   │
│  │   def check(item, qa_idx) -> (bool, str):                               │   │
│  │       # True, "原因" = 过滤掉                                            │   │
│  │       # False, "" = 保留                                                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Rewriter (规则重写器)                             │   │
│  │                                                                          │   │
│  │   process(item: DataItem) -> Result:                                    │   │
│  │     result = Result(item_idx=item.idx)                                  │   │
│  │     for qa in item.qa_pairs:                                            │   │
│  │         new_content = self.rewrite(item, qa.idx)  # 子类实现             │   │
│  │         if new_content is not None:                                     │   │
│  │             result.add_rewrite(qa.idx, new_answer=new_content)          │   │
│  │     return result                                                       │   │
│  │                                                                          │   │
│  │   # 子类只需实现 rewrite():                                              │   │
│  │   def rewrite(item, qa_idx) -> Optional[str]:                           │   │
│  │       # 返回新答案 = 修改                                                │   │
│  │       # 返回 None = 不修改                                               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       MLLMOperator (大模型算子)                          │   │
│  │                                                                          │   │
│  │   process_batch(items: List[DataItem]) -> List[Result]:                 │   │
│  │     │                                                                    │   │
│  │     ├─▶ requests = self._build_requests(items)                          │   │
│  │     │       │                                                            │   │
│  │     │       └─▶ RequestBuilder.build_requests(item, batch_qa)           │   │
│  │     │               │                                                    │   │
│  │     │               ├─▶ 读取 prompt 模板文件                              │   │
│  │     │               ├─▶ 格式化 QA 内容到 prompt                           │   │
│  │     │               └─▶ 构建 payload: [{type: "text/image", value: ...}] │   │
│  │     │                                                                    │   │
│  │     ├─▶ responses = self.model.generate(payloads)                       │   │
│  │     │       │                                                            │   │
│  │     │       └─▶ OpenAIAPI.generate()                                    │   │
│  │     │               │                                                    │   │
│  │     │               ├─▶ 多线程并发请求                                    │   │
│  │     │               ├─▶ 自动重试失败请求                                  │   │
│  │     │               └─▶ 返回 List[str] (JSON 字符串)                      │   │
│  │     │                                                                    │   │
│  │     └─▶ for req, resp in zip(requests, responses):                      │   │
│  │             parsed = RequestBuilder.parse_response(resp)                │   │
│  │             self._add_decision(result, qa_idx, parsed)  # 子类实现       │   │
│  │                                                                          │   │
│  │   # MLLMFilter._add_decision(): 根据 parsed["result"] 添加 FilterDecision│   │
│  │   # MLLMRewriter._add_decision(): 根据 parsed["answer"] 添加 RewriteDecision│
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.5 数据保存流程 (StandardDataSaver)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          StandardDataSaver                                      │
│                                                                                 │
│   __call__(datas: List[Dict]):                                                 │
│     │                                                                           │
│     └─▶ 按 rejected 字段分类累积到 self.datas / self.rejected_datas            │
│                                                                                 │
│   save():                                                                       │
│     │                                                                           │
│     ├─▶ SimpleSaver.save(rejected_datas, is_rejected=True)                     │
│     │       │                                                                   │
│     │       ├─▶ 按 source_file 分组数据                                         │
│     │       ├─▶ 确定输出路径: output_dir/{source}/rejected/{filename}           │
│     │       └─▶ FormatRegistry.save(data, path)  # 保持原格式                   │
│     │                                                                           │
│     └─▶ SimpleSaver.save(datas, is_rejected=False)                             │
│             │                                                                   │
│             ├─▶ 按 source_file 分组数据                                         │
│             ├─▶ 确定输出路径: output_dir/{source}/{filename}                    │
│             └─▶ FormatRegistry.save(data, path)                                │
│                                                                                 │
│   save_yaml():                                                                  │
│     │                                                                           │
│     └─▶ YamlConfigGenerator.generate()  # 生成数据集配置 YAML                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.6 核心数据结构转换

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            数据结构转换流程                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   原始 JSON/JSONL                                                               │
│   ─────────────────                                                             │
│   {                                                                             │
│     "id": "001",                                                                │
│     "image": "img.jpg",                                                         │
│     "conversations": [                                                          │
│       {"from": "human", "value": "Q1"},                                        │
│       {"from": "gpt", "value": "A1"},                                          │
│       {"from": "human", "value": "Q2"},                                        │
│       {"from": "gpt", "value": "A2"}                                           │
│     ]                                                                           │
│   }                                                                             │
│         │                                                                       │
│         │  DataLoader 加载                                                      │
│         ▼                                                                       │
│   Dict (内存中)                                                                 │
│   ─────────────                                                                 │
│   {                                                                             │
│     "id": "001",                                                                │
│     "image": "img.jpg",                                                         │
│     "image_pil": <PIL.Image>,        # 预加载的图像                             │
│     "source_file": "/path/to/file",  # 来源文件                                 │
│     "conversations": [...]                                                      │
│   }                                                                             │
│         │                                                                       │
│         │  SubPipeline 包装                                                     │
│         ▼                                                                       │
│   DataItem                                                                      │
│   ────────                                                                      │
│   DataItem(                                                                     │
│     _data = {...},                   # 原始 Dict 引用                           │
│     idx = 0,                         # 批次内索引                               │
│     _qa_pairs = [                    # 解析后的 QA 对                           │
│       QA(idx=0, question="Q1", answer="A1"),                                   │
│       QA(idx=1, question="Q2", answer="A2"),                                   │
│     ]                                                                           │
│   )                                                                             │
│         │                                                                       │
│         │  Operator 处理                                                        │
│         ▼                                                                       │
│   Result                                                                        │
│   ──────                                                                        │
│   Result(                                                                       │
│     item_idx = 0,                                                               │
│     filter_decisions = [                                                        │
│       FilterDecision(qa_idx=1, rejected=True, reason="太短"),                  │
│     ],                                                                          │
│     rewrite_decisions = [                                                       │
│       RewriteDecision(qa_idx=0, new_answer="修改后的A1"),                       │
│     ]                                                                           │
│   )                                                                             │
│         │                                                                       │
│         │  Result.apply_to(item)                                                │
│         ▼                                                                       │
│   处理后的 Dict                                                                  │
│   ─────────────                                                                 │
│   # 保留的数据 (kept)                     # 被过滤的数据 (rejected)              │
│   {                                      {                                      │
│     "id": "001",                           "id": "001",                         │
│     "image": "img.jpg",                    "image": "img.jpg",                  │
│     "conversations": [                     "conversations": [                   │
│       {"from": "human", "value": "Q1"},      {"from": "human", "value": "Q2"}, │
│       {"from": "gpt", "value": "修改后A1"},  {"from": "gpt", "value": "A2"}     │
│     ],                                     ],                                   │
│     "ori_answer": {"0": "A1"},             "rejected": True,                   │
│     "rewrite_ops": {                       "filter_ops": {                      │
│       "MyRewriter": {"0": "rewritten"}       "MyFilter": {"0": "太短"}          │
│     }                                      }                                    │
│   }                                      }                                      │
│         │                                                                       │
│         │  DataSaver 保存                                                       │
│         ▼                                                                       │
│   output/                                                                       │
│   ├── source_name/                                                              │
│   │   └── data.jsonl                 # 保留的数据                               │
│   └── source_name/rejected/                                                     │
│       └── data.jsonl                 # 被过滤的数据                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.7 类依赖关系图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              类依赖关系                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                              ┌─────────┐                                        │
│                              │ Runner  │                                        │
│                              └────┬────┘                                        │
│           ┌──────────────────────┼──────────────────────┐                       │
│           │                      │                      │                       │
│           ▼                      ▼                      ▼                       │
│   ┌───────────────┐      ┌────────────┐      ┌──────────────────┐              │
│   │StandardData   │      │  Pipeline  │      │StandardDataSaver │              │
│   │   Loader      │      └─────┬──────┘      └────────┬─────────┘              │
│   └───────┬───────┘            │                      │                         │
│           │                    │                      │                         │
│           │              ┌─────┴─────┐          ┌─────┴─────┐                   │
│           │              │           │          │           │                   │
│           ▼              ▼           ▼          ▼           ▼                   │
│   ┌───────────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐            │
│   │ShardedLMDB    │  │SubPipe │  │SubPipe │  │Simple  │  │YamlGen │            │
│   │   Manager     │  │ line 1 │  │ line N │  │ Saver  │  │erator  │            │
│   └───────────────┘  └───┬────┘  └────────┘  └────────┘  └────────┘            │
│                          │                                                      │
│                    ┌─────┴─────────────────┐                                    │
│                    │                       │                                    │
│                    ▼                       ▼                                    │
│              ┌──────────┐           ┌──────────┐                                │
│              │ Operator │           │ Operator │                                │
│              └────┬─────┘           └──────────┘                                │
│                   │                                                             │
│         ┌─────────┼─────────┐                                                   │
│         │         │         │                                                   │
│         ▼         ▼         ▼                                                   │
│   ┌──────────┐ ┌──────┐ ┌────────────┐                                         │
│   │  Filter  │ │Rewri │ │MLLMOperator│                                         │
│   │          │ │ ter  │ │            │                                         │
│   └────┬─────┘ └──┬───┘ └─────┬──────┘                                         │
│        │          │           │                                                 │
│        │          │     ┌─────┴─────┐                                           │
│        │          │     │           │                                           │
│        ▼          ▼     ▼           ▼                                           │
│   ┌─────────┐ ┌─────┐ ┌───────┐ ┌──────────────┐                               │
│   │DataItem │ │Resul│ │Request│ │  OpenAIAPI   │                               │
│   │   QA    │ │  t  │ │Builder│ │   (Model)    │                               │
│   └─────────┘ └─────┘ └───────┘ └──────────────┘                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、核心概念

### 3.1 数据格式

输入数据是 JSON/JSONL 格式的对话数据：

```json
{
    "id": "unique_id",
    "image": "path/to/image.jpg",
    "conversations": [
        {"from": "human", "value": "这张图片里有什么？"},
        {"from": "gpt", "value": "图片显示..."}
    ]
}
```

**核心抽象**：
- **DataItem**: 单条数据的包装类，提供便捷的 QA 对访问
- **QA**: 一个问答对（human + gpt 各一条消息）

```python
# DataItem 使用示例
item = DataItem(raw_data, idx=0)
for qa in item.qa_pairs:
    print(f"Q: {qa.question}")
    print(f"A: {qa.answer}")
```

### 3.2 Operator（算子）

所有数据处理逻辑都是 Operator，分为两类：

| 类型 | 作用 | 返回 |
|------|------|------|
| **Filter** | 判断数据是否应该被过滤 | `(rejected: bool, reason: str)` |
| **Rewriter** | 修改数据内容 | `new_answer: str` 或 `None` |

```python
# Filter 示例
class MyFilter(Filter):
    def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
        if len(item.get_answer(qa_idx)) < 10:
            return True, "答案太短"  # 过滤掉
        return False, ""  # 保留

# Rewriter 示例
class MyRewriter(Rewriter):
    def rewrite(self, item: DataItem, qa_idx: int) -> Optional[str]:
        answer = item.get_answer(qa_idx)
        return answer.strip()  # 返回新答案
```

### 3.3 Result（结果）

Operator 不直接修改数据，而是返回 **Result**：

```python
Result
├── filter_decisions: List[FilterDecision]   # 过滤决策
│   └── FilterDecision(qa_idx, rejected, reason)
└── rewrite_decisions: List[RewriteDecision] # 重写决策
    └── RewriteDecision(qa_idx, new_answer, message)
```

Pipeline 统一调用 `result.apply_to(item)` 来应用修改。

### 3.4 Pipeline（管线）

```
Pipeline
├── SubPipeline 1 (priority=0)  ◀── 先执行
│   ├── ConvLengthFilter
│   └── ImageSizeFilter
├── SubPipeline 2 (priority=1)
│   └── MLLMFilter (图文一致性)
├── SubPipeline 3 (priority=2)
│   └── MLLMRewriter (重写答案)
└── SubPipeline 4 (priority=3)  ◀── 后执行
    └── ResponseTagFilter
```

---

## 四、目录结构速查

```
datastudio/
├── datasets/              # 数据加载/保存
│   ├── data_loader.py     # StandardDataLoader - 加载数据 + LMDB 图像缓存
│   ├── data_saver.py      # StandardDataSaver - 保存处理结果
│   ├── formatters/        # JSON/JSONL 格式处理器
│   └── config/            # YAML 配置加载
│
├── operators/             # 数据处理算子
│   ├── core/              # 核心抽象
│   │   ├── data_item.py   # DataItem, QA - 数据封装
│   │   ├── operator.py    # Operator, Filter, Rewriter - 基类
│   │   └── result.py      # Result, FilterDecision, RewriteDecision
│   ├── filters/           # 规则过滤器
│   │   ├── conv_length.py # 对话轮数过滤
│   │   ├── image_size.py  # 图像尺寸过滤
│   │   └── text_repeat.py # 文本重复检测
│   ├── rewriters/         # 规则重写器
│   │   ├── norm_prompt.py # 标准化 prompt
│   │   ├── remove_think.py# 移除思考内容
│   │   └── split.py       # 拆分多轮对话
│   └── mllm/              # MLLM 算子
│       ├── base.py        # MLLMOperator 基类
│       ├── filter.py      # MLLMFilter - 用大模型过滤
│       ├── rewriter.py    # MLLMRewriter - 用大模型重写
│       └── request.py     # RequestBuilder - 构建模型请求
│
├── pipelines/             # 管线编排
│   ├── pipeline.py        # Pipeline - 主管线
│   └── sub_pipeline.py    # SubPipeline - 子管线
│
├── models/                # 模型封装
│   ├── base.py            # BaseAPI - 基类（重试、并行）
│   └── openai_api.py      # OpenAIAPI - OpenAI 兼容 API
│
└── utils/                 # 工具函数
    ├── registry.py        # 组件注册表
    ├── checkpoint.py      # 断点续传
    ├── database.py        # LMDB 图像缓存
    └── vision.py          # 图像处理
```

---

## 五、配置文件解读

配置文件是 Python 文件（不是 YAML），支持继承：

```python
# configs/my_config.py

# 继承基础配置
_base_ = [
    "@/_base_/models/local_api_model.py",
    "@/_base_/dataset.py",
]

# 工作目录
work_dir = "work_dirs/my_experiment"

# 日志
logger = dict(type="Logger", log_file="logs/run.txt")

# 数据集配置
dataset_yaml = "/path/to/dataset.yaml"
dataloader = dict(
    dataset=dataset_yaml,
    batch_size=512,
    use_image=True,
    cache_dir="~/cache/images_lmdb",
)

# 数据保存
datasaver = dict(
    dataset=dataset_yaml,
    output_dir="output",
)

# 模型（用于 MLLM 算子）
model = dict(model="Qwen3-VL-8B-Instruct", thread_num=512)

# 管线定义
pipeline = dict(
    type="Pipeline",
    operations={
        # 规则过滤（优先级 0，最先执行）
        "basic_filter": dict(
            cfg=dict(
                type="SubPipeline",
                operators=[
                    dict(type="ConvLengthFilter", min_length=1, max_length=20),
                    dict(type="ImageSizeFilter", min_size=100),
                ],
            ),
            priority=0,
        ),
        # MLLM 过滤（优先级 1）
        "mllm_filter": dict(
            cfg=dict(
                type="SubPipeline",
                operators=[
                    dict(
                        type="MLLMFilter",
                        request_builder=dict(
                            type="RequestBuilder",
                            prompt="prompts/filter/question_image_consist_v2.txt",
                            with_image=True,
                        ),
                    ),
                ],
            ),
            priority=1,
        ),
    },
)
```

---

## 六、运行方式

```bash
# 运行处理
python run.py -c configs/my_config.py

# 输出结构
work_dirs/my_experiment/
├── config.yaml          # 保存的配置
├── logs/run.txt         # 日志
└── checkpoint.json      # 断点信息

output/
├── source_name/                # 按 source 分目录
│   ├── data.jsonl              # 保留的数据
│   └── rejected/               # 被过滤的数据
│       └── data.jsonl
├── output.yaml                 # 保留数据的 YAML 索引
└── output_rejected.yaml        # 过滤数据的 YAML 索引
```

---

## 七、关键类速查表

| 类 | 文件 | 作用 |
|----|------|------|
| `Runner` | `run.py` | 主入口，协调各组件 |
| `StandardDataLoader` | `datasets/data_loader.py` | 加载数据，管理 LMDB 缓存 |
| `StandardDataSaver` | `datasets/data_saver.py` | 保存处理结果 |
| `Pipeline` | `pipelines/pipeline.py` | 主管线，按优先级执行子管线 |
| `SubPipeline` | `pipelines/sub_pipeline.py` | 子管线，顺序执行算子 |
| `DataItem` | `operators/core/data_item.py` | 数据封装，提供 QA 访问 |
| `Filter` | `operators/core/operator.py` | 过滤器基类 |
| `Rewriter` | `operators/core/operator.py` | 重写器基类 |
| `Result` | `operators/core/result.py` | 算子执行结果 |
| `MLLMOperator` | `operators/mllm/base.py` | MLLM 算子基类 |
| `RequestBuilder` | `operators/mllm/request.py` | 构建 MLLM 请求 |

---

## 八、扩展指南

### 添加新的 Filter

```python
# datastudio/operators/filters/my_filter.py
from datastudio.operators.core import Filter, DataItem
from datastudio.utils.registry import OPERATORS

@OPERATORS.register_module()
class MyCustomFilter(Filter):
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
    
    def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
        answer = item.get_answer(qa_idx)
        if some_condition(answer, self.threshold):
            return True, "不符合条件"
        return False, ""
```

### 添加新的 Rewriter

```python
# datastudio/operators/rewriters/my_rewriter.py
from datastudio.operators.core import Rewriter, DataItem
from datastudio.utils.registry import OPERATORS

@OPERATORS.register_module()
class MyCustomRewriter(Rewriter):
    def rewrite(self, item: DataItem, qa_idx: int) -> Optional[str]:
        answer = item.get_answer(qa_idx)
        new_answer = transform(answer)
        return new_answer if new_answer != answer else None
```

### 在配置中使用

```python
pipeline = dict(
    type="Pipeline",
    operations={
        "my_step": dict(
            cfg=dict(
                type="SubPipeline",
                operators=[
                    dict(type="MyCustomFilter", threshold=0.8),
                    dict(type="MyCustomRewriter"),
                ],
            ),
            priority=0,
        ),
    },
)
```

---

## 九、核心设计原则

1. **算子不直接修改数据** - 返回 Result，由 Pipeline 统一应用
2. **配置驱动** - 所有组件通过配置实例化，支持复现
3. **注册表模式** - 组件通过 `@OPERATORS.register_module()` 注册
4. **断点续传** - 支持 checkpoint，大数据集处理可中断恢复
5. **图像缓存** - LMDB 分片存储，避免重复读取图像

---

## 十、常见问题

**Q: 如何只运行部分数据测试？**
A: 修改 `dataloader.batch_size` 为小值，或在数据文件中只保留少量样本

**Q: 如何跳过图像处理？**
A: 设置 `dataloader.use_image=False`

**Q: 如何查看处理日志？**
A: 查看 `work_dir/logs/` 目录下的日志文件

**Q: 如何从断点恢复？**
A: 直接重新运行相同命令，会自动读取 checkpoint 继续处理
