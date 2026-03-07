# DataStudio Architecture Guide

## 1. Project Overview

**DataStudio** is a multimodal data processing toolkit designed for:
- Cleaning and filtering training data (Filter)
- Rewriting and standardizing data formats (Rewriter)
- Intelligent data processing using MLLMs

---

## 2. Core Data Flow and Class Relationships

### 2.1 Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 run.py                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                              Runner                                      │   │
│  │                                                                          │   │
│  │   __init__():                                                            │   │
│  │     1. cfg = get_cfg()                    # Load config                  │   │
│  │     2. self.logger = build_from_cfg(...)  # Build logger                 │   │
│  │     3. self.model = build_from_cfg(...)   # Build model (OpenAIAPI)      │   │
│  │     4. self.dataloaders = build_dataloaders(...)  # Build data loaders   │   │
│  │     5. self.datasaver = build_from_cfg(...)       # Build data saver     │   │
│  │     6. self.pipeline = build_from_cfg(...)        # Build pipeline       │   │
│  │                                                                          │   │
│  │   run():                                                                 │   │
│  │     for dataloader in self.dataloaders:                                  │   │
│  │         for batch in dataloader:          # Iterate batches              │   │
│  │             datas = self.pipeline(batch)  # Pipeline processing          │   │
│  │             self.datasaver(datas)         # Accumulate results           │   │
│  │         self.datasaver.save()             # Save to disk                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Loading (StandardDataLoader)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          StandardDataLoader                                     │
│                                                                                 │
│   __init__(data_root, dataset, batch_size, ...):                               │
│     │                                                                           │
│     ├─▶ FormatRegistry.load(data_file)      # Auto-select format by extension  │
│     │       │                                                                   │
│     │       ├─▶ JsonFormat.load()           # .json files                      │
│     │       └─▶ JsonlFormat.load()          # .jsonl files                     │
│     │                                                                           │
│     ├─▶ ShardedLMDBManager.get_instance()   # Get LMDB singleton               │
│     │                                                                           │
│     └─▶ write_images_to_sharded_lmdb()      # Pre-process images to cache      │
│             │                                                                   │
│             └─▶ Multi-threaded read → resize → write to LMDB                   │
│                                                                                 │
│   __iter__() / __next__():                                                     │
│     │                                                                           │
│     └─▶ Returns List[Dict]                  # Each Dict is one raw data entry  │
│             │                                                                   │
│             └─▶ Dict structure:                                                │
│                   {                                                             │
│                     "id": "xxx",                                                │
│                     "image": "path/to/img.jpg",                                 │
│                     "image_pil": <PIL.Image>,    # Pre-loaded image object      │
│                     "conversations": [...]                                      │
│                   }                                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Pipeline Processing (Pipeline → SubPipeline → Operator)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               Pipeline                                          │
│                                                                                 │
│   __call__(datas: List[Dict]) -> List[Dict]:                                   │
│     │                                                                           │
│     │  # Execute SubPipelines in priority order                                │
│     │                                                                           │
│     for sub_pipeline, name in self.sub_pipelines:  # By priority               │
│         │                                                                       │
│         ├─▶ result = sub_pipeline(datas)                                       │
│         │       │                                                               │
│         │       │  ┌─────────────────────────────────────────────────────────┐ │
│         │       │  │              SubPipeline.__call__()                     │ │
│         │       │  │                                                         │ │
│         │       │  │  # Step 1: Wrap raw data                                │ │
│         │       │  │  items = [DataItem(d, i) for i, d in enumerate(datas)]  │ │
│         │       │  │                                                         │ │
│         │       │  │  # Step 2: Execute each Operator sequentially           │ │
│         │       │  │  for op in self.operators:                              │ │
│         │       │  │      │                                                  │ │
│         │       │  │      ├─▶ results = op.process_batch(items)              │ │
│         │       │  │      │                                                  │ │
│         │       │  │      │  # Step 3: Apply results to data                 │ │
│         │       │  │      └─▶ for item, result in zip(items, results):       │ │
│         │       │  │              └─▶ kept, rejected = result.apply_to(item) │ │
│         │       │  │                                                         │ │
│         │       │  │  # Step 4: Separate kept and rejected data              │ │
│         │       │  │  items = [i for i in result if not i.is_rejected]       │ │
│         │       │  │  rejected.extend([i for i in result if i.is_rejected])  │ │
│         │       │  │                                                         │ │
│         │       │  │  return [item.data for item in items + rejected]        │ │
│         │       │  └─────────────────────────────────────────────────────────┘ │
│         │       │                                                               │
│         │       └─▶ List[Dict]  # Contains kept and rejected                   │
│         │                                                                       │
│         ├─▶ datas = [d for d in result if not d.get("rejected")]               │
│         └─▶ all_rejected.extend([d for d in result if d.get("rejected")])      │
│                                                                                 │
│     return datas + all_rejected                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Operator Execution Details

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Operator Types and Execution Flow                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Filter (Rule-based)                             │   │
│  │                                                                          │   │
│  │   process(item: DataItem) -> Result:                                    │   │
│  │     result = Result(item_idx=item.idx)                                  │   │
│  │     for qa in item.qa_pairs:                                            │   │
│  │         rejected, reason = self.check(item, qa.idx)  # Subclass impl   │   │
│  │         if rejected:                                                    │   │
│  │             result.add_filter(qa.idx, rejected=True, reason=reason)     │   │
│  │     return result                                                       │   │
│  │                                                                          │   │
│  │   # Subclasses only need to implement check():                          │   │
│  │   def check(item, qa_idx) -> (bool, str):                               │   │
│  │       # True, "reason" = filter out                                     │   │
│  │       # False, "" = keep                                                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Rewriter (Rule-based)                            │   │
│  │                                                                          │   │
│  │   process(item: DataItem) -> Result:                                    │   │
│  │     result = Result(item_idx=item.idx)                                  │   │
│  │     for qa in item.qa_pairs:                                            │   │
│  │         new_content = self.rewrite(item, qa.idx)  # Subclass impl      │   │
│  │         if new_content is not None:                                     │   │
│  │             result.add_rewrite(qa.idx, new_answer=new_content)          │   │
│  │     return result                                                       │   │
│  │                                                                          │   │
│  │   # Subclasses only need to implement rewrite():                        │   │
│  │   def rewrite(item, qa_idx) -> Optional[str]:                           │   │
│  │       # Return new answer = modify                                      │   │
│  │       # Return None = no change                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       MLLMOperator (LLM-powered)                        │   │
│  │                                                                          │   │
│  │   process_batch(items: List[DataItem]) -> List[Result]:                 │   │
│  │     │                                                                    │   │
│  │     ├─▶ requests = self._build_requests(items)                          │   │
│  │     │       └─▶ RequestBuilder.build_requests(item, batch_qa)           │   │
│  │     │               ├─▶ Read prompt template file                       │   │
│  │     │               ├─▶ Format QA content into prompt                   │   │
│  │     │               └─▶ Build payload: [{type: "text/image", ...}]      │   │
│  │     │                                                                    │   │
│  │     ├─▶ responses = self.model.generate(payloads)                       │   │
│  │     │       └─▶ OpenAIAPI.generate()                                    │   │
│  │     │               ├─▶ Multi-threaded concurrent requests              │   │
│  │     │               ├─▶ Automatic retry on failure                      │   │
│  │     │               └─▶ Returns List[str] (JSON strings)                │   │
│  │     │                                                                    │   │
│  │     └─▶ for req, resp in zip(requests, responses):                      │   │
│  │             parsed = RequestBuilder.parse_response(resp)                │   │
│  │             self._add_decision(result, qa_idx, parsed)  # Subclass impl│   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.5 Data Saving (StandardDataSaver)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          StandardDataSaver                                      │
│                                                                                 │
│   __call__(datas: List[Dict]):                                                 │
│     └─▶ Classify and accumulate by rejected field                              │
│                                                                                 │
│   save():                                                                       │
│     ├─▶ SimpleSaver.save(rejected_datas, is_rejected=True)                     │
│     │       ├─▶ Group data by source_file                                      │
│     │       ├─▶ Determine output path: output_dir/{source}/rejected/{filename} │
│     │       └─▶ FormatRegistry.save(data, path)  # Keep original format        │
│     │                                                                           │
│     └─▶ SimpleSaver.save(datas, is_rejected=False)                             │
│             ├─▶ Group data by source_file                                      │
│             ├─▶ Determine output path: output_dir/{source}/{filename}          │
│             └─▶ FormatRegistry.save(data, path)                                │
│                                                                                 │
│   save_yaml():                                                                  │
│     └─▶ YamlConfigGenerator.generate()  # Generate dataset config YAML         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.6 Core Data Structure Transformations

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Data Structure Transformation Flow                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Raw JSON/JSONL                                                                │
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
│         │  DataLoader loads                                                     │
│         ▼                                                                       │
│   Dict (in memory)                                                              │
│   ─────────────                                                                 │
│   {                                                                             │
│     "id": "001",                                                                │
│     "image": "img.jpg",                                                         │
│     "image_pil": <PIL.Image>,        # Pre-loaded image                        │
│     "source_file": "/path/to/file",  # Source file path                        │
│     "conversations": [...]                                                      │
│   }                                                                             │
│         │                                                                       │
│         │  SubPipeline wraps                                                    │
│         ▼                                                                       │
│   DataItem                                                                      │
│   ────────                                                                      │
│   DataItem(                                                                     │
│     _data = {...},                   # Reference to raw Dict                   │
│     idx = 0,                         # Index within batch                      │
│     _qa_pairs = [                    # Parsed QA pairs                         │
│       QA(idx=0, question="Q1", answer="A1"),                                   │
│       QA(idx=1, question="Q2", answer="A2"),                                   │
│     ]                                                                           │
│   )                                                                             │
│         │                                                                       │
│         │  Operator processes                                                   │
│         ▼                                                                       │
│   Result                                                                        │
│   ──────                                                                        │
│   Result(                                                                       │
│     item_idx = 0,                                                               │
│     filter_decisions = [                                                        │
│       FilterDecision(qa_idx=1, rejected=True, reason="Too short"),             │
│     ],                                                                          │
│     rewrite_decisions = [                                                       │
│       RewriteDecision(qa_idx=0, new_answer="Modified A1"),                     │
│     ]                                                                           │
│   )                                                                             │
│         │                                                                       │
│         │  Result.apply_to(item)                                                │
│         ▼                                                                       │
│   Processed Dict                                                                │
│   ─────────────                                                                 │
│   # Kept data                              # Rejected data                     │
│   {                                      {                                      │
│     "id": "001",                           "id": "001",                         │
│     "image": "img.jpg",                    "image": "img.jpg",                  │
│     "conversations": [                     "conversations": [                   │
│       {"from": "human", "value": "Q1"},      {"from": "human", "value": "Q2"}, │
│       {"from": "gpt", "value": "Mod A1"},    {"from": "gpt", "value": "A2"}    │
│     ],                                     ],                                   │
│     "ori_answer": {"0": "A1"},             "rejected": True,                   │
│     "rewrite_ops": {                       "filter_ops": {                      │
│       "MyRewriter": {"0": "rewritten"}       "MyFilter": {"0": "Too short"}    │
│     }                                      }                                    │
│   }                                      }                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.7 Class Dependency Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Class Dependencies                                    │
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

## 3. Key Class Reference

| Class | File | Purpose |
|-------|------|---------|
| `Runner` | `run.py` | Main entry point, coordinates all components |
| `StandardDataLoader` | `datasets/data_loader.py` | Loads data, manages LMDB cache |
| `StandardDataSaver` | `datasets/data_saver.py` | Saves processing results |
| `Pipeline` | `pipelines/pipeline.py` | Main pipeline, executes sub-pipelines by priority |
| `SubPipeline` | `pipelines/sub_pipeline.py` | Sub-pipeline, executes operators sequentially |
| `DataItem` | `operators/core/data_item.py` | Data wrapper, provides QA access |
| `Filter` | `operators/core/operator.py` | Filter base class |
| `Rewriter` | `operators/core/operator.py` | Rewriter base class |
| `Result` | `operators/core/result.py` | Operator execution result |
| `MLLMOperator` | `operators/mllm/base.py` | MLLM operator base class |
| `RequestBuilder` | `operators/mllm/request.py` | Builds MLLM requests |

---

## 4. Directory Structure

```
datastudio/
├── datasets/              # Data loading/saving
│   ├── data_loader.py     # StandardDataLoader - load data + LMDB image cache
│   ├── data_saver.py      # StandardDataSaver - save processing results
│   ├── formatters/        # JSON/JSONL format handlers
│   └── config/            # YAML config loading
│
├── operators/             # Data processing operators
│   ├── core/              # Core abstractions
│   │   ├── data_item.py   # DataItem, QA - data wrapper
│   │   ├── operator.py    # Operator, Filter, Rewriter - base classes
│   │   └── result.py      # Result, FilterDecision, RewriteDecision
│   ├── filters/           # Rule-based filters
│   │   ├── conv_length.py # Conversation turn count filter
│   │   ├── image_size.py  # Image size filter
│   │   └── text_repeat.py # Text repetition detection
│   ├── rewriters/         # Rule-based rewriters
│   │   ├── norm_prompt.py # Normalize prompts
│   │   ├── remove_think.py# Remove think content
│   │   └── split.py       # Split multi-turn conversations
│   └── mllm/              # MLLM operators
│       ├── base.py        # MLLMOperator base class
│       ├── filter.py      # MLLMFilter - LLM-powered filtering
│       ├── rewriter.py    # MLLMRewriter - LLM-powered rewriting
│       └── request.py     # RequestBuilder - build model requests
│
├── pipelines/             # Pipeline orchestration
│   ├── pipeline.py        # Pipeline - main pipeline
│   └── sub_pipeline.py    # SubPipeline - sub-pipeline
│
├── models/                # Model wrappers
│   ├── base.py            # BaseAPI - base class (retry, parallelism)
│   └── openai_api.py      # OpenAIAPI - OpenAI-compatible API
│
└── utils/                 # Utilities
    ├── registry.py        # Component registry
    ├── checkpoint.py      # Checkpoint resume
    ├── database.py        # LMDB image cache
    └── vision.py          # Image processing
```

---

## 5. Core Design Principles

1. **Operators don't modify data directly** — They return Result objects; the Pipeline applies changes uniformly
2. **Config-driven** — All components are instantiated via config, ensuring reproducibility
3. **Registry pattern** — Components are registered via `@OPERATORS.register_module()`
4. **Checkpoint resume** — Supports checkpoints for interruptible processing of large datasets
5. **Image caching** — LMDB sharded storage avoids repeated image reads
