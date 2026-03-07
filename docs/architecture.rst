架构设计
========

核心模块
--------

DataStudio 采用模块化设计：

.. code-block:: text

   datastudio/
   ├── datasets/      # 数据集加载与保存
   ├── models/        # 模型封装 (GPT, Qwen 等)
   ├── operators/     # 数据操作算子
   │   ├── core/      # 核心类型定义
   │   ├── filters/   # 过滤器
   │   └── mllm/      # 多模态模型算子
   ├── pipelines/     # 处理流水线
   └── utils/         # 工具函数

类层级图
--------

Result 类型
~~~~~~~~~~~

.. inheritance-diagram:: datastudio.operators.core.result.FilterDecision
                         datastudio.operators.core.result.RewriteDecision
                         datastudio.operators.core.result.OperatorResult
   :parts: 1
   :caption: 操作结果类层级

设计原则
--------

1. **不可变决策对象** - FilterDecision 和 RewriteDecision 使用 frozen dataclass
2. **清晰分离** - 过滤 (filter) 与重写 (rewrite) 逻辑分离
3. **组合结果** - OperatorResult 可同时包含两种决策
