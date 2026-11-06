# DataStudio 图片 CoT v2 代码审查

日期：2026-11-06。审查提交：`bea9e3252996857148469c9994fbf94c37b6f26a`，分支 `codex/image-cot-v1`。
本轮为代码审查与离线复现，没有修改 DataStudio 实现、提交或推送修复，没有调用真实教师、访问服务器或发布训练数据。

## 1. 结论与证据等级

**当前不能判定实现无问题。发现 5 类可复现缺陷，建议先修复输入校验和来源政策失效，再接真实付费教师。**

| 编号 | 优先级 | 问题 | 证据 |
|---|---|---|---|
| R1 | P1 | 独立 reference_field 替换原 assistant 消息，抹掉工具字段与非法角色 | 两个输入变体都 accepted_cot，导出重验通过 |
| R2 | P1 | Honey 自定义 category_field 未投影，category_overrides 静默失效 | 显式 weak 来源按 strong 处理并接受 |
| R3 | P2 | v2 视觉联合审核可被旧 visual_evidence_required=False 绕过证据要求 | 零视觉 check 仍接受，导出重验通过 |
| R4 | P2 | quota 预检未校验微批需求，可能先生成、后在审核阶段中止 | check 通过；两次评级、两次生成后抛配置错误 |
| R5 | P2 | Honey 每个展开对话都复制完整图组原数据，导致平方级膨胀 | 1→8 条对话，准备输入大小约增至 61.6 倍 |

P1 表示影响应当优先关闭的输入/质量合同；P2 表示明确的质量契约、可靠性或规模问题。未发现已复现的默认 high 封顶突破到 max 的问题。

审查依据为当前源码与用户既定合同，不将新增检查脚本的通过当成模型语义质量证据。
既有测试：本轮重跑 `tests/image_cot`，**104 passed，35.40 秒**。此前的 571 项合并回归记录仍属于前一轮验证；本轮没有重复声明已重跑所有 571 项。
新增复现：6 个场景，覆盖 5 类缺陷，全部在合成小图、mock 回执下复现。

本地复现脚本：`Dataset/scripts/research/review_datastudio_cot_v2_20261106.py`（本次仅发布审查报告，脚本未随附）。
本地证据：`Dataset/x_temp/datastudio_cot_v2_review_20261106_02/review_results.json`（私有运行记录，未上传）。
脚本 SHA256：`9af15f053a374cd78976a94da99e4b9f0ce4e2a4e0f16dd341488c65859f56df`。
临时输出含合成 QA、图片及回执，保存在忽略的 x_temp，不能提交到代码备份。

## 2. R1：替换参考答案前没有完整验证原对话

位置：`datastudio/image_cot/adapters.py:100`，相关校验在同文件 108–120 行，以及 `contracts.py:450`。

当标准来源配置 `reference_field="gold"` 时，adapter 把第二条消息重建为：

```python
{"from": "gpt", "value": reference}
```

随后才检查 projected conversations 的 tool_calls/function_call 和角色。因此原消息里的工具字段丢失；原角色不论是否为 tool，也会被重写成 gpt。

实际复现：

1. 原第二条消息为 gpt，带 tool_calls，同时有合法 gold。
2. 原第二条消息角色为 tool，同时有合法 gold。

两种输入都生成新候选，终态 accepted_cot，并通过 selected_text 的导出重验。它们本应在调用教师前 input_rejected。
这里不是复用工具答案作为新回答，而是**原本应排除的工具依赖/非法对话通过了输入门槛**，违反用户“排掉 tool”的规则。

建议修复：先在不可变原 conversations 上验证单轮角色、每条消息工具字段与工具依赖，再提取独立参考答案。
参考答案单独赋给 SourceRecord.reference，不通过覆盖原消息来修复结构。检查输出前后，原始字段应保持可追溯。

必要回归：reference_field 有/无 × gpt/tool/system 角色 × assistant tool_calls/function_call。
所有不支持的来源应零教师调用，且保留明确拒绝原因。以上是拟议修复，尚未实现。

## 3. R2：Honey 的 category_field 丢失，来源政策不生效

位置：`datastudio/image_cot/adapters.py:76`，相关 `contracts.py:395–397`。

Honey adapter 只把内层 category 写到 projected["category"]；validate_input 却按来源声明的 category_field 去 projected 中读取。
如果配置 category_field="source_subset"，实际内层 source_subset 没进入 projected，得到 None，category_overrides 因而不会命中。

复现配置：

```python
category_field = "source_subset"
category_overrides = {"weak-source": {"reference_grade": "weak"}}
```

Honey 内层对话 source_subset 为 weak-source。实际 SourceRecord.policy.reference_grade 仍为 strong，终态 accepted_cot。
按已声明政策，它本应 needs_review，且不调用生成教师。这不依赖教师误判，纯粹是适配层漏应用用户政策。
同理，使用自定义分类字段配置的 task_type、格式和其他覆盖也可能失效。

建议修复：统一从原始 Honey 内层对话解析分类值（必要时按明确约定回退外层），把同一个值用于覆盖、范围过滤和审计记录。
避免一处用原对话、一处用投影 dict。对不存在的显式 category_field 给出可见配置/数据错误，不静默当成没有覆盖。

必要回归：category_field 为 category/source_subset/自定义字段；strong→weak、open→pure_ocr 等覆盖必须实际生效，不能仅验证默认 category。

## 4. R3：视觉审核模式没有强制视觉证据合同

位置：`datastudio/image_cot/operations.py:403`；相关导出重验也调用相同 parser。

parse_judgment 仅在 source.policy.visual_evidence_required 为 True 时要求 supported 的 image:0 visual check。
v2 没有拒绝或覆盖旧来源的 False。于是 review_mode="visual_joint" 仍可接受没有任何 visual check 的全通过结果。

复现：配置 visual_joint，来源 visual_evidence_required=False；mock joint judge 只保留 answer/reasoning/instruction 检查。
结果 accepted_cot，visual_audit="checked"，selected_text 重验通过，supported_visual_checks=0。

这里的 checked 确实代表模型接收过图片，并不是没有发送图片；缺陷是它不再提供新手册所承诺的**材料性视觉证据检查**。
这与已明确支持的 text_only/not_checked 模式不同，不能用“模型见过图”代替审核结果的证据覆盖。

建议修复：v2 的审核模式决定必要证据，visual_joint/text_then_visual 必须检查 image:0，禁止用兼容 v1 的来源开关旁路。
如果确实需要不要求视觉证据，应使用明确的文本模式或另行定义不同合同，不能继续当视觉联合模式通过。

必要回归：三种模式 × visual_evidence_required 缺省/True/False × visual check 缺失/错误来源/真实 image:0。
同时验证生成端和导出端，不只修运行时。

## 5. R4：微批与配额不匹配直到付费生成后才暴露

位置：`datastudio/image_cot/runner.py:303–309`；预检 `contracts.py:366–372` 只检查配额值为正整数。

预检不检查 requests_per_microbatch × 每角色 max_output_tokens 与 output_tpm 的关系。
实际 reserve 遇到整批超配额，直接抛 ConfigurationError，而不是缩小该阶段微批或返回受控 pending。
这会中止整个 run；即使单个请求完全放得进配额也无法继续。

复现：2 条题、微批 2；同池 output_tpm=1500；评级每请求上限 128，生成上限 512，文本审核上限 1024。
配置预检通过；两次评级和两次 generate_low 已完成、回执已保存；随后审核的 2×1024 超过 1500，抛 microbatch_exceeds_quota_window。
本轮均为 mock，不涉及真实费用；在真实服务中相同顺序意味着问题是在付费生成之后才被发现。

即时保存仍有效，候选没有丢；缺陷是启动前未发现不可执行配置，阶段调度也没有适配更小配额。

建议修复：先对所有实际角色做最小单请求可执行性预检；对微批不适配情况，优先按该角色 RPM/TPM 细分微批，配额窗口按既有预算器等待。
如果第一版不做细分，也必须在任何模型调用之前拒绝不可执行的配置，并指出需调整的角色/批量。
总预算耗尽应受控 pending；单请求本身超过配额应为配置错误，两者应区分。

## 6. R5：同图组原链被每个子对话重复保存

位置：`datastudio/image_cot/adapters.py:163`，准备输入每记录 canonical 序列化；训练字段保留又遍历完整 source.original。

每展开一个 data 对话，SourceRecord.original 都是完整外层图组，包含该图组全部 data[] 和旧链。
原始对象在内存中可能共享，但序列化到 JSONL 时会完整重复，不能靠 Python 引用共享节省磁盘。
设一组 k 个长链对话、每条长度 L，原注释为 O(kL)，准备输入会达到 O(k²L)。

合成实测：

| 一图的对话数 | 原行字节 | 展开后准备记录总字节 |
|---|---:|---:|
| 1 | 21,225 | 22,099 |
| 4 | 84,738 | 342,448 |
| 8 | 169,422 | 1,362,368 |

1→8 条时，原输入约增 8 倍，准备记录约增 61.6 倍。该开销还可能进入平铺训练元数据，增加 JSON 解析、IO 和分片压力。
这不改变 high 的封顶行为，但在 Honey 长链/多 rollout 场景中是无意义重复，与保持框架精简和提高速率的目标冲突。

建议修复：准备/运行阶段对完整原图组归档一次，以 group ID、dialog index、摘要和引用关联 SourceRecord。
不能直接删除原始字段。训练导出应按实际交付合同明确“原始行/原始对话”粒度并保留可追溯性，不能未经模板核对就静默删减 original_data。
本轮重新读完当前 data_format.md；没有改转换输出或发布真实数据，不声称上述拟议格式已经符合交付模板。

## 7. 已确认的有效边界与尚未验证的事项

本轮既有测试仍覆盖：评级 high 不越到 max、首个合格停止、未知执行不自动重发、截断/不确定不自动升级、逐 HTTP 响应落盘、改裁判版本复用已付费新候选。
这些有效行为不抵消本轮的新发现。没有将 text_only 无法独立核验图片这一已记录的模型能力边界，重复记为代码 bug。

未验证真实 Kimi/GLM 质量、服务端 effort 生效、生产吞吐、目标共享存储容错、学生训练和正式交付模板合规。
本轮目标 DataStudio 工作树保持干净，提交仍为 bea9e32；新增审查文档和复现脚本位于外层 Dataset 工作区，没有修改或夹带其既有无关改动。

建议修复顺序：R1/R2 → R3 → R4 → R5；新增反例先转为目标仓库的正式回归，再实施最小修复。
修复后复跑本轮 6 个场景、104 项扩展测试与原 467 项相关上游测试；通过前不把真实付费放量视为已验收。

在持有上述本地复现脚本的 Dataset 工作区重跑（不是从本仓库根运行；--output-dir 必须是新的私有目录）：

```powershell
.codex_worktrees/DataStudio/.venv/Scripts/python.exe scripts/research/review_datastudio_cot_v2_20261106.py --repo .codex_worktrees/DataStudio --output-dir x_temp/cot_review_new
```

脚本仅使用新生成的合成数据与 mock，保留全部产物，不覆盖或删除既有目录。
