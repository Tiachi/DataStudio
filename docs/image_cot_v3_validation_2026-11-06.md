# 图片 CoT v3 验证记录

日期：2026-11-06，Asia/Shanghai。仓库 Tiachi/DataStudio，分支 codex/image-cot-v1。
实现前基线：51ff7b197e9181373c39b6c70bb3c45b0f2b6b30。

## 1. 状态与结果

已实现并本地验证四档评级、工具排除、两次纯文本审核和逐阶段保存统计。
没有调用真实付费教师，没有发布真实训练数据，没有实测评级准确率、生产吞吐或学生训练收益。

| 检查 | 实际结果 |
|---|---|
| 合并相关 pytest 回归 | **617 passed，1 warning，67.40 秒** |
| 图片 CoT 扩展 | 150 项，其中本轮新增 46 项 |
| 既有算子/模型/Pipeline/schema/registry | 467 项 |
| Ruff check | PASS：扩展模块、CLI、新配置、新测试 |
| compileall | PASS：扩展模块、CLI、新配置 |
| CLI help | 包含 rating-report 和原有命令 |
| Git diff --check | PASS |

范围：tests/image_cot、tests/operators、tests/models、tests/pipelines、
tests/datasets/test_schema_converter.py、tests/utils/test_registry.py。不是全仓库测试。
环境：Windows、Python 3.13.9。只有合成小图、mock 和 127.0.0.1 HTTP。
唯一 warning 来自既有 OpenAIAPI 测试的 AsyncMock 未 await，没有为消除警告而修改上游通用实现。
67.40 秒仅为回归测试耗时，不能换算真实生成速率。

测试代码字节身份（implementation_identity 的模块/CLI摘要）：

```text
f3aad1cf711c9e70d5f311710a8107555f45d19bf1662b8479a9b33465f234ea
```

测试时为未提交工作树；正式提交后 commit/clean 改变，以上源码摘要应一致。
完整合成运行记录留在外层 Dataset/x_temp/cot_v3_final_05，不提交生成的图片、QA 或回执。

## 2. 用户五项要求的实际覆盖

| 要求 | 已验证内容 |
|---|---|
| 只要工具就排除 | 来源工具声明、嵌套字段、tool/function 角色、协议标记；提取 reference_field 前检查；任一模型阶段真实工具调用拒绝且保留回执 |
| 四个评级 | none/low/high/max；none 零生成/零审核；从 low 首个合格即停；high 失败无 max 请求 |
| 评级区分坏题 | invalid 题目拒绝、参考 invalid 待复核、uncertain 待复核；均不生成；矛盾/残缺评级不会默认 max |
| 纯文本 CoT 审核 | 模板无 visual/image/picture/视觉/图片等审核词；无图片请求；reasoning/consistency 独立检查 |
| 答案与 CoT 分开 | 两个独立 role、task ID、请求、结果与回执；答案裁判不见 CoT；CoT 只在答案通过后调用 |
| 阶段保存统计 | 每阶段 raw→results/commit→statistics 分片→总 report；理由、failure_kind、token usage、四档与质量分布；恢复不重发、不重复累加 |

null/空工具占位字段在服务响应中不代表实际工具操作；来源工具字段仍严格排除。
没有把普通问题提到手工工具的自然语言当工具调用。已包含这两类反例。

## 3. 其他关键回归

- 答案明确错误可在 cap 内升级，且省去该候选的 CoT 审核；CoT 明确错误可升级。
- reference/input 问题或不确定结果停止；截断、坏 JSON、未知执行不靠高档补救。
- 本地终答与 gold 不一致仍到独立答案裁判，裁判发现参考问题后停止。
- malformed 审核响应不会丢掉已保存的强教师 reasoning/content，不继续高档。
- 预算只够评级/生成/答案阶段时，CoT 阶段未派发；之前两份产物和统计保留。
- 导出缺一份审核即失败；重验评级 cap、原始候选和两个审核的候选绑定。
- Honey 自定义 source_subset 覆盖规则实际生效；weak 参考不再静默按 strong 接受。
- 历史 v2 的 visual_evidence_required=False 不能绕过其既定视觉合同；v3 本身不调用视觉联合审核。
- 按角色配额细分微批，单请求不适配时 HTTP 预检拒绝；合成时钟验证配额等待，不等待真实一分钟或调用真实接口。
- 实际 loopback HTTP 顺序：评级→生成→答案验证→CoT 审核；只有前两者传图；答案 payload 不含候选推理，CoT payload 包含推理。
- rating-report 只读保存的评级与人工标签，能统计把人工 low 评为 high 等混淆，也覆盖评级预算未派发、没有回执的情况；这不是模型真实准确率的证据。

## 4. 未完成的运行验证

1. 真实评级服务对题目/参考质量的判断与 none/low 边界、high/max 高估率；必须用独立标签和未参与调参的批次确认。
2. 纯文本答案/CoT 裁判的误接收、误拒绝与未知率；文本审核不能证明图中事实。
3. 实际服务的 effort 参数生效、可见推理、结构化 JSON 遵从率、usage 与配额。
4. 服务器共享存储的 fsync/rename、真实异常恢复、备份和吞吐。
5. Honey 原图组重复保存的 R5 规模问题；本轮没有静默删原始字段来规避它。
6. 真实学生编码与训练，以及实际交付 JSON/媒体/README 对当前 data_format 的严格验证。

本轮没有把“评级必须准确”的需求改写为已经实现的准确率承诺。工程上已加入质量分流、审计依据和校准统计；
真实效果仍需数据验证。完整运行说明见 [v3 操作手册](image_cot_v3_operations.md)。
