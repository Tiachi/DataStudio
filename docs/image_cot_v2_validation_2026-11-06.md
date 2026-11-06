# 图片 CoT v2 实现验证记录

日期：2026-11-06，Asia/Shanghai。仓库 Tiachi/DataStudio，分支 codex/image-cot-v1。
基线 HEAD：3d2600b23924e2285b048ee125403e69f190d3c6；上游基线：676bcb8d5494d65c8fb85185ac26d98da6c92842。

状态：已实现且本地验证；未调用真实付费教师，未生产真实训练数据，未运行学生训练，未测生产吞吐。
原有 schema /1 路线保持兼容；新路线使用 image_cot.config/2。

## 1. 最终结果与范围

Windows / Python 3.13.9，pytest 8.4.2。仅合成小图、合成 QA 与 127.0.0.1 HTTP。

| 检查 | 结果 |
|---|---|
| 合并 pytest 回归 | **571 passed，1 warning，35.91 秒** |
| 图片 CoT 扩展 | 104 项，其中本轮新增 48 项 |
| 既有算子、模型、Pipeline、格式转换、registry | 467 项 |
| Ruff check：扩展模块、CLI、新配置、新测试 | PASS |
| compileall：扩展模块、CLI、新配置 | PASS |
| CLI help | 包含 check/profile/prepare/run/resume/report/export |
| Git diff --check | PASS |

测试范围不是全仓库：

```bash
python -m pytest tests/image_cot tests/operators tests/models tests/pipelines tests/datasets/test_schema_converter.py tests/utils/test_registry.py -q
```

唯一 warning 来自既有 tests/models/test_openai_api.py 的 AsyncMock 未 await；未为隐藏警告而修改上游模型实现。
35.91 秒是合成回归耗时，不能换算真实教师产线速率。

最终运行源码摘要（implementation_identity 的模块/CLI字节摘要；配置、测试与文档不在该摘要内）：

```text
5fd4ad40b620f68ed98ba13bb48cf0f55caf4daec746e7362ef9811707c0aff2
```

测试时为未提交工作树，clean=false；代码提交后 commit/clean 改变，字节摘要应一致。
最终回归使用私有临时根 Dataset/x_temp/cot_v2_final_06，不提交其中生成的 QA、图片、日志或回执。

## 2. 核心验收

| 合同 | 实际覆盖 |
|---|---|
| 评级是硬上限 | low 封顶失败仅调用 low；high 封顶失败仅调用 low/high；零 max 请求 |
| 首个合格即停 | max 上限下 low 通过后不调用 high/max；low 失败 high 通过即选 high |
| 真正失败才升档 | 本地终答失败或裁判明确推理失败可升级；不确定、截断、坏 JSON、未知执行不升级 |
| 统一重生成 | Honey 同图两个独立对话分别产生新候选；旧链不进入教师 payload 或选中训练回答 |
| 输入范围 | math/code、工具、多轮、坏 think 拒绝；参考缺失待复核且零教师调用 |
| 可选图题检查 | 来源单独开关；跳过不产生伪造回执；评级关闭用显式固定 cap |
| 上限只能收紧 | 评级、全局、来源、固定 cap 取最小值；旧准备输入不能绕过改后的来源/global cap |
| 评级请求身份 | supported_caps 改变即改变评级配方摘要，不复用不同输出契约的评级 |
| 审核区分能力 | 文本裁判不接图片，不能返回 visual_status/image:0；导出标记 not_checked |
| 视觉审核 | visual_joint/text_then_visual 都验证实际对应回执，后者两份审核都须通过 |
| 证据完整性 | 缺答案/推理/指令依据的 v2 all-pass 进入 uncertain，不靠标志放行 |
| 原生响应 | 保留 reasoning_content/content，生成只用原图和问题，不夹 gold/旧链 |
| 即时保存 | 同批慢请求等待时，快请求回执已 fsync；真实 loopback HTTP 覆盖 |
| 保存错误 | 人工注入磁盘错误会抛出并停止，不能被父引擎吞掉后继续当成功 |
| 恢复 | 生成回执落盘后批次抛异常，恢复不再调用该教师；请求/token 重建覆盖全部 v2 阶段 |
| 重审复用 | 修改裁判版本，新运行显式 reuse_from，只发新审核、不重复强教师调用 |
| 导出 | 重验 cap、候选文本哈希、审核角色/候选绑定；篡改 effort 或视觉范围拒绝 |
| 格式 | answer_line/plain/box/boxed 平衡括号；参考格式单独声明且完整保留原字段 |
| 不浪费调用 | 参考处理器不就绪先待复核；已识别范围外任务不继续高档；预算停保留进度 |

## 3. 复检中修复的实际问题

1. 原有 Budget 只恢复固定五个旧阶段，新增档位可能漏计。现在从统一阶段注册表恢复。
2. 原 API 批次结束才统一写回执，慢请求或批次崩溃可能带走已完成强模型输出。改为逐响应完成回调写入私有日志。
3. 旧审核要求 visual_status 与 image:0，直接接文本裁判会语义造假。新增独立文本审核 schema 和明确视觉未检查标记。
4. 准备输入若未冻结全局/来源开关和上限，使用旧 input_manifest 可违背新 cap。现在不一致即要求重新 prepare。
5. 标准格式自动识别无法解释 Honey 的 messages/from/value。专用 adapter 展开，不修改通用 schema converter。
6. 纯文本审核解码图像属于无意义 IO，现已省掉；继承异步调度，不新增并发/数据库体系。

## 4. 剩余运行验证

- Kimi 实际账号模型名、图文协议、三档生效及 reasoning 可见性；GLM/其他裁判真实 JSON 遵从率与配额。
- 代表性来源、不同 effort 与审核模式的独立人审；视觉幻觉和错误参考造成的误接收率。解析器不能证明证据自然语言真实。
- 与视觉联合审核对照；text_only 不具备逐条视觉正确性证明。
- Linux/目标共享存储的 fsync、rename、异常恢复与备份；本地 durable 测试不是服务器灾难恢复证明。
- 实际输出/token分布、限流、吞吐及成本。现在没有美元硬预算，不能承诺未提供配额下的产能。
- 目标学生训练器的真实编码、loss mask、forward/backward 和训练收益。
- 实际 JSON/媒体/中文 README 按当前 data_format.md 和权威模板逐项验证；本轮未发布真实数据，不声称严格交付合规。

完整操作与质量边界见 [v2 手册](image_cot_v2_operations.md)。
