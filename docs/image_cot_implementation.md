# DataStudio 图片 CoT：实现、使用与验收边界

日期：2026-11-05。上游基线：`676bcb8d5494d65c8fb85185ac26d98da6c92842`。
分支：`codex/image-cot-v1`。可选入口为[run_image_cot.py](../run_image_cot.py)；
原run.py、默认算子、LMDB与多进程引擎保留。没有调用付费API，没有训练。

## 1. 实现状态

| 能力 | 状态 | 证据边界 |
|---|---|---|
| 单图单轮输入、不可变来源、稳定ID | 已实现 | 严格JSONL、非法输入与媒体身份有本地测试 |
| 短链→本地检查→一次联合审核→选择性长链 | 已实现 | mock及真实loopback HTTP测试，不代表模型质量 |
| IP/端口/model与DeepSeek官方API profile | 已实现 | 自部署协议本地验证；没有探测真实DeepSeek端点 |
| intent/receipt/commit、单owner、恢复、显式复用 | 已实现 | 本地崩溃/磁盘失败/损坏/缺响应测试；SFS待测 |
| 人工裁定、质量门禁、预览/无覆盖发布 | 已实现 | 正式发布必须提供真实质量/训练/格式证据 |
| 实际学生训练 | 仅协议及门禁 | checkpoint、tokenizer、trainer未定，未运行 |
| 真实吞吐与CoT严重错误率 | 未验证 | 规模/期限/配额未定，不承诺够快或已合格 |

采用Honey-Data-V2的短/长生成与筛选思路，不声称复现未完全公开的生产配方。
首版每题选一条合格链；OCR/框旁路独立计数，不伪造推理。
额外视觉/关键过程审核与可靠提交协议是本扩展。

## 2. 代码责任及实施取舍

| 文件 | 责任 |
|---|---|
| [contracts.py](../datastudio/image_cot/contracts.py) | 配置、输入、类型、稳定ID、共享配额 |
| [storage.py](../datastudio/image_cot/storage.py) | 追加日志、完整性、单写入者、part提交与恢复 |
| [audited_api.py](../datastudio/image_cot/audited_api.py) | 继承OpenAIAPI，保留回执，约束请求/响应 |
| [operations.py](../datastudio/image_cot/operations.py) | 继承MLLM算子，严格审核、终答与人工裁定 |
| [runner.py](../datastudio/image_cot/runner.py) | prepare、短长DAG、唯一dispatch、预算及分片合流 |
| [export.py](../datastudio/image_cot/export.py) | 审核身份重验、训练JSON、预览、门禁与发布 |

复用Pipeline/SubPipeline、DataItem/Result、RequestBuilder、MLLMRewriter/MLLMFilter、
OpenAIAPI的预编码/连接池/信号量/gather和图像规则算子；没有第二套HTTP客户端。
输入采用严格有界SourceRecord适配器及小批PIL投影，没有直接使用会自动检测schema、
跳过坏行/加载整文件的StandardDataLoader，避免额外投影文件和静默误读；原loader未改。

P0不启用LMDB缓存，图片用PNG无损编码、不缩放；跨阶段会重新读图/编码，已记录耗时。
跨microbatch连接池沿用上游创建方式。只有真实profile证明值得时才加缓存/会话优化。
重复文本检测复用上游算子但只检查8192字符尾部，防止长CoT引发二次复杂度；
补充重复行检查使用线性Counter。图像最小边14、最大比例200沿用示例起点，可由源政策调整。

## 3. 环境和验证

Python >=3.10，Windows/Linux使用异步HTTP路径，不运行依赖fork的MP后端。

```bash
python -m pip install -r requirements-image-cot-dev.txt
python -m pytest tests/image_cot -q
```

测试只用合成小图和127.0.0.1模拟服务，无需API key。
最终在同一进程合并验收：**523项通过，其中56项扩展测试、467项上游回归**。
覆盖CLI、真实loopback HTTP、失败路由、持久化、复用、裁定/发布门禁与默认算子兼容性。
完整范围和代码摘要见[本地验证记录](image_cot_validation_2026-11-05.md)。
上游回归有一条未await AsyncMock的RuntimeWarning，原测试/代码没有为此修改。
mock只证明控制流，不能证明严重错误率、真实吞吐或学生收益。

## 4. 输入和配置

从[base.py](../configs/image_cot/base.py)复制私有配置到代码仓库之外；
DeepSeek小规模验证从[pilot.py](../configs/image_cot/pilot.py)开始。
None预算故意保留，未填写不能调用真实接口。

一期只接显式standard_qa_v1 JSONL，一行一题：

```json
{"id":"source_1","conversations":[{"from":"human","value":"<image>\nHow many groups are shown?"},{"from":"gpt","value":"2"}],"images":["images/my_source/1.png"],"category":"counting"}
```

images相对于source.media_root；多图、多轮、tools、Honey嵌套data/messages记录为
input_rejected，不猜字段。坏JSON/空行报行号并停止，不静默丢行。
原标注和媒体只读；路径不能逃逸已声明媒体根。

```python
image_cot["sources"] = [{
    "schema": "standard_qa_v1", "name": "my_source", "revision": "immutable-revision",
    "path": "/private/annotations/my_source.jsonl",
    "media_root": "/mnt/sfs/unimoe_dataset/training",
    "max_image_pixels": 16000000,
    "task_policy": {"task_type": "numeric", "reference_grade": "strong"},
    "category_overrides": {"ocr": {"task_type": "pure_ocr"}},
}]
```

像素数仅展示配置形状，需按实际图像/内存调整。题型可为numeric/mcq/open/pure_ocr/
box_grounding，参考等级显式strong/weak/unknown。弱/歧义参考先review，不用教师救坏源。
数值默认精确比较，绝对/相对容差由源政策声明；选项要求明确字母映射。
框旁路需登记xyxy_normalized或xyxy_absolute及合法坐标。
唯一非恒等题面变换为remove_exact_suffix，需给出确切remove_suffix；原文保留，审核看前后题面。

prepare --limit限制处理题数和图片数，仍读取源标注文件计算来源hash。
只检查标注allowlist的媒体，不扫描整个共享存储目录树。

## 5. 两种教师连接

最低服务要求为OpenAI-compatible图文Chat Completions：

```python
image_cot["connections"]["lan_teacher"] = {
    "provider_profile": "openai_compatible",
    "ip": "10.0.0.8", "port": 8000, "model": "actual-served-model-name",
    "auth": {"mode": "none"},
}
```

也可给endpoint=http://10.0.0.8:8000，默认api_prefix=/v1。
反向代理路径显式配置，不在endpoint中写/chat/completions或重复/v1。
鉴权用env_bearer/key_env；配置与manifest不记录key值。
无鉴权连接不会误用全局OPENAI_API_KEY；四角色通过connection_ref引用连接。

DeepSeek官方端点为https://api.deepseek.com，prefix为空，model=deepseek-flash，
当前展示名DeepSeek-V4.1-Flash。key从DEEPSEEK_API_KEY读取。
pilot显式关闭short/question_check/joint_verify的thinking，long开启并先用high。
直接HTTP发送顶层thinking，不套SDK extra_body或继承Qwen采样默认项。
DeepSeek系统消息适配为字符串，图片用data:image/png;base64，reasoning_content单独保留。
这些只是已实现的profile，没有完成用户真实服务的图文探针。

官方别名可更新，记录返回model/fingerprint，不假称固定checkpoint。
同一key即使从不同环境变量引用，也不能为不同角色重复分配完整账户配额。
参考：[模型列表](https://api-docs.deepseek.com/api/list-models/)、
[视觉API](https://api-docs.deepseek.com/guides/vision/)、
[thinking](https://api-docs.deepseek.com/guides/thinking_mode/)。

## 6. 运行、恢复与复用

填写private_work_root，各角色输出token/读超时/批超时，以及limits的请求数、累计输出token、
每次执行时长、请求/响应字节和解码像素上限。HTTP还需quota_pools：RPM、输入/输出TPM、
保守input_tokens_per_request和角色池绑定。估算必须覆盖图片、候选和提示词；实测超出会停止。
未知usage保留预留，不能算零。货币成本硬上限尚未实现，配置此类字段会被拒绝。

```bash
python run_image_cot.py check -c /private/cot_config.py
python run_image_cot.py prepare -c /private/cot_config.py --limit 100
python run_image_cot.py run -c /private/cot_config.py --limit 100
python run_image_cot.py report --run /private/work/runs/JOB_ID
python run_image_cot.py resume --run /private/work/runs/JOB_ID
python run_image_cot.py export --run /private/work/runs/JOB_ID --preview
```

check只查配置，不发探针。run --backend mock是合成传输，不能正式发布。
已设置input_manifest时不能再用run --limit隐式截断。

只有一份发送协议：预留→intent fsync→API→已关联receipt fsync→覆盖校验→判定。
不以zip顺序关联；缺少/重复/未知ID阻断批次，已知回执保全，异常回执隔离。
part的commit和引用完整性通过后才消费，不用旧checkpoint。

| 恢复现场 | 动作 |
|---|---|
| 无intent | 预算内发送 |
| intent有、receipt无 | unresolved，不自动重发 |
| receipt有、未commit | 本地重解析，不重生成 |
| 尾部半行 | 保留原段，新段继续；不据此推定未执行 |
| 合法commit | 直接复用 |
| commit引用损坏 | 停止，保留证据，不覆盖重算 |

超时、429/未知5xx保守视为执行状态未知，无隐藏10次重试。
schema错不触发long。需重审时修改joint_verify.request_revision并显式reuse_from旧run，
只建立新审核任务，不必无意义改prompt；更改生成revision是新生成，不是未决请求的安全重发。
code升级保守失效；仅改导出请直接export -c，无须重跑生成。
回执/源快照/工作日志只留私有根，不进Git；缓存按当前分片清空。

## 7. 质量与人工裁定

先查完成状态、think、重复/占位符和确定性终答，再一次联合看图审核
answer/visual/reasoning/instruction。缺键、非法枚举、外围文字、重复key、NaN、
自相矛盾证据都不放行。终答正确不能掩盖关键图值、实体、公式和连接错误。
审核指令在system，候选/参考为明确数据字段；同一模型自审仍须随机人审校准。

short允许空think兼容表示，解释留正文；long保留非空原生推理。
残缺、嵌套、重复think不修补；原生reasoning与内联think并存不自动删重拼接。
审核后不改事实或删除推理。

review_resolutions JSONL最多4096条，schema=image_cot.review_resolution/1，绑定
candidate_id/candidate_text_hash/judge_signature，包含reviewer、带时区timestamp、
具体evidence和pass/fail outcome。只能裁定有效内容审核，不能把技术错误或截断响应改成pass，
也不能覆盖确定错误的终答核验。裁定作用于新run派生选择；旧commit只读，通过显式复用避免重付费。
弱源参考应形成受审源政策/版本，不能只改一个“通过”标签。

## 8. 交付与发布门禁

训练输出为human/gpt conversations、相对images和modal_mask=1。
原字段按original_field_mapping保留顶层；id/conversations/images冲突需明确映射，
不静默覆盖。token统计和_cot不混入训练JSON；OCR/框在selection独立保留，不混入CoT。

可给export单独的私有配置`image_cot={"export": ...}`，不会请求教师：

```bash
python run_image_cot.py export --run /private/work/runs/JOB_ID -c /private/export.py --preview
python run_image_cot.py export --run /private/work/runs/JOB_ID -c /private/export.py --stage-release
python run_image_cot.py export --run /private/work/runs/JOB_ID -c /private/export.py --publish
```

媒体必须已在training_media_root，逐条核验原SHA，不复制、移动、裁剪或转码原图。
preview的README标未发布，不替代最终模板。正式发布须同时提供：

1. 非mock、清洁代码commit、无pending/review的选择集合。
2. image_cot.quality_gate/1：绑定job_id/selection_digest，status=pass且real_execution=true。
   human_audit含independent_groups、severe_errors、one_sided_confidence、reviewer、sampling_plan、
   representative_strata、groups_independent。本版只接至少300个独立组零严重错误，
   一侧95%精确上界≤1%；审计数不得超过总体。模型自报分数不能替代。
3. image_cot.training_probe/1：同job/selection；真实学生model/revision、tokenizer_revision、
   trainer/revision及sample_digest。image_order、bos_eos、reasoning_loss、answer_loss、
   no_final_truncation、no_duplicate_think、actual_forward_backward均为true。
   TrainingProbe是接训练器的协议，本仓库没有假称已运行DeepSeek/GLM学生训练。
4. 实际data_format_path、权威readme_template、完成的release_readme和
   image_cot.delivery_validation/1报告。绑定selection_digest、data_sha256、readme_sha256、
   readme_template_sha256、format_spec_sha256、training_media_root和四个output目的路径；
   actual_json_checked/media_paths_checked/authoritative_readme_template_checked全true。
   不为适配本程序重新定义权威README模板，也不靠自定义schema宣称严格合规。

学生probe与精确格式验证来自实际训练器/项目验证流程，本轮没有生成这些真实证据。
文件“存在”不够，必须验证身份及检查内容。
publish仅复制派生JSONL/README/侧车到显式output_jsonl/output_readme/output_provenance/
output_manifest绝对路径。相同内容可复用，不同内容拒绝覆盖；manifest最后提交。
原子无覆盖发布需目标文件系统支持同目录临时文件hard-link，否则失败，不降级覆盖。
共享存储目录同步及故障恢复仍需实际服务器验证。

## 9. 统计与后续验证

report.json按源题对账六种终态，不混淆图片/请求/候选/合格题。
reports/execution-*.json记录本次解码、模型批次、日志fsync、commit耗时、缓存命中和
可用的峰值RSS（Windows可unknown）。累计token账本和本次耗时分开，不能把resume
缓存速度当生成吞吐；实际货币成本unknown。

限制：单图单轮、每阶段一个候选；不做跨源语义去重/评测污染检测、PRM、交错裁图、
GRPO、自动重发未知请求、跨框架学生训练、Files API或大规模生产复现。
下一道门禁是配置真实端点与明确小批预算，先一题图文探针，再固定小样本校准。
此前不宣称DeepSeek真实连通、严重错误率合格、吞吐达标或学生SFT有效。
