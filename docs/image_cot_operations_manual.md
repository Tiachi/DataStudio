# 图片 CoT 数据构建操作手册

更新日期：2026-11-05。适用分支：`codex/image-cot-v1`，入口：`run_image_cot.py`。
这是当前代码的操作说明。用户自己的真实服务、原始数据和训练器仍需按实际情况配置。

## 阅读路线

| 目标 | 直接阅读 |
|---|---|
| 第一次运行，不调用 API | [离线 mock 示例](#quickstart) |
| 接自部署 IP 服务或 DeepSeek | [教师接入](#teachers) |
| 规划成本与速度 | [预算与配额](#budget) |
| 中断、升级或改审核 | [恢复与复用](#recovery) |
| 人工裁定不确定样本 | [人工复核](#review) |
| 拿到可检查的训练 JSONL | [预览与字段映射](#export) |
| 准备正式训练交付 | [发布门禁](#release) |
| 排查失败 | [故障排查](#troubleshooting) |

本手册中的 `/path/to`、`/private`、`JOB_ID` 都是需要替换的示例。
Python 配置是会被执行的可信代码，只加载自己维护、经过审查的文件。
命令路径含空格时加引号；Windows的Python配置建议使用`D:/...`或raw string表示路径。

## 1. 安装、版本与运行目录

```bash
git clone --branch codex/image-cot-v1 https://github.com/Tiachi/DataStudio.git
cd DataStudio
git status --short --branch
git rev-parse HEAD
```

已有 checkout 先核对分支、remote 和工作树；干净时才能 fast-forward 更新，不能为更新清理或覆盖他人修改。
一次正式任务应固定一个干净 commit。任务运行期间不要在这个 checkout 做 pull、commit、改 prompt 或开发。
需要升级时另用一个 checkout，保留旧运行环境和原路径。

使用 Python >=3.10。当前已完成的本地验证为 Windows/Python 3.13.9，不能直接推断 Linux/SFS 已验收。

| 系统 | 创建环境 | 安装依赖 | 后续解释器 |
|---|---|---|---|
| Linux/macOS | `python3 -m venv .venv` | `.venv/bin/python -m pip install -r requirements-image-cot-dev.txt` | `.venv/bin/python` |
| Windows PowerShell | `python -m venv .venv` | `.\.venv\Scripts\python.exe -m pip install -r requirements-image-cot-dev.txt` | `.\.venv\Scripts\python.exe` |

以下统一写 `python`，实际使用表中的解释器或已激活的虚拟环境。命令在仓库根目录执行。
安装依赖需要网络；服务器使用允许的直连/批准网络，不自动启用个人代理。

```bash
python -m pytest tests/image_cot -q
python run_image_cot.py --help
```

56 项扩展测试只用合成输入和本地 HTTP 服务。历史完整合并验收为 523 项通过，
详见[验证记录](image_cot_validation_2026-11-05.md)，它不是生产质量/速度报告。

## 2. 先分清四类路径

| 路径 | 内容 | 约束 |
|---|---|---|
| 代码 checkout | Python、prompt、公共模板 | 不放真实数据、key、回执和工作输出 |
| 私有配置目录 | 自己的 Python 配置、外部 prompt、审计文件 | 放在 checkout 外；运行使用的版本保持不变 |
| source `media_root` | 原图与相对路径基准 | 原图只读，不自动复制/移动/转码 |
| `private_work_root` | prepared、runs、回执、选择、预览 | 不在 checkout 内，不能与标注/媒体路径互相包含 |

例如代码、输入、配置、工作根是四个并列目录：

```text
/path/to/DataStudio/
/path/to/cot-input/data.jsonl
/path/to/cot-input/images/example.png
/path/to/cot-config/cot_demo.py
/path/to/cot-work/
```

Windows 可分别使用 `D:/src/DataStudio`、`D:/cot-input`、`D:/cot-config`、`D:/cot-work`。
这些是布局示例，不代表目录已存在。配置中使用绝对路径，样本的 images 仍使用相对路径。

<a id="quickstart"></a>
## 3. 第一次：离线 mock 跑通完整链路

### 3.1 准备一条合成输入

准备 `cot-input/images/example.png` 和 `cot-input/data.jsonl`。图像应是可解码的单帧图，
默认最短边至少 14 像素。下面的 JSONL 必须是一整行：

<!-- DOC-CHECK: input-example -->
```json
{"id":"doc_example_1","conversations":[{"from":"human","value":"<image>\nHow many groups are shown?"},{"from":"gpt","value":"2"}],"images":["images/example.png"],"category":"counting"}
```

mock 教师是固定合成逻辑，示例参考答案 2 与它的预设相配；它不看图、不代表模型实际求解。
要验证真实图像质量，必须切换真实教师并做人审。不要将此 mock 输出混入训练集。

### 3.2 保存完整私有配置

保存为 `cot-config/cot_demo.py`，只需先替换开头三个路径。此配置使用 `mode="mock"`，
即使误把后端写成 http 也会在发送前拒绝；真实验证时按下一节显式切换。
预算数字用于一题流程演示，不是教师最优参数或账户真实配额。

<!-- DOC-CHECK: tiny-config -->
```python
from pathlib import Path
import runpy

REPO = Path("/path/to/DataStudio").resolve()
DATA = Path("/path/to/cot-input").resolve()
WORK = Path("/path/to/cot-work").resolve()

image_cot = runpy.run_path(str(REPO / "configs/image_cot/base.py"))["image_cot"]
image_cot.update(
    mode="mock",
    private_work_root=str(WORK),
    input_manifest=None,
    reuse_from=[],
    review_resolutions=None,
)
image_cot["sources"] = [{
    "schema": "standard_qa_v1",
    "name": "doc_example",
    "revision": "synthetic-v1",
    "path": str(DATA / "data.jsonl"),
    "media_root": str(DATA),
    "max_image_pixels": 1_000_000,
    "task_policy": {"task_type": "numeric", "reference_grade": "strong"},
}]
image_cot["limits"].update(
    source_records_per_part=16,
    source_bytes_per_part=4 * 1024 * 1024,
    requests_per_microbatch=1,
    max_inflight=1,
    max_response_bytes=2 * 1024 * 1024,
    max_request_bytes=4 * 1024 * 1024,
    max_decoded_pixels_per_batch=2_000_000,
    max_requests_total=5,
    max_output_tokens_total=16_000,
    max_wall_time_s=600,
)
output_caps = {"question_check": 512, "short": 2048, "long": 8192, "joint_verify": 1536}
for role, model in image_cot["models"].items():
    model.update(
        connection_ref=None,
        request_revision="1",
        max_output_tokens=output_caps[role],
        read_timeout_s=120,
        batch_deadline_s=180,
        effective_options={},
        transport_retry=1,
        quota_pool="teacher_account",
    )
image_cot["quota_pools"] = {}
image_cot["export"].update(
    dataset_name="docs_demo",
    training_media_root=str(DATA),
    original_field_mapping={
        "id": "original_id",
        "conversations": "original_conversations",
        "images": "original_images",
        "modal_mask": "original_modal_mask",
        "videos": "original_videos",
        "audios": "original_audios",
    },
)
```

这里的 prompt 来自 base.py，路径已经指向 checkout 内对应文件。
首次离线验证可用当前开发 checkout；正式/长任务则必须冻结代码与这些 prompt。

### 3.3 执行预检、构建、查看与预览

```bash
python run_image_cot.py check -c /path/to/cot-config/cot_demo.py --backend mock
python run_image_cot.py run -c /path/to/cot-config/cot_demo.py --backend mock --limit 1
```

`check` 只做配置预检。成功时打印 `PASS: configuration preflight only...`。
`run` 成功后先打印完整 run 路径，再打印一行构建摘要。记录这个路径，替换后面的 RUN。

```bash
python run_image_cot.py report --run /path/to/cot-work/runs/JOB_ID
python run_image_cot.py export --run /path/to/cot-work/runs/JOB_ID --preview
```

这个特定合成示例通常为 accepted_cot=1、requests_reserved=3（图题检查、短链、联合审核）。
预览返回 `export_staging/RELEASE_ID`，其中有 data.jsonl、provenance.jsonl、README.md、manifest.json。
manifest 必须是 preview=true、published=false；mock 的正式发布会被拒绝。

需要一条命令自动构造上述合成图/配置并验证时，见[文档示例验证](#doc-validation)。

<a id="teachers"></a>
## 4. 切换真实教师

先复制离线配置为新的私有文件。保留旧配置与工作记录，改成 `mode="pilot"`，
明确设置连接、配额和小批上限。`run` 的默认后端是 http，真实命令会请求服务。

### 4.1 IP＋端口＋模型名

```python
image_cot["mode"] = "pilot"
image_cot["connections"]["lan_teacher"] = {
    "provider_profile": "openai_compatible",
    "ip": "10.0.0.8", "port": 8000,
    "api_prefix": "/v1",
    "model": "actual-served-model-name",
    "auth": {"mode": "none"},
}
for model in image_cot["models"].values():
    model["connection_ref"] = "lan_teacher"
```

得到 `POST http://10.0.0.8:8000/v1/chat/completions`。
也可使用 endpoint 而不写 ip/port。endpoint 是服务根，api_prefix 是额外 API 前缀：

| endpoint | api_prefix | 完整地址 |
|---|---|---|
| `http://10.0.0.8:8000` | `/v1` | `/v1/chat/completions` |
| `https://teacher.example.org/proxy` | `/v1` | `/proxy/v1/chat/completions` |
| `http://10.0.0.8:8000/v1` | 空字符串 | `/v1/chat/completions` |

不要在 endpoint 再写 /chat/completions；默认前缀与已有 /v1 重复时会报 duplicate_api_prefix。
IPv6 可通过 ip/port 配置，由规范化逻辑加方括号。

模型名必须是服务实际暴露的 ID，不从 HF 目录名猜测。最低协议要求是图文 messages、
model、非流式 Chat Completions 和完整完成状态。不是任意 IP 服务都能直接接。
本版本不自动调用 GET /models；能力发现和最小图文探针需操作者核实。

带鉴权时只改 auth：

```python
image_cot["connections"]["lan_teacher"]["auth"] = {
    "mode": "env_bearer", "key_env": "TEACHER_API_KEY"
}
```

不同角色可连接不同模型/端点。例如 long 指向 Thinking 服务；同一账户仍要共享 quota pool。
通用 profile 不盲发 enable_thinking、top_k 等参数，只发送已登记的 effective_options。
具体 thinking 控制是否被服务支持，要看该服务协议；返回 400 时先修配置，不靠重复请求尝试。

若endpoint前面还有LLMRouter或其它网关，客户端retry=1不会关闭网关内部重试。
例如[LLMRouter配置](https://github.com/Open-Bee/LLMRouter#configuration)另有num_retries；
实际接入需检查超时/5xx的网关重试与failover，防止未知执行在中间层又被派发。
本扩展不能据客户端配置保证整个代理链只执行一次。

### 4.2 DeepSeek 官方小规模验证

```python
image_cot["mode"] = "pilot"
for role, model in image_cot["models"].items():
    model["connection_ref"] = "deepseek_validation"
    model["effective_options"] = {"thinking": {"type": "enabled" if role == "long" else "disabled"}}
    if role == "long":
        model["effective_options"]["reasoning_effort"] = "high"
```

继承的 connection 为 https://api.deepseek.com，api_prefix 为空，model=deepseek-flash，
从 DEEPSEEK_API_KEY 环境变量读取 key。也可直接从仓库 pilot.py 开始，但须填完预算与来源。
这是本分支内置 profile；正式使用时核对供应商模型 ID、能力和配额，不假定别名锁定 checkpoint。

Linux 交互式终端可隐藏输入并导出变量：

```bash
read -r -s -p 'DeepSeek API key: ' DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
```

PowerShell 7.1+：

```powershell
$env:DEEPSEEK_API_KEY = Read-Host -MaskInput "DeepSeek API key"
```

不要将真实 key 写进配置、README、命令参数或 Git；配置中只有变量名。
无鉴权连接会显式清空 key，不误用全局 OPENAI_API_KEY。

DeepSeek 直接 HTTP payload 使用顶层 thinking，不发送 SDK 的 extra_body 包裹；
system content 转成字符串，图像用 PNG data URL，reasoning_content/content 分别保存。
API开关控制教师，不决定学生 SFT 中的推理字段/loss mask。

官方参考：[模型发现](https://api-docs.deepseek.com/api/list-models/)、
[图片输入](https://api-docs.deepseek.com/guides/vision/)、
[thinking](https://api-docs.deepseek.com/guides/thinking_mode/)。

### 4.3 配额绑定与第一次真实请求

下面是形状示例，必须替换为账户确认过的限额和保守输入估算，不能当作服务实际配额：

```python
image_cot["quota_pools"] = {
    "teacher_account": {"rpm": 30, "input_tpm": 600000, "output_tpm": 100000,
                        "input_tokens_per_request": 20000}
}
```

先只验证一题，保留示例的 microbatch=1、并发=1、请求数=5 等显式上限：

```bash
python run_image_cot.py check -c /private/config/real_pilot.py --backend http
python run_image_cot.py prepare -c /private/config/real_pilot.py --limit 1
python run_image_cot.py run -c /private/config/real_pilot.py --backend http --limit 1
```

check 不发送 API，不证明真实鉴权/模型存在/图像被有效读取。
只有实际 run 和回执/人工核图能验证这些。成功 HTTP 状态也不等于 CoT 通过。

## 5. 来源与任务政策

| source 字段 | 含义 |
|---|---|
| schema | 固定 standard_qa_v1 |
| name / revision | 明确来源名与不可变版本，不能只写可漂移的“最新” |
| path | JSONL 标注文件，prepare 会计算完整文件摘要 |
| media_root | 输入 images 相对路径基准 |
| max_image_pixels | 每张图像的显式像素上限 |
| task_policy | task_type、reference_grade及本任务规则 |
| category_field | 默认 category |
| category_overrides | 按类别精确覆盖 task_policy，不是任意Python表达式路由 |

一条 human/gpt、单图和占位符数量必须严格匹配。输入若自带 modal_mask，只接受整数1。
不接受 videos/audios、动画多帧、路径逃逸、含混 data/messages/tools，坏行不静默跳过。
图片原始SHA进入记录，运行/导出时再次检查；标注文件在prepare前后摘要不同会停止，
后续媒体与记录SHA不符也会停止。

| task_type | 参考与本地规则 | 审核/处理 |
|---|---|---|
| numeric | 唯一 Answer: 收束；数值和单位可解析 | 数值核验后仍联合审核过程 |
| mcq | options 是明确字母→文本字典，或由 options_field 指定 | 不重排选项；标签或唯一完整选项文本精确匹配 |
| open | 可信参考；本地不假装做语义真值判断 | 联合看图审核完整结论和理由 |
| pure_ocr | 可信且非空原标注 | 图题检查后保留原标注，不进入CoT文件 |
| box_grounding | 可信JSON坐标字符串、明确box_format | 检查格式和边界后旁路 |

reference_grade=weak/unknown 先needs_review；仅仅非空不等于strong。
数值/选项参考若自身不能被登记规则解释，也先review，不先付费生成。

numeric 容差默认0，可用字符串声明 absolute_tolerance、relative_tolerance；规则为
`abs(actual-reference) <= max(abs_tol, rel_tol*abs(reference))`。单位字符串必须一致，
不自动进行 cm→m 等单位换算，也不做substring或统一5%容差。
数值/选项候选终答抽取支持 Answer、Final Answer、最终答案、答案的单独收束行；
多个收束或无歧义抽取失败进入review，不能靠猜测取最后一个数。

box_format 支持xyxy_normalized或xyxy_absolute，原gpt.value示意为字符串`"[0.1,0.2,0.8,0.9]"`；
也可为多框JSON数组。须满足x1<x2、y1<y2且在归一化/图像边界内。

可配置 min_image_size（默认14）、max_image_ratio（默认200）、max_candidate_chars。
唯一非恒等题面变换为question_transform=remove_exact_suffix＋remove_suffix，
只移除精确后缀；原题保留，审核比较变换前后。不要用它删事实、单位或选项。

<a id="budget"></a>
## 6. 预算、批次与速率

| limits 字段 | 单位/作用 |
|---|---|
| source_records_per_part | 源分片最多条数；限制元数据/候选驻留规模 |
| source_bytes_per_part | 原始行及准备后SourceRecord行/分片字节边界，不是整库数据量 |
| requests_per_microbatch | 一次派发最多请求数，不超过源part条数 |
| max_inflight | 模型实例并发上限；本runner分阶段执行 |
| max_response_bytes | 单响应体上限，含非JSON响应 |
| max_request_bytes | 编码后的请求体上限，含base64图片 |
| max_decoded_pixels_per_batch | 当前批次累计解码像素边界 |
| max_requests_total | job累计请求预留数；不等于成功条数 |
| max_output_tokens_total | job累计输出token账本上限，未知usage保守计预留 |
| max_wall_time_s | 本次调度的派发/等待时间预算，resume会有新的一次时长窗口 |

每角色另配max_output_tokens、read_timeout_s、batch_deadline_s。
时长预算从调度器创建Budget时开始，不是整条命令包含prepare的操作系统强制截止；
准备、摘要计算和同步文件IO不能靠此值强行中断。execution观察的elapsed包含更完整的本次耗时。
传输retry固定1，不用隐藏10次重试。货币成本硬上限未实现，max_cost_total等未知预算字段会报错。
请求/token上限不等于账单金额保证，实际费用需要供应商记录。

配额按共享账户池预留，必须同时满足：

```text
microbatch请求数 <= RPM
microbatch请求数 * input_tokens_per_request <= input TPM
microbatch请求数 * 本角色max_output_tokens <= output TPM
```

若一个批次本身装不进一分钟窗口，直接报microbatch_exceeds_quota_window，不能靠无限等待解决。
已预留窗口不足则有界等待。相同服务/凭证不能为各角色重复分配完整账户配额；
input_tokens_per_request要覆盖图片、提示词和被审核长候选，不能只估问题文本。
收到实际输入usage超出估算会停止并要求校准，不能继续声称TPM受控。

--limit N只控制准备多少题，不会自动增加预算。普通通过题通常3次请求；
有短长两条都审核的题最多5次。保守预估N题时可用5N次请求和
`N*(q_max+short_max+2*judge_max+long_max)`输出预留作容量上界，再按实际失败/旁路比例校准。
它们不是合格产量或速度承诺。

先1题检查协议，再固定小样本统计接受率、升级率、严重错误与每条合格成本。
并发、批次、像素和token上限分开调整；不要因为API进度条在动就同时放大所有参数。

## 7. 分步运行及输入冻结

`run`可以自动prepare；单独prepare适合先验收输入、不请求教师：

```bash
python run_image_cot.py prepare -c /private/config/pilot.py --limit 100
```

返回`prepared/INPUT_DIGEST/input_manifest.json`。它绑定源说明、标注hash、分片hash、数量和准备版本。
源标注按清单读取，媒体逐条校验，输出不覆盖原始文件。不是把整张媒体目录搬进工作根。
--limit限制处理题数和图片数，仍会对配置列出的标注文件先计算完整摘要；
一题pilot应只列需要的源，避免不必要的大标注IO。

有两种运行方式：

- 保持input_manifest=None，`run --limit 100`会准备相同选择后运行，已有合法prepare可复用。
- 在新私有配置中将input_manifest设为刚返回的完整路径，再`run`，此时不能再带--limit。

后一方式固定的是该次选择，不表示每次运行都重新读最新原始标注。
改变限制、源政策或源文件会形成新的准备身份；同一source身份重复列两次会拒绝，避免重复曝光。

```bash
python run_image_cot.py run -c /private/config/frozen_pilot.py --backend http
```

运行结束时记录实际run路径、job_id、代码commit、backend和终态计数。
退出0表示本阶段完成，不代表数据已经符合生产质量门槛；退出4表示还有pending/review。
未完成的请求不会因为run结束而消失，结果不会依靠一个领先于落盘的索引推进。

prepare接受CLI的--backend参数，但准备函数本身不派发模型请求；
不要把`prepare --backend http`理解成一次真实服务验证。

## 8. 工作目录怎么读

```text
private_work_root/
├── prepared/INPUT_DIGEST/
│   ├── input_manifest.json
│   ├── sources-000000.jsonl
│   └── owner.lock
└── runs/JOB_ID/
    ├── run.json
    ├── owner.lock
    ├── parts/input-000000/STAGE/PART_ID/
    │   ├── plan.json
    │   ├── journal-000000.jsonl
    │   ├── results-SHA256.jsonl
    │   └── commit.json
    ├── selection/input-000000.jsonl
    ├── report.json
    ├── reports/execution-UUID.json
    └── export_staging/RELEASE_ID/
        ├── data.jsonl
        ├── provenance.jsonl
        ├── README.md
        └── manifest.json
```

STAGE为question_check、short、short_verify、long、long_verify中实际执行的阶段。
未执行long时，没有long目录是正常情况。发生半行/恢复可多出日志段，异常回执有quarantine文件。

| 文件 | 查看重点 |
|---|---|
| run.json | 实际配置快照、代码commit/code_digest/clean状态、backend、输入manifest摘要 |
| plan.json | 本part预期task/attempt集合、recipe、上游commit引用 |
| journal | 请求intent与原始receipt，一份权威付费/执行证据；不是公开训练数据 |
| results | 结构化结果与原因；是否已提交必须看commit |
| commit.json | 文件hash、数量、父提交和结果覆盖；唯一part完成依据 |
| selection | 每题终态、选中candidate/text_hash/审核引用；失败件通常无candidate_id |
| report.json | 派生构建摘要，可在完整run后读取 |
| execution观察 | 本次执行耗时/缓存/新预留数；不是整个job的独立吞吐基准 |

中断可能尚无report.json。仅列自己的work/runs一级目录，打开run.json确认任务身份，
再恢复对应job；不要为了找任务递归扫描媒体根，也不要随手删owner.lock。
lock文件存在不等于进程仍活跃；独占性由操作系统锁保证，不按文件时间自动抢锁。

## 9. 六种终态及计数

| terminal | 含义 | 默认能进入CoT训练JSON吗 |
|---|---|---|
| accepted_cot | 有通过本地核验与内容审核的候选，并完成选择 | 可以导出预览，正式发布另过门禁 |
| bypass_verified | 保留可信OCR/框原标注，未重写 | 不进入本扩展CoT文件，保留selection/源快照 |
| input_rejected | 输入不支持、媒体无效或图题检查明确失败 | 否 |
| candidate_rejected | 短/长候选均未满足内容门槛 | 否 |
| needs_review | 弱/歧义参考、不确定内容或待裁定 | 否 |
| technical_pending | 请求/格式/预算等技术问题，不能当成难题 | 否 |

六种终态按输入题目对账，不把图片数、模型请求数或多候选数量混成合格题数。
参考等级不足可在不发送教师前进入review；OCR旁路也仍需要有效输入和通过图题检查。

`report.quality_verified`、`throughput_verified`、`student_training_verified`在构建报告中固定为false。
它们不是发布后自动改真的状态；正式证据在外部审计/probe和export manifest中。
`report`命令只读摘要，不等于重新验证每个提交文件。

<a id="recovery"></a>
## 10. 中断、恢复、重审与版本升级

### 10.1 恢复同一个job

```bash
python run_image_cot.py resume --run /private/work/runs/JOB_ID
```

恢复要求完整implementation身份相同：commit、实现代码摘要、clean标志都必须一致。
仅改README后提交也会改变commit；不要据“只是文档”假定旧job可以直接在新HEAD恢复。
工作区干净与否也影响身份，私有配置/输出应放在仓库外。

恢复使用run.json的配置快照，但prompt和review_resolutions仍从记录的绝对路径读取。
HTTP后端的resume仍做角色配置、key环境变量和配额预检，即使有完整回执也不能假定无需原环境。
只想离线查看已有结果可用report/preview；它们不要求教师key。
**保留原checkout与原文件字节；不要在旧路径直接改prompt或裁定文件。**
仅新建一个旧commit的worktree，也不会自动改写冻结配置中的绝对路径。
路径和内容若已变，先恢复可核实的运行环境，不手改run.json/commit/hash来绕过检查。

| 证据状态 | 恢复动作 |
|---|---|
| 无durable intent | 仍可能在预算内发送这个从未派发的任务 |
| intent有、receipt无 | 标为未决，不自动发送相同请求 |
| receipt有、未commit | 本地解析/应用结果，不重发已完成生成 |
| 合法commit | 复用并验证引用 |
| journal尾部半行 | 保留原段，在新段继续；不据此推断供应商没执行 |
| commit缺文件/坏hash | 停止，保留现场，不用重跑覆盖证据 |

所以“未决不重发”不等于resume完全零API。后续从未发送的有效任务仍可能产生请求。
timeout、429、未知5xx无法证明未计费，当前保守停在technical_pending。
没有--force-resend或自动查账功能；先查供应商记录并保留裁定，再明确规划后续任务。

### 10.2 换审核，复用生成

复制一份新配置，不修改旧run。绑定原输入manifest、原来源/教师/生成prompt和显式复用来源：

```python
image_cot["input_manifest"] = "/private/work/prepared/INPUT_DIGEST/input_manifest.json"
image_cot["reuse_from"] = ["/private/work/runs/OLD_JOB_ID"]
image_cot["models"]["joint_verify"]["request_revision"] = "audit-2"
```

```bash
python run_image_cot.py check -c /private/config/rejudge.py
python run_image_cot.py run -c /private/config/rejudge.py
```

新run可以复用签名相同的图题/生成回执，只为新审核付费；不是覆盖旧判断。
无需为了重审无意义改prompt正文。若原问题卡在question_check的技术格式错误，
应更改question_check的revision；改joint_verify不会修复尚未通过图题检查的题。

复用不是全局自动cache：只看显式reuse_from中、对应分片/阶段的合法提交。
改变分片布局可能减少可复用命中。更换源、教师/端点、生成预算、预处理或代码可能使生成失效；
只改鉴权变量名不改变教师语义身份。模型配置变动前先确认哪些阶段会重新请求。
新run有自己的预算账本，不能代替账户历史账单汇总。

### 10.3 增加预算或升级代码

增加请求/token上限属于新配置，通常是新job，不能用旧resume命令偷偷改上限。
用新配置显式reuse_from复用旧的完整成果；旧pending仍保留。
代码升级采取保守失效，不承诺跨代码版本复用生成。旧job保留原环境，升级在新checkout完成。
只改训练输出映射/名称时直接export -c，不必run一遍。

<a id="review"></a>
## 11. 人工复核与裁定文件

先按selection和results的原因取样，定位对应生成原文、图片和联合审核。
accepted件的selection提供receipt_ref；review/拒绝件通常没有这个快捷引用。
先用selection的question_id/source_part定位本输入分片，在对应short/short_verify等results里取
candidate_id、task_id和reason，再找同part journal中对应task的receipt。
只检查该job的已知分片，不对整个数据池递归搜长链文本；selection.input_reason为空也不表示通过。
不要只看最终答案。重点查不可读图值、视觉专名、单位、公式、漏小问以及已经被修正的探索。
通过件也要随机抽检，不能只看模型自报uncertain的题。

每条裁定绑定实际候选和审核版本。以下是结构示意，不是可直接使用的真实审计证据：

candidate_text_hash和judge_signature从实际结果原样取，不能把普通文件SHA或自己计算的
裸文本SHA混作这些身份；它们使用本项目的规范序列化与签名约定。

```json
{"schema_version":"image_cot.review_resolution/1","candidate_id":"REPLACE_ACTUAL_CANDIDATE_ID","candidate_text_hash":"REPLACE_ACTUAL_TEXT_HASH","judge_signature":"REPLACE_ACTUAL_JUDGE_SIGNATURE","reviewer":"actual-reviewer","timestamp":"2026-11-05T12:00:00+08:00","evidence":"填写基于图片与问题的实际裁定依据","outcome":"pass"}
```

保存为新的私有JSONL，最多4096条且candidate_id不能重复。不要在原裁定文件原地追加后继续旧resume。
新配置指定review_resolutions和reuse_from，再运行形成新派生选择。

```python
image_cot["review_resolutions"] = "/private/audit/review-v1.jsonl"
image_cot["reuse_from"] = ["/private/work/runs/OLD_JOB_ID"]
```

裁定可以处理有效内容审核的pass/fail/uncertain；不能把技术error、截断响应或确定错误的
数值/选项核验改为pass。弱源参考可能还没有候选，应先完成受审源政策/来源版本，
不能给不存在的链伪造candidate_id。

<a id="export"></a>
## 12. 预览与训练字段映射

```bash
python run_image_cot.py export --run /private/work/runs/JOB_ID --preview
```

只使用accepted_cot并重新核验生成/审核引用、正文hash和本地终答。
旁路件计入bypass_count_separate，不写进data.jsonl；有pending的job也可做私有preview。
提供有效export配置时，零条合格允许空预览文件，但不能把“文件生成成功”当作有了训练数据。

| 预览文件 | 用途 |
|---|---|
| data.jsonl | 供检查的训练格式；gpt.value为通过审核的完整规范正文 |
| provenance.jsonl | id→selection/source原行号与摘要，含提交/审核引用 |
| README.md | PREVIEW标记、字段概要与处理信息；不是权威最终README |
| manifest.json | count、hash、selection_digest、preview/published状态 |

导出可单独用私有配置，只需image_cot.export，不必携带教师参数：

```python
image_cot = {"export": {
    "dataset_name": "my_image_cot",
    "training_media_root": "/mnt/sfs/unimoe_dataset/training",
    "original_field_mapping": {
        "id": "original_id", "conversations": "original_conversations",
        "images": "original_images", "modal_mask": "original_modal_mask",
        "videos": "original_videos", "audios": "original_audios"
    }
}}
```

```bash
python run_image_cot.py export --run /private/work/runs/JOB_ID -c /private/config/export.py --preview
```

dataset_name限字母、数字、下划线、点和连字符，输出id=`dataset_name_candidate_id`。
images仍相对training_media_root，逐条检查原始SHA。该路径下需要已经有对应图片，
程序不替你搬图、裁图、建立新媒体布局或替换原文件。

输入所有原始顶层字段都保留。与id/conversations/images等目标字段冲突时必须显式映射；
映射目标不能再撞其他原字段，也不能是禁止的token统计或_cot字段。
源自带特殊字段要先登记可逆映射，不能简单删掉。内部回执/成本不插进训练JSON。

只包含图片时modal_mask=1，不生成音视频时长。正式交付的JSON、媒体和README必须遵守
项目当前权威格式文档；本项目的preview schema检查不替代完整规范核验。

short可以有空think，简洁解释在标签外；long保留非空推理。原生reasoning与内联think
并存、残缺/重复块不自动拼接修复。学生训练器必须实际验证标签/原生reasoning编码、
图像顺序和loss mask，不能从教师API能否关闭thinking推断训练行为。

<a id="release"></a>
## 13. 正式发布：准备三类真实证据

### 13.1 先冻结选择集合

从实际report.json取job_id、selection_digest、accepted_cot计数。
正式发布要求非mock、构建时绑定清洁commit、status=completed且无needs_review/technical_pending。
被明确拒绝的输入/候选可以留在覆盖统计中，不必把所有拒绝件改为通过。
partial_release=true当前不支持；不能绕过未决覆盖范围。

选择集合、审核/编码政策或输出字段变了，旧证据不能无条件沿用。
report中的固定false标志不会因外部证据生成而回写，它不是生产门禁唯一状态来源。

### 13.2 人审质量报告

以下是**待填写结构**，故意为pending/real_execution=false，不可用于发布：

```json
{
  "schema_version": "image_cot.quality_gate/1",
  "job_id": "REPLACE_FROM_REPORT",
  "selection_digest": "REPLACE_FROM_REPORT",
  "status": "pending",
  "real_execution": false,
  "human_audit": {
    "independent_groups": 0,
    "severe_errors": 0,
    "one_sided_confidence": 0.95,
    "reviewer": "REPLACE_ACTUAL_REVIEWER",
    "sampling_plan": "REPLACE_ACTUAL_PLAN_AND_EVIDENCE",
    "representative_strata": [],
    "groups_independent": false
  }
}
```

本版接受至少300个独立题组零严重错误，对应一侧95%精确上界
`1-0.05**(1/n) <= 0.01`，且n不能大于accepted_cot数。当前不处理非零错误下的发布门槛计算。
按图片/题组考虑聚类，不把同图多个回答冒充独立样本；按source、题型、短/长链分层抽取。
300题零错误只支持该抽样总体的区间结论，不保证整个数据集永远无错。

至少记录关键图值/实体、关系/公式、结论和指令错误。还要报告误拒绝和覆盖，
否则只提高通过件的表面精度可能丢掉大量有价值题目。
候选自评和同一教师自审不能替代人审；字段签名一致也不能证明审计真的执行过。

### 13.3 真实学生训练probe

本仓库提供TrainingProbe协议与证据门禁，不包含已跑通的DeepSeek/GLM训练器。
需在目标模型的真实tokenizer/processor/trainer上对固定代表样本检查并运行一次小批forward/backward。
尤其检查默认drop_thinking、推理/答案loss mask、图像顺序和最终答案截断。

```json
{
  "schema_version": "image_cot.training_probe/1",
  "job_id": "REPLACE_FROM_REPORT",
  "selection_digest": "REPLACE_FROM_REPORT",
  "status": "pending",
  "real_execution": false,
  "student": {
    "profile": "deepseek_v41_flash",
    "model": "REPLACE_MODEL",
    "model_revision": "REPLACE_REVISION",
    "tokenizer_revision": "REPLACE_REVISION",
    "trainer": "REPLACE_TRAINER",
    "trainer_revision": "REPLACE_REVISION"
  },
  "sample_digest": "REPLACE_ACTUAL_PROBE_SAMPLE_DIGEST",
  "checks": {
    "image_order": false,
    "bos_eos": false,
    "reasoning_loss": false,
    "answer_loss": false,
    "no_final_truncation": false,
    "no_duplicate_think": false,
    "actual_forward_backward": false
  }
}
```

student.profile必须与export.student_profile一致；所有检查都应来自实际结果。
仅解析template、伪造mock训练报告或手改布尔值，不构成通过证据。

### 13.4 项目格式与权威README验证

先用最终dataset_name、字段映射和training_media_root生成preview，取得实际data.jsonl摘要。
然后完整读取项目当前data_format.md和实际README模板，完成最终release_readme。
验证实际JSON、每条媒体路径、原始字段、禁止字段、占位符、README及路径对应。
不要把本程序自动生成的preview README直接改一个标题就当成模板合规。

```json
{
  "schema_version": "image_cot.delivery_validation/1",
  "status": "pending",
  "real_execution": false,
  "selection_digest": "REPLACE_FROM_REPORT",
  "data_sha256": "REPLACE_ACTUAL_EXPORTED_DATA_SHA256",
  "readme_sha256": "REPLACE_COMPLETED_README_SHA256",
  "readme_template_sha256": "REPLACE_AUTHORITATIVE_TEMPLATE_SHA256",
  "format_spec_sha256": "REPLACE_CURRENT_FORMAT_SPEC_SHA256",
  "training_media_root": "/actual/resolved/training/root",
  "destinations": {
    "output_jsonl": "/actual/json/dataset.jsonl",
    "output_readme": "/actual/readmes/dataset.md",
    "output_provenance": "/actual/audit/provenance.jsonl",
    "output_manifest": "/actual/audit/manifest.json"
  },
  "checks": {
    "actual_json_checked": false,
    "media_paths_checked": false,
    "authoritative_readme_template_checked": false
  }
}
```

上述pending结构同样不能发布。程序会核验文件SHA、身份和检查声明，
外部验证流程必须真正对照权威规范；这不是自动证明所有格式要求的通用验证器。

training_media_root在报告中须为本机`str(Path(root).resolve())`的结果；
destinations与export配置四个目的路径按字段值精确对应，建议一开始就用无`..`的规范绝对路径。
Windows与Linux分隔符不同，不能不经核对直接复用另一台机器的路径声明。

若采用本项目的UniMoE交付约定，JSON位于omni/json、媒体相对training、README位于omni/readmes；
这些是规范中的目的布局，本轮没有核实你的服务器实际目录。
标题、完成日期/维护人、中文章节、原字段表和处理/路径说明，均以当前权威模板逐项核验。
仅图片没有音视频时，不制造video_duration_sec/audio_duration_sec。

### 13.5 组合export配置并发布

保存独立私有export.py，例如：

```python
image_cot = {"export": {
    "dataset_name": "my_image_cot",
    "training_media_root": "/mnt/sfs/unimoe_dataset/training",
    "original_field_mapping": {
        "id": "original_id", "conversations": "original_conversations",
        "images": "original_images", "modal_mask": "original_modal_mask",
        "videos": "original_videos", "audios": "original_audios"
    },
    "student_profile": "deepseek_v41_flash",
    "quality_gate": "/private/audit/quality_gate.json",
    "training_probe": "/private/audit/training_probe.json",
    "data_format_path": "/private/contracts/data_format.md",
    "readme_template": "/private/contracts/_TEMPLATE.md",
    "release_readme": "/private/audit/my_image_cot.md",
    "delivery_validation": "/private/audit/delivery_validation.json",
    "completion_date": "2026-11-05",
    "maintainer": "REPLACE_ACTUAL_MAINTAINER",
    "output_jsonl": "/mnt/sfs/unimoe_dataset/omni/json/my_image_cot.jsonl",
    "output_readme": "/mnt/sfs/unimoe_dataset/omni/readmes/my_image_cot.md",
    "output_provenance": "/private/releases/my_image_cot/r1/provenance.jsonl",
    "output_manifest": "/private/releases/my_image_cot/r1/manifest.json"
}}
```

路径和日期为示例，须替换。审计侧车放在批准的非训练扫描目录，
避免训练加载器把provenance.jsonl误当成另一份训练数据。
这些证据路径可以在构建结束后通过独立export配置提供，无需重跑教师。

```bash
python run_image_cot.py export --run /private/work/runs/JOB_ID -c /private/config/export.py --stage-release
python run_image_cot.py export --run /private/work/runs/JOB_ID -c /private/config/export.py --publish
```

stage-release写私有staging，manifest仍published=false；publish才复制派生JSONL、README和侧车到目的路径。
原图不移动。现有相同内容校验复用，不同内容拒绝覆盖。
个别文件先出现、manifest最后出现，**不是跨目录多文件事务**；
只有最终manifest及其hash全部通过后才登记/混入训练，不能让glob扫描器提前消费部分发布。

目标文件系统需支持同目录临时文件hard-link原子无覆盖发布，Linux/目标SFS仍需实际故障验证。
不支持时停止，不降级为覆盖。失败后保留工作根；匹配的完整成果可以在重跑发布时复用，
不要删除已有正式文件去强行让新内容占用旧路径。

## 14. 看懂日志、报告和速率

| 字段/观察 | 正确解释 |
|---|---|
| requests_reserved | 当前job累计请求预留，不等于供应商实际计费成功数 |
| output_tokens_accounted | 有usage取completion_tokens；未知保留输出预留 |
| monetary_cost | 当前为unknown，没有价格/账单硬上限实现 |
| new_requests_reserved | 本次执行新增预留，复用/恢复可能为0 |
| committed_stage_cache_hits | 提交结果复用，不是模型生成速度 |
| phase_timings_s | decode、model批次、intent/receipt fsync、part commit等聚合 |
| peak_rss_bytes | 可用平台的峰值RSS，Windows可能null |
| API进度条ok | 底层交付了回执，不等于HTTP/业务/CoT合格 |

用合格且人审认可的独立题数/实际端到端小时衡量产量，报告升级率、接受率、
误拒绝、严重错误区间、调用数与单条合格成本。
不要拿pytest耗时、mock吞吐、缓存resume速度或单API tokens/s当生产端到端收益。

当前跨阶段仍重复读图/PNG编码，连接池按microbatch创建。固定分片已避免累计重写前缀，
代码签名每run计算，日志/结果缓存按分片清空；更进一步的LMDB/会话/调度优化应由实际profile决定。

## 15. 常用参数与退出码

| 子命令 | 参数 |
|---|---|
| check | -c/--config必填，--backend http或mock，默认http |
| prepare | 同上，额外--limit；准备过程不请求教师 |
| run | 同上，--limit只在没有input_manifest时使用 |
| resume | --run必填，无--force、--limit或新配置参数 |
| report | --run必填 |
| export | --run必填，-c可选；--preview/--stage-release/--publish三选一 |

| 退出码 | 当前CLI含义 |
|---|---|
| 0 | 本阶段成功；不是生产质量保证 |
| 2 | argparse错误，或非export阶段的CoTError，包括配置/契约类失败 |
| 3 | export阶段的CoTError，例如门禁、身份或字段映射失败 |
| 4 | 构建完成但有technical_pending或needs_review |
| 5 | CLI捕获的OSError/KeyError/TypeError/ValueError等IO/配置异常 |
| 130 | Ctrl+C中断，保留证据后恢复 |

其它未被CLI捕获的Python异常可能打印堆栈，不应一律按上述某个业务状态解释。

<a id="troubleshooting"></a>
## 16. 故障排查

| 错误/现象 | 先检查 | 操作 |
|---|---|---|
| positive_integer_required | None预算、0或错误类型 | 填明确上限，不继承上游8192并发默认值 |
| connection_required | 从mock切到http但未绑连接 | 设置四角色connection_ref与mode=pilot |
| api_key_env_missing | 当前进程没有指定变量 | 在该进程环境注入key；不要输出key |
| duplicate_api_prefix | endpoint已带/v1 | 改为根endpoint，或显式空prefix |
| work_root_overlaps_code/input | 工作根包含代码或源路径 | 选择并列私有根，不移动/删除原图解决 |
| malformed_source_json_line | 源文件不是合法逐行JSON | 创建受审适配后的新源文件，不覆盖原标注 |
| single_turn/image_required、unsupported schema | 多轮、多图、Honey data/messages | 一期不支持，不能只改字段名猜语义 |
| original_field_mapping_conflict | 源字段撞目标/其它映射 | 完整登记可逆映射，再单独export |
| unit_mismatch / ambiguous_final_extraction | 单位、参考、收束格式 | 核对源政策和候选，不用宽松substring放行 |
| question_check/judge_schema_error | 非JSON、缺键、非法枚举、截断 | 看私有回执，修审核配方并显式重审；不重生成全部题 |
| technical_pending，429/503/超时 | 执行与计费状态未知 | 不盲重跑，查provider记录，保留原证据 |
| microbatch_exceeds_quota_window | 批次最坏token/RPM装不进窗口 | 降microbatch或校准确认后的配额，不乱调并发 |
| input_token_estimate_not_conservative... | 实际prompt_tokens超预留 | 停止，重估图片/候选/提示词开销 |
| resume_requires_original_code_identity | commit/hash/clean状态不同 | 回到可核实的原冻结环境，不改manifest伪造一致 |
| report.json不存在 | run尚未完成到摘要阶段 | 核对该run.json、parts现场后恢复 |
| commit_integrity/parent/hash错误 | 文件缺失或被改 | 停止消费、保留现场，不自动覆盖重算 |
| mock_cannot_publish | 原run是mock | 只能preview，真实合成需另建真实run |
| release_gate_identity_or_status | 报告缺失、pending、签名错 | 执行实际验证，不手改true绕门禁 |
| human_quality_gate_insufficient | 不足300独立组或有严重错误 | 校准并补实际证据；当前版本不接受其它统计口径 |
| delivery_path_correspondence | 报告路径与配置不一致 | 核对resolved媒体根和四个目标字面值 |
| release_target_conflict | 目标已有不同内容 | 用新release命名，不删除正式数据强行覆盖 |
| link/fsync不支持 | 目标文件系统语义不同 | 做目标环境验证，不降级覆盖发布 |

先读具体私有原因和回执，不把所有拒绝都当成模型能力不足。
文件系统磁盘满时先停止发送后续请求，完整回执已落盘则可本地恢复；
不要把“API已返回”误认为“所有回执已经durable”。

<a id="doc-validation"></a>
## 17. 文档示例验证与日常维护

静态检查本README、手册、实现指南及验证文档中的本地链接/锚点、JSON和Python代码块：

```bash
python tools/check_image_cot_docs.py
```

用一个全新、仓库外、不会覆盖任何现有内容的目录执行手册mock示例：

```bash
python tools/check_image_cot_docs.py --smoke-root /private/cot-doc-smoke-new
```

检查器只提取本手册标记的合成输入/私有配置，生成小图后调用CLI的mock模式，
验证check、prepare、run、report、resume、preview和mock禁止正式发布。
它不发外部HTTP、不接真实key、不证明质量/速度/完整模板合规。
输出和简要报告在指定私有目录，目录已有时拒绝覆盖。

维护文档时先核对CLI与contracts/runner/export实际行为，再改命令和字段。
只改文档无需重跑整个模型/数据流程；应运行文档检查及有意义的mock示例，
保留旧的完整实现验收记录。Git提交只放代码、公共模板、文档和合成测试，不放真实数据/日志/回执。

进一步阅读：[项目首页](../README.md)、[实现与取舍](image_cot_implementation.md)、
[本地验证范围](image_cot_validation_2026-11-05.md)。
