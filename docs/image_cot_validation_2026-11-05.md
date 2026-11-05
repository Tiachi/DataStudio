# 图片 CoT 扩展本地验证记录

日期：2026-11-05，Asia/Shanghai。仓库：Tiachi/DataStudio；分支：codex/image-cot-v1。
上游基线：676bcb8d5494d65c8fb85185ac26d98da6c92842。

状态：本地实现验证通过；没有调用付费API、没有训练、没有发布真实训练数据。

## 验证对象与环境

Python 3.13.9 / Windows；pytest 8.4.2、pytest-asyncio 1.4.1.post1。
独立虚拟环境安装上游依赖与requirements-image-cot-dev.txt，未修改全局环境。

实现代码摘要（image_cot模块与CLI的SHA256组合，计算方式见implementation_identity）：

```text
0d9dab8557e1f4abba6f83502315b0cb82f5a18fc8df2b10160e20abc7f38c95
```

测试时基线HEAD未提交feature，工作树含本轮代码；最终提交的这些代码字节摘要相同。
新增路径固定LF，防止Windows/Linux的换行转换无意义地改变任务/代码身份。
这是源码身份记录，不代表运行过真实教师或学生。

## 实际执行结果

| 检查 | 结果 |
|---|---|
| 扩展与上游测试同一进程 | 523 passed，19.66秒；1条上游RuntimeWarning |
| 其中image_cot测试 | 56 passed |
| 原算子/模型/Pipeline/schema/registry | 467 passed |
| Ruff，全部新增Python文件 | PASS |
| compileall | PASS |
| Git whitespace check | 提交前执行，PASS后交付 |

合并运行范围：tests/image_cot、tests/operators、tests/models、tests/pipelines、
tests/datasets/test_schema_converter.py、tests/utils/test_registry.py。
不声称运行了仓库内所有测试。耗时只是合成测试耗时，不能换算真实产线吞吐。

上游RuntimeWarning来自test_process_single_preprocess_failure中的AsyncMock未await；
对应原测试和OpenAIAPI没有为消除warning而修改。新增HTTP集成测试实际走aiohttp和父类异步引擎。

## 关键覆盖

- 缺键、bool/非法枚举、重复JSON key、NaN、外围代码围栏、自相矛盾判定不放行。
- 生成payload不含参考答案；审核有实际system角色，并收到原题/变换题/参考等级。
- 原生reasoning保留；截断、空正文、重复/残缺think、重复编码不伪装修复。
- 终答正确但关键过程/视觉审核失败，短链与长链均失败时不产生正向SFT。
- 400/429/503、缺响应不触发难题升级或隐式重发；乱序按ID回连，异常回执隔离。
- intent/receipt/results/commit前后8个边界的故障注入；单owner、半行、坏hash和恢复。
- 无durable receipt不能commit pass；日志追加不按flush次数累计重读/重写输出前缀。
- durable生成回执后中断可以本地恢复；换judge显式复用生成，只新增审核请求。
- OCR旁路独立计数、不改写；坐标任务需登记格式和合法边界。
- 预算停止、凭证作用域/连接配置、导出字段映射、mock拒绝发布、质量/学生门禁身份。
- CLI子进程检查、生成、独立导出配置；无覆盖发布与相同内容复用。
- 使用合成门禁替身的文件发布协议测试只检查IO及README字节/验证hash绑定，
  不构成真实人审、学生训练或用户权威模板合规证据。

## 未关闭的运行门禁

1. 用户实际IP教师与DeepSeek官方端点图文探针、配额/usage校准。
2. 代表性随机人审、严重错误率区间、误拒绝与覆盖。
3. Linux及目标SFS的fsync/rename/link、故障恢复和真实IO/峰值内存。
4. 真实目标学生/训练器的编码、loss mask与一次forward/backward。
5. 目标JSON/媒体/完整README按当前data_format及真实权威模板逐项验证。

操作说明与限制见[实现指南](image_cot_implementation.md)。
