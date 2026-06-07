# 阶段六执行计划 - 线上观测与 Bad Case 闭环

## 1. 一句话定位

阶段六负责把阶段五接回业务后的回答效果变成可观测、可复盘、可沉淀的数据闭环。

```text
阶段五解决“能不能切到 SFT”。
阶段六解决“切上去以后好不好、哪里不好、如何回归优化”。
```

阶段六默认不改变回答结果，不调整检索策略，不自动训练模型，只记录线上 trace 和可回归指标。

## 2. 背景与目标

### 2.1 背景

Base / SFT 接入后，需要持续回答以下问题：

```text
1. SFT 是否让拒答、引用和忠实性持续改善？
2. 是否出现误拒、慢回答、引用异常、无上下文硬答？
3. 哪些真实问题应该进入下一轮 holdout / golden set？
4. 出问题时能否按 provider、model、prompt、上下文快速定位？
```

### 2.2 阶段六目标

```text
1. 新增默认关闭的回答 trace 记录能力；
2. 记录 provider、model、延迟、上下文数量、引用、拒答等字段；
3. 默认不记录完整问题和答案，避免泄露业务内容；
4. 支持通过环境变量打开完整文本记录用于本地排查；
5. 提供本地检查脚本和单测；
6. 为后续 bad case 入库、RAGAS 回归、线上看板预留字段。
```

### 2.3 非目标

```text
1. 不接真实监控系统；
2. 不新增数据库表；
3. 不做自动标注；
4. 不自动触发训练；
5. 不改变回答 prompt；
6. 不替代阶段四离线评估。
```

## 3. 总体架构

```text
Query Processor
  -> Retriever / Reranker
  -> AnswerOutPutNode
       -> Base / SFT Answer LLM
       -> AnswerTraceRecorder
            -> fine_tuning/data/online/answer_traces.jsonl
            -> bad case mining / golden set update
            -> next stage eval / retrain
```

## 4. 核心模块

| 模块 | 职责 | 输入 | 输出 |
|---|---|---|---|
| `answer_trace.py` | 构造并写入回答 trace | state / latency | JSONL trace |
| `AnswerOutPutNode` | 回答完成后调用 trace | answer state | trace side effect |
| `check_stage6_observability.py` | 检查 trace 配置和 sample 写入 | env / args | 检查结果 |
| `test_stage6_answer_trace.py` | 验证脱敏、字段和写入逻辑 | fake state | 测试结果 |

## 5. 配置设计

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `ANSWER_TRACE_ENABLED` | `false` | 是否记录回答 trace |
| `ANSWER_TRACE_PATH` | `fine_tuning/data/online/answer_traces.jsonl` | trace 输出路径 |
| `ANSWER_TRACE_INCLUDE_TEXT` | `false` | 是否记录 query / answer 原文 |
| `ANSWER_TRACE_INCLUDE_CONTEXT` | `false` | 是否记录上下文正文摘要 |
| `ANSWER_TRACE_MAX_TEXT_CHARS` | `500` | 文本截断长度 |

## 6. Trace Schema

默认脱敏记录：

```json
{
  "ts": "2026-06-07T12:00:00+08:00",
  "task_id": "abc123",
  "provider": "sft",
  "model_name": "kb-sft",
  "is_stream": false,
  "latency_ms": 1200,
  "query_hash": "sha256:...",
  "answer_hash": "sha256:...",
  "query_chars": 18,
  "answer_chars": 120,
  "prompt_chars": 2400,
  "context_count": 4,
  "used_citations": ["1", "2"],
  "is_refusal": false,
  "has_answer": true,
  "error": ""
}
```

开启原文记录后额外写入：

```json
{
  "query": "RS-12 如何测量电阻？",
  "answer_preview": "根据资料..."
}
```

## 7. Bad Case 闭环

```text
线上 trace
  -> 按规则筛选 bad case
       -> 无上下文但非拒答
       -> 有上下文但无引用
       -> SFT 误拒疑似
       -> 延迟过高
       -> 用户反馈低分
  -> 人工复核
  -> 加入 holdout / golden set
  -> 阶段四回归评估
  -> 阶段二/三再造数再训练
```

## 8. 验收命令

默认关闭检查：

```bash
uv run python fine_tuning/scripts/check_stage6_observability.py --check-only
```

写入一条 sample trace：

```bash
ANSWER_TRACE_ENABLED=true \
ANSWER_TRACE_PATH=/private/tmp/stage6_answer_traces.jsonl \
uv run python fine_tuning/scripts/check_stage6_observability.py --write-sample
```

单测：

```bash
uv run python fine_tuning/tests/test_stage6_answer_trace.py
```

## 9. 阶段六通过标准

```text
1. 默认 ANSWER_TRACE_ENABLED=false 时不写 trace；
2. 开启后能写 JSONL；
3. 默认不记录 query / answer 原文；
4. trace 包含 provider / model / latency / citation / refusal；
5. trace 产物不进入 Git；
6. AnswerOutPutNode 语法检查通过；
7. trace 失败不影响正常回答。
```

## 10. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 记录原文泄露敏感信息 | 数据安全风险 | 默认关闭原文，按需临时打开 |
| 写 trace 异常影响回答 | 线上故障 | trace 捕获异常，不向上抛 |
| trace 文件过大 | 磁盘增长 | 后续接日志轮转或对象存储 |
| 指标误判 | 错误优化方向 | 规则筛选后人工复核 |
| 线上问题未入评估集 | 离线好线上差 | 定期抽样 trace 加入 holdout |
