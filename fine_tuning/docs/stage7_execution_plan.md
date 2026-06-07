# 阶段七执行计划 - Bad Case 挖掘与 Golden Set 候选

## 1. 一句话定位

阶段七负责把阶段六记录的线上 answer trace，转成可复盘的 bad case 和可进入回归评估的 Golden Set 候选。

```text
阶段六解决“记录什么”。
阶段七解决“哪些记录值得复盘，如何进入下一轮评估集”。
```

阶段七是离线数据治理，不改线上回答链路，不调用模型，不自动训练。

## 2. 背景与目标

### 2.1 背景

阶段六已经可以记录：

```text
provider / model_name / latency_ms / context_count
used_citations / is_refusal / query_hash / answer_hash
```

这些字段可以先做规则化筛选，找出最值得人工复核的问题：

```text
1. 没有上下文却没有拒答；
2. 有上下文却没有引用；
3. 空回答或模型错误；
4. SFT 疑似过度拒答；
5. 延迟明显过高；
6. prompt / answer 长度异常。
```

### 2.2 阶段七目标

```text
1. 提供离线 bad case 挖掘脚本；
2. 输出 bad_cases.jsonl；
3. 输出 golden_candidates.jsonl；
4. 输出 Markdown 汇总报告；
5. 支持阈值配置，如 latency、answer length；
6. 保持默认脱敏，不要求 trace 里有问题/答案原文；
7. 为阶段四回归评估和下一轮造数提供候选集。
```

### 2.3 非目标

```text
1. 不做 LLM judge；
2. 不自动写入训练集；
3. 不自动修改 holdout；
4. 不改线上查询链路；
5. 不处理用户反馈 API；
6. 不提交线上 trace 和挖掘产物。
```

## 3. 总体架构

```text
fine_tuning/data/online/answer_traces.jsonl
  -> mine_stage7_bad_cases.py
       -> 规则分类
       -> 严重度排序
       -> Golden Set 候选筛选
  -> fine_tuning/data/online/bad_cases.jsonl
  -> fine_tuning/data/online/golden_candidates.jsonl
  -> fine_tuning/data/online/_stage7_bad_case_report.md
```

## 4. Bad Case 规则

| 类型 | 触发规则 | 严重度 | 典型根因 |
|---|---|---:|---|
| `empty_answer` | `has_answer=false` 或 `answer_chars=0` | 3 | 模型调用失败 |
| `model_error` | `error` 非空 | 3 | 模型服务异常 |
| `no_context_answered` | `context_count=0` 且非拒答 | 3 | 无召回仍硬答 |
| `missing_citation` | `context_count>0`、非拒答、无引用 | 2 | 引用约束失败 |
| `sft_over_refusal_candidate` | `provider=sft` 且 `is_refusal=true` 且有上下文 | 2 | SFT 过度拒答疑似 |
| `high_latency` | `latency_ms` 超阈值 | 2 | 模型服务慢 / prompt 过长 |
| `long_prompt` | `prompt_chars` 超阈值 | 1 | 上下文过长 |
| `short_answer` | 非拒答且答案太短 | 1 | 回答不完整 |

一条 trace 可以命中多个规则，脚本保留所有 `bad_types`，严重度取最大值。

## 5. Golden Set 候选策略

进入 `golden_candidates.jsonl` 的优先条件：

```text
1. severity >= 2；
2. 有 query_hash；
3. 去重后保留每个 query_hash 最近一条；
4. 保留 provider / model / bad_types / trace_id；
5. 如果 trace 包含 query 原文，则同步保留 query，便于人工补 ground_truth。
```

候选样本不会直接成为训练集，必须人工复核：

```text
golden_candidate
  -> 人工补 ground_truth / expected_sources
  -> 加入正式 Golden Set
  -> 阶段四回归评估
```

## 6. 输出 Schema

### 6.1 bad_cases.jsonl

```json
{
  "trace_id": "sha256:...",
  "task_id": "abc123",
  "provider": "sft",
  "model_name": "kb-sft",
  "bad_types": ["missing_citation"],
  "severity": 2,
  "latency_ms": 1300,
  "query_hash": "sha256:...",
  "answer_hash": "sha256:...",
  "context_count": 4,
  "used_citations": [],
  "suggestion": "检查回答 prompt 的引用约束，必要时加入引用缺失回归样本。"
}
```

### 6.2 golden_candidates.jsonl

```json
{
  "candidate_id": "golden-candidate-000001",
  "source_trace_id": "sha256:...",
  "query_hash": "sha256:...",
  "provider": "sft",
  "model_name": "kb-sft",
  "bad_types": ["missing_citation"],
  "needs_human_label": true,
  "label_fields": ["question", "ground_truth", "expected_sources", "category"]
}
```

## 7. 验收命令

使用 sample trace 验证：

```bash
uv run python fine_tuning/scripts/mine_stage7_bad_cases.py --sample
```

读取真实线上 trace：

```bash
uv run python fine_tuning/scripts/mine_stage7_bad_cases.py
```

单测：

```bash
uv run python fine_tuning/tests/test_stage7_bad_case_mining.py
```

## 8. 阶段七通过标准

```text
1. 无 trace 文件时能给出清晰提示；
2. sample 模式可生成 bad_cases / golden_candidates / report；
3. 单条 trace 可命中多个 bad_types；
4. Golden 候选按 query_hash 去重；
5. 默认不要求原文 query / answer；
6. 产物写入 fine_tuning/data/online/，不进入 Git。
```

## 9. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 规则误判 | 人工复核成本增加 | severity 分级，先看高优先级 |
| trace 默认脱敏 | 无法直接补 ground truth | 通过 query_hash 找日志，必要时临时打开原文 |
| 候选进入训练集过快 | 训练污染 | Golden 候选必须人工确认 |
| 只看规则不看语义 | 漏掉复杂 bad case | 后续阶段加入 LLM judge / 用户反馈 |
| 文件膨胀 | 本地磁盘增长 | data/online 忽略，后续接日志轮转 |
