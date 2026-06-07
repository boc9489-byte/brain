# 阶段六测试记录 - 线上观测与 Bad Case 闭环

## 1. 测试目标

验证阶段六新增的回答 trace 能力是否满足：

```text
1. 默认关闭，不影响回答链路；
2. 开启后可写 JSONL；
3. 默认脱敏，不记录完整 query / answer；
4. 可记录 provider / model / latency / citation / refusal 等字段；
5. trace 产物不进入 Git。
```

## 2. 测试环境

```text
项目路径：/Users/bob/PycharmProjects/shopkeeper_brain
uv 环境：.venv
默认 trace：关闭
默认输出：fine_tuning/data/online/answer_traces.jsonl
```

## 3. 已执行命令

语法检查：

```bash
uv run python -m py_compile \
  knowledge/utils/observability/answer_trace.py \
  knowledge/processor/query_processor/nodes/answer_output_node.py \
  fine_tuning/scripts/check_stage6_observability.py \
  fine_tuning/tests/test_stage6_answer_trace.py
```

实际结果：

```text
通过
```

单测：

```bash
uv run python fine_tuning/tests/test_stage6_answer_trace.py
```

实际输出：

```text
[ok] stage6 answer trace
```

默认关闭检查：

```bash
uv run python fine_tuning/scripts/check_stage6_observability.py --check-only
```

实际输出摘要：

```text
[stage6] answer trace settings:
{
  "enabled": false,
  "path": "/Users/bob/PycharmProjects/shopkeeper_brain/fine_tuning/data/online/answer_traces.jsonl",
  "include_text": false,
  "include_context": false,
  "max_text_chars": 500
}
[stage6] sample trace:
{
  "provider": "base",
  "model_name": "qwen-flash",
  "context_count": 1,
  "used_citations": ["1"],
  "is_refusal": false,
  "has_answer": true
}
[stage6] check passed
```

写入 sample trace：

```bash
ANSWER_TRACE_ENABLED=true \
ANSWER_TRACE_PATH=/private/tmp/stage6_answer_traces.jsonl \
uv run python fine_tuning/scripts/check_stage6_observability.py --write-sample
```

实际输出摘要：

```text
[stage6] answer trace settings:
{
  "enabled": true,
  "path": "/private/tmp/stage6_answer_traces.jsonl",
  "include_text": false,
  "include_context": false,
  "max_text_chars": 500
}
[stage6] sample trace written -> /private/tmp/stage6_answer_traces.jsonl
```

结论：

```text
通过。sample trace 默认不包含 query / answer_preview 原文字段。
```

## 4. 验收标准

```text
1. 所有命令退出码为 0；
2. 默认关闭时 enabled=false；
3. sample trace 能写入 JSONL；
4. 默认记录中没有 query / answer_preview 原文字段；
5. Git 忽略 fine_tuning/data/online/；
6. trace 写入失败不会影响 AnswerOutPutNode 返回。
```
