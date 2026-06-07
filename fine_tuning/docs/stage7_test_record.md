# 阶段七测试记录 - Bad Case 挖掘与 Golden Set 候选

## 1. 测试目标

验证阶段七脚本能从 answer trace 中离线挖掘 bad case，并输出 Golden Set 候选。

```text
answer_traces.jsonl
  -> mine_stage7_bad_cases.py
  -> bad_cases.jsonl
  -> golden_candidates.jsonl
  -> _stage7_bad_case_report.md
```

## 2. 测试环境

```text
项目路径：/Users/bob/PycharmProjects/shopkeeper_brain
uv 环境：.venv
默认输入：fine_tuning/data/online/answer_traces.jsonl
默认输出：fine_tuning/data/online/
```

## 3. 已执行命令

语法检查：

```bash
uv run python -m py_compile \
  fine_tuning/scripts/mine_stage7_bad_cases.py \
  fine_tuning/tests/test_stage7_bad_case_mining.py
```

实际结果：

```text
通过
```

单测：

```bash
uv run python fine_tuning/tests/test_stage7_bad_case_mining.py
```

实际输出：

```text
[ok] stage7 bad case mining
```

sample 挖掘：

```bash
uv run python fine_tuning/scripts/mine_stage7_bad_cases.py --sample
```

实际输出：

```text
[stage7] traces=4 bad_cases=4 golden_candidates=4
[stage7] bad_cases -> /Users/bob/PycharmProjects/shopkeeper_brain/fine_tuning/data/online/bad_cases.jsonl
[stage7] golden_candidates -> /Users/bob/PycharmProjects/shopkeeper_brain/fine_tuning/data/online/golden_candidates.jsonl
[stage7] report -> /Users/bob/PycharmProjects/shopkeeper_brain/fine_tuning/data/online/_stage7_bad_case_report.md
```

报告摘要：

```text
trace_total=4
bad_case_total=4
golden_candidate_total=4
类型分布：
- no_context_answered=1
- missing_citation=1
- sft_over_refusal_candidate=1
- high_latency=1
- long_prompt=1
- short_answer=1
```

真实 trace 挖掘：

```bash
uv run python fine_tuning/scripts/mine_stage7_bad_cases.py
```

实际结果：

```text
未执行真实线上 trace 挖掘。当前使用 --sample 验证阶段七工程通路。
```

## 4. 验收标准

```text
1. 语法检查退出码为 0；
2. 单测通过；
3. sample 模式能产出 bad case；
4. report 中包含 bad case 类型统计；
5. golden_candidates.jsonl 去重；
6. fine_tuning/data/online/ 仍被 Git 忽略。
```
