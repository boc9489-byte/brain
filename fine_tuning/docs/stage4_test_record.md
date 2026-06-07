# 阶段四测试记录 - Base vs SFT 离线评估

## 1. 测试目标

验证阶段四评估脚本是否可以在本地离线模式下完成：

```text
sft_holdout.jsonl
  -> eval_before_after.py --stub
  -> eval_report.md
  -> _eval_metrics.json
  -> predictions.jsonl
  -> bad_cases.jsonl
```

本次测试不加载真实模型，不加载 adapter。

## 2. 测试环境

```text
项目路径：/Users/bob/PycharmProjects/shopkeeper_brain
conda 环境：langchain
输入数据：fine_tuning/data/processed/sft_holdout.jsonl
输出目录：fine_tuning/data/eval
```

## 3. 语法检查

测试命令：

```bash
python -m py_compile fine_tuning/scripts/eval_before_after.py
```

验收标准：

```text
命令退出码为 0
```

实际结果：

```text
通过
```

指标单测：

```bash
python fine_tuning/tests/test_stage4_eval.py
```

实际输出：

```text
[ok] stage4 eval metrics direction is correct
```

## 4. 输入检查

测试命令：

```bash
conda run --no-capture-output -n langchain python fine_tuning/scripts/eval_before_after.py --check-only
```

验收标准：

```text
1. 能读取 sft_holdout.jsonl；
2. messages role 顺序为 system / user / assistant；
3. 输出 [eval-check] passed。
```

实际输出：

```text
[config] fine_tuning/configs/config.yaml not found, using config.example.yaml with environment fallbacks.
[eval-check] samples=5
[eval-check] output_dir=/Users/bob/PycharmProjects/shopkeeper_brain/fine_tuning/data/eval
[eval-check] first_roles=['system', 'user', 'assistant']
[eval-check] passed
```

结论：

```text
通过
```

## 5. stub 评估

测试命令：

```bash
conda run --no-capture-output -n langchain python fine_tuning/scripts/eval_before_after.py --stub
```

验收标准：

```text
1. 生成 eval_report.md；
2. 生成 _eval_metrics.json；
3. 生成 predictions.jsonl；
4. 生成 bad_cases.jsonl；
5. Base 与 SFT 指标可形成 before/after 对比。
```

实际输出摘要：

```text
[eval] stub mode: base=naive, sft=oracle
[eval] samples=5

| 指标 | Base | SFT | Δ |
|---|---:|---:|---:|
| 拒答 Recall | 0.00 | 1.00 | +1.00 |
| 拒答 Precision | 0.00 | 1.00 | +1.00 |
| 拒答 F1 | 0.00 | 1.00 | +1.00 |
| 误拒率 | 0.00 | 0.00 | +0.00 |
| weak_recall Recall | 0.00 | 1.00 | +1.00 |
| 引用有效率 | 1.00 | 1.00 | +0.00 |
```

生成产物：

```text
fine_tuning/data/eval/eval_report.md
fine_tuning/data/eval/_eval_metrics.json
fine_tuning/data/eval/predictions.jsonl
fine_tuning/data/eval/bad_cases.jsonl
```

结论：

```text
通过
```

说明：

```text
当前 holdout 只有 5 条 dry-run 样本，只能验证评估脚本通路。
正式评估前需要用强模型正式造数，并保证 holdout 至少 100-150 条。
```

## 6. 正式评估前置条件

```text
1. 阶段二正式强模型造数完成；
2. sft_holdout.jsonl 不是 dry-run stub 数据；
3. 阶段三 LoRA adapter 已生成；
4. adapter 路径为 fine_tuning/outputs/kb-sft 或 config 中指定路径；
5. 评估输出目录 fine_tuning/data/eval/ 不进入 Git。
```
