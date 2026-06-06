# 阶段二测试记录 - 强模型造数链路

## 1. 测试目标

验证阶段二新增链路是否能在不调用强模型的情况下跑通：

```text
kb_chunks.jsonl
  -> expand_dataset.py --retriever local --dry-run
  -> sft_train.jsonl / sft_holdout.jsonl
  -> validate_messages_dataset.py
```

本次测试只验证工程通路，不代表正式训练数据质量。

## 2. 测试环境

```text
项目路径：/Users/bob/PycharmProjects/shopkeeper_brain
conda 环境：langchain
配置来源：fine_tuning/configs/config.example.yaml + knowledge/.env 兜底
raw chunk：fine_tuning/data/raw/kb_chunks.jsonl
```

## 3. 语法与轻量测试

测试命令：

```bash
python -m py_compile fine_tuning/scripts/_common.py fine_tuning/scripts/expand_dataset.py fine_tuning/scripts/validate_messages_dataset.py fine_tuning/tests/test_stage2_pipeline.py
python fine_tuning/tests/test_stage2_pipeline.py
```

实际输出：

```text
[ok] stage2 local/stub builders cover all types and pass inline validation
```

结论：

```text
通过
```

## 4. 阶段二 dry-run 造数

测试命令：

```bash
conda run --no-capture-output -n langchain python fine_tuning/scripts/expand_dataset.py --retriever local --dry-run --total 40
```

实际输出摘要：

```text
requested_total=40
actual_total=40
train=35
holdout=5
by_type:
  faithful=12
  multi_hop=7
  cite=7
  refuse=10
  format=4
refuse_subtypes:
  no_recall=4
  weak_recall=3
  conflict=3
dropped:
  duplicate=4
mode=stub
retriever=local
```

结论：

```text
通过
```

说明：

```text
local + dry-run 只用于验证阶段二 messages 造数通路。
正式训练前必须配置强模型，重新生成 SFT 数据，并做人工抽检。
```

## 5. messages 数据校验

测试命令：

```bash
conda run --no-capture-output -n langchain python fine_tuning/scripts/validate_messages_dataset.py
```

实际输出摘要：

```text
train=35
holdout=5
total=40
类型分布: cite=7, faithful=12, format=4, multi_hop=7, refuse=10
refuse 子类: conflict=3, no_recall=4, weak_recall=3
硬错误: 0
软告警: 0
```

结论：

```text
通过
```

## 6. 当前阶段二状态

| 检查项 | 状态 | 说明 |
|---|---|---|
| 阶段二设计文档 | 通过 | `stage2_execution_plan.md` 已新增 |
| local dry-run 造数 | 通过 | 生成 40 条 messages 样本 |
| 五类能力覆盖 | 通过 | faithful / multi_hop / cite / refuse / format 均覆盖 |
| 拒答子类覆盖 | 通过 | no_recall / weak_recall / conflict 均覆盖 |
| messages 校验 | 通过 | 硬错误 0，软告警 0 |
| Git 保护 | 通过 | sft_train / sft_holdout / 统计报告均被忽略 |

## 7. 后续正式造数入口

```bash
cp fine_tuning/configs/config.example.yaml fine_tuning/configs/config.yaml
# 填写 llm.base_url / llm.api_key / llm.model
conda run --no-capture-output -n langchain python fine_tuning/scripts/expand_dataset.py --retriever local
conda run --no-capture-output -n langchain python fine_tuning/scripts/validate_messages_dataset.py
```

如要验证 Milvus 召回版本：

```bash
conda run --no-capture-output -n langchain python fine_tuning/scripts/expand_dataset.py --retriever milvus
```

注意当前项目 Milvus 向量字段为 `dense_vector`。

