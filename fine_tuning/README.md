# fine_tuning - 掌柜智库知识库 SFT 阶段一

本目录独立于 `knowledge/processor` 主链路，用来完成知识库生成模型微调的第一阶段：数据闭环。

第一阶段目标不是训练模型，而是打通从真实知识库到可训练数据的链路：

```text
Milvus 真实 chunk -> SFT 样本 -> 数据校验 -> messages 训练格式
```

新作战方案的最终目标是让 Qwen3-8B 在 RAG 上下文下具备：

```text
忠实回答 / 多片段整合 / 引用标注 / 资料不足拒答 / 格式适配
```

## 目录结构

```text
fine_tuning/
├── configs/
│   └── config.example.yaml
├── scripts/
│   ├── _common.py
│   ├── export_kb_chunks.py
│   ├── build_sft_dataset.py
│   ├── validate_dataset.py
│   └── convert_to_messages.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── train/
├── eval/
├── screenshots/
└── outputs/
```

## 文档入口

| 文档 | 作用 |
|---|---|
| `docs/enterprise_engineering_solution.md` | 企业工程总方案，串联定位、架构、数据、测试、验收和路线图 |
| `docs/stage1_execution_plan.md` | 阶段一执行计划，说明边界、命令、风险和增强项 |
| `docs/code_reading_stage1.md` | 代码阅读文档，说明脚本职责、数据 schema 和主链路关系 |
| `docs/stage1_engineering_report.md` | 阶段一工程汇报，用于阶段复盘和对外说明 |
| `docs/stage1_test_record.md` | 测试记录，记录 Milvus 导出和后续脚本验收 |
| `docs/stage1_acceptance_checklist.md` | 阶段一验收清单，用于提交前逐项确认 |
| `docs/daily_progress_2026-06-06.md` | 当天任务进度记录 |

## 本地运行

在仓库根目录执行：

```bash
cp fine_tuning/configs/config.example.yaml fine_tuning/configs/config.yaml
```

`config.yaml` 会被 Git 忽略。`milvus.uri` 和 `milvus.collection` 留空时，会默认读取 `knowledge/.env` 中的：

```text
MILVUS_URL
CHUNKS_COLLECTION
```

然后跑第一阶段流水线：

```bash
python fine_tuning/scripts/export_kb_chunks.py
python fine_tuning/scripts/build_sft_dataset.py --dry-run
python fine_tuning/scripts/validate_dataset.py
python fine_tuning/scripts/convert_to_messages.py
```

命令说明：

| 命令 | 作用 | 是否需要外部模型 |
|---|---|---|
| export_kb_chunks.py | 从 Milvus 导出真实 chunk | 否 |
| build_sft_dataset.py --dry-run | 使用 stub 构造冷启动样本 | 否 |
| validate_dataset.py | 校验引用、拒答、重复和长度 | 否 |
| convert_to_messages.py | 转成 system/user/assistant 训练格式 | 否 |

`--dry-run` 只用于验证流水线。正式训练前要配置强模型重新生成高质量样本。

## 产物

运行后会生成：

```text
fine_tuning/data/raw/kb_chunks.jsonl
fine_tuning/data/raw/_export_stats.json
fine_tuning/data/processed/train.jsonl
fine_tuning/data/processed/holdout.jsonl
fine_tuning/data/processed/messages_train.jsonl
fine_tuning/data/processed/messages_holdout.jsonl
fine_tuning/data/processed/_validation_report.md
```

这些数据文件默认不会提交到 Git。

## 阶段一验收

```text
1. 能从 Milvus 导出真实知识库 chunk
2. 能构造 answerable / multi_chunk / unanswerable / conflicting 四类样本
3. validate_dataset.py 无硬错误
4. 能生成 messages_train.jsonl 和 messages_holdout.jsonl
5. 不污染 knowledge 主链路
```

## 与新作战方案的能力映射

当前阶段一是冷启动版本，先实现四类样本，再映射到作战方案五类能力：

| 当前样本 type | 作战方案能力 | 当前状态 |
|---|---|---|
| answerable | faithful + cite | 已实现 |
| multi_chunk | multi_hop | 已实现 |
| unanswerable | refuse | 已实现 |
| conflicting | refuse / 冲突召回 | 已实现 |
| format | format | 阶段一增强项，待补 |

`convert_to_messages.py` 会在 `meta.battle_capabilities` 中写入这个映射，方便后续评估时按五类能力分桶。

## 后续阶段

阶段二接强模型生成正式 SFT 数据；阶段三在云 GPU 上进行 Qwen3-8B QLoRA；阶段四做 Base vs SFT + RAGAS / 拒答准确率评测；阶段五通过 vLLM LoRA 接回业务。
