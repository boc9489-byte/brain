# fine_tuning - 掌柜智库知识库 SFT

本目录独立于 `knowledge/processor` 主链路，用来完成知识库生成模型微调的数据工程、造数、校验和后续训练准备。

阶段一目标不是训练模型，而是打通从真实知识库到可训练数据的链路：

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
│   ├── convert_to_messages.py
│   ├── expand_dataset.py
│   └── validate_messages_dataset.py
├── src/
│   ├── train_sft.py
│   └── merge_lora.py
├── data/
│   ├── raw/
│   ├── seed/
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
| `docs/stage2_execution_plan.md` | 阶段二强模型正式造数方案，说明架构、数据流、验收和风险 |
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

然后跑阶段一 bootstrap 流水线：

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

## 阶段二造数

阶段二新增正式造数候选链路：

```text
kb_chunks.jsonl
  -> expand_dataset.py
  -> sft_train.jsonl / sft_holdout.jsonl
  -> validate_messages_dataset.py
```

本地离线验证：

```bash
python fine_tuning/scripts/expand_dataset.py --retriever local --dry-run --total 40
python fine_tuning/scripts/validate_messages_dataset.py
```

正式强模型造数：

```bash
# 在 fine_tuning/configs/config.yaml 填写 llm.base_url / llm.api_key / llm.model
python fine_tuning/scripts/expand_dataset.py --retriever local
python fine_tuning/scripts/validate_messages_dataset.py
```

Milvus 召回造数：

```bash
python fine_tuning/scripts/expand_dataset.py --retriever milvus
python fine_tuning/scripts/validate_messages_dataset.py
```

注意：

```text
1. 当前项目的 Milvus 向量字段是 dense_vector，不是 embedding；
2. local + dry-run 仍只验证工程通路，不能直接训练；
3. 正式数据需要强模型生成，并做至少 10% 人工抽检。
```

## 阶段三训练

阶段三新增 QLoRA 训练入口：

```text
sft_train.jsonl
  -> train_sft.py
  -> fine_tuning/outputs/kb-sft
```

本地只做检查，不加载模型：

```bash
python fine_tuning/src/train_sft.py --check-only
python fine_tuning/src/merge_lora.py --check-only
```

GPU 环境训练：

```bash
conda create -n kb-sft python=3.10
conda activate kb-sft
pip install -r fine_tuning/requirements-train.txt

python fine_tuning/src/train_sft.py --config fine_tuning/configs/config.yaml
```

可选合并 LoRA：

```bash
python fine_tuning/src/merge_lora.py --config fine_tuning/configs/config.yaml
```

注意：

```text
1. 阶段三训练必须使用阶段二正式强模型造数后的 sft_train.jsonl；
2. local + dry-run 生成的数据仍不能训练；
3. adapter、checkpoint、merged model 都输出到 fine_tuning/outputs/，不会提交 Git；
4. 阶段三只训练，不做 Base vs SFT 评估，评估放阶段四。
```

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
fine_tuning/data/processed/sft_train.jsonl
fine_tuning/data/processed/sft_holdout.jsonl
fine_tuning/data/processed/_expand_stats.json
fine_tuning/data/processed/_messages_validation_report.md
fine_tuning/outputs/kb-sft/
fine_tuning/outputs/kb-sft-merged/
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
