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
│   ├── validate_messages_dataset.py
│   ├── eval_before_after.py
│   ├── check_stage5_serving.py
│   ├── check_stage6_observability.py
│   └── mine_stage7_bad_cases.py
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
| `docs/zero_to_gpu_finetune.md` | 0 基础从本地造数到 GPU 微调的完整操作手册 |
| `docs/enterprise_engineering_solution.md` | 企业工程总方案，串联定位、架构、数据、测试、验收和路线图 |
| `docs/gpu_deployment_runbook.md` | GPU / vLLM / LoRA 上线操作手册 |
| `docs/release_acceptance_checklist.md` | 发布前、GPU、业务、线上验收清单 |
| `docs/stage1_execution_plan.md` | 阶段一执行计划，说明边界、命令、风险和增强项 |
| `docs/code_reading_stage1.md` | 代码阅读文档，说明脚本职责、数据 schema 和主链路关系 |
| `docs/stage1_engineering_report.md` | 阶段一工程汇报，用于阶段复盘和对外说明 |
| `docs/stage1_test_record.md` | 测试记录，记录 Milvus 导出和后续脚本验收 |
| `docs/stage1_acceptance_checklist.md` | 阶段一验收清单，用于提交前逐项确认 |
| `docs/stage2_execution_plan.md` | 阶段二强模型正式造数方案，说明架构、数据流、验收和风险 |
| `docs/stage2_test_record.md` | 阶段二 dry-run 造数和 messages 校验记录 |
| `docs/stage3_execution_plan.md` | 阶段三 QLoRA 训练方案，说明训练输入、输出、配置和风险 |
| `docs/stage3_test_record.md` | 阶段三训练脚本 check-only 测试记录 |
| `docs/stage4_execution_plan.md` | 阶段四 Base vs SFT 离线评估方案 |
| `docs/stage4_test_record.md` | 阶段四 stub 评估和指标测试记录 |
| `docs/stage5_execution_plan.md` | 阶段五 vLLM LoRA 接回业务方案 |
| `docs/stage5_test_record.md` | 阶段五配置开关和接入检查记录 |
| `docs/stage6_execution_plan.md` | 阶段六线上观测与 Bad Case 闭环方案 |
| `docs/stage6_test_record.md` | 阶段六 trace 脱敏、写入和检查记录 |
| `docs/stage7_execution_plan.md` | 阶段七 Bad Case 挖掘与 Golden Set 候选方案 |
| `docs/stage7_test_record.md` | 阶段七规则挖掘、候选生成和报告记录 |
| `docs/stage8_execution_plan.md` | 阶段八 Query Intent Analyzer 与检索路由优化方案 |
| `docs/daily_progress_2026-06-06.md` | 当天任务进度记录 |

## 本地运行

在仓库根目录执行：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r fine_tuning/requirements-runtime.txt
```

准备本地配置：

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
uv run python fine_tuning/scripts/export_kb_chunks.py
uv run python fine_tuning/scripts/build_sft_dataset.py --dry-run
uv run python fine_tuning/scripts/validate_dataset.py
uv run python fine_tuning/scripts/convert_to_messages.py
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
uv run python fine_tuning/scripts/expand_dataset.py --retriever local --dry-run --total 40
uv run python fine_tuning/scripts/validate_messages_dataset.py
```

正式强模型造数：

```bash
# 在 fine_tuning/configs/config.yaml 填写 llm.base_url / llm.api_key / llm.model
uv run python fine_tuning/scripts/expand_dataset.py --retriever local
uv run python fine_tuning/scripts/validate_messages_dataset.py
```

Milvus 召回造数：

```bash
uv run python fine_tuning/scripts/expand_dataset.py --retriever milvus
uv run python fine_tuning/scripts/validate_messages_dataset.py
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
uv run python fine_tuning/src/train_sft.py --check-only
uv run python fine_tuning/src/merge_lora.py --check-only
```

GPU 环境训练：

```bash
uv venv --python 3.10 .venv-kb-sft
source .venv-kb-sft/bin/activate
uv pip install -r fine_tuning/requirements-train.txt

uv run --active python fine_tuning/src/train_sft.py --config fine_tuning/configs/config.yaml
```

训练前还必须准备基座模型。推荐提前下载到 GPU 本地目录：

```bash
mkdir -p /usr-data/models

huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
  --local-dir /usr-data/models/Qwen2.5-3B-Instruct \
  --local-dir-use-symlinks False
```

如果 HuggingFace 网络不稳定，可以用 ModelScope：

```bash
modelscope download \
  --model Qwen/Qwen2.5-3B-Instruct \
  --local_dir /usr-data/models/Qwen2.5-3B-Instruct
```

然后把 `fine_tuning/configs/config.yaml` 中的 `train.base_model` 改为本地路径：

```yaml
train:
  base_model: "/usr-data/models/Qwen2.5-3B-Instruct"
```

0 基础完整流程见：

```text
fine_tuning/docs/zero_to_gpu_finetune.md
```

可选合并 LoRA：

```bash
uv run --active python fine_tuning/src/merge_lora.py --config fine_tuning/configs/config.yaml
```

注意：

```text
1. 阶段三训练必须使用阶段二正式强模型造数后的 sft_train.jsonl；
2. local + dry-run 生成的数据仍不能训练；
3. adapter、checkpoint、merged model 都输出到 fine_tuning/outputs/，不会提交 Git；
4. 阶段三只训练，不做 Base vs SFT 评估，评估放阶段四。
```

## 阶段四评估

阶段四新增 Base vs SFT 离线评估入口：

```text
sft_holdout.jsonl
  -> eval_before_after.py
  -> fine_tuning/data/eval/
```

本地 stub 验证：

```bash
uv run python fine_tuning/scripts/eval_before_after.py --stub
```

只检查输入输出，不加载模型：

```bash
uv run python fine_tuning/scripts/eval_before_after.py --check-only
```

真实模型评估：

```bash
uv run --active python fine_tuning/scripts/eval_before_after.py \
  --base Qwen/Qwen2.5-3B-Instruct \
  --adapter fine_tuning/outputs/kb-sft
```

可选 LLM judge：

```bash
uv run --active python fine_tuning/scripts/eval_before_after.py \
  --base Qwen/Qwen2.5-3B-Instruct \
  --adapter fine_tuning/outputs/kb-sft \
  --judge
```

阶段四重点看：

```text
refusal_recall
false_refusal
citation_validity
faithfulness
completeness
bad_cases
```

## 阶段五接入

阶段五新增 Base / SFT 回答模型开关：

```text
AnswerOutPutNode
  -> AIClients.get_answer_llm_client()
  -> ANSWER_MODEL_PROVIDER=base | sft
  -> vLLM OpenAI-compatible service
```

默认回退到 base：

```bash
ANSWER_MODEL_PROVIDER=base
ANSWER_OPENAI_API_BASE=http://127.0.0.1:8000/v1
ANSWER_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
```

启用 SFT：

```bash
ANSWER_MODEL_PROVIDER=sft
ANSWER_OPENAI_API_BASE=http://127.0.0.1:8000/v1
ANSWER_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
ANSWER_SFT_MODEL=kb-sft
```

vLLM LoRA 启动参考：

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2.5-3B-Instruct \
  --served-model-name Qwen/Qwen2.5-3B-Instruct \
  --enable-lora \
  --lora-modules kb-sft=/path/to/fine_tuning/outputs/kb-sft
```

本地配置检查：

```bash
uv run python fine_tuning/scripts/check_stage5_serving.py --check-only
```

可选服务健康检查：

```bash
uv run python fine_tuning/scripts/check_stage5_serving.py --health
```

阶段五只应在阶段四离线评估通过后打开 `ANSWER_MODEL_PROVIDER=sft`。

## 阶段六观测

阶段六新增默认关闭的回答 trace：

```text
AnswerOutPutNode
  -> intent_type normalization / fallback classification
  -> record_answer_trace()
  -> fine_tuning/data/online/answer_traces.jsonl
  -> bad case / golden set / next eval
```

查询链路会记录粗粒度意图字段 `intent_type`。上游 LLM 商品名提取如果已经输出
`intent_type`，回答节点会优先使用；如果没有，则使用本地规则分类兜底。当前支持：

```text
install_config
troubleshooting
parameter
operation
image_request
comparison
after_sales
general
```

这个字段会进入回答 prompt 和 answer trace，用于按意图分析召回失败、误拒、缺少引用、
高延迟等 bad case。

默认关闭：

```bash
ANSWER_TRACE_ENABLED=false
ANSWER_TRACE_PATH=fine_tuning/data/online/answer_traces.jsonl
ANSWER_TRACE_INCLUDE_TEXT=false
```

本地检查：

```bash
uv run python fine_tuning/tests/test_query_intent.py
uv run python fine_tuning/scripts/check_stage6_observability.py --check-only
```

写入一条 sample：

```bash
ANSWER_TRACE_ENABLED=true \
ANSWER_TRACE_PATH=/private/tmp/stage6_answer_traces.jsonl \
uv run python fine_tuning/scripts/check_stage6_observability.py --write-sample
```

默认 trace 只记录 hash、长度、provider、model、延迟、引用和拒答等字段，不记录完整问题和答案。

## 阶段七挖掘

阶段七把阶段六 trace 离线挖成 bad case 和 Golden Set 候选：

```text
answer_traces.jsonl
  -> mine_stage7_bad_cases.py
  -> bad_cases.jsonl
  -> golden_candidates.jsonl
  -> _stage7_bad_case_report.md
```

无真实 trace 时用 sample 验证：

```bash
uv run python fine_tuning/scripts/mine_stage7_bad_cases.py --sample
```

读取真实线上 trace：

```bash
uv run python fine_tuning/scripts/mine_stage7_bad_cases.py
```

主要规则：

```text
no_context_answered
missing_citation
sft_over_refusal_candidate
high_latency
long_prompt
short_answer
empty_answer
model_error
```

`golden_candidates.jsonl` 只是人工复核候选，不能直接进入训练集。

## 阶段八规划

阶段八把当前轻量 `intent_type` 升级为完整 Query Analyzer：

```text
original_query + history
  -> LLM candidate item / intent_type / rewritten_query
  -> Milvus item_name_collection alignment
  -> confidence threshold / score gap filtering
  -> clarification or retrieval routing
  -> hybrid / HyDE / web / image-aware retrieval
```

规划文档见：

```text
fine_tuning/docs/stage8_execution_plan.md
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
fine_tuning/data/eval/eval_report.md
fine_tuning/data/eval/_eval_metrics.json
fine_tuning/data/eval/bad_cases.jsonl
fine_tuning/data/eval/predictions.jsonl
fine_tuning/data/online/answer_traces.jsonl
fine_tuning/data/online/bad_cases.jsonl
fine_tuning/data/online/golden_candidates.jsonl
fine_tuning/data/online/_stage7_bad_case_report.md
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
