# 阶段五执行计划 - vLLM LoRA 接回业务

## 1. 一句话定位

阶段五负责把阶段三训练出的 LoRA adapter，通过 vLLM OpenAI-compatible 服务以可开关、可回滚的方式接回知识库回答链路。

```text
阶段四证明 SFT 是否有效。
阶段五只在评估通过后，把 Base / SFT 切换能力接入业务回答节点。
```

阶段五不是重新训练，也不是改检索策略；它只负责模型服务化和业务生成层切换。

## 2. 背景与目标

### 2.1 背景

当前主项目回答生成由 `AnswerOutPutNode` 调用 `AIClients.get_llm_client()`，底层使用 OpenAI-compatible `ChatOpenAI`。

阶段三产出：

```text
fine_tuning/outputs/kb-sft
```

阶段四产出：

```text
fine_tuning/data/eval/eval_report.md
fine_tuning/data/eval/_eval_metrics.json
```

阶段五的前提是阶段四评估显示 SFT 在拒答、引用和忠实性上有收益，且误拒率没有明显升高。

### 2.2 阶段五目标

```text
1. 支持 Base / SFT 回答模型开关；
2. 支持 vLLM OpenAI-compatible endpoint；
3. 支持 LoRA adapter alias 作为 SFT model name；
4. 支持一键回滚到 Base；
5. 提供本地配置检查脚本；
6. 不改导入链路，不改 Milvus 入库，不改召回和 rerank。
```

### 2.3 非目标

```text
1. 不训练模型；
2. 不合并 LoRA；
3. 不强制启动 vLLM；
4. 不修改检索策略；
5. 不做线上 A/B 自动化平台；
6. 不把 adapter、checkpoint、eval 产物提交 Git。
```

## 3. 总体架构

```text
Frontend / API
  -> Query Processor
  -> Retriever / Reranker
  -> AnswerOutPutNode
       -> AIClients.get_answer_llm_client()
       -> ANSWER_MODEL_PROVIDER=base
            -> vLLM base model
       -> ANSWER_MODEL_PROVIDER=sft
            -> vLLM base model + LoRA alias
  -> SSE / Task Result
```

vLLM 侧推荐：

```text
vLLM OpenAI Server
  base model: Qwen/Qwen2.5-3B-Instruct
  lora alias: kb-sft
  endpoint: http://<model-server>:8000/v1
```

## 4. 核心模块

| 模块 | 职责 | 输入 | 输出 |
|---|---|---|---|
| `answer_model_config.py` | 从环境变量解析 Base/SFT 回答模型配置 | env | `AnswerModelSettings` |
| `AIClients.get_answer_llm_client()` | 创建回答模型客户端 | settings | `ChatOpenAI` |
| `AnswerOutPutNode` | 使用回答模型生成最终答案 | prompt | answer |
| `check_stage5_serving.py` | 本地检查阶段五配置和可选模型服务健康 | env / args | 检查结果 |
| `stage5_test_record.md` | 记录本地检查和验收命令 | 测试输出 | 测试结论 |

## 5. 配置设计

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `ANSWER_MODEL_PROVIDER` | `base` | `base` 或 `sft` |
| `ANSWER_OPENAI_API_BASE` | `OPENAI_API_BASE` | 回答模型 OpenAI-compatible base url |
| `ANSWER_OPENAI_API_KEY` | `OPENAI_API_KEY` 或 `EMPTY` | 回答模型 API key |
| `ANSWER_BASE_MODEL` | `LLM_DEFAULT_MODEL` / `MODEL` | Base 模型名 |
| `ANSWER_SFT_MODEL` | `kb-sft` | vLLM LoRA alias |
| `ANSWER_TEMPERATURE` | `0` | 回答温度 |
| `ANSWER_MAX_TOKENS` | `1024` | 最大生成 tokens |
| `ANSWER_TIMEOUT_SEC` | `60` | 请求超时 |

### 5.1 Base 模式

```bash
ANSWER_MODEL_PROVIDER=base
ANSWER_OPENAI_API_BASE=http://127.0.0.1:8000/v1
ANSWER_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
```

### 5.2 SFT 模式

```bash
ANSWER_MODEL_PROVIDER=sft
ANSWER_OPENAI_API_BASE=http://127.0.0.1:8000/v1
ANSWER_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
ANSWER_SFT_MODEL=kb-sft
```

## 6. vLLM 启动参考

Base 服务：

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2.5-3B-Instruct \
  --served-model-name Qwen/Qwen2.5-3B-Instruct
```

LoRA 服务：

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2.5-3B-Instruct \
  --served-model-name Qwen/Qwen2.5-3B-Instruct \
  --enable-lora \
  --lora-modules kb-sft=/path/to/fine_tuning/outputs/kb-sft
```

业务侧只切 `ANSWER_MODEL_PROVIDER`，不需要修改 prompt 或检索代码。

## 7. 回滚流程

```text
发现 SFT 误拒率升高、回答变慢、引用异常或模型服务不稳定
  -> 设置 ANSWER_MODEL_PROVIDER=base
  -> 重启业务 API 服务
  -> 保留 bad case 和日志
  -> 回到阶段四重新评估
```

## 8. 验收命令

无网络本地检查：

```bash
uv run python fine_tuning/scripts/check_stage5_serving.py --check-only
```

检查当前配置是否可切 SFT：

```bash
ANSWER_MODEL_PROVIDER=sft \
ANSWER_OPENAI_API_BASE=http://127.0.0.1:8000/v1 \
ANSWER_SFT_MODEL=kb-sft \
uv run python fine_tuning/scripts/check_stage5_serving.py --check-only
```

可选模型服务健康检查：

```bash
uv run python fine_tuning/scripts/check_stage5_serving.py --health
```

## 9. 阶段五通过标准

```text
1. 默认 ANSWER_MODEL_PROVIDER=base 时业务行为不变；
2. ANSWER_MODEL_PROVIDER=sft 时模型名切为 ANSWER_SFT_MODEL；
3. 配置检查脚本能输出 provider/base_url/model；
4. vLLM /v1/models 健康检查可选通过；
5. AnswerOutPutNode 能通过语法检查；
6. 可通过环境变量回滚到 base；
7. 只有阶段四评估通过后才允许打开 sft。
```

## 10. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 阶段四未评估就接入 | 无法证明收益 | 默认 base，文档强制阶段四门禁 |
| vLLM 未启用 LoRA | SFT model alias 不存在 | `check_stage5_serving.py --health` 查 `/v1/models` |
| SFT 过度拒答 | 用户可回答问题被拒 | 监控 false_refusal，快速回滚 base |
| adapter 路径错误 | vLLM 启动失败 | 启动命令显式 `--lora-modules alias=path` |
| 回答延迟升高 | 用户体验下降 | 记录 latency，必要时降 max_tokens 或回滚 |
| 模型服务公网暴露 | 安全风险 | 仅内网访问，保留 API key / Nginx 控制 |
