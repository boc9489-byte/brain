# GPU 上线 Runbook - 掌柜智库 SFT

## 1. 目标

把训练后的 LoRA adapter 通过 vLLM OpenAI-compatible 服务接回业务回答链路，并支持：

```text
1. base / sft 环境变量切换；
2. GPU 模型服务健康检查；
3. 业务服务联调；
4. 线上 trace 观测；
5. 快速回滚到 base。
```

## 2. 服务器规划

| 服务器 | 角色 | 服务 | 端口 |
|---|---|---|---:|
| 业务服务器 | FastAPI / RAG | `knowledge/api/import_router.py` | 8000 |
| GPU 服务器 | vLLM LLM | `/v1/chat/completions`、`/v1/models` | 8000 |
| 存储服务器 | Milvus / MinIO / MongoDB | 知识库数据依赖 | 按环境 |

生产建议模型服务只暴露在内网，不直接暴露公网。

## 3. 代码同步

### 3.1 有 Git 网络

```bash
cd /usr-data/apps
git clone https://github.com/boc9489-byte/brain.git shopkeeper_brain
cd shopkeeper_brain
git checkout main
git pull origin main
```

已有项目：

```bash
cd /usr-data/apps/shopkeeper_brain
git fetch origin
git pull origin main
```

### 3.2 无 Git 网络

本地打包：

```bash
cd /Users/bob/PycharmProjects/shopkeeper_brain
git archive --format=tar.gz -o /private/tmp/shopkeeper_brain.tar.gz main
```

上传到服务器：

```bash
scp /private/tmp/shopkeeper_brain.tar.gz user@SERVER:/usr-data/apps/
```

服务器解压：

```bash
cd /usr-data/apps
mkdir -p shopkeeper_brain
tar -xzf shopkeeper_brain.tar.gz -C shopkeeper_brain
```

## 4. GPU 环境准备

```bash
nvidia-smi
cd /usr-data/apps/shopkeeper_brain
uv venv --python 3.10 .venv-vllm
source .venv-vllm/bin/activate
uv pip install vllm
```

模型目录建议：

```text
/usr-data/models/Qwen2.5-3B-Instruct
/usr-data/adapters/kb-sft
/usr-data/logs/vllm
```

## 5. 启动 vLLM

Base + LoRA：

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /usr-data/models/Qwen2.5-3B-Instruct \
  --served-model-name Qwen/Qwen2.5-3B-Instruct \
  --enable-lora \
  --lora-modules kb-sft=/usr-data/adapters/kb-sft
```

健康检查：

```bash
curl http://127.0.0.1:8000/v1/models
```

预期：

```text
能看到 Qwen/Qwen2.5-3B-Instruct 和 kb-sft。
```

## 6. 业务环境准备

```bash
cd /usr-data/apps/shopkeeper_brain
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r knowledge/requirements.txt
uv pip install -r fine_tuning/requirements-runtime.txt
cp knowledge/.env.example knowledge/.env
```

编辑 `knowledge/.env`：

```bash
ANSWER_MODEL_PROVIDER=base
ANSWER_OPENAI_API_BASE=http://GPU_SERVER_IP:8000/v1
ANSWER_OPENAI_API_KEY=EMPTY
ANSWER_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
ANSWER_SFT_MODEL=kb-sft

ANSWER_TRACE_ENABLED=true
ANSWER_TRACE_PATH=fine_tuning/data/online/answer_traces.jsonl
ANSWER_TRACE_INCLUDE_TEXT=false
ANSWER_TRACE_INCLUDE_CONTEXT=false
```

## 7. 上线前检查

```bash
uv run python fine_tuning/scripts/check_stage5_serving.py --check-only
uv run python fine_tuning/scripts/check_stage5_serving.py --health
uv run python fine_tuning/scripts/check_stage6_observability.py --check-only
```

语法与单测：

```bash
uv run python -m py_compile \
  knowledge/utils/client/answer_model_config.py \
  knowledge/utils/client/ai_clients.py \
  knowledge/utils/observability/answer_trace.py \
  knowledge/processor/query_processor/nodes/answer_output_node.py

uv run python fine_tuning/tests/test_stage5_answer_config.py
uv run python fine_tuning/tests/test_stage6_answer_trace.py
uv run python fine_tuning/tests/test_stage7_bad_case_mining.py
```

## 8. 启动业务服务

```bash
uv run python -c "from knowledge.api.import_router import create_app; import uvicorn; uvicorn.run(create_app(), host='0.0.0.0', port=8000)"
```

检查：

```bash
curl http://127.0.0.1:8000/hello
```

前端：

```text
http://SERVER_IP:8000/front/import.html
```

## 9. SFT 灰度切换

第一步只上 base：

```bash
ANSWER_MODEL_PROVIDER=base
```

阶段四真实评估通过后再切 SFT：

```bash
ANSWER_MODEL_PROVIDER=sft
```

切换后重启业务服务，再检查：

```bash
uv run python fine_tuning/scripts/check_stage5_serving.py --check-only
uv run python fine_tuning/scripts/check_stage5_serving.py --health
```

## 10. 线上观测

查看回答 trace：

```bash
tail -f fine_tuning/data/online/answer_traces.jsonl
```

挖掘 bad case：

```bash
uv run python fine_tuning/scripts/mine_stage7_bad_cases.py
```

产物：

```text
fine_tuning/data/online/bad_cases.jsonl
fine_tuning/data/online/golden_candidates.jsonl
fine_tuning/data/online/_stage7_bad_case_report.md
```

## 11. 回滚

如果 SFT 出现误拒、延迟升高、引用异常或 vLLM 不稳定：

```bash
ANSWER_MODEL_PROVIDER=base
```

重启业务服务。保留：

```text
fine_tuning/data/online/answer_traces.jsonl
fine_tuning/data/online/bad_cases.jsonl
业务服务日志
vLLM 日志
```

回到阶段四重新做 Base vs SFT 评估。

## 12. 常见问题

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `/v1/models` 看不到 `kb-sft` | vLLM 没有启用 LoRA | 检查 `--enable-lora` 和 `--lora-modules` |
| SFT 调用失败 | `ANSWER_SFT_MODEL` 与 LoRA alias 不一致 | 统一设置为 `kb-sft` |
| 业务回答仍走旧模型 | `ANSWER_MODEL_PROVIDER` 未切换或服务未重启 | 修改 `.env` 后重启业务 |
| trace 没有写入 | `ANSWER_TRACE_ENABLED=false` | 改为 `true` 并确认目录权限 |
| 延迟高 | prompt 过长或 GPU 负载高 | 看 `latency_ms`、`prompt_chars`，调整 top-k / max_tokens |
