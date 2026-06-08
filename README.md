# shopkeeper_brain - 掌柜智库

掌柜智库是一个面向企业文档知识库的 RAG 项目，当前已经覆盖：

```text
文档导入
  -> PDF / Markdown 解析
  -> Markdown 图片摘要
  -> Chunk 切分
  -> 商品名识别
  -> BGE-M3 向量化
  -> Milvus 入库

知识库微调
  -> Milvus chunk 导出
  -> SFT 数据构造
  -> QLoRA 训练
  -> Base vs SFT 离线评估
  -> vLLM LoRA 接回业务
  -> 线上 trace 与 Bad Case 闭环
```

## 1. 目录结构

```text
shopkeeper_brain/
├── knowledge/                  # 业务知识库导入、查询和客户端配置
│   ├── api/                    # FastAPI 入口
│   ├── front/                  # 导入页面
│   ├── processor/              # 导入 / 查询处理节点
│   ├── prompt/                 # Prompt 模板
│   ├── service/                # 上传服务
│   ├── test/                   # 连接测试和实验脚本
│   └── utils/                  # 客户端、SSE、任务、观测工具
├── fine_tuning/                # 知识库 SFT 专项
│   ├── configs/
│   ├── docs/
│   ├── scripts/
│   ├── src/
│   └── tests/
└── file/                       # 本地资料目录，Git 忽略
```

## 2. 文档入口

| 文档 | 作用 |
|---|---|
| `fine_tuning/README.md` | fine_tuning 模块总入口 |
| `fine_tuning/docs/enterprise_engineering_solution.md` | 微调专项企业工程总方案 |
| `fine_tuning/docs/gpu_deployment_runbook.md` | GPU / vLLM / LoRA 上线操作手册 |
| `fine_tuning/docs/release_acceptance_checklist.md` | 发布前、GPU、业务、线上验收清单 |
| `fine_tuning/docs/stage1_execution_plan.md` | 阶段一：数据闭环 |
| `fine_tuning/docs/stage2_execution_plan.md` | 阶段二：强模型正式造数 |
| `fine_tuning/docs/stage3_execution_plan.md` | 阶段三：QLoRA 训练 |
| `fine_tuning/docs/stage4_execution_plan.md` | 阶段四：Base vs SFT 离线评估 |
| `fine_tuning/docs/stage5_execution_plan.md` | 阶段五：vLLM LoRA 接回业务 |
| `fine_tuning/docs/stage6_execution_plan.md` | 阶段六：线上 trace 与 Bad Case 闭环 |
| `fine_tuning/docs/stage7_execution_plan.md` | 阶段七：Bad Case 挖掘与 Golden Set 候选 |

## 3. 本地环境

项目统一使用 `uv` 管理 Python 环境。

```bash
cd /Users/bob/PycharmProjects/shopkeeper_brain
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r knowledge/requirements.txt
uv pip install -r fine_tuning/requirements-runtime.txt
```

准备环境变量：

```bash
cp knowledge/.env.example knowledge/.env
```

`knowledge/.env` 会被 Git 忽略，不要提交真实密钥。

## 4. 启动导入服务

```bash
uv run python -c "from knowledge.api.import_router import create_app; import uvicorn; uvicorn.run(create_app(), host='0.0.0.0', port=8000)"
```

健康检查：

```bash
curl http://127.0.0.1:8000/hello
```

前端导入页：

```text
http://127.0.0.1:8000/front/import.html
```

## 5. fine_tuning 快速检查

阶段一到阶段七的轻量检查：

```bash
uv run python fine_tuning/scripts/export_kb_chunks.py
uv run python fine_tuning/scripts/build_sft_dataset.py --dry-run
uv run python fine_tuning/scripts/validate_dataset.py
uv run python fine_tuning/scripts/convert_to_messages.py
uv run python fine_tuning/scripts/expand_dataset.py --retriever local --dry-run --total 40
uv run python fine_tuning/scripts/validate_messages_dataset.py
uv run python fine_tuning/src/train_sft.py --check-only
uv run python fine_tuning/src/merge_lora.py --check-only
uv run python fine_tuning/scripts/eval_before_after.py --check-only
uv run python fine_tuning/scripts/check_stage5_serving.py --check-only
uv run python fine_tuning/scripts/check_stage6_observability.py --check-only
uv run python fine_tuning/scripts/mine_stage7_bad_cases.py --sample
```

说明：

```text
dry-run / stub 只验证工程通路，不能作为正式训练数据。
正式训练前必须完成阶段二强模型造数、人工抽检、阶段三训练和阶段四评估。
```

## 6. GPU 上线摘要

模型服务建议与业务服务分离：

```text
业务服务器
  -> FastAPI / RAG
  -> Milvus / MinIO / MongoDB
  -> 调用 GPU 机器 vLLM OpenAI-compatible API

GPU 服务器
  -> vLLM base model
  -> LoRA alias: kb-sft
```

vLLM LoRA 启动示例：

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /usr-data/models/Qwen2.5-3B-Instruct \
  --served-model-name Qwen/Qwen2.5-3B-Instruct \
  --enable-lora \
  --lora-modules kb-sft=/usr-data/adapters/kb-sft
```

业务侧配置：

```bash
ANSWER_MODEL_PROVIDER=base
ANSWER_OPENAI_API_BASE=http://GPU_SERVER_IP:8000/v1
ANSWER_OPENAI_API_KEY=EMPTY
ANSWER_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
ANSWER_SFT_MODEL=kb-sft
```

先用 `base` 上线，阶段四真实评估通过后再切：

```bash
ANSWER_MODEL_PROVIDER=sft
```

详细步骤见：

```text
fine_tuning/docs/gpu_deployment_runbook.md
fine_tuning/docs/release_acceptance_checklist.md
```

## 7. Git 安全

可以提交：

```text
knowledge/**/*.py
knowledge/.env.example
fine_tuning/scripts/*.py
fine_tuning/src/*.py
fine_tuning/tests/*.py
fine_tuning/docs/*.md
fine_tuning/README.md
README.md
.gitignore
```

禁止提交：

```text
knowledge/.env
knowledge/temp_data/
fine_tuning/configs/config.yaml
fine_tuning/data/
fine_tuning/outputs/
*.safetensors
checkpoint-*/
.venv/
.venv-*/
```

提交前检查：

```bash
git status --short --ignored fine_tuning knowledge .gitignore README.md
git diff --check
```
