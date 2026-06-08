# 0 基础 GPU 微调快跑手册 - 掌柜智库 SFT

本文面向第一次租 GPU 做微调的人，目标是从本地项目出发，在 GPU 机器上完成：

```text
本地造数 -> 上传代码和数据 -> 下载基座模型 -> QLoRA 训练 -> 产出 LoRA adapter -> 离线评估
```

## 1. 先理解你要准备什么

正式训练前需要四类东西：

| 资源 | 作用 | 建议位置 |
|---|---|---|
| 项目代码 | 训练脚本、配置、评估脚本 | `/usr-data/apps/shopkeeper_brain` |
| 训练数据 | `sft_train.jsonl` / `sft_holdout.jsonl` | `fine_tuning/data/processed/` |
| 基座模型 | Qwen2.5 / Qwen3 等原始模型权重 | `/usr-data/models/Qwen2.5-3B-Instruct` |
| 训练环境 | PyTorch、Transformers、TRL、PEFT、bitsandbytes | `.venv-kb-sft` |

注意：

```text
1. 项目代码不会自动包含模型权重；
2. Git 不会提交 fine_tuning/data/ 和 fine_tuning/outputs/；
3. dry-run / stub 数据只能验证流程，不能用于正式训练；
4. 第一次建议用 Qwen2.5-3B-Instruct 跑通闭环，再升级到 7B / 8B。
```

## 2. 本地生成正式训练数据

在本地仓库根目录：

```bash
cd /Users/bob/PycharmProjects/shopkeeper_brain

uv venv --python 3.12
source .venv/bin/activate
uv pip install -r knowledge/requirements.txt
uv pip install -r fine_tuning/requirements-runtime.txt

cp knowledge/.env.example knowledge/.env
cp fine_tuning/configs/config.example.yaml fine_tuning/configs/config.yaml
```

编辑 `knowledge/.env`，确保 Milvus 相关配置可用：

```bash
MILVUS_URL=http://127.0.0.1:19530
CHUNKS_COLLECTION=kb_chunks
```

导出知识库 chunk：

```bash
uv run python fine_tuning/scripts/export_kb_chunks.py
```

先用 dry-run 验证工程链路：

```bash
uv run python fine_tuning/scripts/expand_dataset.py --retriever local --dry-run --total 40
uv run python fine_tuning/scripts/validate_messages_dataset.py
```

正式训练前，编辑 `fine_tuning/configs/config.yaml`，填写强模型造数配置：

```yaml
llm:
  base_url: "https://your-openai-compatible-endpoint/v1"
  api_key: "YOUR_API_KEY"
  model: "your-strong-model"
```

生成正式 SFT 数据：

```bash
uv run python fine_tuning/scripts/expand_dataset.py --retriever local
uv run python fine_tuning/scripts/validate_messages_dataset.py
```

必须确认存在：

```bash
ls -lh fine_tuning/data/processed/sft_train.jsonl
ls -lh fine_tuning/data/processed/sft_holdout.jsonl
```

并人工抽检至少 10% 样本，重点看：

```text
1. 回答是否只依据检索资料；
2. 引用编号是否存在且不越界；
3. 资料不足时是否拒答；
4. train / holdout 不是 dry-run 生成的模板数据。
```

## 3. 租 GPU 和登录服务器

建议第一轮配置：

| 目标模型 | 推荐 GPU | 训练方式 |
|---|---|---|
| Qwen2.5-3B-Instruct | 12GB-24GB 显存 | QLoRA |
| Qwen2.5-7B-Instruct | RTX 4090 24GB / A10 24GB | QLoRA |
| Qwen3-8B | RTX 4090 24GB / A100 | QLoRA |

登录 GPU：

```bash
ssh root@GPU_SERVER_IP
nvidia-smi
```

如果看不到 GPU，先不要继续训练。

## 4. 上传项目代码

如果 GPU 服务器可以访问 GitHub：

```bash
mkdir -p /usr-data/apps
cd /usr-data/apps
git clone https://github.com/boc9489-byte/brain.git shopkeeper_brain
cd shopkeeper_brain
git checkout main
```

如果 GPU 服务器不能访问 GitHub，本地打包上传：

```bash
cd /Users/bob/PycharmProjects/shopkeeper_brain
git archive --format=tar.gz -o /private/tmp/shopkeeper_brain.tar.gz main
scp /private/tmp/shopkeeper_brain.tar.gz root@GPU_SERVER_IP:/usr-data/apps/
```

GPU 上解压：

```bash
cd /usr-data/apps
mkdir -p shopkeeper_brain
tar -xzf shopkeeper_brain.tar.gz -C shopkeeper_brain
cd shopkeeper_brain
```

## 5. 单独上传配置和训练数据

因为 `fine_tuning/configs/config.yaml` 和 `fine_tuning/data/` 被 Git 忽略，需要单独上传：

```bash
ssh root@GPU_SERVER_IP "mkdir -p /usr-data/apps/shopkeeper_brain/fine_tuning/configs /usr-data/apps/shopkeeper_brain/fine_tuning/data/processed"

scp fine_tuning/configs/config.yaml \
  root@GPU_SERVER_IP:/usr-data/apps/shopkeeper_brain/fine_tuning/configs/

scp fine_tuning/data/processed/sft_train.jsonl \
    fine_tuning/data/processed/sft_holdout.jsonl \
  root@GPU_SERVER_IP:/usr-data/apps/shopkeeper_brain/fine_tuning/data/processed/
```

GPU 上确认：

```bash
cd /usr-data/apps/shopkeeper_brain
ls -lh fine_tuning/configs/config.yaml
ls -lh fine_tuning/data/processed/sft_train.jsonl
ls -lh fine_tuning/data/processed/sft_holdout.jsonl
```

## 6. 下载基座模型

训练脚本里的 `train.base_model` 可以写模型名，也可以写本地路径。

如果写：

```yaml
train:
  base_model: "Qwen/Qwen2.5-3B-Instruct"
```

Transformers 会在训练时自动下载模型。新手不建议依赖自动下载，因为 GPU 平台网络经常不稳定。建议提前下载到本地路径。

GPU 上创建模型目录：

```bash
mkdir -p /usr-data/models
```

方式一：HuggingFace 下载：

```bash
pip install -U huggingface_hub

huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
  --local-dir /usr-data/models/Qwen2.5-3B-Instruct \
  --local-dir-use-symlinks False
```

方式二：ModelScope 下载：

```bash
pip install -U modelscope

modelscope download \
  --model Qwen/Qwen2.5-3B-Instruct \
  --local_dir /usr-data/models/Qwen2.5-3B-Instruct
```

下载后检查：

```bash
ls -lh /usr-data/models/Qwen2.5-3B-Instruct
```

至少应看到：

```text
config.json
tokenizer.json 或 tokenizer.model
model-*.safetensors
model.safetensors.index.json
```

然后编辑 GPU 上的 `fine_tuning/configs/config.yaml`：

```yaml
train:
  base_model: "/usr-data/models/Qwen2.5-3B-Instruct"
  output_dir: "fine_tuning/outputs/kb-sft"
```

## 7. 搭建 GPU 训练环境

GPU 上执行：

```bash
cd /usr-data/apps/shopkeeper_brain

uv venv --python 3.10 .venv-kb-sft
source .venv-kb-sft/bin/activate
uv pip install -r fine_tuning/requirements-train.txt
```

如果没有 `uv`：

```bash
pip install uv
```

检查 PyTorch 和 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期：

```text
torch.cuda.is_available() 输出 True。
```

## 8. 训练前检查

```bash
cd /usr-data/apps/shopkeeper_brain
source .venv-kb-sft/bin/activate

uv run --active python fine_tuning/src/train_sft.py \
  --config fine_tuning/configs/config.yaml \
  --check-only
```

必须看到：

```text
[train-check] rows=...
[train-check] passed
```

如果 rows 为 0 或检查失败，回到第 5 步检查数据上传。

## 9. 开始 QLoRA 微调

```bash
uv run --active python fine_tuning/src/train_sft.py \
  --config fine_tuning/configs/config.yaml
```

训练完成后检查 adapter：

```bash
ls -lh fine_tuning/outputs/kb-sft
```

应看到：

```text
adapter_config.json
adapter_model.safetensors
training_summary.json
tokenizer files
```

这些文件不要提交 Git。

## 10. 常见训练问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 下载模型很慢或失败 | GPU 平台网络不稳定 | 先用 HuggingFace / ModelScope 下载到 `/usr-data/models` |
| CUDA 不可用 | 镜像不含 GPU PyTorch 或驱动异常 | 换 CUDA/PyTorch 镜像，先确认 `nvidia-smi` |
| `bitsandbytes` 报错 | CUDA 与 bitsandbytes 不匹配 | 换官方 PyTorch CUDA 镜像或重装 requirements |
| OOM | 上下文太长或模型太大 | `max_seq_len` 调到 1024，保持 `per_device_batch=1` |
| loss mask 报错 | assistant 模板与模型 chat template 不匹配 | 检查 `assistant_template` 是否适配当前模型 |
| 训练数据像模板 | 使用了 dry-run 数据 | 重新执行正式强模型造数 |

## 11. 训练后评估

```bash
uv run --active python fine_tuning/scripts/eval_before_after.py \
  --base /usr-data/models/Qwen2.5-3B-Instruct \
  --adapter fine_tuning/outputs/kb-sft
```

查看报告：

```bash
ls -lh fine_tuning/data/eval
```

重点看：

```text
eval_report.md
_eval_metrics.json
bad_cases.jsonl
```

只有阶段四评估确认 SFT 有收益后，才进入 vLLM LoRA 接回业务。

## 12. 最小成功标准

第一次跑通时，只要满足：

```text
1. sft_train.jsonl / sft_holdout.jsonl 存在且校验通过；
2. 基座模型在 /usr-data/models 下可读取；
3. train_sft.py --check-only 通过；
4. fine_tuning/outputs/kb-sft 下生成 adapter；
5. eval_before_after.py 能生成评估报告。
```

就算完成了微调项目的最小闭环。
