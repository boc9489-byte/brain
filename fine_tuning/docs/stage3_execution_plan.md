# 阶段三执行计划 - QLoRA 训练闭环

## 1. 一句话定位

阶段三负责把阶段二验收通过的正式 messages SFT 数据，送入 QLoRA 训练流程，产出可评估的 LoRA adapter。

```text
阶段二解决“高质量训练数据从哪里来”。
阶段三解决“如何可复现地训练出 adapter”。
```

阶段三仍然不接线上业务，不改 RAG Answer Node，不部署 vLLM。

## 2. 背景与目标

### 2.1 背景

阶段二已经新增：

```text
expand_dataset.py
  -> sft_train.jsonl
  -> sft_holdout.jsonl
  -> validate_messages_dataset.py
```

阶段三只接受校验通过的数据进入训练。

### 2.2 阶段三目标

```text
1. 提供可复现的 QLoRA 训练脚本；
2. 只对 assistant 答案段计算 loss；
3. 支持 Qwen2.5/Qwen3 Instruct 类模型；
4. 输出 LoRA adapter 到 fine_tuning/outputs/；
5. 保存训练配置快照和训练摘要；
6. 提供 LoRA 合并脚本，供后续阶段四/五使用。
```

### 2.3 非目标

```text
1. 不在 Mac CPU 环境真实训练；
2. 训练脚本不负责预下载模型；
3. 不提交模型权重、adapter、checkpoint；
4. 不做 Base vs SFT 评估；
5. 不接 vLLM；
6. 不修改 knowledge 查询链路。
```

说明：阶段三训练必须能读取基座模型。`train.base_model` 可以是 HuggingFace 模型名，也可以是 GPU 本地路径；为了降低租用 GPU 上的网络失败风险，推荐提前下载到 `/usr-data/models/`，再把 `train.base_model` 改成本地路径。

## 3. 总体架构

```text
sft_train.jsonl
  -> validate_messages_dataset.py
  -> train_sft.py
       -> tokenizer.apply_chat_template
       -> assistant-only label mask
       -> 4-bit NF4 base model
       -> LoRA target modules
       -> SFTTrainer
  -> fine_tuning/outputs/kb-sft
       -> adapter_model.safetensors
       -> adapter_config.json
       -> tokenizer files
       -> training_summary.json

可选：
fine_tuning/outputs/kb-sft
  -> merge_lora.py
  -> fine_tuning/outputs/kb-sft-merged
```

## 4. 核心模块

| 模块 | 职责 | 输入 | 输出 | 依赖 |
|---|---|---|---|---|
| `train_sft.py` | QLoRA 训练入口 | `sft_train.jsonl`、config | LoRA adapter | torch、transformers、trl、peft、bitsandbytes |
| `merge_lora.py` | 合并 adapter 到 base | base model、adapter | merged model | transformers、peft |
| `requirements-train.txt` | 训练环境依赖 | 无 | 依赖清单 | uv |
| `config.example.yaml` | 训练参数模板 | 无 | train 配置 | YAML |
| `stage3_test_record.md` | 本地检查记录 | check 命令输出 | 验收记录 | 无 |

## 5. 数据流设计

### 5.1 主流程

```text
读取 config
  -> 定位 sft_train.jsonl
  -> 可选执行 --check-only
  -> 加载 tokenizer
  -> messages 渲染为模型 chat template 文本
  -> 4-bit 加载 base model
  -> 注入 LoRA 配置
  -> 使用 completion-only collator 做 assistant-only loss
  -> trainer.train()
  -> 保存 adapter / tokenizer / training_summary
```

### 5.2 check-only 流程

```text
读取 sft_train.jsonl
  -> 检查 messages 三段结构
  -> 检查样本数量
  -> 检查 train 配置
  -> 不加载模型、不下载模型、不启动训练
```

check-only 用于本地 Mac 或无 GPU 环境做提交前检查。

## 6. 数据模型

训练输入沿用阶段二 messages schema：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "【检索资料】...【问题】..."},
    {"role": "assistant", "content": "...[1]。"}
  ],
  "meta": {
    "id": "sft-train-000001",
    "type": "faithful",
    "battle_capabilities": ["faithful", "cite"]
  }
}
```

训练时使用 tokenizer 的 chat template 渲染文本，并只对 assistant 段计算 loss。

## 7. 配置设计

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `train.base_model` | `Qwen/Qwen2.5-3B-Instruct` | 第一版建议 3B 跑通 |
| `train.output_dir` | `fine_tuning/outputs/kb-sft` | LoRA adapter 输出目录 |
| `train.max_seq_len` | `2048` | 最大训练长度 |
| `train.load_in_4bit` | `true` | QLoRA NF4 |
| `train.lora_r` | `16` | LoRA rank |
| `train.lora_alpha` | `32` | LoRA alpha |
| `train.lora_dropout` | `0.05` | dropout |
| `train.learning_rate` | `0.0002` | 学习率 |
| `train.epochs` | `3` | 训练轮数 |
| `train.per_device_batch` | `1` | 单卡 batch |
| `train.grad_accum` | `16` | 梯度累积 |
| `train.assistant_template` | `<\|im_start\|>assistant\n` | labels mask 起点 |

## 8. 环境设计

建议单独创建 uv 训练环境，不污染本地导入/造数环境：

```bash
uv venv --python 3.10 .venv-kb-sft
source .venv-kb-sft/bin/activate
uv pip install -r fine_tuning/requirements-train.txt
```

训练环境建议：

```text
Linux + NVIDIA GPU
显存 3B：8-12GB 起步
显存 7B：16-24GB 起步
CUDA / PyTorch / bitsandbytes 版本匹配
```

### 8.1 基座模型准备

训练脚本执行到 `AutoTokenizer.from_pretrained` 和 `AutoModelForCausalLM.from_pretrained` 时会读取 `train.base_model`。

如果配置为模型名：

```yaml
train:
  base_model: "Qwen/Qwen2.5-3B-Instruct"
```

Transformers 会尝试从远程仓库下载。租用 GPU 环境容易遇到网络不稳定，推荐提前下载：

```bash
mkdir -p /usr-data/models
uv pip install -U huggingface_hub

uv run huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
  --local-dir /usr-data/models/Qwen2.5-3B-Instruct \
  --local-dir-use-symlinks False
```

或使用 ModelScope：

```bash
uv pip install -U modelscope

uv run modelscope download \
  --model Qwen/Qwen2.5-3B-Instruct \
  --local_dir /usr-data/models/Qwen2.5-3B-Instruct
```

然后把 `fine_tuning/configs/config.yaml` 改为：

```yaml
train:
  base_model: "/usr-data/models/Qwen2.5-3B-Instruct"
```

下载后检查：

```bash
ls -lh /usr-data/models/Qwen2.5-3B-Instruct
```

预期至少包含：

```text
config.json
tokenizer.json 或 tokenizer.model
model-*.safetensors
```

## 9. 验收命令

本地 check：

```bash
uv run python fine_tuning/src/train_sft.py --check-only
uv run python fine_tuning/src/merge_lora.py --check-only
```

GPU 训练：

```bash
uv run --active python fine_tuning/src/train_sft.py --config fine_tuning/configs/config.yaml
```

可选合并：

```bash
uv run --active python fine_tuning/src/merge_lora.py --config fine_tuning/configs/config.yaml
```

## 10. 阶段三通过标准

```text
1. train_sft.py --check-only 通过；
2. 训练环境依赖文件齐全；
3. GPU 环境可启动训练；
4. adapter 输出到 fine_tuning/outputs/kb-sft；
5. training_summary.json 记录训练配置；
6. adapter / checkpoint / merged model 不进入 Git；
7. 阶段四可读取 adapter 做 Base vs SFT 评估。
```

## 11. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 本地无 GPU | 无法训练 | 本地只跑 `--check-only` |
| 依赖版本漂移 | SFTTrainer API 报错 | 固定 `requirements-train.txt` |
| chat template 不匹配 | loss mask 失败 | 默认 Qwen assistant template，可配置覆盖 |
| 数据过长 | OOM 或截断严重 | 调 `max_seq_len`，回看 validate 长度告警 |
| 过度拒答 | 误拒率升高 | 阶段四评估 false_refusal |
| adapter 被误提交 | 仓库膨胀/泄露 | `.gitignore` 保护 outputs、safetensors、checkpoint |

## 12. 后续阶段入口

阶段三完成后，进入阶段四：

```text
Base model
  vs
Base model + LoRA adapter
  -> refusal_recall
  -> false_refusal
  -> citation_validity
  -> faithfulness
  -> answer_completeness
```
