# 阶段三测试记录 - QLoRA 训练脚本

## 1. 测试目标

验证阶段三新增训练脚本在本地环境能完成静态检查：

```text
sft_train.jsonl
  -> train_sft.py --check-only
  -> merge_lora.py --check-only
```

本次测试不加载模型、不下载模型、不启动 GPU 训练。

## 2. 测试环境

```text
项目路径：/Users/bob/PycharmProjects/shopkeeper_brain
本地环境：当前开发环境
训练数据：fine_tuning/data/processed/sft_train.jsonl
输出目录：fine_tuning/outputs/kb-sft
```

## 3. 语法检查

测试命令：

```bash
uv run python -m py_compile fine_tuning/src/train_sft.py fine_tuning/src/merge_lora.py
```

验收标准：

```text
命令退出码为 0
```

实际结果：

```text
通过
```

## 4. train_sft.py check-only

测试命令：

```bash
uv run python fine_tuning/src/train_sft.py --check-only
```

验收标准：

```text
1. 能读取 sft_train.jsonl；
2. 能识别 train 配置；
3. 不加载模型；
4. 输出 [train-check] passed。
```

实际输出：

```text
[config] fine_tuning/configs/config.yaml not found, using config.example.yaml with environment fallbacks.
[train-check] train_path=/Users/bob/PycharmProjects/shopkeeper_brain/fine_tuning/data/processed/sft_train.jsonl
[train-check] rows=35
[train-check] base_model=Qwen/Qwen2.5-3B-Instruct
[train-check] output_dir=/Users/bob/PycharmProjects/shopkeeper_brain/fine_tuning/outputs/kb-sft
[train-check] max_seq_len=2048
[train-check] load_in_4bit=True
[train-check] passed
```

结论：

```text
通过
```

## 5. merge_lora.py check-only

测试命令：

```bash
uv run python fine_tuning/src/merge_lora.py --check-only
```

验收标准：

```text
1. 能识别 base_model；
2. 能识别 adapter_dir；
3. adapter 未生成时只提示，不失败；
4. 输出 [merge-check] passed。
```

实际输出：

```text
[config] fine_tuning/configs/config.yaml not found, using config.example.yaml with environment fallbacks.
[merge-check] base_model=Qwen/Qwen2.5-3B-Instruct
[merge-check] adapter_dir=/Users/bob/PycharmProjects/shopkeeper_brain/fine_tuning/outputs/kb-sft
[merge-check] output_dir=/Users/bob/PycharmProjects/shopkeeper_brain/fine_tuning/outputs/kb-sft-merged
[merge-check] adapter does not exist yet; run train_sft.py first.
[merge-check] passed
```

结论：

```text
通过
```

## 6. 正式训练前置条件

```text
1. 阶段二正式强模型造数完成；
2. validate_messages_dataset.py 硬错误为 0；
3. sft_train.jsonl 不是 dry-run stub 数据；
4. GPU 训练环境已安装 requirements-train.txt；
5. fine_tuning/outputs/ 确认不会进入 Git。
```
